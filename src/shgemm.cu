#include <shgemm/shgemm.hpp>
#include <cutf/cuda.hpp>
#include <cutf/cp_async.hpp>
#include <wmma_extension/tcec/tcec.hpp>
#include <cassert>
#include "shgemm_core.hpp"
#include "dmem_accessor.hpp"
#include "shgemm_pipeline_core.hpp"

namespace {
template <unsigned SIZE, unsigned BLOCK_SIZE>
__device__ void mem_fill_zero(
		float* const ptr
		) {
	for (unsigned i = 0; i < SIZE; i += BLOCK_SIZE) {
		ptr[i + threadIdx.x] = 0.f;
	}
}

template<
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	class A_DMEM_LOADER,
	class B_DMEM_LOADER,
	class C_DMEM_STORER,
	class SHGEMM_CORE,
	unsigned NUM_STAGES,
	unsigned NUM_UNROLLINGS,
	unsigned BLOCK_SIZE,
	class TC_T
	>
__global__ void shgemm_kernel(
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const float alpha,
		const float* const a_ptr, const std::size_t lda,
		const half * const b_ptr, const std::size_t ldb,
		const float beta,
		float* const c_ptr, const std::size_t ldc,
		float* const w_ptr = nullptr, const unsigned num_k_slices = 1
		) {

	extern __shared__ float smem[];
	float* const a_smem_ptr = smem;
	half * const b_smem_ptr = reinterpret_cast<half*>(a_smem_ptr + mtk::shgemm::device::get_A_smem_size<SMEM_M, SMEM_K, typename A_DMEM_LOADER::layout>::value * NUM_STAGES);

	mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, TC_T, void, mtk::shgemm::device::A_Policy<TC_T>> frag_c[(SMEM_M * SMEM_N) / (FRAG_M * FRAG_N) / (BLOCK_SIZE / mtk::shgemm::utils::warp_size)];

	const auto end_k = k * (blockIdx.y + 1) / num_k_slices;
	const auto start_k = k * blockIdx.y / num_k_slices;
	mtk::shgemm::device::shgemm_pipeline_core<
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		A_DMEM_LOADER,
		B_DMEM_LOADER,
		SHGEMM_CORE,
		NUM_STAGES,
		NUM_UNROLLINGS,
		BLOCK_SIZE,
		TC_T
		> pipeline_core;
	pipeline_core(
			m,
			n,
			end_k - start_k,
			a_ptr + mtk::shgemm::device::get_mem_index<typename A_DMEM_LOADER::layout>{}(0, start_k, lda), lda,
			b_ptr + mtk::shgemm::device::get_mem_index<typename B_DMEM_LOADER::layout>{}(start_k, 0, ldb), ldb,
			a_smem_ptr,
			b_smem_ptr,
			frag_c
			);

	// Store frag C to smem
	__syncthreads();
	constexpr unsigned num_submatrices = (SMEM_M / FRAG_M) * (SMEM_N / FRAG_N);
	float* const c_smem_ptr = smem;
	for (unsigned matrix_id_offset = 0; matrix_id_offset < num_submatrices; matrix_id_offset += BLOCK_SIZE / mtk::shgemm::utils::warp_size) {
		const unsigned matrix_id = matrix_id_offset + (threadIdx.x / mtk::shgemm::utils::warp_size);
		const unsigned matrix_id_m = matrix_id % (SMEM_M / FRAG_M);
		const unsigned matrix_id_n = matrix_id / (SMEM_M / FRAG_M);
		mtk::wmma::tcec::store_matrix_sync(
				c_smem_ptr + matrix_id_m * FRAG_M + matrix_id_n * FRAG_N * (SMEM_M + mtk::shgemm::device::C_smem_skew),
				frag_c[matrix_id_offset / (BLOCK_SIZE / mtk::shgemm::utils::warp_size)],
				SMEM_M + mtk::shgemm::device::C_smem_skew,
				nvcuda::wmma::mem_col_major);
	}

	// Store smem C to dmem
	__syncthreads();
	C_DMEM_STORER c_dmem_storer;
	if (beta == 0.0f || num_k_slices == 1) {
		c_dmem_storer(c_ptr, ldc,
				mtk::shgemm::device::get_m_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_M, mtk::shgemm::device::get_n_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_N,
				m, n,
				c_smem_ptr,
				alpha, beta);
	} else {
		c_dmem_storer(w_ptr, (beta == 0.0f ? ldc : m),
				mtk::shgemm::device::get_m_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_M, mtk::shgemm::device::get_n_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_N,
				m, n,
				c_smem_ptr,
				alpha, beta);
	}
}

__global__ void init_working_memory_kernel(
		float* const w_ptr,
		const unsigned m,
		const unsigned n,
		const std::size_t ld
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= m * n) {
		return;
	}
	const auto im = tid % m;
	const auto in = tid / m;
	w_ptr[im + in * ld] = 0.f;
}

void init_working_memory(
		float* const w_ptr,
		const unsigned m,
		const unsigned n,
		const std::size_t ld,
		cudaStream_t cuda_stream) {
	const std::size_t block_size = 256;
	const std::size_t size = m * n;
	init_working_memory_kernel<<<(size + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(w_ptr, m, n, ld);
}


template <class T>
constexpr unsigned size_of = 0;
template <> constexpr unsigned size_of<float> = 4;
template <> constexpr unsigned size_of<half > = 2;

template <unsigned SMEM_M, unsigned SMEM_N, unsigned SMEM_K, unsigned NUM_STAGES, class A_layout, class B_layout>
unsigned get_shared_memory_size_in_byte(
		) {
	return std::max(NUM_STAGES * (mtk::shgemm::device::get_A_smem_size<SMEM_M, SMEM_K, A_layout>::value) * size_of<float> +
		NUM_STAGES * (mtk::shgemm::device::get_B_smem_size<SMEM_K, SMEM_N, B_layout>::value) * size_of<half>,
		(SMEM_M + mtk::shgemm::device::C_smem_skew) * SMEM_N * size_of<float>);
}

template <mtk::shgemm::operation_t op, class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct loader_selector {};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct loader_selector<mtk::shgemm::op_n, T, SMEM_M, SMEM_N, BLOCK_SIZE> {
	using type = mtk::shgemm::device::dmem_loader_col_major<T, SMEM_M, SMEM_N, BLOCK_SIZE>;
};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct loader_selector<mtk::shgemm::op_t, T, SMEM_M, SMEM_N, BLOCK_SIZE> {
	using type = mtk::shgemm::device::dmem_loader_row_major<T, SMEM_M, SMEM_N, BLOCK_SIZE>;
};

template<
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned NUM_STAGES,
	unsigned NUM_UNROLLINGS,
	unsigned BLOCK_SIZE,
	class TC_T,
	mtk::shgemm::operation_t op_a, mtk::shgemm::operation_t op_b
	>
struct kernel_ptr {
	using A_DMEM_LOADER = typename loader_selector<op_a, float, SMEM_M, SMEM_K, BLOCK_SIZE>::type;
	using B_DMEM_LOADER = typename loader_selector<op_b, half , SMEM_K, SMEM_N, BLOCK_SIZE>::type;
	using C_DMEM_STORER = mtk::shgemm::device::dmem_storer_n<float, SMEM_M, SMEM_N, BLOCK_SIZE>;
	using SHGEMM_CORE = mtk::shgemm::device::shgemm_core<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, TC_T, typename A_DMEM_LOADER::layout, typename B_DMEM_LOADER::layout>;

	constexpr static mtk::shgemm::detail::kernel_func_t func =
				&(shgemm_kernel<
					SMEM_M, SMEM_N, SMEM_K,
					FRAG_M, FRAG_N, FRAG_K,
					A_DMEM_LOADER,
					B_DMEM_LOADER,
					C_DMEM_STORER,
					SHGEMM_CORE,
					NUM_STAGES,
					NUM_UNROLLINGS,
					BLOCK_SIZE,
					TC_T
					>);
};

template<
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned NUM_STAGES,
	unsigned NUM_UNROLLINGS,
	unsigned BLOCK_SIZE,
	class TC_T,
	mtk::shgemm::operation_t op_a, mtk::shgemm::operation_t op_b
	>
struct atomic_kernel_ptr {
	using A_DMEM_LOADER = typename loader_selector<op_a, float, SMEM_M, SMEM_K, BLOCK_SIZE>::type;
	using B_DMEM_LOADER = typename loader_selector<op_b, half , SMEM_K, SMEM_N, BLOCK_SIZE>::type;
	using C_DMEM_STORER = mtk::shgemm::device::dmem_atomic_storer_n<float, SMEM_M, SMEM_N, BLOCK_SIZE>;
	using SHGEMM_CORE = mtk::shgemm::device::shgemm_core<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, TC_T, typename A_DMEM_LOADER::layout, typename B_DMEM_LOADER::layout>;

	constexpr static mtk::shgemm::detail::kernel_func_t func =
				&(shgemm_kernel<
					SMEM_M, SMEM_N, SMEM_K,
					FRAG_M, FRAG_N, FRAG_K,
					A_DMEM_LOADER,
					B_DMEM_LOADER,
					C_DMEM_STORER,
					SHGEMM_CORE,
					NUM_STAGES,
					NUM_UNROLLINGS,
					BLOCK_SIZE,
					TC_T
					>);
};

template<
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned NUM_STAGES,
	unsigned NUM_UNROLLINGS,
	unsigned BLOCK_SIZE,
	class TC_T,
	mtk::shgemm::operation_t op_a, mtk::shgemm::operation_t op_b
	>
struct pipline_kernel_ptr {
	using A_DMEM_LOADER = typename loader_selector<op_a, float, SMEM_M, SMEM_K, BLOCK_SIZE>::type;
	using B_DMEM_LOADER = typename loader_selector<op_b, half , SMEM_K, SMEM_N, BLOCK_SIZE>::type;
	using C_DMEM_STORER = mtk::shgemm::device::dmem_storer_n<float, SMEM_M, SMEM_N, BLOCK_SIZE>;
	using SHGEMM_CORE = mtk::shgemm::device::shgemm_core_pipeline<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, TC_T, typename A_DMEM_LOADER::layout, typename B_DMEM_LOADER::layout>;

	constexpr static mtk::shgemm::detail::kernel_func_t func =
				&(shgemm_kernel<
					SMEM_M, SMEM_N, SMEM_K,
					FRAG_M, FRAG_N, FRAG_K,
					A_DMEM_LOADER,
					B_DMEM_LOADER,
					C_DMEM_STORER,
					SHGEMM_CORE,
					NUM_STAGES,
					NUM_UNROLLINGS,
					BLOCK_SIZE,
					TC_T
					>);
};

template<
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned NUM_STAGES,
	unsigned NUM_UNROLLINGS,
	unsigned BLOCK_SIZE,
	class TC_T,
	mtk::shgemm::operation_t op_a, mtk::shgemm::operation_t op_b
	>
struct atomic_pipline_kernel_ptr {
	using A_DMEM_LOADER = typename loader_selector<op_a, float, SMEM_M, SMEM_K, BLOCK_SIZE>::type;
	using B_DMEM_LOADER = typename loader_selector<op_b, half , SMEM_K, SMEM_N, BLOCK_SIZE>::type;
	using C_DMEM_STORER = mtk::shgemm::device::dmem_atomic_storer_n<float, SMEM_M, SMEM_N, BLOCK_SIZE>;
	using SHGEMM_CORE = mtk::shgemm::device::shgemm_core_pipeline<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, TC_T, typename A_DMEM_LOADER::layout, typename B_DMEM_LOADER::layout>;

	constexpr static mtk::shgemm::detail::kernel_func_t func =
				&(shgemm_kernel<
					SMEM_M, SMEM_N, SMEM_K,
					FRAG_M, FRAG_N, FRAG_K,
					A_DMEM_LOADER,
					B_DMEM_LOADER,
					C_DMEM_STORER,
					SHGEMM_CORE,
					NUM_STAGES,
					NUM_UNROLLINGS,
					BLOCK_SIZE,
					TC_T
					>);
};

template <mtk::shgemm::operation_t op_t>
struct op_t2layout {using type = void;};
template <> struct op_t2layout<mtk::shgemm::op_n> {using type = mtk::shgemm::utils::col_major;};
template <> struct op_t2layout<mtk::shgemm::op_t> {using type = mtk::shgemm::utils::row_major;};

template<
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned NUM_STAGES,
	unsigned NUM_UNROLLINGS,
	unsigned BLOCK_SIZE,
	class TC_T,
	mtk::shgemm::operation_t op_a, mtk::shgemm::operation_t op_b,
	unsigned USE_PIPELINE_CORE,
	unsigned USE_ATOMIC_STORER
	>
void set_kernel(
		mtk::shgemm::detail::kernel& kernel,
		const unsigned num_sm
		) {
	mtk::shgemm::detail::kernel_func_t func;
	if constexpr (USE_ATOMIC_STORER) {
		if constexpr (USE_PIPELINE_CORE) {
			func = atomic_pipline_kernel_ptr<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, 2, NUM_UNROLLINGS, BLOCK_SIZE, TC_T, op_a, op_b>::func;
		} else {
			func = atomic_kernel_ptr<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, 2, NUM_UNROLLINGS, BLOCK_SIZE, TC_T, op_a, op_b>::func;
		}
	} else {
		if constexpr (USE_PIPELINE_CORE) {
			func = pipline_kernel_ptr<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, 2, NUM_UNROLLINGS, BLOCK_SIZE, TC_T, op_a, op_b>::func;
		} else {
			func = kernel_ptr<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, 2, NUM_UNROLLINGS, BLOCK_SIZE, TC_T, op_a, op_b>::func;
		}
	}
	const unsigned smem_size = get_shared_memory_size_in_byte<SMEM_M, SMEM_N, SMEM_K, NUM_STAGES, typename op_t2layout<op_a>::type, typename op_t2layout<op_b>::type>();

	int max_block_per_ms;
	CUTF_CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_block_per_ms, *func, BLOCK_SIZE, smem_size));

	CUTF_CHECK_ERROR(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	kernel.func = func;
	kernel.num_blocks_filling = std::max(1, max_block_per_ms) * num_sm;
	kernel.smem_m = SMEM_M;
	kernel.smem_n = SMEM_N;
	kernel.smem_k = SMEM_K;
	kernel.block_size = BLOCK_SIZE;
	kernel.smem_size = smem_size;
}
} // noname namespace

void mtk::shgemm::create(
		mtk::shgemm::shgemmHandle_t &handle
		) {
	handle.cuda_stream = 0;
	handle.fixed_lernel_level = detail::num_levels;

	const auto debug_env = getenv("MTK_SHGEMM_DEBUG");
	if (debug_env != nullptr && debug_env[0] == '1') {
		handle.debug_mode = 1;
	}
	const auto kernel_level_env = getenv("MTK_SHGEMM_KERNEL_LEVEL");
	if (kernel_level_env != nullptr && (kernel_level_env[0] == '0' || kernel_level_env[0] == '1') && kernel_level_env[1] == '\0') {
		handle.fixed_lernel_level = static_cast<mtk::shgemm::detail::kernel_level>(std::stoul(kernel_level_env));
	}

	// working memory for small-mn/large k GEMM
	cudaMalloc(reinterpret_cast<void**>(&handle.w_ptr), handle.max_working_memory_num_elements * sizeof(float));

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	const unsigned num_sm = prop.multiProcessorCount;
	/*=======================================
		TF32-NN
		=====================================*/
	{
		constexpr unsigned BLOCK_SIZE = 128;
		constexpr unsigned SMEM_M = 32, SMEM_N = 32, SMEM_K = 128;
		constexpr unsigned FRAG_M = 16, FRAG_N = 16, FRAG_K = 128;
		constexpr unsigned USE_PIPELINE_CORE = 0;
		constexpr unsigned NUM_UNROLLINGS = 4;

		using TC_T = nvcuda::wmma::precision::tf32;
		constexpr auto OP_A = mtk::shgemm::op_n;
		constexpr auto OP_B = mtk::shgemm::op_n;
		constexpr unsigned USE_ATOMIC_STORER = 0;

		auto& kernel = handle.tf32_nn_kernel[mtk::shgemm::detail::P0];

		set_kernel<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, 2, NUM_UNROLLINGS, BLOCK_SIZE, TC_T, OP_A, OP_B, USE_PIPELINE_CORE, USE_ATOMIC_STORER>(
				kernel, num_sm
				);
	}
	{
		constexpr unsigned BLOCK_SIZE = 128;
		constexpr unsigned SMEM_M = 32, SMEM_N = 64, SMEM_K = 64;
		constexpr unsigned FRAG_M = 16, FRAG_N = 32, FRAG_K = 32;
		constexpr unsigned USE_PIPELINE_CORE = 1;
		constexpr unsigned NUM_UNROLLINGS = 4;

		using TC_T = nvcuda::wmma::precision::tf32;
		constexpr auto OP_A = mtk::shgemm::op_n;
		constexpr auto OP_B = mtk::shgemm::op_n;
		constexpr unsigned USE_ATOMIC_STORER = 0;

		auto& kernel = handle.tf32_nn_kernel[mtk::shgemm::detail::P1];

		set_kernel<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, 2, NUM_UNROLLINGS, BLOCK_SIZE, TC_T, OP_A, OP_B, USE_PIPELINE_CORE, USE_ATOMIC_STORER>(
				kernel, num_sm
				);
	}
	{
		constexpr unsigned BLOCK_SIZE = 128;
		constexpr unsigned SMEM_M = 64, SMEM_N = 64, SMEM_K = 64;
		constexpr unsigned FRAG_M = 32, FRAG_N = 32, FRAG_K = 64;
		constexpr unsigned USE_PIPELINE_CORE = 0;
		constexpr unsigned NUM_UNROLLINGS = 4;

		using TC_T = nvcuda::wmma::precision::tf32;
		constexpr auto OP_A = mtk::shgemm::op_n;
		constexpr auto OP_B = mtk::shgemm::op_n;
		constexpr unsigned USE_ATOMIC_STORER = 1;

		auto& kernel = handle.tf32_nn_k_slicing_kernel;

		set_kernel<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, 2, NUM_UNROLLINGS, BLOCK_SIZE, TC_T, OP_A, OP_B, USE_PIPELINE_CORE, USE_ATOMIC_STORER>(
				kernel, num_sm
				);
	}
	/*=======================================
		FP16-NN
		=====================================*/
	{
		constexpr unsigned BLOCK_SIZE = 128;
		constexpr unsigned SMEM_M = 32, SMEM_N = 64, SMEM_K = 128;
		constexpr unsigned FRAG_M = 16, FRAG_N = 32, FRAG_K = 128;
		constexpr unsigned USE_PIPELINE_CORE = 0;
		constexpr unsigned NUM_UNROLLINGS = 4;

		using TC_T = half;
		constexpr auto OP_A = mtk::shgemm::op_n;
		constexpr auto OP_B = mtk::shgemm::op_n;
		constexpr unsigned USE_ATOMIC_STORER = 0;

		auto& kernel = handle.fp16_nn_kernel[mtk::shgemm::detail::P0];

		set_kernel<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, 2, NUM_UNROLLINGS, BLOCK_SIZE, TC_T, OP_A, OP_B, USE_PIPELINE_CORE, USE_ATOMIC_STORER>(
				kernel, num_sm
				);
	}
	{
		constexpr unsigned BLOCK_SIZE = 128;
		constexpr unsigned SMEM_M = 32, SMEM_N = 64, SMEM_K = 128;
		constexpr unsigned FRAG_M = 16, FRAG_N = 32, FRAG_K = 128;
		constexpr unsigned USE_PIPELINE_CORE = 0;
		constexpr unsigned NUM_UNROLLINGS = 4;

		using TC_T = half;
		constexpr auto OP_A = mtk::shgemm::op_n;
		constexpr auto OP_B = mtk::shgemm::op_n;
		constexpr unsigned USE_ATOMIC_STORER = 0;

		auto& kernel = handle.fp16_nn_kernel[mtk::shgemm::detail::P1];

		set_kernel<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, 2, NUM_UNROLLINGS, BLOCK_SIZE, TC_T, OP_A, OP_B, USE_PIPELINE_CORE, USE_ATOMIC_STORER>(
				kernel, num_sm
				);
	}
	{
		constexpr unsigned BLOCK_SIZE = 128;
		constexpr unsigned SMEM_M = 64, SMEM_N = 64, SMEM_K = 64;
		constexpr unsigned FRAG_M = 32, FRAG_N = 32, FRAG_K = 64;
		constexpr unsigned USE_PIPELINE_CORE = 0;
		constexpr unsigned NUM_UNROLLINGS = 4;

		using TC_T = half;
		constexpr auto OP_A = mtk::shgemm::op_n;
		constexpr auto OP_B = mtk::shgemm::op_n;
		constexpr unsigned USE_ATOMIC_STORER = 1;

		auto& kernel = handle.fp16_nn_k_slicing_kernel;

		set_kernel<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, 2, NUM_UNROLLINGS, BLOCK_SIZE, TC_T, OP_A, OP_B, USE_PIPELINE_CORE, USE_ATOMIC_STORER>(
				kernel, num_sm
				);
	}
}

void mtk::shgemm::destroy(
		mtk::shgemm::shgemmHandle_t &handle
		) {
	cudaFree(handle.w_ptr);
}

void mtk::shgemm::set_cuda_stream(
		mtk::shgemm::shgemmHandle_t &handle,
		cudaStream_t const cuda_stream
		) {
	handle.cuda_stream = cuda_stream;
}

void mtk::shgemm::enable_kernel_level_fixing(
		mtk::shgemm::shgemmHandle_t &handle,
		const mtk::shgemm::detail::kernel_level kernel_level) {
	handle.fixed_lernel_level = kernel_level;
}

void mtk::shgemm::disable_kernel_level_fixing(
		mtk::shgemm::shgemmHandle_t& handle
		) {
	handle.fixed_lernel_level = mtk::shgemm::detail::num_levels;
}

mtk::shgemm::detail::kernel_level mtk::shgemm::shgemm(
		const mtk::shgemm::shgemmHandle_t handle,
		const mtk::shgemm::operation_t op_a,
		const mtk::shgemm::operation_t op_b,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const float* const alpha_ptr,
		const float* const a_ptr, const std::size_t lda,
		const half * const b_ptr, const std::size_t ldb,
		const float* const beta_ptr,
		float* const c_ptr, const std::size_t ldc,
		const tc_t compute_type
		) {
	if (m * n <= handle.max_working_memory_num_elements && k >= 4096 && *beta_ptr == 0.f) {
		mtk::shgemm::detail::kernel kernel;
		if (op_a == mtk::shgemm::op_n && op_b == mtk::shgemm::op_n) {
			if (compute_type == mtk::shgemm::fp16) {
				kernel = handle.fp16_nn_k_slicing_kernel;
			} else if (compute_type == mtk::shgemm::tf32) {
				kernel = handle.tf32_nn_k_slicing_kernel;
			}
		}

		const auto num_blocks = ((m + kernel.smem_m - 1) / kernel.smem_m) * ((n + kernel.smem_n - 1) / kernel.smem_n);
		unsigned num_k_slices = 1;
		for (;num_k_slices <= k / kernel.smem_k; num_k_slices <<= 1) {
			if (num_blocks * num_k_slices >= kernel.num_blocks_filling * (2048 / kernel.smem_k)) {
				break;
			}
		}

		float* w_ptr;
		std::size_t ldw;
		if (*beta_ptr == 0.0f) {
			w_ptr = c_ptr;
			ldw = ldc;
		} else {
			w_ptr = handle.w_ptr;
			ldw = m;
		}
		init_working_memory(w_ptr, m, n, ldw, handle.cuda_stream);

		if (handle.debug_mode) {
			std::printf("[shape=(%lu,%lu,%lu),op_a=%s,op_b=%s,compute_type=%s] kernel_ptr = %p, num_l_slicing = %u\n",
					m, n, k,
					op_a == mtk::shgemm::op_n ? "N" : "T",
					op_b == mtk::shgemm::op_n ? "N" : "T",
					compute_type == mtk::shgemm::tf32 ? "TF32" : "FP16",
					kernel.func,
					num_k_slices
					);
		}

		const dim3 grid_size((n + kernel.smem_n - 1) / kernel.smem_n * (m + kernel.smem_m - 1) / kernel.smem_m, num_k_slices);
		const dim3 block_size(kernel.block_size);

		kernel.func<<<grid_size, block_size, kernel.smem_size, handle.cuda_stream>>>
			(
			 m, n, k,
			 *alpha_ptr,
			 a_ptr, lda,
			 b_ptr, ldb,
			 *beta_ptr,
			 c_ptr, ldc,
			 w_ptr, num_k_slices
			);
		// TODO:
		// axpy
		return static_cast<mtk::shgemm::detail::kernel_level>(mtk::shgemm::detail::P0);

	} else {
		mtk::shgemm::detail::kernel* kernel_list = nullptr;
		if (op_a == mtk::shgemm::op_n && op_b == mtk::shgemm::op_n) {
			if (compute_type == mtk::shgemm::fp16) {
				kernel_list = (mtk::shgemm::detail::kernel*)handle.fp16_nn_kernel;
			} else if (compute_type == mtk::shgemm::tf32) {
				kernel_list = (mtk::shgemm::detail::kernel*)handle.tf32_nn_kernel;
			}
		}

		unsigned kernel_level = mtk::shgemm::detail::num_levels - 1;
		if (handle.fixed_lernel_level >= detail::num_levels) {
			kernel_level = mtk::shgemm::detail::num_levels - 1;
			for (; kernel_level > 0; kernel_level--) {
				const auto kernel = kernel_list[kernel_level];
				const auto num_blocks = ((m + kernel.smem_m - 1) / kernel.smem_m) * ((n + kernel.smem_n - 1) / kernel.smem_n);
				if (num_blocks >= kernel.num_blocks_filling * 2) {
					break;
				}
			}
		} else {
			kernel_level = handle.fixed_lernel_level;
		}
		auto kernel = kernel_list[kernel_level];

		if (handle.debug_mode) {
			std::printf("[shape=(%lu,%lu,%lu),op_a=%s,op_b=%s,compute_type=%s] kernel_ptr = %p, kernel_level = %u\n",
					m, n, k,
					op_a == mtk::shgemm::op_n ? "N" : "T",
					op_b == mtk::shgemm::op_n ? "N" : "T",
					compute_type == mtk::shgemm::tf32 ? "TF32" : "FP16",
					kernel.func,
					kernel_level
					);
		}

		const dim3 grid_size((n + kernel.smem_n - 1) / kernel.smem_n * (m + kernel.smem_m - 1) / kernel.smem_m);
		const dim3 block_size(kernel.block_size);

		kernel.func<<<grid_size, block_size, kernel.smem_size, handle.cuda_stream>>>
			(
			 m, n, k,
			 *alpha_ptr,
			 a_ptr, lda,
			 b_ptr, ldb,
			 *beta_ptr,
			 c_ptr, ldc,
			 nullptr, 1
			);
		return static_cast<mtk::shgemm::detail::kernel_level>(kernel_level);
	}
	return mtk::shgemm::detail::num_levels;
}

void mtk::shgemm::set_debug_mode(mtk::shgemm::shgemmHandle_t& handle, const unsigned on) {
	handle.debug_mode = on;
}
