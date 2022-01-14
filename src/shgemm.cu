#include <shgemm/shgemm.hpp>
#include <wmma_extension/tcec/tcec.hpp>
#include <cassert>
#include "wmmae_shgemm.hpp"

namespace {
constexpr unsigned warp_size = 32;

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
	unsigned BLOCK_SIZE,
	class TC_T
	>
__device__ void shgemm_core(
		const float* const a_ptr,
		const half * const b_ptr,
		float* const c_ptr
		) {
	constexpr unsigned num_submatrices = (SMEM_M / FRAG_M) * (SMEM_N / FRAG_N);
	static_assert(num_submatrices * warp_size % BLOCK_SIZE == 0, "the number of reg-level sub matrices must be a multiple of (BLOCK_SIZE / warp_size)");

	using A_Policy = typename mtk::wmma::tcec::detail::default_policy<TC_T, mtk::wmma::tcec::op_with_error_correction   , mtk::wmma::tcec::op_mma>::type;
	using B_Policy = typename mtk::wmma::tcec::detail::default_policy<TC_T, mtk::wmma::tcec::op_without_error_correction, mtk::wmma::tcec::op_mma>::type;

	for (unsigned matrix_id_offset = 0; matrix_id_offset < num_submatrices; matrix_id_offset += BLOCK_SIZE / warp_size) {
		const unsigned matrix_id = matrix_id_offset + (threadIdx.x / warp_size);
		const unsigned matrix_id_m = matrix_id_m % (SMEM_M / FRAG_M);
		const unsigned matrix_id_n = matrix_id_m / (SMEM_M / FRAG_M);

		mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, TC_T, void, A_Policy> frag_c;
		mtk::wmma::tcec::fill_zero(frag_c);

		for (unsigned k = 0; k < SMEM_K; k += FRAG_K) {
			mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, TC_T, nvcuda::wmma::row_major, A_Policy> frag_a;
			mtk::wmma::tcec::load_matrix_sync(frag_a, a_ptr + matrix_id_m * FRAG_M * SMEM_K + k, SMEM_K);

			mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, TC_T, nvcuda::wmma::col_major, B_Policy> frag_b;
			mtk::wmma::tcec::load_matrix_sync(frag_b, b_ptr + matrix_id_n * FRAG_N * SMEM_K + k, SMEM_K);

			mtk::shgemm::mma_sync(frag_c, frag_a, frag_b, frag_c);
		}

		mtk::wmma::tcec::store_matrix_sync(c_ptr + matrix_id_m * SMEM_N * FRAG_M + matrix_id_n * FRAG_N, frag_c, SMEM_N, nvcuda::wmma::mem_col_major);
	}
}

template <class T, unsigned SMEM_M, unsigned SMEM_N>
struct dmem_loader_n {
	__device__ void operator()(
			T* const smem_ptr, const std::size_t lds,
			const std::size_t dmem_start_m, const std::size_t dmem_start_n,
			const std::size_t dmem_size_m, const std::size_t dmem_size_n,
			const T* const dmem_ptr, const std::size_t ldd
			) {

	}
};

template <class T, unsigned SMEM_M, unsigned SMEM_N>
struct dmem_storer_n {
	__device__ void operator()(
			T* const dmem_ptr, const std::size_t ldd,
			const std::size_t dmem_start_m, const std::size_t dmem_start_n,
			const std::size_t dmem_size_m, const std::size_t dmem_size_n,
			const T* const smem_ptr, const std::size_t lds,
			const float alpha, const float beta
			) {

	}
};

template<
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	class A_DMEM_LOADER,
	class B_DMEM_LOADER
	>
__global__ void shgemm_kernel(
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const float* const alpha_ptr,
		const float* const a_ptr, const std::size_t lda,
		const half * const b_ptr, const std::size_t ldb,
		const float* const beta_ptr,
		const float* const c_ptr, const std::size_t ldc
		) {
	constexpr unsigned NUM_STAGES = 2;

	extern __shared__ float smem[];
	float* const a_smem_ptr = smem;
	float* const c_smem_ptr = smem + NUM_STAGES * SMEM_M * SMEM_K;
	half * const b_smem_ptr = reinterpret_cast<half*>(c_smem_ptr + SMEM_M * SMEM_N);

	A_DMEM_LOADER a_dram_loader;
	B_DMEM_LOADER b_dram_loader;
}

template <class T>
constexpr unsigned size_of = 0;
template <> constexpr unsigned size_of<float> = 4;
template <> constexpr unsigned size_of<half > = 2;

constexpr unsigned get_shared_memory_size_in_byte(
		const unsigned NUM_STAGES,
		const unsigned SMEM_M,
		const unsigned SMEM_N,
		const unsigned SMEM_K
		) {
	return NUM_STAGES * SMEM_M * SMEM_K * size_of<float> +
		NUM_STAGES * SMEM_K * SMEM_N * size_of<half> +
		SMEM_M * SMEM_N * size_of<float>;
}

void shgemm_tn(
		const mtk::shgemm::shgemmHandle_t handle,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const float* const alpha_ptr,
		const float* const a_ptr, const std::size_t lda,
		const half * const b_ptr, const std::size_t ldb,
		const float* const beta_ptr,
		const float* const c_ptr, const std::size_t ldc
		) {
	constexpr unsigned NUM_STAGES = 2;
	constexpr unsigned SMEM_M = 128;
	constexpr unsigned SMEM_N = 128;
	constexpr unsigned SMEM_K = 128;
	constexpr unsigned FRAG_M = 128;
	constexpr unsigned FRAG_N = 128;
	constexpr unsigned FRAG_K = 128;

	constexpr auto smem_size = get_shared_memory_size_in_byte(NUM_STAGES, SMEM_M, SMEM_N, SMEM_K);
	const dim3 grid_size(1);
	const dim3 block_size(1);

	shgemm_kernel<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, dmem_loader_n<float, SMEM_K, SMEM_M>, dmem_loader_n<half, SMEM_K, SMEM_N>>
		<<<grid_size, block_size, smem_size, handle.cuda_stream>>>
		(
		 m, n, k,
		 alpha_ptr,
		 a_ptr, lda,
		 b_ptr, ldb,
		 beta_ptr,
		 c_ptr, ldc
		 );
}
} // noname namespace

void mtk::shgemm::set_cuda_stream(
		mtk::shgemm::shgemmHandle_t &handle,
		cudaStream_t const cuda_stream
		) {
	handle.cuda_stream = cuda_stream;
}

void mtk::shgemm::shgemm(
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
		const float* const c_ptr, const std::size_t ldc
		) {
	if (op_a == mtk::shgemm::op_t && op_b == mtk::shgemm::op_n) {
		shgemm_tn(
				handle,
				m, n, k,
				alpha_ptr,
				a_ptr, lda,
				b_ptr, ldb,
				beta_ptr,
				c_ptr, ldc
				);
	}
}
