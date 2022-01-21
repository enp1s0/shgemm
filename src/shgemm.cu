#include <shgemm/shgemm.hpp>
#include <cutf/cuda.hpp>
#include <cutf/cp_async.hpp>
#include <wmma_extension/tcec/tcec.hpp>
#include <cassert>
#include "shgemm_core.hpp"
#include "dmem_accessor.hpp"

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
	unsigned BLOCK_SIZE,
	class TC_T
	>
__global__ void shgemm_kernel(
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const float alpha,
		const float* const a_ptr, const std::size_t lda,
		const half * const b_ptr, const std::size_t ldb,
		const float beta,
		float* const c_ptr, const std::size_t ldc
		) {
	constexpr unsigned NUM_STAGES = 2;

	extern __shared__ float smem[];
	float* const a_smem_ptr = smem;
	float* const c_smem_ptr = smem + NUM_STAGES * SMEM_M * SMEM_K;
	half * const b_smem_ptr = reinterpret_cast<half*>(c_smem_ptr + SMEM_M * SMEM_N);

	mem_fill_zero<SMEM_M * SMEM_N, BLOCK_SIZE>(c_smem_ptr);

	A_DMEM_LOADER a_dram_loader;
	B_DMEM_LOADER b_dram_loader;
	SHGEMM_CORE shgemm_core;

	std::size_t block_k = 0;
	a_dram_loader(a_smem_ptr,
			block_k, blockIdx.y * SMEM_M,
			k, m,
			a_ptr, lda
			);
	b_dram_loader(b_smem_ptr,
			block_k, blockIdx.x * SMEM_N,
			k, n,
			b_ptr, ldb
			);
	block_k += SMEM_K;
	cutf::cp_async::wait_all();
	__syncthreads();

	for (; block_k < k; block_k += SMEM_K) {
		a_dram_loader(a_smem_ptr + ((block_k / SMEM_K) & 0x1) * SMEM_K * SMEM_M,
				block_k, blockIdx.y * SMEM_M,
				k, m,
				a_ptr, lda
				);
		b_dram_loader(b_smem_ptr + ((block_k / SMEM_K) & 0x1) * SMEM_K * SMEM_N,
				block_k, blockIdx.x * SMEM_N,
				k, n,
				b_ptr, ldb
				);

		shgemm_core(c_smem_ptr,
				a_smem_ptr + (1 - ((block_k / SMEM_K) & 0x1)) * SMEM_K * SMEM_M,
				b_smem_ptr + (1 - ((block_k / SMEM_K) & 0x1)) * SMEM_K * SMEM_N
				);
	cutf::cp_async::wait_all();
		__syncthreads();
	}

	shgemm_core(c_smem_ptr,
			a_smem_ptr + (1 - ((block_k / SMEM_K) & 0x1)) * SMEM_K * SMEM_M,
			b_smem_ptr + (1 - ((block_k / SMEM_K) & 0x1)) * SMEM_K * SMEM_N
			);

	__syncthreads();
	C_DMEM_STORER c_dmem_storer;
	c_dmem_storer(c_ptr, ldc,
			blockIdx.y * SMEM_M, blockIdx.x * SMEM_N,
			m, n,
			c_smem_ptr,
			alpha, beta);
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
		float* const c_ptr, const std::size_t ldc
		) {
	constexpr unsigned NUM_STAGES = 2;
	constexpr unsigned SMEM_M = 128;
	constexpr unsigned SMEM_N = 64;
	constexpr unsigned SMEM_K = 32;
	constexpr unsigned FRAG_M = 32;
	constexpr unsigned FRAG_N = 64;
	constexpr unsigned FRAG_K = 16;
	constexpr unsigned BLOCK_SIZE = 128;
	using TC_T = half;

	constexpr auto smem_size = get_shared_memory_size_in_byte(NUM_STAGES, SMEM_M, SMEM_N, SMEM_K);
	const dim3 grid_size((n + SMEM_N - 1) / SMEM_N, (m + SMEM_M - 1) / SMEM_M);
	const dim3 block_size(BLOCK_SIZE);

	CUTF_CHECK_ERROR(cudaFuncSetAttribute(
				&(shgemm_kernel<
					SMEM_M, SMEM_N, SMEM_K,
					FRAG_M, FRAG_N, FRAG_K,
					mtk::shgemm::device::dmem_loader_n<float, SMEM_K, SMEM_M, BLOCK_SIZE>,
					mtk::shgemm::device::dmem_loader_n<half , SMEM_K, SMEM_N, BLOCK_SIZE>,
					mtk::shgemm::device::dmem_storer_n<float, SMEM_M, SMEM_N, BLOCK_SIZE>,
					mtk::shgemm::device::shgemm_core<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, TC_T>,
					BLOCK_SIZE,
					TC_T
					>)
				, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

	shgemm_kernel<
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		mtk::shgemm::device::dmem_loader_n<float, SMEM_K, SMEM_M, BLOCK_SIZE>,
		mtk::shgemm::device::dmem_loader_n<half , SMEM_K, SMEM_N, BLOCK_SIZE>,
		mtk::shgemm::device::dmem_storer_n<float, SMEM_M, SMEM_N, BLOCK_SIZE>,
		mtk::shgemm::device::shgemm_core<SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, TC_T>,
		BLOCK_SIZE,
		TC_T
	>
		<<<grid_size, block_size, smem_size, handle.cuda_stream>>>
		(
		 m, n, k,
		 *alpha_ptr,
		 a_ptr, lda,
		 b_ptr, ldb,
		 *beta_ptr,
		 c_ptr, ldc
		 );
}
} // noname namespace

void mtk::shgemm::create(
		mtk::shgemm::shgemmHandle_t &handle
		) {
	handle.cuda_stream = 0;
}

void mtk::shgemm::destroy(
		mtk::shgemm::shgemmHandle_t &handle
		) {
}

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
		float* const c_ptr, const std::size_t ldc
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
