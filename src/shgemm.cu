#include <shgemm/shgemm.hpp>

namespace {
template <class T, unsigned SMEM_M, unsigned SMEM_N>
struct dmem_loader_n {
	__device__ void operator()(
			T* const smem_ptr, const std::size_t lds,
			const std::size_t dmem_m, const std::size_t dmem_n,
			const T* const dmem_ptr, const std::size_t ldd
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
