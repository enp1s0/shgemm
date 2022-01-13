#include <shgemm/shgemm.hpp>

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
		const float* const c_ptr, const std::size_t ldv
		) {

}
