#ifndef __SHEGMM_SHGEMM_HPP__
#define __SHEGMM_SHGEMM_HPP__
#include <cstdint>
#include <cuda_fp16.h>

namespace mtk {
namespace shgemm {
enum operation_t {
	op_n,
	op_t
};

struct shgemmHandle_t {
	cudaStream_t cuda_stream;
};

void set_cuda_stream(shgemmHandle_t& handle, cudaStream_t const cuda_stream);

// main function
void shgemm(
		const shgemmHandle_t handle,
		const operation_t op_a,
		const operation_t op_b,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const float* const alpha_ptr,
		const float* const a_ptr, const std::size_t lda,
		const half * const b_ptr, const std::size_t ldb,
		const float* const beta_ptr,
		float* const c_ptr, const std::size_t ldc
		);

} // namespace shgemm
} // namespace mtk
#endif
