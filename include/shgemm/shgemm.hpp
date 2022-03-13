#ifndef __SHEGMM_SHGEMM_HPP__
#define __SHEGMM_SHGEMM_HPP__
#include <cstdint>
#include <cuda_fp16.h>

namespace mtk {
namespace shgemm {
namespace detail {
typedef void (*kernel_func_t)(
			const std::size_t,
			const std::size_t,
			const std::size_t,
			const float,
			const float* const, const std::size_t,
			const half * const, const std::size_t,
			const float,
			float* const , const std::size_t
			);

struct kernel {
	unsigned smem_m, smem_n;
	unsigned num_blocks_filling;
	unsigned block_size;
	unsigned smem_size;

	kernel_func_t func;
};

enum kernel_level {
	P0 = 0, // small
	P1 = 1, // Large
	num_levels
};
}
enum operation_t {
	op_n,
	op_t
};

enum tc_t {
	fp16,
	tf32
};

struct shgemmHandle_t {
	cudaStream_t cuda_stream;

	detail::kernel fp16_nn_kernel[detail::num_levels];
	detail::kernel fp16_tn_kernel[detail::num_levels];
	detail::kernel fp16_nt_kernel[detail::num_levels];
	detail::kernel fp16_tt_kernel[detail::num_levels];
	detail::kernel tf32_nn_kernel[detail::num_levels];
	detail::kernel tf32_tn_kernel[detail::num_levels];
	detail::kernel tf32_nt_kernel[detail::num_levels];
	detail::kernel tf32_tt_kernel[detail::num_levels];
};

void create(shgemmHandle_t& handle);
void destroy(shgemmHandle_t& handle);

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
		float* const c_ptr, const std::size_t ldc,
		const tc_t compute_type
		);

} // namespace shgemm
} // namespace mtk
#endif
