#ifndef __SHEGMM_SHGEMM_HPP__
#define __SHEGMM_SHGEMM_HPP__
#include <cstdint>
#include <cuda_fp16.h>

namespace mtk {
namespace shgemm {
namespace detail {
typedef void (*kernel_func_t)(
			const unsigned,
			const unsigned,
			const unsigned,
			const float,
			const float* const, const unsigned,
			const half * const, const unsigned,
			const float,
			float* const, const unsigned,
			float* const, const unsigned
			);

struct kernel {
	unsigned smem_m, smem_n, smem_k;
	unsigned num_blocks_filling = 0xffffffffu;
	unsigned block_size;
	unsigned smem_size;

	kernel_func_t func;
};

enum kernel_level {
	P0 = 0, // Small
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

	detail::kernel fp16_nn_k_slicing_kernel;
	detail::kernel fp16_tn_k_slicing_kernel;
	detail::kernel fp16_nt_k_slicing_kernel;
	detail::kernel fp16_tt_k_slicing_kernel;
	detail::kernel tf32_nn_k_slicing_kernel;
	detail::kernel tf32_tn_k_slicing_kernel;
	detail::kernel tf32_nt_k_slicing_kernel;
	detail::kernel tf32_tt_k_slicing_kernel;

	// Transposed C
	detail::kernel fp16_nn_t_kernel[detail::num_levels];
	detail::kernel fp16_tn_t_kernel[detail::num_levels];
	detail::kernel fp16_nt_t_kernel[detail::num_levels];
	detail::kernel fp16_tt_t_kernel[detail::num_levels];
	detail::kernel tf32_nn_t_kernel[detail::num_levels];
	detail::kernel tf32_tn_t_kernel[detail::num_levels];
	detail::kernel tf32_nt_t_kernel[detail::num_levels];
	detail::kernel tf32_tt_t_kernel[detail::num_levels];

	detail::kernel fp16_nn_t_k_slicing_kernel;
	detail::kernel fp16_tn_t_k_slicing_kernel;
	detail::kernel fp16_nt_t_k_slicing_kernel;
	detail::kernel fp16_tt_t_k_slicing_kernel;
	detail::kernel tf32_nn_t_k_slicing_kernel;
	detail::kernel tf32_tn_t_k_slicing_kernel;
	detail::kernel tf32_nt_t_k_slicing_kernel;
	detail::kernel tf32_tt_t_k_slicing_kernel;

	unsigned debug_mode = 0;

	detail::kernel_level fixed_lernel_level;

	const std::size_t max_working_memory_num_elements = 64 * 64 * 108 * 2; // 1024 * 864
	float* w_ptr;
};

void create(shgemmHandle_t& handle);
void destroy(shgemmHandle_t& handle);

void enable_kernel_level_fixing(shgemmHandle_t& handle, const detail::kernel_level kernel_level);
void disable_kernel_level_fixing(shgemmHandle_t& handle);

void set_debug_mode(shgemmHandle_t& handle, const unsigned on);

void set_cuda_stream(shgemmHandle_t& handle, cudaStream_t const cuda_stream);

// main function
mtk::shgemm::detail::kernel_level shgemm(
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
		const tc_t compute_type,
		const operation_t op_c = op_n
		);

inline mtk::shgemm::detail::kernel_level hsgemm(
		const shgemmHandle_t handle,
		const operation_t op_a,
		const operation_t op_b,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const float* const alpha_ptr,
		const half * const a_ptr, const std::size_t lda,
		const float* const b_ptr, const std::size_t ldb,
		const float* const beta_ptr,
		float* const c_ptr, const std::size_t ldc,
		const tc_t compute_type,
		const operation_t op_c = op_n
		) {
	const auto inv_op = [&](const mtk::shgemm::operation_t op) {
		if (op == op_n) {
			return op_t;
		}
		return op_n;
	};
	return shgemm(
			handle,
			inv_op(op_a),
			inv_op(op_b),
			n, m, k,
			alpha_ptr,
			b_ptr, ldb,
			a_ptr, lda,
			beta_ptr,
			c_ptr, ldc,
			compute_type,
			inv_op(op_c)
			);
}

} // namespace shgemm
} // namespace mtk
#endif
