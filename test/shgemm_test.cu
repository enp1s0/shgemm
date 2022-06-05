#include <iostream>
#include <chrono>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/curand.hpp>
#include <mateval/comparison_cuda.hpp>
#include <shgemm/shgemm.hpp>

constexpr std::size_t test_count = 1lu << 6;
constexpr std::size_t min_log_DIM = 5;
constexpr std::size_t max_log_DIM = 14;
constexpr std::size_t log_DIM_interval = 3;
constexpr auto compute_type = mtk::shgemm::tf32;
constexpr auto op_a = mtk::shgemm::op_n;
constexpr auto op_b = mtk::shgemm::op_n;

mtk::mateval::layout_t convert_op_shgemm2mateval(
		const mtk::shgemm::operation_t op
		) {
	if (op == mtk::shgemm::op_n) {
		return mtk::mateval::col_major;
	}
	return mtk::mateval::row_major;
}

std::string op_name_str(
		const mtk::shgemm::operation_t op
		) {
	if (op == mtk::shgemm::op_n) {
		return "N";
	}
	return "T";
}

void test_shgemm_core(
		mtk::shgemm::shgemmHandle_t shgemm_handle,
		mtk::shgemm::operation_t op_a,
		mtk::shgemm::operation_t op_b,
		const float* const a_fp32_ptr,
		const float* const b_fp32_ptr,
		const half * const b_fp16_ptr,
		float* const c_fp32_ptr,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const mtk::shgemm::tc_t compute_type
		) {
	const float alpha = 1.0f, beta = 0.0f;
	const auto level = mtk::shgemm::shgemm(
			shgemm_handle,
			op_a, op_b,
			m, n, k,
			&alpha,
			a_fp32_ptr, (op_a == mtk::shgemm::op_n ? m : k),
			b_fp16_ptr, (op_b == mtk::shgemm::op_n ? k : n),
			&beta,
			c_fp32_ptr, m,
			compute_type
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	const auto error = mtk::mateval::cuda::get_error_AxB(
			mtk::mateval::max_relative_error | mtk::mateval::relative_residual,
			m, n, k,
			convert_op_shgemm2mateval(op_a),
			convert_op_shgemm2mateval(op_b),
			mtk::mateval::col_major,
			a_fp32_ptr, (op_a == mtk::shgemm::op_n ? m : k),
			b_fp32_ptr, (op_b == mtk::shgemm::op_n ? k : n),
			c_fp32_ptr, m
			);

	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto start_clock = std::chrono::system_clock::now();
	for (std::size_t test_c = 0; test_c < test_count; test_c++) {
		mtk::shgemm::shgemm(
				shgemm_handle,
				op_a, op_b,
				m, n, k,
				&alpha,
				a_fp32_ptr, (op_a == mtk::shgemm::op_n ? m : k),
				b_fp16_ptr, (op_b == mtk::shgemm::op_n ? k : n),
				&beta,
				c_fp32_ptr, m,
				compute_type
				);
	}
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / test_count;

	const auto throughput = 2 * m * n * k / elapsed_time * 1e-12; // TFlop/s

	std::printf("%s,%lu,%lu,%lu,%s,%s,%e,%e,%e,%u\n",
			(compute_type == mtk::shgemm::fp16 ? "fp16" : "tf32"),
			m, n, k,
			op_name_str(op_a).c_str(),
			op_name_str(op_b).c_str(),
			error.at(mtk::mateval::max_relative_error),
			error.at(mtk::mateval::relative_residual),
			throughput,
			static_cast<unsigned>(level)
			);
	std::fflush(stdout);
}

__global__ void convert_B_to_fp16_kernel(
		half* const fp16_ptr,
		float* const fp32_ptr,
		const std::size_t N
		) {
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= N) return;
	const auto fp16 = cutf::type::cast<half>(fp32_ptr[tid]);
	fp16_ptr[tid] = fp16;
	fp32_ptr[tid] = cutf::type::cast<float>(fp16);
}

void convert_B_to_fp16(
		half* const fp16_ptr,
		float* const fp32_ptr,
		const std::size_t N
		) {
	constexpr unsigned block_size = 256;
	convert_B_to_fp16_kernel<<<(N + block_size - 1) / block_size, block_size>>>(
			fp16_ptr,
			fp32_ptr,
			N
			);
	cudaDeviceSynchronize();
}

int main() {
	const auto max_N = 1lu << max_log_DIM;
	auto a_fp32_uptr = cutf::memory::get_device_unique_ptr<float>(max_N * max_N);
	auto b_fp32_uptr = cutf::memory::get_device_unique_ptr<float>(max_N * max_N);
	auto b_fp16_uptr = cutf::memory::get_device_unique_ptr<half >(max_N * max_N);
	auto c_fp32_uptr = cutf::memory::get_device_unique_ptr<float>(max_N * max_N);

	const auto seed = 10lu;
	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));

	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), a_fp32_uptr.get(), max_N * max_N));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), b_fp32_uptr.get(), max_N * max_N));
	convert_B_to_fp16(b_fp16_uptr.get(), b_fp32_uptr.get(), max_N * max_N);

	mtk::shgemm::shgemmHandle_t shgemm_handle;
	mtk::shgemm::create(shgemm_handle);

	std::printf("tc_t,m,n,k,op_a,op_b,residual,relative_max_error,throughput_in_tflops,kernel_level\n");
	std::fflush(stdout);
	for (std::size_t log_M = min_log_DIM; log_M <= max_log_DIM; log_M += log_DIM_interval) {
		for (std::size_t log_N = min_log_DIM; log_N <= max_log_DIM; log_N += log_DIM_interval) {
			for (std::size_t log_K = min_log_DIM; log_K <= max_log_DIM; log_K += log_DIM_interval) {
				const auto m = 1lu << log_M;
				const auto n = 1lu << log_N;
				const auto k = 1lu << log_K;
				test_shgemm_core(
						shgemm_handle,
						op_a,
						op_b,
						a_fp32_uptr.get(),
						b_fp32_uptr.get(),
						b_fp16_uptr.get(),
						c_fp32_uptr.get(),
						m, n, k,
						compute_type
						);
			}
		}
	}
	mtk::shgemm::destroy(shgemm_handle);
}
