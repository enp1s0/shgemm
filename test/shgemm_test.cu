#include <iostream>
#include <chrono>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/curand.hpp>
#include <mateval/comparison_cuda.hpp>
#include <shgemm/shgemm.hpp>

constexpr std::size_t test_count = 1lu << 6;
constexpr std::size_t min_log_DIM = 6;
constexpr std::size_t max_log_DIM = 14;

void test_shgemm_core(
		mtk::shgemm::shgemmHandle_t shgemm_handle,
		const float* const a_fp32_ptr,
		const float* const b_fp32_ptr,
		const half * const b_fp16_ptr,
		float* const c_fp32_ptr,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k
		) {
	const float alpha = 1.0f, beta = 0.0f;
	mtk::shgemm::shgemm(
			shgemm_handle,
			mtk::shgemm::op_t, mtk::shgemm::op_n,
			m, n, k,
			&alpha,
			a_fp32_ptr, k,
			b_fp16_ptr, k,
			&beta,
			c_fp32_ptr, m
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	const auto [relative_max_error, residual] = mtk::mateval::cuda::max_relative_error_and_residual_AxB(
			m, n, k,
			mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major,
			a_fp32_ptr, k,
			b_fp32_ptr, k,
			c_fp32_ptr, m
			);

	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto start_clock = std::chrono::system_clock::now();
	for (std::size_t test_c = 0; test_c < test_count; test_c++) {
	mtk::shgemm::shgemm(
			shgemm_handle,
			mtk::shgemm::op_t, mtk::shgemm::op_n,
			m, n, k,
			&alpha,
			a_fp32_ptr, k,
			b_fp16_ptr, k,
			&beta,
			c_fp32_ptr, m
			);
	}
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / test_count;

	const auto throughput = 2 * m * n * k / elapsed_time * 1e-12; // TFlop/s

	std::printf("%lu,%lu,%lu,%e,%e,%e\n",
			m, n, k,
			residual,
			relative_max_error,
			throughput
			);
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
	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));

	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), a_fp32_uptr.get(), max_N * max_N));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), b_fp32_uptr.get(), max_N * max_N));
	convert_B_to_fp16(b_fp16_uptr.get(), b_fp32_uptr.get(), max_N * max_N);

	mtk::shgemm::shgemmHandle_t shgemm_handle;
	mtk::shgemm::create(shgemm_handle);
	for (std::size_t log_M = min_log_DIM; log_M <= max_log_DIM; log_M++) {
		for (std::size_t log_N = min_log_DIM; log_N <= max_log_DIM; log_N++) {
			for (std::size_t log_K = min_log_DIM; log_K <= max_log_DIM; log_K++) {
				const auto m = 1lu << log_M;
				const auto n = 1lu << log_N;
				const auto k = 1lu << log_K;
				test_shgemm_core(
						shgemm_handle,
						a_fp32_uptr.get(),
						b_fp32_uptr.get(),
						b_fp16_uptr.get(),
						c_fp32_uptr.get(),
						m, n, k
						);
			}
		}
	}
}
