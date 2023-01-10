#include <iostream>
#include <chrono>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/curand.hpp>
#include <cutf/cublas.hpp>
#include <mateval/comparison_cuda.hpp>
#include <shgemm/shgemm.hpp>

//#define TEST_ALL

constexpr std::size_t test_count = 1lu << 6;
constexpr std::size_t min_log_DIM = 10;
constexpr std::size_t max_log_DIM = 14;
constexpr std::size_t log_DIM_interval = 2;

mtk::mateval::layout_t convert_op_shgemm2mateval(
		const mtk::shgemm::operation_t op
		) {
	if (op == mtk::shgemm::op_n) {
		return mtk::mateval::col_major;
	}
	return mtk::mateval::row_major;
}

mtk::mateval::layout_t convert_op_shgemm2mateval(
		const cublasOperation_t op
		) {
	if (op == CUBLAS_OP_N) {
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

std::string op_name_str(
		const cublasOperation_t op
		) {
	if (op == CUBLAS_OP_N) {
		return "N";
	}
	return "T";
}

cublasOperation_t op_to_cublas(
		const mtk::shgemm::operation_t op
		) {
	if (op == mtk::shgemm::op_n) {
		return CUBLAS_OP_N;
	}
	return CUBLAS_OP_T;
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
		const mtk::shgemm::tc_t compute_type,
		mtk::shgemm::operation_t op_c
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
			c_fp32_ptr, (op_c == mtk::shgemm::op_n ? m : n),
			compute_type,
			op_c
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	const auto error = mtk::mateval::cuda::get_error_AxB(
			mtk::mateval::max_relative_error | mtk::mateval::relative_residual,
			m, n, k,
			convert_op_shgemm2mateval(op_a),
			convert_op_shgemm2mateval(op_b),
			convert_op_shgemm2mateval(op_c),
			a_fp32_ptr, (op_a == mtk::shgemm::op_n ? m : k),
			b_fp32_ptr, (op_b == mtk::shgemm::op_n ? k : n),
			c_fp32_ptr, (op_c == mtk::shgemm::op_n ? m : n)
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
				c_fp32_ptr, (op_c == mtk::shgemm::op_n ? m : n),
				compute_type,
				op_c
				);
	}
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / test_count;

	const auto throughput = 2 * m * n * k / elapsed_time * 1e-12; // TFlop/s

	std::printf("%s,%lu,%lu,%lu,%s,%s,%s,%e,%e,%e,%u\n",
			(compute_type == mtk::shgemm::fp16 ? "fp16" : "tf32"),
			m, n, k,
			op_name_str(op_a).c_str(),
			op_name_str(op_b).c_str(),
			op_name_str(op_c).c_str(),
			error.at(mtk::mateval::max_relative_error),
			error.at(mtk::mateval::relative_residual),
			throughput,
			static_cast<unsigned>(level)
			);
	std::fflush(stdout);
}

void test_cublas_core(
		cublasHandle_t cublas_handle,
		cublasOperation_t op_a,
		cublasOperation_t op_b,
		const float* const a_fp32_ptr,
		const float* const b_fp32_ptr,
		float* const c_fp32_ptr,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const std::string compute_type
		) {
	if (compute_type == "TF32") {
		cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
	} else {
		cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
	}

	const float alpha = 1.0f, beta = 0.0f;
	cublasSgemm(
			cublas_handle,
			op_a, op_b,
			m, n, k,
			&alpha,
			a_fp32_ptr, (op_a == CUBLAS_OP_N ? m : k),
			b_fp32_ptr, (op_b == CUBLAS_OP_N ? k : n),
			&beta,
			c_fp32_ptr, m
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	const auto error = mtk::mateval::cuda::get_error_AxB(
			mtk::mateval::max_relative_error | mtk::mateval::relative_residual,
			m, n, k,
			convert_op_shgemm2mateval(op_a),
			convert_op_shgemm2mateval(op_b),
			mtk::mateval::col_major,
			a_fp32_ptr, (op_a == CUBLAS_OP_N ? m : k),
			b_fp32_ptr, (op_b == CUBLAS_OP_N ? k : n),
			c_fp32_ptr, m
			);

	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto start_clock = std::chrono::system_clock::now();
	for (std::size_t test_c = 0; test_c < test_count; test_c++) {
		cublasSgemm(
				cublas_handle,
				op_a, op_b,
				m, n, k,
				&alpha,
				a_fp32_ptr, (op_a == CUBLAS_OP_N ? m : k),
				b_fp32_ptr, (op_b == CUBLAS_OP_N ? k : n),
				&beta,
				c_fp32_ptr, m
				);
	}
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / test_count;

	const auto throughput = 2 * m * n * k / elapsed_time * 1e-12; // TFlop/s

	std::printf("cublas-%s,%lu,%lu,%lu,%s,%s,%e,%e,%e,%u\n",
			compute_type.c_str(),
			m, n, k,
			op_name_str(op_a).c_str(),
			op_name_str(op_b).c_str(),
			error.at(mtk::mateval::max_relative_error),
			error.at(mtk::mateval::relative_residual),
			throughput,
			0u
			);
	cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
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
	auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();

	std::vector<mtk::shgemm::operation_t> op_a_list = {
		mtk::shgemm::op_n,
		mtk::shgemm::op_t,
	};

	std::vector<mtk::shgemm::operation_t> op_b_list = {
		mtk::shgemm::op_n,
		mtk::shgemm::op_t,
	};

	std::vector<mtk::shgemm::operation_t> op_c_list = {
		mtk::shgemm::op_n,
		mtk::shgemm::op_t,
	};

	std::printf("tc_t,m,n,k,op_a,op_b,op_c,residual,relative_max_error,throughput_in_tflops,kernel_level\n");
	std::fflush(stdout);
	for (std::size_t log_M = min_log_DIM; log_M <= max_log_DIM; log_M += log_DIM_interval) {
		for (const auto op_a : op_a_list) {
			for (const auto op_b : op_b_list) {
				for (const auto op_c : op_c_list) {
					const auto m = 1lu << log_M;
					const auto n = 1lu << log_M;
					const auto k = 1lu << log_M;
					test_shgemm_core(
							shgemm_handle,
							op_a,
							op_b,
							a_fp32_uptr.get(),
							b_fp32_uptr.get(),
							b_fp16_uptr.get(),
							c_fp32_uptr.get(),
							m, n, k,
							mtk::shgemm::tf32,
							op_c
							);
					test_shgemm_core(
							shgemm_handle,
							op_a,
							op_b,
							a_fp32_uptr.get(),
							b_fp32_uptr.get(),
							b_fp16_uptr.get(),
							c_fp32_uptr.get(),
							m, n, k,
							mtk::shgemm::fp16,
							op_c
							);
				}
#ifdef TEST_ALL
					test_cublas_core(
							*cublas_handle_uptr.get(),
							op_to_cublas(op_a),
							op_to_cublas(op_b),
							a_fp32_uptr.get(),
							b_fp32_uptr.get(),
							c_fp32_uptr.get(),
							m, n, k,
							"TF32"
							);
					test_cublas_core(
							*cublas_handle_uptr.get(),
							op_to_cublas(op_a),
							op_to_cublas(op_b),
							a_fp32_uptr.get(),
							b_fp32_uptr.get(),
							c_fp32_uptr.get(),
							m, n, k,
							"FP32"
							);
#endif
			}
		}
	}
	mtk::shgemm::destroy(shgemm_handle);
}
