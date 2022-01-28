#ifndef __SHGEMM_SHGEMM_CORE_HPP__
#define __SHGEMM_SHGEMM_CORE_HPP__
#include "utils.hpp"
#include "wmmae_shgemm.hpp"

namespace mtk {
namespace shgemm {
namespace device {

template<
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	class TC_T,
	class A_LAYOUT,
	class B_LAYOUT
	>
struct shgemm_core {
	__device__ void operator()(
			mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, TC_T, void, A_Policy<TC_T>> frag_c[(SMEM_M * SMEM_N) / (FRAG_M * FRAG_N) / (BLOCK_SIZE / utils::warp_size)],
			const float* const a_ptr,
			const half * const b_ptr
			) {
		constexpr unsigned num_submatrices = (SMEM_M / FRAG_M) * (SMEM_N / FRAG_N);
		static_assert(num_submatrices * mtk::shgemm::utils::warp_size % BLOCK_SIZE == 0, "the number of reg-level sub matrices must be a multiple of (BLOCK_SIZE / warp_size)");

		for (unsigned matrix_id_offset = 0; matrix_id_offset < num_submatrices; matrix_id_offset += BLOCK_SIZE / mtk::shgemm::utils::warp_size) {
			const unsigned matrix_id = matrix_id_offset + (threadIdx.x / mtk::shgemm::utils::warp_size);
			const unsigned matrix_id_m = matrix_id % (SMEM_M / FRAG_M);
			const unsigned matrix_id_n = matrix_id / (SMEM_M / FRAG_M);

			for (unsigned k = 0; k < SMEM_K; k += FRAG_K) {
				mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, TC_T, nvcuda::wmma::row_major, A_Policy<TC_T>> frag_a;
				mtk::shgemm::device::load_matrix<A_LAYOUT, SMEM_M, SMEM_K>(frag_a, a_ptr + matrix_id_m * FRAG_M * SMEM_K + k);
				mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, TC_T, nvcuda::wmma::col_major, B_Policy<TC_T>> frag_b;
				mtk::shgemm::device::load_matrix<B_LAYOUT, SMEM_K, SMEM_N>(frag_b, b_ptr + matrix_id_n * FRAG_N * SMEM_K + k);

				mtk::shgemm::device::mma_sync(frag_c[matrix_id_offset / (BLOCK_SIZE / mtk::shgemm::utils::warp_size)],
						frag_a,
						frag_b,
						frag_c[matrix_id_offset / (BLOCK_SIZE / mtk::shgemm::utils::warp_size)]);
			}
		}
	}
};

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
struct shgemm_core_pipeline {
	__device__ void operator()(
			mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, TC_T, void, A_Policy<TC_T>> frag_c[(SMEM_M * SMEM_N) / (FRAG_M * FRAG_N) / (BLOCK_SIZE / utils::warp_size)],
			const float* const a_ptr,
			const half * const b_ptr
			) {
		constexpr unsigned num_submatrices = (SMEM_M / FRAG_M) * (SMEM_N / FRAG_N);
		static_assert(num_submatrices * mtk::shgemm::utils::warp_size % BLOCK_SIZE == 0, "the number of reg-level sub matrices must be a multiple of (BLOCK_SIZE / warp_size)");

		for (unsigned matrix_id_offset = 0; matrix_id_offset < num_submatrices; matrix_id_offset += BLOCK_SIZE / mtk::shgemm::utils::warp_size) {
			const unsigned matrix_id = matrix_id_offset + (threadIdx.x / mtk::shgemm::utils::warp_size);
			const unsigned matrix_id_m = matrix_id % (SMEM_M / FRAG_M);
			const unsigned matrix_id_n = matrix_id / (SMEM_M / FRAG_M);

			mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, TC_T, nvcuda::wmma::row_major, A_Policy<TC_T>> frag_a[2];
			mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, TC_T, nvcuda::wmma::col_major, B_Policy<TC_T>> frag_b[2];

			mtk::wmma::tcec::load_matrix_sync(frag_a[0], a_ptr + matrix_id_m * FRAG_M * SMEM_K, SMEM_K, false);
			mtk::wmma::tcec::load_matrix_sync(frag_b[0], b_ptr + matrix_id_n * FRAG_N * SMEM_K, SMEM_K, false);

			unsigned k = FRAG_K;
			for (; k < SMEM_K; k += FRAG_K) {
				mtk::wmma::tcec::load_matrix_sync(frag_a[(k / FRAG_K) & 0x1], a_ptr + matrix_id_m * FRAG_M * SMEM_K + k, SMEM_K, false);
				mtk::wmma::tcec::load_matrix_sync(frag_b[(k / FRAG_K) & 0x1], b_ptr + matrix_id_n * FRAG_N * SMEM_K + k, SMEM_K, false);

				mtk::shgemm::device::mma_sync(frag_c[matrix_id_offset / (BLOCK_SIZE / mtk::shgemm::utils::warp_size)],
						frag_a[1 - ((k / FRAG_K) & 0x1)],
						frag_b[1 - ((k / FRAG_K) & 0x1)],
						frag_c[matrix_id_offset / (BLOCK_SIZE / mtk::shgemm::utils::warp_size)]);
			}

			mtk::shgemm::device::mma_sync(frag_c[matrix_id_offset / (BLOCK_SIZE / mtk::shgemm::utils::warp_size)],
					frag_a[1 - ((k / FRAG_K) & 0x1)],
					frag_b[1 - ((k / FRAG_K) & 0x1)],
					frag_c[matrix_id_offset / (BLOCK_SIZE / mtk::shgemm::utils::warp_size)]);
		}
	}
};
} // namespace device
} // namespace shgemm
} // namespace mtk
#endif
