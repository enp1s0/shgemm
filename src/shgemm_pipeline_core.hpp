#ifndef __SHGEMM_PIPELINE_CORE_HPP__
#define __SHGEMM_PIPELINE_CORE_HPP__
#include <cstdint>
#include <cuda_fp16.h>
#include "shgemm_core.hpp"
#include "dmem_accessor.hpp"
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
	class A_DMEM_LOADER,
	class B_DMEM_LOADER,
	class SHGEMM_CORE,
	unsigned NUM_STAGES,
	unsigned NUM_UNROLLINGS,
	unsigned BLOCK_SIZE,
	class TC_T
	>
struct shgemm_pipeline_core {
	__device__ void operator() (
			const unsigned m,
			const unsigned n,
			const unsigned k,
			const float* const a_dmem_ptr, const unsigned lda,
			const half * const b_dmem_ptr, const unsigned ldb,
			float* const a_smem_ptr,
			half * const b_smem_ptr,
			mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, TC_T, void, mtk::shgemm::device::A_Policy<TC_T>> frag_c[(SMEM_M * SMEM_N) / (FRAG_M * FRAG_N) / (BLOCK_SIZE / mtk::shgemm::utils::warp_size)]
			) {
	A_DMEM_LOADER a_dram_loader;
	B_DMEM_LOADER b_dram_loader;
	SHGEMM_CORE shgemm_core;

	constexpr unsigned A_smem_size = mtk::shgemm::device::get_A_smem_size<SMEM_M, SMEM_K, typename A_DMEM_LOADER::layout>::value;
	constexpr unsigned B_smem_size = mtk::shgemm::device::get_B_smem_size<SMEM_K, SMEM_N, typename B_DMEM_LOADER::layout>::value;

	a_dram_loader(a_smem_ptr,
			mtk::shgemm::device::get_m_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_M, 0,
			m, k,
			a_dmem_ptr, lda
			);
	b_dram_loader(b_smem_ptr,
			0, mtk::shgemm::device::get_n_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_N,
			k, n,
			b_dmem_ptr, ldb
			);
	unsigned block_k = SMEM_K;

	// Initialize frag C
	constexpr unsigned frag_c_length = (SMEM_M * SMEM_N) / (FRAG_M * FRAG_N) / (BLOCK_SIZE / mtk::shgemm::utils::warp_size);
#pragma unroll
	for (unsigned i = 0; i < frag_c_length; i++) {
		mtk::wmma::tcec::fill_zero(frag_c[i]);
	}

	// MMA
#pragma unroll NUM_UNROLLINGS
	for (; block_k < k; block_k += SMEM_K) {
		a_dram_loader(a_smem_ptr + ((block_k / SMEM_K) % NUM_STAGES) * A_smem_size,
				mtk::shgemm::device::get_m_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_M, block_k,
				m, k,
				a_dmem_ptr, lda
				);
		b_dram_loader(b_smem_ptr + ((block_k / SMEM_K) % NUM_STAGES) * B_smem_size,
				block_k, mtk::shgemm::device::get_n_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_N,
				k, n,
				b_dmem_ptr, ldb
				);

		cutf::cp_async::wait_group<2>();
		__syncthreads();
		shgemm_core(frag_c,
				a_smem_ptr + (((block_k / SMEM_K) + NUM_STAGES - 1) % NUM_STAGES) * A_smem_size,
				b_smem_ptr + (((block_k / SMEM_K) + NUM_STAGES - 1) % NUM_STAGES) * B_smem_size
				);
	}
	cutf::cp_async::wait_all();
	__syncthreads();

	shgemm_core(frag_c,
			a_smem_ptr + (((block_k / SMEM_K) + NUM_STAGES - 1) % NUM_STAGES) * A_smem_size,
			b_smem_ptr + (((block_k / SMEM_K) + NUM_STAGES - 1) % NUM_STAGES) * B_smem_size
			);
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
	class B_DMEM_LOADER,
	class SHGEMM_CORE,
	unsigned NUM_UNROLLINGS,
	unsigned BLOCK_SIZE,
	class TC_T
	>
struct shgemm_pipeline_core<
	SMEM_M, SMEM_N, SMEM_K,
	FRAG_M, FRAG_N, FRAG_K,
	A_DMEM_LOADER,
	B_DMEM_LOADER,
	SHGEMM_CORE,
	2,
	NUM_UNROLLINGS,
	BLOCK_SIZE,
	TC_T
	> {
	__device__ void operator() (
			const unsigned m,
			const unsigned n,
			const unsigned k,
			const float* const a_dmem_ptr, const unsigned lda,
			const half * const b_dmem_ptr, const unsigned ldb,
			float* const a_smem_ptr,
			half * const b_smem_ptr,
			mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, TC_T, void, mtk::shgemm::device::A_Policy<TC_T>> frag_c[(SMEM_M * SMEM_N) / (FRAG_M * FRAG_N) / (BLOCK_SIZE / mtk::shgemm::utils::warp_size)]
			) {
	A_DMEM_LOADER a_dram_loader;
	B_DMEM_LOADER b_dram_loader;
	SHGEMM_CORE shgemm_core;

	constexpr unsigned A_smem_size = mtk::shgemm::device::get_A_smem_size<SMEM_M, SMEM_K, typename A_DMEM_LOADER::layout>::value;
	constexpr unsigned B_smem_size = mtk::shgemm::device::get_B_smem_size<SMEM_K, SMEM_N, typename B_DMEM_LOADER::layout>::value;

	a_dram_loader(a_smem_ptr,
			mtk::shgemm::device::get_m_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_M, 0,
			m, k,
			a_dmem_ptr, lda
			);
	b_dram_loader(b_smem_ptr,
			0, mtk::shgemm::device::get_n_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_N,
			k, n,
			b_dmem_ptr, ldb
			);
	unsigned block_k = SMEM_K;

	// Initialize frag C
	constexpr unsigned frag_c_length = (SMEM_M * SMEM_N) / (FRAG_M * FRAG_N) / (BLOCK_SIZE / mtk::shgemm::utils::warp_size);
#pragma unroll
	for (unsigned i = 0; i < frag_c_length; i++) {
		mtk::wmma::tcec::fill_zero(frag_c[i]);
	}

	cutf::cp_async::wait_all();
	__syncthreads();
	// MMA
#pragma unroll NUM_UNROLLINGS
	for (; block_k < k; block_k += SMEM_K) {
		a_dram_loader(a_smem_ptr + ((block_k / SMEM_K) & 0x1) * A_smem_size,
				mtk::shgemm::device::get_m_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_M, block_k,
				m, k,
				a_dmem_ptr, lda
				);
		b_dram_loader(b_smem_ptr + ((block_k / SMEM_K) & 0x1) * B_smem_size,
				block_k, mtk::shgemm::device::get_n_block_id<SMEM_M, SMEM_N>(m, n) * SMEM_N,
				k, n,
				b_dmem_ptr, ldb
				);

		shgemm_core(frag_c,
				a_smem_ptr + (1 - ((block_k / SMEM_K) & 0x1)) * A_smem_size,
				b_smem_ptr + (1 - ((block_k / SMEM_K) & 0x1)) * B_smem_size
				);
		cutf::cp_async::wait_all();
		__syncthreads();
	}

	shgemm_core(frag_c,
			a_smem_ptr + (1 - ((block_k / SMEM_K) & 0x1)) * A_smem_size,
			b_smem_ptr + (1 - ((block_k / SMEM_K) & 0x1)) * B_smem_size
			);
	}
};
} // namespace device
} // namespace shgemm
} // namespace mtk
#endif
