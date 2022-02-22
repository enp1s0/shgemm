#ifndef __SHGEMM_WMMAE_SHGEMM_HPP__
#define __SHGEMM_WMMAE_SHGEMM_HPP__
#include <type_traits>
#include <wmma_extension/tcec/tcec.hpp>
#include "utils.hpp"

namespace mtk {
namespace shgemm {
namespace device {

template <class MEM_Layout, unsigned SMEM_M, unsigned SMEM_N, int m, int n, int k, class T, class MEM_T>
__device__ void load_matrix(
		mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_a   , m, n, k, T, nvcuda::wmma::row_major, mtk::shgemm::device::A_Policy<T>>& frag,
		const MEM_T* const ptr
		) {
	if constexpr (std::is_same<MEM_Layout, mtk::shgemm::utils::row_major>::value) {
		mtk::wmma::tcec::load_matrix_sync(frag, ptr, SMEM_N, false);
	} else {
		using Policy = mtk::shgemm::device::A_Policy<T>;
		using Use = nvcuda::wmma::matrix_a;
		constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<Use, Policy::m, Policy::k, Policy::m>::value;
		constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<Use, Policy::k, Policy::n, Policy::n>::value;

		mtk::wmma::tcec::detail::foreach_ij_wrapper<Use, T, nvcuda::wmma::row_major, Policy>{}(
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
					for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
						for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
							const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::col_major>{}(i, j, SMEM_M, bm * frag_m, bn * frag_n);
							const auto v = ptr[mem_offset];
							const auto hv = mtk::wmma::detail::common::cast<T>(v);
							const auto dhv = mtk::wmma::detail::common::cast<T>(mtk::wmma::tcec::detail::correction_scale_0<T>(v - mtk::wmma::detail::common::cast<float>(hv)));
							for (unsigned f = 0; f < frag_index_count; f++) {
								const auto frag_index = frag_index_list[f];
								frag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
								frag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = dhv;
							}
						}
					}
				});
	}
}

template <class MEM_Layout, unsigned SMEM_M, unsigned SMEM_N, int m, int n, int k, class T, class MEM_T>
__device__ void load_matrix(
		mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_b   , m, n, k, T, nvcuda::wmma::col_major, mtk::shgemm::device::B_Policy<T>>& frag,
		const MEM_T* const ptr
		) {
	if constexpr (std::is_same<MEM_Layout, mtk::shgemm::utils::col_major>::value) {
		mtk::wmma::tcec::load_matrix_sync(frag, ptr, SMEM_M, false);
	} else {
		using Policy = mtk::shgemm::device::B_Policy<T>;
		using Use = nvcuda::wmma::matrix_b;
		constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<Use, Policy::m, Policy::k, Policy::m>::value;
		constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<Use, Policy::k, Policy::n, Policy::n>::value;

		mtk::wmma::tcec::detail::foreach_ij_wrapper<Use, T, nvcuda::wmma::col_major, Policy>{}(
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
					for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
						for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
							const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, nvcuda::wmma::row_major>{}(i, j, SMEM_N, bm * frag_m, bn * frag_n);
							const auto v = ptr[mem_offset];
							const auto hv = mtk::wmma::detail::common::cast<T>(v);
							for (unsigned f = 0; f < frag_index_count; f++) {
								const auto frag_index = frag_index_list[f];
								frag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
							}
						}
					}
				});
	}
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T>
__device__ void mma_sync(
		mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator      , m, n, k, T, void    , mtk::shgemm::device::A_Policy<T>>& frag_d,
		const mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_a   , m, n, k, T, A_Layout, mtk::shgemm::device::A_Policy<T>>& frag_a,
		const mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_b   , m, n, k, T, B_Layout, mtk::shgemm::device::B_Policy<T>>& frag_b,
		const mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, m, n, k, T, void    , mtk::shgemm::device::A_Policy<T>>& frag_c) {
	constexpr unsigned num_m_block = frag_d.num_sub_frag_m;
	constexpr unsigned num_n_block = frag_d.num_sub_frag_n;
	constexpr unsigned num_k_block = frag_a.num_sub_frag_n;

	mtk::wmma::tcec::detail::mma_sync_wrapper<T, A_Layout, B_Layout, float, mtk::shgemm::device::A_Policy<T>> mma_op;
	mtk::wmma::tcec::detail::fill_zero_wrapper<nvcuda::wmma::accumulator, float, void, mtk::shgemm::device::A_Policy<T>> zero_op;

	for (unsigned bm = 0; bm < num_m_block; bm++) {
		for (unsigned bn = 0; bn < num_n_block; bn++) {
			typename mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::shgemm::device::A_Policy<T>>::sub_frag_t tmp;
			zero_op(tmp);
			mma_op(
					tmp,
					frag_a.sub_frag[bm + 0  * num_m_block],
					frag_b.sub_frag[0  + bn * num_k_block],
					tmp
					);
			for (unsigned i = 0; i < tmp.num_elements; i++) {
				frag_d.sub_frag[bm + bn * num_m_block].x[i] = frag_c.sub_frag[bm + bn * num_m_block].x[i] + tmp.x[i];
			}
			mma_op(
					frag_d.sub_d_frag[bm + bn * num_m_block],
					frag_a.sub_d_frag[bm + 0  * num_m_block],
					frag_b.sub_frag  [0  + bn * num_k_block],
					frag_c.sub_d_frag[bm + bn * num_m_block]
					);
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				zero_op(tmp);
				mma_op(
						tmp,
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						tmp
						);
				for (unsigned i = 0; i < tmp.num_elements; i++) {
					frag_d.sub_frag[bm + bn * num_m_block].x[i] += tmp.x[i];
				}
				mma_op(
						frag_d.sub_d_frag[bm + bn * num_m_block],
						frag_a.sub_d_frag[bm + bk * num_m_block],
						frag_b.sub_frag  [bk + bn * num_k_block],
						frag_d.sub_d_frag[bm + bn * num_m_block]
						);
			}
		}
	}
}
} // namespace device
} // namespace shgemm
} // namespace mtk
#endif
