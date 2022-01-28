#ifndef __SHGEMM_WMMAE_SHGEMM_HPP__
#define __SHGEMM_WMMAE_SHGEMM_HPP__
#include <wmma_extension/tcec/tcec.hpp>
#include "utils.hpp"

namespace mtk {
namespace shgemm {
namespace device {
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
