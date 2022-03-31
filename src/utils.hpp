#ifndef __SHGEMM_UTILS_HPP__
#define __SHGEMM_UTILS_HPP__
#include <wmma_extension/tcec/tcec.hpp>
namespace mtk {
namespace shgemm {
namespace utils {
constexpr unsigned warp_size = 32;

class col_major;
class row_major;
} // namespace utils

namespace device {
template <class TC_T>
using A_Policy = typename mtk::wmma::tcec::detail::default_policy<TC_T, mtk::wmma::tcec::op_with_error_correction   , mtk::wmma::tcec::op_mma>::type;
template <class TC_T>
using B_Policy = typename mtk::wmma::tcec::detail::default_policy<TC_T, mtk::wmma::tcec::op_without_error_correction, mtk::wmma::tcec::op_mma>::type;

// CAUTION: THESE VALUES MUST BE 0 or 8.
constexpr unsigned A_smem_skew = 8;
constexpr unsigned B_smem_skew = 8;

template <unsigned SMEM_M, unsigned SMEM_K, class Layout>
struct get_A_smem_size{const static unsigned value = 0;};
template <unsigned SMEM_M, unsigned SMEM_K>
struct get_A_smem_size<SMEM_M, SMEM_K, mtk::shgemm::utils::row_major> {const static unsigned value = (SMEM_K + mtk::shgemm::device::A_smem_skew) * SMEM_M;};
template <unsigned SMEM_M, unsigned SMEM_K>
struct get_A_smem_size<SMEM_M, SMEM_K, mtk::shgemm::utils::col_major> {const static unsigned value = (SMEM_M + mtk::shgemm::device::A_smem_skew) * SMEM_K;};

template <unsigned SMEM_K, unsigned SMEM_N, class Layout>
struct get_B_smem_size{const static unsigned value = 0;};
template <unsigned SMEM_K, unsigned SMEM_N>
struct get_B_smem_size<SMEM_K, SMEM_N, mtk::shgemm::utils::row_major> {const static unsigned value = (SMEM_N + mtk::shgemm::device::B_smem_skew) * SMEM_K;};
template <unsigned SMEM_K, unsigned SMEM_N>
struct get_B_smem_size<SMEM_K, SMEM_N, mtk::shgemm::utils::col_major> {const static unsigned value = (SMEM_K + mtk::shgemm::device::B_smem_skew) * SMEM_N;};

template <class LAYOUT>
struct get_mem_index {__device__ std::size_t operator()(const unsigned m, const unsigned n, const unsigned ld);};
template <>
struct get_mem_index<mtk::shgemm::utils::col_major> {__device__ std::size_t operator()(const unsigned m, const unsigned n, const unsigned ld) {return m + n * static_cast<std::size_t>(ld);};};
template <>
struct get_mem_index<mtk::shgemm::utils::row_major> {__device__ std::size_t operator()(const unsigned m, const unsigned n, const unsigned ld) {return m * static_cast<std::size_t>(ld) + n;};};
} // namespace device
} // namespace shgemm
} // namespace mtk
#endif
