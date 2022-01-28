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
} // namespace device
} // namespace shgemm
} // namespace mtk
#endif
