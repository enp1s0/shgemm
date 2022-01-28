#ifndef __SHGEMM_UTILS_HPP__
#define __SHGEMM_UTILS_HPP__
namespace mtk {
namespace shgemm {
namespace utils {
constexpr unsigned warp_size = 32;

class col_major;
class row_major;
} // namespace utils
} // namespace shgemm
} // namespace mtk
#endif
