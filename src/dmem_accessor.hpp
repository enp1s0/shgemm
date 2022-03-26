#ifndef __SHGEMM_DMEM_ACCESSOR_HPP__
#define __SHGEMM_DMEM_ACCESSOR_HPP__
#include <cassert>
#include <cutf/type.hpp>
#include <cutf/cp_async.hpp>
#include "utils.hpp"

namespace mtk {
namespace shgemm {
namespace device {

// -----------------------------------------
// N - loader
// -----------------------------------------
template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct dmem_loader_n {
	__device__ void operator()(
			T* const smem_ptr,
			const std::size_t dmem_start_m, const std::size_t dmem_start_n,
			const std::size_t dmem_size_m, const std::size_t dmem_size_n,
			const T* const dmem_ptr, const std::size_t ldd
			) {
		static_assert(SMEM_M * SMEM_N >= BLOCK_SIZE * 8, "SMEM_M * SMEM_N >= BLOCK_SIZE must be satisfied");
		if (dmem_start_m + SMEM_M <= dmem_size_m && dmem_start_n + SMEM_N <= dmem_size_n) {
			if ((ldd & 0x3) == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 4) {
					const auto i = i_offset + threadIdx.x * 4;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::A_smem_skew);

					// 128 bit memory access
					cutf::cp_async::cp_async<16>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else if ((ldd & 0x1) == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 2) {
					const auto i = i_offset + threadIdx.x * 2;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::A_smem_skew);

					// 64 bit memory access
					cutf::cp_async::cp_async<8>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::A_smem_skew);

					cutf::cp_async::cp_async<4>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			}
		} else {
			for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
				const auto i = i_offset + threadIdx.x;
				const auto m = (i % SMEM_M) + dmem_start_m;
				const auto n = (i / SMEM_M) + dmem_start_n;
				const auto dmem_index = m + n * ldd;
				const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::A_smem_skew);

				if (m < dmem_size_m && n < dmem_size_n) {
					cutf::cp_async::cp_async<4>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				} else {
					smem_ptr[smem_index] = 0.f;
				}
			}
		}
		cutf::cp_async::commit();
	}
};

template <unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct dmem_loader_n<half, SMEM_M, SMEM_N, BLOCK_SIZE> {
	__device__ void operator()(
			half* const smem_ptr,
			const std::size_t dmem_start_m, const std::size_t dmem_start_n,
			const std::size_t dmem_size_m, const std::size_t dmem_size_n,
			const half* const dmem_ptr, const std::size_t ldd
			) {
		static_assert(SMEM_M * SMEM_N >= BLOCK_SIZE * 8, "SMEM_M * SMEM_N >= BLOCK_SIZE must be satisfied");
		if (dmem_start_m + SMEM_M <= dmem_size_m && dmem_start_n + SMEM_N <= dmem_size_n) {
			if ((ldd & 0x7) == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 8) {
					const auto i = i_offset + threadIdx.x * 8;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::B_smem_skew);

					cutf::cp_async::cp_async<16>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else if ((ldd & 0x3) == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 4) {
					const auto i = i_offset + threadIdx.x * 4;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::B_smem_skew);

					cutf::cp_async::cp_async<8>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else if ((ldd & 0x1) == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 2) {
					const auto i = i_offset + threadIdx.x * 2;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::B_smem_skew);

					cutf::cp_async::cp_async<4>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::B_smem_skew);

					smem_ptr[smem_index] = dmem_ptr[dmem_index];
				}
			}
		} else {
			for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
				const auto i = i_offset + threadIdx.x;
				const auto m = (i % SMEM_M) + dmem_start_m;
				const auto n = (i / SMEM_M) + dmem_start_n;
				const auto dmem_index = m + n * ldd;
				const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::B_smem_skew);

				auto v = cutf::type::cast<half>(0);
				if (m < dmem_size_m && n < dmem_size_n) {
					v = dmem_ptr[dmem_index];
				}

				smem_ptr[smem_index] = v;
			}
		}
		cutf::cp_async::commit();
	}
};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct dmem_loader_row_major {
	using layout = mtk::shgemm::utils::row_major;
	__device__ void operator()(
			T* const smem_ptr,
			const std::size_t dmem_start_m, const std::size_t dmem_start_n,
			const std::size_t dmem_size_m, const std::size_t dmem_size_n,
			const T* const dmem_ptr, const std::size_t ldd
			) {
		dmem_loader_n<T, SMEM_N, SMEM_M, BLOCK_SIZE>{}(
				smem_ptr,
				dmem_start_n, dmem_start_m,
				dmem_size_n, dmem_size_m,
				dmem_ptr, ldd
				);
	}
};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct dmem_loader_col_major {
	using layout = mtk::shgemm::utils::col_major;
	__device__ void operator()(
			T* const smem_ptr,
			const std::size_t dmem_start_m, const std::size_t dmem_start_n,
			const std::size_t dmem_size_m, const std::size_t dmem_size_n,
			const T* const dmem_ptr, const std::size_t ldd
			) {
		dmem_loader_n<T, SMEM_M, SMEM_N, BLOCK_SIZE>{}(
				smem_ptr,
				dmem_start_m, dmem_start_n,
				dmem_size_m, dmem_size_n,
				dmem_ptr, ldd
				);
	}
};

// -----------------------------------------
// N - storer
// -----------------------------------------
template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct dmem_storer_n {
	__device__ void operator()(
			T* const dmem_ptr, const std::size_t ldd,
			const std::size_t dmem_start_m, const std::size_t dmem_start_n,
			const std::size_t dmem_size_m, const std::size_t dmem_size_n,
			const T* const smem_ptr,
			const float alpha, const float beta
			) {
		if (beta == 0.f) {
			if (dmem_start_m + SMEM_M < dmem_size_m && dmem_start_n + SMEM_N < dmem_size_n) {
				if (ldd & 0x3 == 0) {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 4) {
						const auto i = i_offset + threadIdx.x * 4;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * ldd;

						// 128 bit memory access
						auto v = *reinterpret_cast<const float4*>(&smem_ptr[i]);
						v.x *= alpha;
						v.y *= alpha;
						v.z *= alpha;
						v.w *= alpha;
						*reinterpret_cast<float4*>(&dmem_ptr[dmem_index]) = v;
					}
				} else if (ldd & 0x1 == 0) {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 2) {
						const auto i = i_offset + threadIdx.x * 2;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * ldd;

						// 64 bit memory access
						auto v = *reinterpret_cast<const float2*>(&smem_ptr[i]);
						v.x *= alpha;
						v.y *= alpha;
						*reinterpret_cast<float2*>(&dmem_ptr[dmem_index]) = v;
					}
				} else {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
						const auto i = i_offset + threadIdx.x;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * ldd;

						dmem_ptr[dmem_index] = smem_ptr[i] * alpha;
					}
				}
			} else {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					if (m >= dmem_size_m || n >= dmem_size_n) {
						continue;
					}

					dmem_ptr[dmem_index] = smem_ptr[i] * alpha;
				}
			}
		} else {
			if (dmem_start_m + SMEM_M < dmem_size_m && dmem_start_n + SMEM_N < dmem_size_n) {
				if (ldd & 0x3 == 0) {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 4) {
						const auto i = i_offset + threadIdx.x * 4;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * ldd;

						// 128 bit memory access
						auto v = *reinterpret_cast<const float4*>(&smem_ptr[i]);
						const auto w = *reinterpret_cast<float4*>(&dmem_ptr[dmem_index]);
						v.x = v.x * alpha + w.x * beta;
						v.y = v.y * alpha + w.y * beta;
						v.z = v.z * alpha + w.z * beta;
						v.w = v.w * alpha + w.w * beta;
						*reinterpret_cast<float4*>(&dmem_ptr[dmem_index]) = v;
					}
				} else if (ldd & 0x1 == 0) {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 2) {
						const auto i = i_offset + threadIdx.x * 2;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * ldd;

						// 64 bit memory access
						auto v = *reinterpret_cast<const float2*>(&smem_ptr[i]);
						const auto w = *reinterpret_cast<float2*>(&dmem_ptr[dmem_index]);
						v.x = v.x * alpha + w.x * beta;
						v.y = v.y * alpha + w.y * beta;
						*reinterpret_cast<float2*>(&dmem_ptr[dmem_index]) = v;
					}
				} else {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
						const auto i = i_offset + threadIdx.x;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * ldd;

						dmem_ptr[dmem_index] = smem_ptr[i] * alpha + dmem_ptr[dmem_index] + beta;
					}
				}
			} else {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					if (m >= dmem_size_m || n >= dmem_size_n) {
						continue;
					}

					dmem_ptr[dmem_index] = smem_ptr[i] * alpha + dmem_ptr[dmem_index] * beta;
				}
			}
		}
	}
};
} // namespace device
} // namespace shgemm
} // namespace mtk
#endif
