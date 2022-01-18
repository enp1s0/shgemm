#ifndef __SHGEMM_DMEM_ACCESSOR_HPP__
#define __SHGEMM_DMEM_ACCESSOR_HPP__
#include <cassert>
#include <cutf/type.hpp>
#include "utils.hpp"

namespace mtk {
namespace shgemm {
namespace device {

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
			if (ldd & 0x3 == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 4) {
					const auto i = i_offset + threadIdx.x * 4;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					// 128 bit memory access
					*reinterpret_cast<float4*>(&smem_ptr[i]) = *reinterpret_cast<const float4*>(&dmem_ptr[dmem_index]);
				}
			} else if (ldd & 0x1 == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 2) {
					const auto i = i_offset + threadIdx.x * 2;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					// 64 bit memory access
					*reinterpret_cast<float2*>(&smem_ptr[i]) = *reinterpret_cast<const float2*>(&dmem_ptr[dmem_index]);
				}
			} else {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					smem_ptr[i] = dmem_ptr[dmem_index];
				}
			}
		} else {
			for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
				const auto i = i_offset + threadIdx.x;
				const auto m = (i % SMEM_M) + dmem_start_m;
				const auto n = (i / SMEM_M) + dmem_start_n;
				const auto dmem_index = m + n * ldd;

				auto v = static_cast<T>(0);
				if (m < dmem_size_m && n < dmem_size_n) {
					v = dmem_ptr[dmem_index];
				}

				smem_ptr[i] = v;
			}
		}
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
			if (ldd & 0x7 == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 8) {
					const auto i = i_offset + threadIdx.x * 8;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					*reinterpret_cast<uint4*>(&smem_ptr[i]) = *reinterpret_cast<const uint4*>(&dmem_ptr[dmem_index]);
				}
			} else if (ldd & 0x3 == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 4) {
					const auto i = i_offset + threadIdx.x * 4;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					*reinterpret_cast<uint2*>(&smem_ptr[i]) = *reinterpret_cast<const uint2*>(&dmem_ptr[dmem_index]);
				}
			} else if (ldd & 0x1 == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 2) {
					const auto i = i_offset + threadIdx.x * 2;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					*reinterpret_cast<half2*>(&smem_ptr[i]) = *reinterpret_cast<const half2*>(&dmem_ptr[dmem_index]);
				}
			} else {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					smem_ptr[i] = dmem_ptr[dmem_index];
				}
			}
		} else {
			for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
				const auto i = i_offset + threadIdx.x;
				const auto m = (i % SMEM_M) + dmem_start_m;
				const auto n = (i / SMEM_M) + dmem_start_n;
				const auto dmem_index = m + n * ldd;

				auto v = cutf::type::cast<half>(0);
				if (m < dmem_size_m && n < dmem_size_n) {
					v = dmem_ptr[dmem_index];
				}

				smem_ptr[i] = v;
			}
		}
	}
};

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
			if (dmem_start_m + SMEM_M < dmem_size_m && dmem_size_n + SMEM_N < dmem_size_n) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					dmem_ptr[dmem_index] = smem_ptr[i] * alpha;
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
			if (dmem_start_m + SMEM_M < dmem_size_m && dmem_size_n + SMEM_N < dmem_size_n) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * ldd;

					dmem_ptr[dmem_index] = smem_ptr[i] * alpha + dmem_ptr[dmem_index] * beta;
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
