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
			const unsigned dmem_start_m, const unsigned dmem_start_n,
			const unsigned dmem_size_m, const unsigned dmem_size_n,
			const T* const dmem_ptr, const unsigned ldd
			) {
		static_assert(SMEM_M * SMEM_N >= BLOCK_SIZE * 8, "SMEM_M * SMEM_N >= BLOCK_SIZE must be satisfied");
		if (dmem_start_m + SMEM_M <= dmem_size_m && dmem_start_n + SMEM_N <= dmem_size_n) {
			if ((ldd & 0x3) == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 4) {
					const auto i = i_offset + threadIdx.x * 4;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::A_smem_skew);

					// 128 bit memory access
					cutf::cp_async::cp_async<16>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else if ((ldd & 0x1) == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 2) {
					const auto i = i_offset + threadIdx.x * 2;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::A_smem_skew);

					// 64 bit memory access
					cutf::cp_async::cp_async<8>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::A_smem_skew);

					cutf::cp_async::cp_async<4>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			}
		} else {
			for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
				const auto i = i_offset + threadIdx.x;
				const auto m = (i % SMEM_M) + dmem_start_m;
				const auto n = (i / SMEM_M) + dmem_start_n;
				const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
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
			const unsigned dmem_start_m, const unsigned dmem_start_n,
			const unsigned dmem_size_m, const unsigned dmem_size_n,
			const half* const dmem_ptr, const unsigned ldd
			) {
		static_assert(SMEM_M * SMEM_N >= BLOCK_SIZE * 8, "SMEM_M * SMEM_N >= BLOCK_SIZE must be satisfied");
		if (dmem_start_m + SMEM_M <= dmem_size_m && dmem_start_n + SMEM_N <= dmem_size_n) {
			if ((ldd & 0x7) == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 8) {
					const auto i = i_offset + threadIdx.x * 8;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::B_smem_skew);

					cutf::cp_async::cp_async<16>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else if ((ldd & 0x3) == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 4) {
					const auto i = i_offset + threadIdx.x * 4;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::B_smem_skew);

					cutf::cp_async::cp_async<8>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else if ((ldd & 0x1) == 0) {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 2) {
					const auto i = i_offset + threadIdx.x * 2;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::B_smem_skew);

					cutf::cp_async::cp_async<4>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::B_smem_skew);

					smem_ptr[smem_index] = dmem_ptr[dmem_index];
				}
			}
		} else {
			for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
				const auto i = i_offset + threadIdx.x;
				const auto m = (i % SMEM_M) + dmem_start_m;
				const auto n = (i / SMEM_M) + dmem_start_n;
				const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
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
			const unsigned dmem_start_m, const unsigned dmem_start_n,
			const unsigned dmem_size_m, const unsigned dmem_size_n,
			const T* const dmem_ptr, const unsigned ldd
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
			const unsigned dmem_start_m, const unsigned dmem_start_n,
			const unsigned dmem_size_m, const unsigned dmem_size_n,
			const T* const dmem_ptr, const unsigned ldd
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
			T* const dmem_ptr, const unsigned ldd,
			const unsigned dmem_start_m, const unsigned dmem_start_n,
			const unsigned dmem_size_m, const unsigned dmem_size_n,
			const T* const smem_ptr,
			const float alpha, const float beta
			) {
		if (beta == 0.f) {
			if (dmem_start_m + SMEM_M < dmem_size_m && dmem_start_n + SMEM_N < dmem_size_n) {
				if ((ldd & 0x3) == 0) {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 4) {
						const auto i = i_offset + threadIdx.x * 4;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
						const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::C_smem_skew);

						// 128 bit memory access
						auto v = *reinterpret_cast<const float4*>(&smem_ptr[smem_index]);
						v.x *= alpha;
						v.y *= alpha;
						v.z *= alpha;
						v.w *= alpha;
						*reinterpret_cast<float4*>(&dmem_ptr[dmem_index]) = v;
					}
				} else if ((ldd & 0x1) == 0) {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 2) {
						const auto i = i_offset + threadIdx.x * 2;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
						const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::C_smem_skew);

						// 64 bit memory access
						auto v = *reinterpret_cast<const float2*>(&smem_ptr[smem_index]);
						v.x *= alpha;
						v.y *= alpha;
						*reinterpret_cast<float2*>(&dmem_ptr[dmem_index]) = v;
					}
				} else {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
						const auto i = i_offset + threadIdx.x;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
						const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::C_smem_skew);

						dmem_ptr[dmem_index] = smem_ptr[smem_index] * alpha;
					}
				}
			} else {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::C_smem_skew);

					if (m >= dmem_size_m || n >= dmem_size_n) {
						continue;
					}

					dmem_ptr[dmem_index] = smem_ptr[smem_index] * alpha;
				}
			}
		} else {
			if (dmem_start_m + SMEM_M < dmem_size_m && dmem_start_n + SMEM_N < dmem_size_n) {
				if ((ldd & 0x3) == 0) {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 4) {
						const auto i = i_offset + threadIdx.x * 4;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
						const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::C_smem_skew);

						// 128 bit memory access
						auto v = *reinterpret_cast<const float4*>(&smem_ptr[smem_index]);
						const auto w = *reinterpret_cast<float4*>(&dmem_ptr[dmem_index]);
						v.x = v.x * alpha + w.x * beta;
						v.y = v.y * alpha + w.y * beta;
						v.z = v.z * alpha + w.z * beta;
						v.w = v.w * alpha + w.w * beta;
						*reinterpret_cast<float4*>(&dmem_ptr[dmem_index]) = v;
					}
				} else if ((ldd & 0x1) == 0) {
					for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE * 2) {
						const auto i = i_offset + threadIdx.x * 2;
						const auto m = (i % SMEM_M) + dmem_start_m;
						const auto n = (i / SMEM_M) + dmem_start_n;
						const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
						const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::C_smem_skew);

						// 64 bit memory access
						auto v = *reinterpret_cast<const float2*>(&smem_ptr[smem_index]);
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
						const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
						const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::C_smem_skew);

						dmem_ptr[dmem_index] = smem_ptr[smem_index] * alpha + dmem_ptr[dmem_index] + beta;
					}
				}
			} else {
				for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
					const auto i = i_offset + threadIdx.x;
					const auto m = (i % SMEM_M) + dmem_start_m;
					const auto n = (i / SMEM_M) + dmem_start_n;
					const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
					const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::C_smem_skew);

					if (m >= dmem_size_m || n >= dmem_size_n) {
						continue;
					}

					dmem_ptr[dmem_index] = smem_ptr[smem_index] * alpha + dmem_ptr[dmem_index] * beta;
				}
			}
		}
	}
};

template <mtk::shgemm::operation_t op, class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct dmem_storer {
	__device__ void operator()(
			T* const dmem_ptr, const unsigned ldd,
			const unsigned dmem_start_m, const unsigned dmem_start_n,
			const unsigned dmem_size_m, const unsigned dmem_size_n,
			const T* const smem_ptr,
			const float alpha, const float beta
			) {
		dmem_storer_n<T, SMEM_M, SMEM_N, BLOCK_SIZE>{}(
				dmem_ptr, ldd,
				dmem_start_m, dmem_start_n,
				dmem_size_m, dmem_size_n,
				smem_ptr,
				alpha, beta
				);
	}
};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct dmem_storer<mtk::shgemm::op_t, T, SMEM_M, SMEM_N, BLOCK_SIZE> {
	__device__ void operator()(
			T* const dmem_ptr, const unsigned ldd,
			const unsigned dmem_start_m, const unsigned dmem_start_n,
			const unsigned dmem_size_m, const unsigned dmem_size_n,
			const T* const smem_ptr,
			const float alpha, const float beta
			) {
		dmem_storer_n<T, SMEM_N, SMEM_M, BLOCK_SIZE>{}(
				dmem_ptr, ldd,
				dmem_start_n, dmem_start_m,
				dmem_size_n, dmem_size_m,
				smem_ptr,
				alpha, beta
				);
	}
};

// -----------------------------------------
// N - atomic storer
// -----------------------------------------
template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct dmem_atomic_storer_n {
	__device__ void operator()(
			T* const dmem_ptr, const unsigned ldd,
			const unsigned dmem_start_m, const unsigned dmem_start_n,
			const unsigned dmem_size_m, const unsigned dmem_size_n,
			const T* const smem_ptr,
			const float alpha, const float beta
			) {
		for (unsigned i_offset = 0; i_offset < SMEM_M * SMEM_N; i_offset += BLOCK_SIZE) {
			const auto i = i_offset + threadIdx.x;
			const auto m = (i % SMEM_M) + dmem_start_m;
			const auto n = (i / SMEM_M) + dmem_start_n;
			const auto dmem_index = m + n * static_cast<std::uint64_t>(ldd);
			const auto smem_index = (i % SMEM_M) + (i / SMEM_M) * (SMEM_M + mtk::shgemm::device::C_smem_skew);

			if (m >= dmem_size_m || n >= dmem_size_n) {
				continue;
			}

			atomicAdd(&dmem_ptr[dmem_index], smem_ptr[smem_index] * alpha);
		}
	}
};

template <mtk::shgemm::operation_t op, class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct dmem_atomic_storer {
	__device__ void operator()(
			T* const dmem_ptr, const unsigned ldd,
			const unsigned dmem_start_m, const unsigned dmem_start_n,
			const unsigned dmem_size_m, const unsigned dmem_size_n,
			const T* const smem_ptr,
			const float alpha, const float beta
			) {
		dmem_atomic_storer_n<T, SMEM_M, SMEM_N, BLOCK_SIZE>{}(
				dmem_ptr, ldd,
				dmem_start_m, dmem_start_n,
				dmem_size_m, dmem_size_n,
				smem_ptr,
				alpha, beta
				);
	}
};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
struct dmem_atomic_storer<mtk::shgemm::op_t, T, SMEM_M, SMEM_N, BLOCK_SIZE> {
	__device__ void operator()(
			T* const dmem_ptr, const unsigned ldd,
			const unsigned dmem_start_m, const unsigned dmem_start_n,
			const unsigned dmem_size_m, const unsigned dmem_size_n,
			const T* const smem_ptr,
			const float alpha, const float beta
			) {
		dmem_atomic_storer_n<T, SMEM_N, SMEM_M, BLOCK_SIZE>{}(
				dmem_ptr, ldd,
				dmem_start_n, dmem_start_m,
				dmem_size_n, dmem_size_m,
				smem_ptr,
				alpha, beta
				);
	}
};
} // namespace device
} // namespace shgemm
} // namespace mtk
#endif
