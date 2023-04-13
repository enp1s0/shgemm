# SHGEMM - Single and Half precision GEMM on Tensor Cores

## Build
```
git clone https://github.com/enp1s0/shgemm
cd shgemm
git submodule update --init --recursive

mkdir build
cd build
cmake ..
make -j4
```

## Usage
```cuda
// sample.cu
// nvcc sample.cu ... -lshgemm ...
#include <shgemm/shgemm.hpp>

mtk::shgemm::shgemmHandle_t shgemm_handle;
mtk::shgemm::create(shgemm_handle);

// Optional
mtk::shgemm::set_cuda_stream(shgemm_handle, cuda_stream);

const auto compute_type = mtk::shgemm::tf32;

// SHGEMM (A=float, B=half)
mtk::shgemm::shgemm(
			shgemm_handle,
			mtk::shgemm::op_n, mtk::shgemm::op_n,
			m, n, k,
			&alpha_fp32,
			a_fp32_ptr, lda,
			b_fp16_ptr, ldb,
			&beta_fp32,
			c_fp32_ptr, ldc,
			compute_type
			);

// HSGEMM (A=half, B=float) is also available
mtk::shgemm::hsgemm(
			shgemm_handle,
			mtk::shgemm::op_n, mtk::shgemm::op_n,
			m, n, k,
			&alpha_fp32,
			a_fp16_ptr, lda,
			b_fp32_ptr, ldb,
			&beta_fp32,
			c_fp32_ptr, ldc,
			compute_type
			);

mtk::shgemm::destroy(shgemm_handle);
```

## Test
Before building the library, please change `BUILD_SHGEMM_TEST` in CMakeLists.txt to `ON` and execute the building commonds again.
```
./build/shgemm.test
```

## Publication
```bibtex
@inproceedings{ootomo_shgemm_2023,
  author = {Ootomo, Hiroyuki and Yokota, Rio},
  title = {Mixed-Precision Random Projection for RandNLA on Tensor Cores},
  year = {2023},
  series = {PASC '23}
}
```

## License
MIT
