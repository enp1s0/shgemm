# shgemm

## Build
```
git clone https://github.com/enp1s0/shgemm --recursive
cd shgemm

cd test/mateval
mkdir build
cd build
cmake ..
make -j4

cd ../../gpu_monitor
mkdir build
cd build
cmake ..
make -j4

cd ../../../
mkdir build
cd build
cmake ..
make -j4
```

## Test
Before building the library, please change `BUILD_SHGEMM_TEST` in CMakeLists.txt to `ON`.
```
cd /path/to/shgemm
./build/shgemm.test
```

## License
MIT
