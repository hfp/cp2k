# OpenCL Backend

The OpenCL backend of DBM is maintained in
[LIBXSTREAM](https://github.com/hfp/libxstream/tree/main/samples/dbm), which is a prerequisite of
CP2K's OpenCL support anyway. CP2K compiles `dbm_opencl.c` from there and generates the kernel
header from `kernels/dbm_multiply.cl`; `dbm_multiply_opencl.c` merely adapts it to the interface
shared with the CUDA and HIP backends. The backend and its environment variables are documented
alongside the code in LIBXSTREAM.
