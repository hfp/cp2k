PROJHOME := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CP2KHOME := $(abspath $(PROJHOME)/../..)
OMP_TRACE := $(wildcard $(PROJHOME)/../base/openmp_trace.c)
ALL_HEADERS := $(shell find . -name "*.h") $(shell find ../offload/ -name "*.h")
ALL_OBJECTS :=  $(OMP_TRACE:.c=.o) \
        ../offload/offload_library.o \
        ../offload/offload_mempool.o \
        ../mpiwrap/cp_mpi.o \
        dbm_distribution.o \
        dbm_library.o \
        dbm_matrix.o \
        dbm_multiply.o \
        dbm_multiply_comm.o \
        dbm_multiply_cpu.o \
        dbm_shard.o

# Optimization level
ifeq (,$(filter-out 0,$(DBG)))
OPT ?= 3
else
OPT ?= 0
endif

# Optimization flag derived from OPT flag
OPTFLAG ?= -O$(patsubst O%,%,$(OPT))

# Common compiler flags
CFLAGS := -fPIC -g $(OPTFLAG) -march=native -Wall -Wextra -Wcast-qual

USM ?= 0
ifneq ($(USM),0)
CFLAGS += -D__OFFLOAD_UNIFIED_MEMORY
endif

# Intel Compiler
ICX := $(shell which icx 2>/dev/null)
INTEL ?= $(if $(ICX),$(if $(filter-out 0,$(GNU)),0,1),0)

# Build with MPI when MPICC is given
MPI ?= $(if $(MPICC),1,0)

ifeq ($(INTEL),0)
MKL_FCRTL := gf
LIBS += $(if $(OMPRT),-l$(OMPRT),-fopenmp)
CFLAGS += -fopenmp -Wno-vla-parameter
CC := $(if $(filter-out 0,$(MPI)),mpicc,gcc)
else
MKL_FCRTL := intel
LIBS += $(if $(OMPRT),-fopenmp -Wno-recommended-option,-qopenmp)
CFLAGS += $(if $(OMPRT),-fopenmp -Wno-recommended-option,-qopenmp)
CC := $(if $(filter-out 0,$(MPI)),mpiicx,icx)
endif

ifneq ($(MPI),0)
CFLAGS += -D__parallel
ifneq ($(MPICC),)
CC := $(MPICC)
endif
endif

ifeq ($(filter-out 0,$(OPENCL)),)
NVCC := $(shell which nvcc 2>/dev/null)
NVARCH := sm_70
NVARCHX := $(if $(NVARCH),-arch $(NVARCH))
NVFLAGS := -g $(OPTFLAG) -lineinfo $(NVARCHX) -Wno-deprecated-gpu-targets -Xcompiler "$(CFLAGS)" -D__OFFLOAD_CUDA

HIPCC := $(shell which hipcc 2>/dev/null)
HIPARCH := gfx90a
ROCM_PATH_ENV := $(shell echo "$${ROCM_PATH}")
ROCM_PATH := $(if $(ROCM_PATH_ENV),$(ROCM_PATH_ENV),/opt/rocm)
HIPFLAGS := -fPIE -g $(OPTFLAG) --offload-arch=$(HIPARCH) -Wall -Wextra -Werror -I${ROCM_PATH}/include -D__OFFLOAD_HIP -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_AMD__
endif

# prefer NVCC or HIPCC over OpenCL
ifneq ($(NVCC)$(HIPCC),)
NVCC_PATH := $(if $(NVCC),$(wildcard $(dir $(NVCC))/..))
CUDA_FILE := $(wildcard $(NVCC_PATH)/../cuda/include/cuda.h)
CUDA_PATH := $(if $(CUDA_FILE),$(NVCC_PATH)/../cuda,$(NVCC_PATH))
CUDA_LIBS := $(if $(wildcard $(CUDA_PATH)/lib64),lib64,lib)
OPENCL := 0
endif

# Make foundational runtimes available
LIBS += -ldl -lstdc++ -lc -lm


# Make BLAS/LAPACK available
ifneq ($(MKLROOT),)
LIBS += -L$(MKLROOT)/lib/intel64 \
        -Wl,--start-group \
        -lmkl_$(MKL_FCRTL)_lp64 \
        -lmkl_core \
        -lmkl_sequential \
        -Wl,--end-group
else ifneq ($(OPENBLAS),0)
LIBS += -lopenblas
else
LIBS += -lblas
endif

.PHONY : all clean

all: dbm_miniapp.x

clean:
	rm -fv $(ALL_OBJECTS) \
        dbm_multiply_gpu.o \
        dbm_multiply_gpu_kernel.o \
        dbm_miniapp.o

realclean: clean
	rm -fv dbm_miniapp.x


# Enable Cuda when nvcc compiler is present.
ifneq ($(NVCC),)
ALL_OBJECTS += dbm_multiply_gpu.o dbm_multiply_gpu_kernel.o
CFLAGS += -I${CUDA_PATH}/include -D__OFFLOAD_CUDA
LIBS += -lcudart -lcuda -lcublas -L${CUDA_PATH}/$(CUDA_LIBS)
ifneq ($(wildcard $(NVCC_PATH)/../math_libs/$(CUDA_LIBS)),)
LIBS += -L$(NVCC_PATH)/../math_libs/$(CUDA_LIBS)
endif

%.o: %.cu $(ALL_HEADERS)
	cd $(dir $<); $(NVCC) -c $(NVFLAGS) $(notdir $<)


# Enable HIP/ROCm when hipcc compiler is present.
else ifneq ($(HIPCC),)
ALL_OBJECTS += dbm_multiply_gpu.o dbm_multiply_gpu_kernel.o
CFLAGS += -I${ROCM_PATH}/include -D__OFFLOAD_HIP -D__HIP_PLATFORM_AMD__
LIBS += -L${ROCM_PATH}/lib -lamdhip64 -lhipfft -lhipblas -lrocblas

%.o: %.cu $(ALL_HEADERS)
	cd $(dir $<); $(HIPCC) -c $(HIPFLAGS) $(notdir $<)
endif

# Make LIBXSMM available
LIBXSMMROOT ?= $(wildcard $(CP2KHOME)/../libxsmm)
ifneq ($(LIBXSMMROOT),)
CFLAGS += -D__LIBXSMM -I$(LIBXSMMROOT)/include
LIBS += $(LIBXSMMROOT)/lib/libxsmm.a \
        $(LIBXSMMROOT)/lib/libxsmmext.a \
        -lpthread -lrt
endif

# Make LIBXS available (wraps BLAS, MKL JIT, and LIBXSMM behind a common API)
LIBXSROOT ?= $(wildcard $(CP2KHOME)/../libxs)
ifneq ($(LIBXSROOT),)
CFLAGS += -I$(LIBXSROOT)/include
endif

# Make LIBXSTREAM available (OpenCL offload backend).
# Auto-detect as sibling of CP2KHOME when OpenCL is confirmed.
LIBXSTREAMROOT ?= $(wildcard $(CP2KHOME)/../libxstream)
ifneq ($(LIBXSTREAMROOT),)
# Detect OpenCL: specific locations take precedence, then confirm implicit.
XSTREAM_OPENCL_INC ?=
XSTREAM_OPENCL_LIB ?=
# 1. OPENCL_ROOT (explicit, takes precedence)
ifeq (,$(XSTREAM_OPENCL_INC)$(XSTREAM_OPENCL_LIB))
ifneq (,$(wildcard $(OPENCL_ROOT)/include/CL/cl.h))
XSTREAM_OPENCL_INC := $(OPENCL_ROOT)/include
XSTREAM_OPENCL_LIB := -L$(OPENCL_ROOT)/$(if $(wildcard \
  $(OPENCL_ROOT)/lib64),lib64,lib) -lOpenCL
endif
endif
# 2. ICX/SYCL tree (specific)
ifeq (,$(XSTREAM_OPENCL_INC)$(XSTREAM_OPENCL_LIB))
ifneq (,$(ICX))
XSTREAM_OCL_ROOT := $(abspath $(dir $(ICX))/..)
XSTREAM_OCL_CLINC := $(firstword $(wildcard \
  $(XSTREAM_OCL_ROOT)/include/sycl/CL/cl.h \
  $(XSTREAM_OCL_ROOT)/include/CL/cl.h))
ifneq (,$(XSTREAM_OCL_CLINC))
XSTREAM_OPENCL_INC := $(abspath $(dir $(XSTREAM_OCL_CLINC))/..)
XSTREAM_OPENCL_LIB := -L$(XSTREAM_OCL_ROOT)/$(if $(wildcard \
  $(XSTREAM_OCL_ROOT)/lib64),lib64,lib) -lOpenCL
endif
endif
endif
# 3. Implicit/OS-packaged: confirm CL/cl.h is reachable and library exists.
ifeq (,$(XSTREAM_OPENCL_INC)$(XSTREAM_OPENCL_LIB))
ifneq (,$(wildcard /usr/include/CL/cl.h))
XSTREAM_OCL_SO := $(shell ldconfig -p 2>/dev/null | grep -m1 libOpenCL | \
  rev | cut -d' ' -f1 | rev)
ifeq (,$(XSTREAM_OCL_SO))
XSTREAM_OCL_SO := $(firstword $(wildcard \
  /usr/lib/x86_64-linux-gnu/libOpenCL.so* \
  /usr/lib64/libOpenCL.so* \
  /usr/lib/libOpenCL.so*))
endif
ifneq (,$(XSTREAM_OCL_SO))
XSTREAM_OPENCL_LIB := $(XSTREAM_OCL_SO)
endif
endif
endif
# Proceed if OpenCL was confirmed.
ifneq (,$(XSTREAM_OPENCL_LIB))
CFLAGS += -D__LIBXSTREAM -I$(LIBXSTREAMROOT)/include
ifneq (,$(XSTREAM_OPENCL_INC))
CFLAGS += -I$(XSTREAM_OPENCL_INC)
endif
LIBS += $(LIBXSTREAMROOT)/lib/libxstream.a $(XSTREAM_OPENCL_LIB)
endif
endif

# LIBXS must be linked last (depended upon by LIBXSTREAM)
ifneq ($(LIBXSROOT),)
LIBS += $(LIBXSROOT)/lib/libxs.a
endif

%.o: %.c $(OMP_TRACE) $(ALL_HEADERS)
	cd $(dir $<); $(CC) -c -std=c11 $(CFLAGS) $(notdir $<)

dbm_miniapp.x: dbm_miniapp.o $(ALL_OBJECTS)
	$(CC) -o $@ $^ $(LIBS)

#EOF
