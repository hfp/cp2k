PROJHOME := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CP2KHOME := $(abspath $(PROJHOME)/../..)
ALL_HEADERS := $(shell find . -name "*.h") $(shell find ../offload/ -name "*.h")
ALL_OBJECTS := ../offload/offload_library.o \
        dbm_distribution.o \
        dbm_library.o \
        dbm_matrix.o \
        dbm_mempool.o \
        dbm_mpi.o \
        dbm_multiply.o \
        dbm_multiply_comm.o \
        dbm_multiply_cpu.o \
        dbm_shard.o

.PHONY : all clean

all: dbm_miniapp.x

clean:
	rm -fv *.o */*.o ../offload/*.o

realclean: clean
	rm -fv *.x

CFLAGS := -fopenmp -g -O3 -march=native -Wall -Wextra -Wcast-qual -Wno-vla-parameter

NVCC := $(shell which nvcc)
NVARCH := sm_70
NVFLAGS := -g -O3 -lineinfo -arch $(NVARCH) -Wno-deprecated-gpu-targets -Xcompiler "$(CFLAGS)" -D__OFFLOAD_CUDA

HIPCC := $(shell which hipcc)
HIPARCH := gfx90a
HIPFLAGS := -fPIE -g -O3 --offload-arch=$(HIPARCH) -Wall -Wextra -Werror -I${ROCM_PATH}/include -D__OFFLOAD_HIP -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_AMD__

# If DBCSR library exists, prefer OpenCL because, e.g., working NVCC implies functioning OpenCL
OPENCL_OFFLOAD := $(firstword $(wildcard $(CP2KHOME)/lib/*/*/exts/dbcsr/libdbcsr.a))


# Make foundational runtimes available
LIBS += -fopenmp -ldl -lstdc++ -lc -lm


# Make BLAS/LAPACK available
ifneq ($(MKLROOT),)
LIBS += \
        -L$(MKLROOT)/lib/intel64 \
        -Wl,--start-group \
        -lmkl_gf_lp64 \
        -lmkl_core \
        -lmkl_sequential \
        -Wl,--end-group
else
LIBS += -lblas
endif


# Enable OpenCL when DBCSR library was prebuilt (assume __DBCSR_ACC).
ifneq ($(OPENCL_OFFLOAD),)
  KERNEL_SRC := $(PROJHOME)/dbm_multiply_gpu_kernel.cl
  KERNEL_GEN := $(PROJHOME)/dbm_multiply_gpu_opencl.h
  KERNEL_SCR := $(CP2KHOME)/exts/dbcsr/src/acc/opencl/acc_opencl.sh
  ALL_HEADERS += $(KERNEL_GEN)

  ALL_OBJECTS += dbm_multiply_gpu.o
  CFLAGS += -I$(CP2KHOME)/exts/dbcsr -D__OFFLOAD_OPENCL -D__DBCSR_ACC
  LIBS += $(OPENCL_OFFLOAD) -lgfortran

  ifneq (Darwin,$(shell uname))
    OPENCL_LIB := $(shell ldconfig -p 2>/dev/null | grep -m1 OpenCL | rev | cut -d' ' -f1 | rev)
    ifeq (,$(OPENCL_LIB))
      OPENCL_LIB := $(wildcard /usr/lib/x86_64-linux-gnu/libOpenCL.so.1)
    endif
    ifeq (,$(CUDATOOLKIT_HOME))
      CUDATOOLKIT_HOME := $(NVSDKCOMPUTE_ROOT)
    endif
    ifeq (,$(CUDATOOLKIT_HOME))
      CUDATOOLKIT_HOME := $(if $(NVCC),$(abspath $(dir $(NVCC))/..))
    endif
    ifneq (,$(CUDATOOLKIT_HOME))
      CUDA_LIBDIR := $(if $(wildcard $(CUDATOOLKIT_HOME)/lib64),lib64,lib)
      ifeq (,$(wildcard $(OPENCL_INC)))
        CLINC := $(strip $(lastword $(sort $(wildcard $(CUDATOOLKIT_HOME)/../cuda/*/targets/x86_64-linux/include/CL/cl.h))))
        ifneq (,$(CLINC))
          OPENCL_INC := $(abspath $(dir $(CLINC))/..)
        else
          CFLAGS += -I$(CUDATOOLKIT_HOME)/include
        endif
      endif
      ifeq (,$(wildcard $(OPENCL_LIB)))
        LIBS += -L$(CUDATOOLKIT_HOME)/$(CUDA_LIBDIR)
        LIBS += -Wl,-rpath=$(CUDATOOLKIT_HOME)/$(CUDA_LIBDIR)
      endif
    endif
    # OPENCL_INC: directory containing CL/cl.h.
    ifneq (,$(wildcard $(OPENCL_INC)))
      CFLAGS += -I$(OPENCL_INC)
    endif
    # OPENCL_LIB: file/library to be linked
    ifneq (,$(wildcard $(OPENCL_LIB)))
      LIBS += $(OPENCL_LIB)
    else
      LIBS += -l:libOpenCL.so.1
    endif
  else # macOS
    LIBS += -framework OpenCL
  endif

$(KERNEL_GEN): $(KERNEL_SCR) $(KERNEL_SRC)
	@$(KERNEL_SCR) $(KERNEL_SRC) /dev/null $@


# Enable Cuda when nvcc compiler is present.
else ifneq ($(NVCC),)
ALL_OBJECTS += dbm_multiply_gpu.o dbm_multiply_gpu_kernel.o
CFLAGS += -I${CUDA_PATH}/include -D__OFFLOAD_CUDA
LIBS += -lcudart -lcuda -lcublas -L${CUDA_PATH}/lib64

%.o: %.cu $(ALL_HEADERS)
	cd $(dir $<); $(NVCC) -c $(NVFLAGS) $(notdir $<)


# Enable HIP/ROCm when hipcc compiler is present.
else ifneq ($(HIPCC),)
ALL_OBJECTS += dbm_multiply_gpu.o
CFLAGS += -I${ROCM_PATH}/include -D__OFFLOAD_HIP -D__HIP_PLATFORM_AMD__
LIBS += -L${ROCM_PATH}/lib -lamdhip64 -lhipfft -lhipblas -lrocblas

%.o: %.cu $(ALL_HEADERS)
	cd $(dir $<); $(HIPCC) -c $(HIPFLAGS) $(notdir $<)
endif


# Make LIBXSMM available
ifneq ($(LIBXSMMROOT),)
CFLAGS += -D__LIBXSMM -I$(LIBXSMMROOT)/include
LIBS += \
        $(LIBXSMMROOT)/lib/libxsmm.a $(LIBXSMMROOT)/lib/libxsmmext.a \
        -lpthread -lrt
endif


# Build with MPI when mpicc compiler is present.
ifneq ($(MPICC),)
%.o: %.c $(ALL_HEADERS)
	cd $(dir $<); $(MPICC) -c -std=c11 $(CFLAGS) -D__parallel $(notdir $<)

dbm_miniapp.x: dbm_miniapp.o $(ALL_OBJECTS)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LIBS)


# Build without MPI otherwise.
else
%.o: %.c $(ALL_HEADERS)
	cd $(dir $<); $(CC) -c -std=c11 $(CFLAGS) $(notdir $<)

dbm_miniapp.x: dbm_miniapp.o $(ALL_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

endif
#EOF
