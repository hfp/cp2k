/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../offload/offload_runtime.h"
#if defined(__OFFLOAD_OPENCL) && !defined(__NO_OFFLOAD_DBM)

#include "dbm_multiply_gpu_kernel.h"
#include "dbm_multiply_opencl.cl.h"

#if !defined(DBM_MULTIPLY_OPENCL_BATCHSIZE)
#define DBM_MULTIPLY_OPENCL_BATCHSIZE 1
#endif

void dbm_multiply_gpu_launch_kernel(const offloadStream_t stream,
                                    const int mnk_range[3][2], double alpha,
                                    int ntasks, const dbm_task_t *tasks,
                                    const double *pack_a_data,
                                    const double *pack_b_data,
                                    double *shard_c_data) {
  static cl_kernel kernel = NULL;
  int result = EXIT_SUCCESS, verbosity = c_dbcsr_acc_opencl_config.verbosity;
  cl_event event, *const perf_event =
                      ((0 <= verbosity && 2 >= verbosity) ? NULL : &event);
  const c_dbcsr_acc_opencl_stream_t *const str = ACC_OPENCL_STREAM(stream);
  const size_t amount = ntasks, wgsize = 0;
  const size_t work_size = /* consider intra-kernel batchsize */
      ((size_t)ntasks * mnk_range[0][1] + DBM_MULTIPLY_OPENCL_BATCHSIZE - 1) /
      DBM_MULTIPLY_OPENCL_BATCHSIZE;
  size_t offset_batch = 0, offset_adata = 0, offset_bdata = 0, offset_cdata = 0;
  c_dbcsr_acc_opencl_info_memptr_t adata, bdata, cdata, batch;
  assert(NULL != pack_a_data && NULL != pack_b_data && NULL != shard_c_data);
  assert(0 < mnk_range[0][0] && 0 < mnk_range[0][1] &&
         mnk_range[0][0] <= mnk_range[0][1]);
  assert(0 < mnk_range[1][0] && 0 < mnk_range[1][1] &&
         mnk_range[1][0] <= mnk_range[1][1]);
  assert(0 < mnk_range[2][0] && 0 < mnk_range[2][1] &&
         mnk_range[2][0] <= mnk_range[2][1]);
  assert(NULL != str && NULL != str->queue);
  assert(0 < ntasks && NULL != tasks);
  /* creating/calling kernel must be consistent across threads */
  ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_main);
  result |= c_dbcsr_acc_opencl_info_devptr_lock(&adata, NULL /*lock*/,
                                                pack_a_data, 1 /*esize*/,
                                                NULL /*amount*/, &offset_adata);
  result |= c_dbcsr_acc_opencl_info_devptr_lock(&bdata, NULL /*lock*/,
                                                pack_b_data, 1 /*esize*/,
                                                NULL /*amount*/, &offset_bdata);
  result |= c_dbcsr_acc_opencl_info_devptr_lock(&cdata, NULL /*lock*/,
                                                shard_c_data, 1 /*esize*/,
                                                NULL /*amount*/, &offset_cdata);
  result |= c_dbcsr_acc_opencl_info_devptr_lock(
      &batch, NULL /*lock*/, tasks /*batch*/, sizeof(dbm_task_t), &amount,
      &offset_batch);
  assert(0 == offset_adata && 0 == offset_bdata && 0 == offset_cdata);
#if defined(OPENCL_DBM_SOURCE_MULTIPLY_OPENCL)
  if (NULL == kernel) { /* first-time check if kernel is present */
    char build_params[ACC_OPENCL_BUFFERSIZE] =
        "-DBS=" #DBM_MULTIPLY_OPENCL_BATCHSIZE " ";
    const char *extensions[] = {NULL, NULL};
    const size_t nextensions = sizeof(extensions) / sizeof(*extensions);
    const libxsmm_timer_tickint start = libxsmm_timer_tick();
    const build_params_size = sizeof(build_params) - strlen(build_params);
    const int nchar = c_dbcsr_acc_opencl_flags_atomics(
        &c_dbcsr_acc_opencl_config.device, c_dbcsr_acc_opencl_atomic_fp_64,
        extensions, nextensions, build_params, build_params_size);
    if (0 < nchar && (int)build_params_size > nchar) {
      const int result_kernel = c_dbcsr_acc_opencl_kernel(
          0 /*source_is_file*/, OPENCL_DBM_SOURCE_MULTIPLY_OPENCL,
          "dbm_multiply", build_params,
          0 == c_dbcsr_acc_opencl_config.debug
              ? "-cl-fast-relaxed-math -cl-denorms-are-zero"
              : NULL,
          NULL /*try*/, NULL /*try_ok*/, extensions, nextensions, &kernel);
      result |= result_kernel;
      if (2 <= verbosity || 0 > verbosity) {
        if (EXIT_SUCCESS == result_kernel) {
          const double duration =
              libxsmm_timer_duration(start, libxsmm_timer_tick());
          fprintf(stderr, "INFO ACC/LIBDBM: DBM-kernel bs=%i ms=%.1f\n",
                  DBM_MULTIPLY_OPENCL_BATCHSIZE, 1E3 * duration);
        } else {
          fprintf(stderr, "INFO ACC/LIBDBM: DBM-kernel failed to generate\n");
        }
      }
    }
  }
#else
#error "OpenCL kernel code not found!"
#endif
  result |= clSetKernelArg(kernel, 0, sizeof(cl_double), &alpha);
  result |= clSetKernelArg(kernel, 1, sizeof(cl_int), &mnk_range[0][1]);
  result |= clSetKernelArg(kernel, 2, sizeof(cl_int), &mnk_range[1][1]);
  result |= clSetKernelArg(kernel, 3, sizeof(cl_int), &offset_batch);
  result |= clSetKernelArg(kernel, 4, sizeof(cl_int), &ntasks);
  result |= c_dbcsr_acc_opencl_set_kernel_ptr(kernel, 5, batch.memory);
  result |= c_dbcsr_acc_opencl_set_kernel_ptr(kernel, 6, adata.memory);
  result |= c_dbcsr_acc_opencl_set_kernel_ptr(kernel, 7, bdata.memory);
  result |= c_dbcsr_acc_opencl_set_kernel_ptr(kernel, 8, cdata.memory);
  result |= clEnqueueNDRangeKernel(str->queue, kernel, 1 /*work_dim*/,
                                   NULL /*offset*/, &work_size,
                                   0 != wgsize ? &wgsize : NULL, 0 /*num_wait*/,
                                   NULL /*wait_list*/, perf_event);
  if (NULL != perf_event && EXIT_SUCCESS == result) {
    cl_ulong begin = 0, end = 0;
    clWaitForEvents(1, perf_event);
    result |= clGetEventProfilingInfo(*perf_event, CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &begin, NULL);
    result |= clGetEventProfilingInfo(*perf_event, CL_PROFILING_COMMAND_END,
                                      sizeof(cl_ulong), &end, NULL);
    if (EXIT_SUCCESS == result) {
      const double duration =
          1E-9 * LIBXSMM_DELTA(begin, end); /* Nanoseconds->seconds */
      const double gflops = (2ULL * mnk_range[0][1] * mnk_range[1][1] *
                             mnk_range[2][1] * ntasks) *
                            1E-9 / duration;
      fprintf(stderr,
              "INFO ACC/LIBDBM: DBM-kernel mnk=%ix%ix%i "
              "ntasks=%i gflops=%.1f ms=%.2g\n",
              mnk_range[0][1], mnk_range[1][1], mnk_range[2][1], ntasks, gflops,
              1E3 * duration);
    }
  }
  ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_main);
  OFFLOAD_CHECK(result);
}

#endif // defined(__OFFLOAD_OPENCL) && !defined(__NO_OFFLOAD_DBM)

// EOF
