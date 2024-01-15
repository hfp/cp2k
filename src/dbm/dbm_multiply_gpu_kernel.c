/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../offload/offload_runtime.h"
#if defined(__OFFLOAD_OPENCL) && !defined(__NO_OFFLOAD_DBM)

#include "dbm_multiply_gpu_kernel.cl.h"
#include "dbm_multiply_gpu_kernel.h"

void dbm_multiply_gpu_launch_kernel(const offloadStream_t stream, int m_max,
                                    int n_max, double alpha, int ntasks,
                                    const dbm_task_t *batch,
                                    const double *pack_a_data,
                                    const double *pack_b_data,
                                    double *shard_c_data) {
  cl_event event, *const perf_event =
                      ((c_dbcsr_acc_opencl_timer_host ==
                            c_dbcsr_acc_opencl_config.timer ||
                        (0 <= c_dbcsr_acc_opencl_config.verbosity &&
                         2 >= c_dbcsr_acc_opencl_config.verbosity))
                           ? NULL
                           : &event);
  const cl_command_queue queue =
      (NULL != stream ? *ACC_OPENCL_STREAM(stream)
                      : c_dbcsr_acc_opencl_stream_default());
  const size_t work_size = (size_t)ntasks * n_max, wgsize = 0;
  cl_kernel kernel = NULL;
  assert(NULL != pack_a_data && NULL != pack_b_data && NULL != shard_c_data);
  assert(0 < ntasks && 0 < m_max && n_max);
  assert(NULL != batch && NULL != queue);
#if defined(_OPENMP)
#pragma omp critical(dbm_multiply_gpu_launch_kernel)
#endif
  { /* creating/calling kernel must be consistent across threads */
    size_t offset = 0;
    if (NULL != c_dbcsr_acc_opencl_config.clmems) {
      void *const handle = c_dbcsr_acc_opencl_info_devptr(batch, NULL /*amount*/, &offset);
      if (NULL != handle && 0 != offset) batch = *(const dbm_task_t**)handle;
    }
#if defined(OPENCL_DBM_SOURCE_MULTIPLY_GPU_KERNEL)
    if (NULL == kernel) { /* first-time check if kernel is present */
      const c_dbcsr_acc_opencl_info_stream_t *const qinfo =
          c_dbcsr_acc_opencl_info_stream(stream);
      const c_dbcsr_acc_opencl_device_t *const devinfo =
          c_dbcsr_acc_opencl_config.device + qinfo->tid;
      char build_params[ACC_OPENCL_BUFFERSIZE];
      const char *extensions[] = {NULL, NULL};
      cl_device_id active_device = NULL;
      int nchar;
      OFFLOAD_CHECK(clGetCommandQueueInfo(
          queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &active_device, NULL));
      nchar = c_dbcsr_acc_opencl_flags_atomics(
          active_device, c_dbcsr_acc_opencl_atomic_fp_64, devinfo, extensions,
          sizeof(extensions) / sizeof(*extensions), build_params,
          sizeof(build_params));
      if (0 < nchar && (int)sizeof(build_params) > nchar) {
        OFFLOAD_CHECK(c_dbcsr_acc_opencl_kernel(
            0 /*source_is_file*/, OPENCL_DBM_SOURCE_MULTIPLY_GPU_KERNEL,
            "process_batch_kernel", build_params,
            "-cl-fast-relaxed-math -cl-denorms-are-zero", NULL /*try*/,
            NULL /*try_ok*/, extensions,
            sizeof(extensions) / sizeof(*extensions), &kernel));
      }
    }
#endif
    OFFLOAD_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_int), &m_max));
    OFFLOAD_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_int), &n_max));
    OFFLOAD_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_double), &alpha));
    OFFLOAD_CHECK(clSetKernelArg(kernel, 3, sizeof(size_t), &offset));
    OFFLOAD_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &batch));
    OFFLOAD_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &pack_a_data));
    OFFLOAD_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &pack_b_data));
    OFFLOAD_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), &shard_c_data));
    OFFLOAD_CHECK(
        clEnqueueNDRangeKernel(queue, kernel, 1 /*work_dim*/, NULL /*offset*/,
                               &work_size, 0 != wgsize ? &wgsize : NULL,
                               0 /*num_wait*/, NULL /*wait_list*/, perf_event));
  }
}

#endif // defined(__OFFLOAD_OPENCL) && !defined(__NO_OFFLOAD_DBM)

// EOF
