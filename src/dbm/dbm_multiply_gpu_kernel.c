/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2023 CP2K developers group <https://cp2k.org>              */
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
#pragma omp critical(c_dbcsr_acc_set_active_device)
#endif
  { /* creating/calling kernel/clSetKernelArg must be consistent across threads */
#if defined(OPENCL_DBM_SOURCE_MULTIPLY_GPU_KERNEL)
    if (NULL == kernel) { /* first-time check if kernel is present */
      OFFLOAD_CHECK(c_dbcsr_acc_opencl_kernel(0 /*source_is_file*/,
        OPENCL_DBM_SOURCE_MULTIPLY_GPU_KERNEL, "dbm_multiply",
        NULL /*build_params*/, NULL /*build_options*/,
        NULL /*try*/, NULL /*try_ok*/, NULL /*extnames*/, 0 /*num_exts*/,
        &kernel));
    }
#endif
    OFFLOAD_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_int), &m_max));
    OFFLOAD_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_int), &n_max));
    OFFLOAD_CHECK(
        clSetKernelArg(kernel, 2, sizeof(cl_double), &alpha));
    OFFLOAD_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &batch));
    OFFLOAD_CHECK(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &pack_a_data));
    OFFLOAD_CHECK(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &pack_b_data));
    OFFLOAD_CHECK(
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &shard_c_data));
    OFFLOAD_CHECK(clEnqueueNDRangeKernel(
        queue, kernel, 1 /*work_dim*/, NULL /*offset*/, &work_size,
        &wgsize, 0 /*num_wait*/, NULL /*wait_list*/, perf_event));
  }
}

#endif // defined(__OFFLOAD_OPENCL) && !defined(__NO_OFFLOAD_DBM)

// EOF
