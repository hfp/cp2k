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

size_t dbm_multiply_gpu_worksize(int ntasks, int split, int *batchsize) {
  const int worksize =
      (int)(((size_t)ntasks * split + *batchsize - 1) / *batchsize);
  *batchsize = (int)(((size_t)ntasks * split + worksize - 1) / worksize);
  if (split < *batchsize) {
    *batchsize = split; /* limit batchsize */
  }
  return ((size_t)ntasks * split + *batchsize - 1) / *batchsize;
}

void dbm_multiply_gpu_launch_kernel(const offloadStream_t stream,
                                    const int m_range[2], const int n_range[2],
                                    double alpha, int ntasks,
                                    const dbm_task_t *tasks,
                                    const double *pack_a_data,
                                    const double *pack_b_data,
                                    double *shard_c_data) {
  static cl_kernel kernel = NULL;
  cl_event event, *const perf_event =
                      ((c_dbcsr_acc_opencl_timer_host ==
                            c_dbcsr_acc_opencl_config.timer ||
                        (0 <= c_dbcsr_acc_opencl_config.verbosity &&
                         2 >= c_dbcsr_acc_opencl_config.verbosity))
                           ? NULL
                           : &event);
  const c_dbcsr_acc_opencl_stream_t *const str = ACC_OPENCL_STREAM(stream);
  int batchsize = 1; /* intra-kernel batch-size */
  const size_t amount = ntasks, wgsize = 0;
  const size_t work_size =
      dbm_multiply_gpu_worksize(ntasks, m_range[1], &batchsize);
  size_t offset_batch = 0, offset_adata = 0, offset_bdata = 0, offset_cdata = 0;
  c_dbcsr_acc_opencl_info_memptr_t adata, bdata, cdata, batch;
  assert(NULL != pack_a_data && NULL != pack_b_data && NULL != shard_c_data);
  assert(0 < m_range[0] && 0 < m_range[1] && m_range[0] <= m_range[1]);
  assert(0 < n_range[0] && 0 < n_range[1] && n_range[0] <= n_range[1]);
  assert(NULL != str && NULL != str->queue);
  assert(0 < ntasks && NULL != tasks);
  /* creating/calling kernel must be consistent across threads */
  ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_main);
  OFFLOAD_CHECK(c_dbcsr_acc_opencl_info_devptr_lock(
      &adata, NULL /*lock*/, pack_a_data, 1 /*esize*/, NULL /*amount*/,
      &offset_adata));
  OFFLOAD_CHECK(c_dbcsr_acc_opencl_info_devptr_lock(
      &bdata, NULL /*lock*/, pack_b_data, 1 /*esize*/, NULL /*amount*/,
      &offset_bdata));
  OFFLOAD_CHECK(c_dbcsr_acc_opencl_info_devptr_lock(
      &cdata, NULL /*lock*/, shard_c_data, 1 /*esize*/, NULL /*amount*/,
      &offset_cdata));
  OFFLOAD_CHECK(c_dbcsr_acc_opencl_info_devptr_lock(
      &batch, NULL /*lock*/, tasks /*batch*/, sizeof(dbm_task_t), &amount,
      &offset_batch));
  assert(0 == offset_adata && 0 == offset_bdata && 0 == offset_cdata);
#if defined(OPENCL_DBM_SOURCE_MULTIPLY_OPENCL)
  if (NULL == kernel) { /* first-time check if kernel is present */
    char build_params[ACC_OPENCL_BUFFERSIZE];
    const char *extensions[] = {NULL, NULL};
    const int nchar = c_dbcsr_acc_opencl_flags_atomics(
        &c_dbcsr_acc_opencl_config.device, c_dbcsr_acc_opencl_atomic_fp_64,
        extensions, sizeof(extensions) / sizeof(*extensions), build_params,
        sizeof(build_params));
    if (0 < nchar && (int)sizeof(build_params) > nchar) {
      OFFLOAD_CHECK(c_dbcsr_acc_opencl_kernel(
          0 /*source_is_file*/, OPENCL_DBM_SOURCE_MULTIPLY_OPENCL,
          "dbm_multiply", build_params,
          0 == c_dbcsr_acc_opencl_config.debug
              ? NULL
              : "-cl-fast-relaxed-math -cl-denorms-are-zero",
          NULL /*try*/, NULL /*try_ok*/, extensions,
          sizeof(extensions) / sizeof(*extensions), &kernel));
    }
  }
#else
#error "OpenCL kernel code not found!"
#endif
  OFFLOAD_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_double), &alpha));
  OFFLOAD_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_int), &m_range[1]));
  OFFLOAD_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), &n_range[1]));
  OFFLOAD_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &batchsize));
  OFFLOAD_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &offset_batch));
  OFFLOAD_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), &ntasks));
  OFFLOAD_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &batch.memory));
  OFFLOAD_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), &adata.memory));
  OFFLOAD_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), &bdata.memory));
  OFFLOAD_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), &cdata.memory));
  OFFLOAD_CHECK(clEnqueueNDRangeKernel(
      str->queue, kernel, 1 /*work_dim*/, NULL /*offset*/, &work_size,
      0 != wgsize ? &wgsize : NULL, 0 /*num_wait*/, NULL /*wait_list*/,
      perf_event));
  ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_main);
}

#endif // defined(__OFFLOAD_OPENCL) && !defined(__NO_OFFLOAD_DBM)

// EOF
