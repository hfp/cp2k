/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "../../exts/dbcsr/src/acc/opencl/common/opencl_atomics.h"
#include "dbm_multiply_internal.h"

/**
 * A * B^T -> C
 * TEST: benchmark_multiply(2, 2, 3, 2, 2, 4, comm)
 */
kernel void process_batch_kernel(double alpha, int ntasks, int n_max,
                                 int task_offset,
                                 global const dbm_task_t *restrict tasks,
                                 global const double *restrict a_data,
                                 global const double *restrict b_data,
                                 global double *restrict c_data) {
  const int work_size = (int)get_global_size(0), idx = (int)get_local_id(0);
  const int batchsize = (ntasks * n_max + work_size - 1) / work_size;
  const int i0 = idx * batchsize, i1 = i0 + batchsize;
  double colvec[4]; /* M-column */

  for (int i = i0; i < i1; ++i) {
    const int tid = i % ntasks, nid = i % n_max;
    const dbm_task_t task = tasks[task_offset + tid];
    if (nid < task.n) {
      for (int m = 0; m < task.m; ++m) {
        colvec[m] = ZERO;
      }
      for (int k = 0; k < task.k; ++k) {
        const double b = b_data[task.k * nid + k + task.offset_b];
        for (int m = 0; m < task.m; ++m) {
          const double a = a_data[task.k * m + k + task.offset_a];
          colvec[m] = MAD(a, b, colvec[m]);
        }
      }
      for (int m = 0; m < task.m; ++m) {
        ACCUMULATE(&c_data[task.m * nid + m + task.offset_c], colvec[m]);
        colvec[m] = ZERO; /* reset */
      }
    }
  }
}
