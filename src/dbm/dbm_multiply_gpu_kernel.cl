/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "../../exts/dbcsr/src/acc/opencl/common/opencl_atomics.h"
#include "dbm_multiply_internal.h"

/* TEST: benchmark_multiply(2, 2, 3, 2, 2, 4, comm) */
kernel void process_batch_kernel(double alpha, int ntasks, int n_max,
                                 int task_offset,
                                 global const dbm_task_t *restrict tasks,
                                 global const double *restrict a_data,
                                 global const double *restrict b_data,
                                 global double *restrict c_data) {
  const int work_size = (int)get_global_size(0), idx = (int)get_local_id(0);
  const int batchsize = (ntasks * n_max + work_size - 1) / work_size;
  double colvec[4]; /* column */

  for (int i = idx; i < (idx + batchsize); ++i) {
    const int tid = i % ntasks, nid = i % n_max;
    const dbm_task_t task = tasks[task_offset + tid];
    if (nid < task.n) {
      printf("tid=%i nid=%i\n", tid, nid);
      for (int k = 0; k < task.k; ++k) {
        const double a = a_data[task.m * k + idx + task.offset_a];
        const double b = b_data[task.n * k + idx + task.offset_b];
        for (int m = 0; m < task.m; ++m) {
          colvec[m] = fma(work_group_broadcast(a, m), b, colvec[m]);
        }
      }
    }
  }
}
