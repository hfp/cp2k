/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "../../exts/dbcsr/src/acc/opencl/common/opencl_atomics.h"
#include "dbm_multiply_internal.h"

kernel void process_batch_kernel(double alpha, int ntasks, int n_max,
                                 int task_offset,
                                 global const dbm_task_t *restrict tasks,
                                 global const double *restrict pack_a_data,
                                 global const double *restrict pack_b_data,
                                 global double *restrict shard_c_data) {
  const int work_size = (int)get_global_size(0), idx = (int)get_local_id(0);
  const int batchsize = (ntasks * n_max + work_size - 1) / work_size;
  int i, j;

  for (i = idx; i < (idx + batchsize); ++i) {
    const int tid = i % ntasks, nid = i % n_max;
    const dbm_task_t task = tasks[task_offset + tid];
    if (nid < task.n) {
      printf("tid=%i nid=%i\n", tid, nid);
    }
  }
}
