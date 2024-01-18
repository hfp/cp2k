/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "../../exts/dbcsr/src/acc/opencl/common/opencl_atomics.h"
#include "dbm_multiply_internal.h"

kernel void process_batch_kernel(int ntasks, const int m_range[2],
                                 const int n_range[2], double alpha,
                                 int task_offset,
                                 global const dbm_task_t *restrict tasks,
                                 global const double *restrict pack_a_data,
                                 global const double *restrict pack_b_data,
                                 global double *restrict shard_c_data) {
  const int idx = (int)get_local_id(0), work_size = (int)get_global_size(0);
  const int task_size = ntasks * n_max;
  const int batchsize = (task_size + work_size - 1) / work_size;
  const int i0 = idx * batchsize, i1 = min(i0 + batchsize, task_size);
  double colvec[4]; /* column */
  int i, j;

  for (i = i0; i < i1; i += n_max) {
    const dbm_task_t task = tasks[task_offset + i / n_max];
    const int n = i % n_max;
    printf("task=%i n=%i MxN=%ix%i\n", i / n_max, task.m, task.n);
  }
  printf("\n");
}