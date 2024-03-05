/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#if !defined(LU) /* impacts opencl/common */
#define LU 0
#endif

#include "../../exts/dbcsr/src/acc/opencl/common/opencl_atomics.h"
#include "dbm_multiply_internal.h"

#if !defined(NN)
#define NN 4
#endif

#define IDX(I, J, K, M, N) ((I) * (N) + (J) + (K))
#define IDT(I, J, K, M, N) IDX(J, I, K, N, M)

void dbm_multiply_kernel(double alpha, const dbm_task_t *task,
                         global const double *restrict a_data,
                         global const double *restrict b_data, double *vec,
                         global double *restrict c_data, int m, int max_n) {
  UNROLL(1)
  for (int j = 0; j < max_n; j += NN) {
    const int nn = min(task->n - j, NN);

    UNROLL_AUTO
    for (int k = 0; k < task->k; ++k) {
      const int ia = IDX(m, k, task->offset_a, task->m, task->k);
      const double a = a_data[ia];

      UNROLL_AUTO
      for (int n = 0; n < nn; ++n) {
        const int ib = IDT(k, n + j, task->offset_b, task->k, task->n);
        const double b = b_data[ib];
        vec[n] = MAD(a, b, vec[n]);
      }
    }

    /* flush private accumulator to global memory using atomics */
    UNROLL_FORCE(NN)
    for (int n = 0; n < NN; ++n) {
      const int ic = IDX(m, n + j, task->offset_c, task->m, task->n);
      ACCUMULATE(&c_data[ic], alpha * vec[n]);
      vec[n] = ZERO; /* reset */
    }
  }
}

kernel void dbm_multiply(double alpha, int max_n, int itask, int ntasks,
                         global const dbm_task_t *tasks,
                         global const double *restrict a_data,
                         global const double *restrict b_data,
                         global double *restrict c_data) {
  double vec[NN] = {0}; /* private accumulator */
  const int i = (int)get_global_id(0);

#if defined(SPLIT_TASK)
  const int size = (int)get_global_size(0);
  if (size != ntasks) {
    const int max_m = size / ntasks, tid = i / max_m;
    const dbm_task_t task = tasks[itask + min(tid, ntasks - 1)]; /* copy */
    const int m = i - tid * max_m;
    dbm_multiply_kernel(alpha, &task, a_data, b_data, vec, c_data, m, max_n);
  } else
#endif
  { /* full matrix multiplication */
    const dbm_task_t task = tasks[itask + i];
    for (int m = 0; m < task.m; ++m) {
      dbm_multiply_kernel(alpha, &task, a_data, b_data, vec, c_data, m, max_n);
    }
  }
}
