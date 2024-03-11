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

#if !defined(BN)
#define BN 4
#endif

#define IDX(I, J, K, M, N) ((I) * (N) + (J) + (K))
#define IDT(I, J, K, M, N) IDX(J, I, K, N, M)

#define DBM_MULTIPLY_KERNEL(ALPHA, TASK, A, B, VEC, C, M, N0)                  \
  do {                                                                         \
    const int task_k = ((N0) < (TASK).n ? (TASK).k : 0);                       \
    const int task_n = min((TASK).n - (N0), BN);                               \
    UNROLL_AUTO                                                                \
    for (int k = 0; k < task_k; ++k) {                                         \
      const int ia = IDT(M, k, (TASK).offset_a, (TASK).m, (TASK).k);           \
      const double a = (A)[ia];                                                \
      UNROLL_AUTO                                                              \
      for (int n = 0; n < task_n; ++n) {                                       \
        const int ib = IDX(k, n + (N0), (TASK).offset_b, (TASK).k, (TASK).n);  \
        const double b = (B)[ib];                                              \
        vec[n] = MAD(a, b, vec[n]);                                            \
      }                                                                        \
    }                                                                          \
    /* flush private accumulator to global memory using atomics */             \
    UNROLL_FORCE(BN)                                                           \
    for (int n = 0; n < (BN); ++n) {                                           \
      const int ic = IDT(M, n + (N0), (TASK).offset_c, (TASK).m, (TASK).n);    \
      ACCUMULATE((C) + ic, (ALPHA) * (VEC)[n]);                                \
      vec[n] = ZERO; /* reset */                                               \
    }                                                                          \
  } while (0)

kernel void dbm_multiply(double alpha, int itask, int ntasks,
                         global const dbm_task_t *tasks,
                         global const double *restrict a_data,
                         global const double *restrict b_data,
                         global double *restrict c_data) {
  double vec[BN] = {0}; /* private accumulator */
  const int i = (int)get_global_id(0);

  const int size = (int)get_global_size(0);
  if (size != ntasks) {
    const int max_m = size / ntasks, tid = i / max_m;
    const dbm_task_t task = tasks[itask + min(tid, ntasks - 1)]; /* copy */
    const int m = i - tid * max_m;
    if (m < task.m) {
      if ((BN) < task.n) {
        UNROLL_FORCE(8)
        for (int n0 = 0; n0 < task.n; n0 += (BN)) {
          DBM_MULTIPLY_KERNEL(alpha, task, a_data, b_data, vec, c_data, m, n0);
        }
      } else {
        DBM_MULTIPLY_KERNEL(alpha, task, a_data, b_data, vec, c_data, m, 0);
      }
    }
  } else { /* full matrix multiplication */
    const dbm_task_t task = tasks[itask + i];
    UNROLL_OUTER(1)
    for (int m = 0; m < task.m; ++m) {
      UNROLL(1)
      for (int n0 = 0; n0 < task.n; n0 += (BN)) {
        DBM_MULTIPLY_KERNEL(alpha, task, a_data, b_data, vec, c_data, m, n0);
      }
    }
  }
}
