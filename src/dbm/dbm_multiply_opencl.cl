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

#if !defined(TRACK_C) && 0
#define TRACK_C
#endif
#if !defined(NN)
#define NN 4
#endif

#define IDX(I, J, K, M, N) ((I) * (N) + (J) + (K))
#define IDT(I, J, K, M, N) IDX(J, I, K, N, M)

kernel void dbm_multiply(double alpha, int m_max, int n_max, int itask,
                         int ntasks, global const dbm_task_t *tasks,
                         global const double *restrict a_data,
                         global const double *restrict b_data,
                         global double *restrict c_data) {
  double vec[NN] = {0}; /* private accumulator */
  const int i = (int)get_global_id(0);
  const int tid = i / m_max, t = min(tid, ntasks - 1);
  const dbm_task_t task = tasks[itask + t]; /* !OOB */
  const int m = i - tid * m_max;            /* i % m_max */

  UNROLL(1)
  for (int j = 0; j < n_max; j += NN) {
    const int nn = min(n_max - j, NN);
#if defined(TRACK_C)
    int tc = -1;
#endif

    if (j < task.n && tid < ntasks) { /* valid task */
      UNROLL_AUTO
      for (int k = 0; k < task.k; ++k) {
        const int ia = IDX(m, k, task.offset_a, task.m, task.k);
        const double a = a_data[ia];

        UNROLL_AUTO
        for (int n = 0; n < nn; ++n) {
          const int ib = IDT(k, n + j, task.offset_b, task.k, task.n);
          const double b = b_data[ib];
          vec[n] = MAD(a, b, vec[n]);
        }
      }
    }

#if defined(TRACK_C)
    if (0 <= tc && task.offset_c != tc)
#endif
    { /* flush private accumulator to global memory using atomics */
      UNROLL_FORCE(NN)
      for (int n = 0; n < NN; ++n) {
        const int ic = IDX(m, n + j, task.offset_c, task.m, task.n);
        ACCUMULATE(&c_data[ic], alpha * vec[n]);
        vec[n] = ZERO; /* reset */
      }
    }

#if defined(TRACK_C)
    tc = task.offset_c; /* track change in C-offset */
#endif
  }
}
