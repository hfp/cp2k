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

#if !defined(TRACK_C)
#define TRACK_C
#endif
#if !defined(NN)
#define NN 4
#endif
#if !defined(NK)
#define NK 16
#endif
#if !defined(BS)
#define BS 1
#endif

#define IDX(I, J, K, M, N) ((I) * (N) + (J) + (K))
#define IDT(I, J, K, M, N) IDX(J, I, K, N, M)

#if (1 < BS)
#define TILE(M, N) tile[M][N]
#else
#define TILE(M, N) vec[N]
#endif

kernel void dbm_multiply(double alpha, int m_max, int n_max, int nbatch,
                         int itask, int ntasks, global const dbm_task_t *tasks,
                         global const double *restrict a_data,
                         global const double *restrict b_data,
                         global double *restrict c_data) {
#if (1 < BS)
  double tile[BS][NN] = {0}; /* private accumulator */
#else
  double vec[NN] = {0}; /* private accumulator */
#endif
  const int i0 = (int)get_global_id(0) * nbatch, i1 = i0 + nbatch;

  UNROLL_OUTER(1)
  for (int j = 0; j < n_max; j += NN) {
    const int nn = min(n_max - j, NN);
#if defined(TRACK_C)
    int tc = -1;
#endif

    UNROLL_OUTER(1)
    for (int i = i0; i < i1; i += m_max) {
      const int tid = i / m_max, t = min(tid, ntasks - 1);
      const dbm_task_t task = tasks[itask + t]; /* !OOB */
      const int m0 = i - tid * m_max;           /* i % m_max */
#if (1 < BS)
      const int mn = min(min(task.m, nbatch), BS);
#else
      const int m = 0;
#endif

      if (j < task.n && tid < ntasks) { /* valid task */
#if (1 < BS)
        UNROLL(BS)
        for (int m = 0; m < mn; ++m)
#endif
        {
          UNROLL_AUTO
          for (int k = 0; k < task.k; ++k) { /* UNROLL(NK) */
            const int ia = IDX(m + m0, k, task.offset_a, task.m, task.k);
            const double a = a_data[ia];

            UNROLL_FORCE(NN)
            for (int n = 0; n < nn; ++n) {
              const int ib = IDT(k, n + j, task.offset_b, task.k, task.n);
              const double b = b_data[ib];
              TILE(m, n) = MAD(a, b, TILE(m, n));
            }
          }
        }
      }

#if defined(TRACK_C)
      if (BS < nbatch || (0 <= tc && task.offset_c != tc) || i1 <= (i + m_max))
#endif
      { /* flush private accumulator to global memory using atomics */
#if (1 < BS)
        UNROLL(BS)
        for (int m = 0; m < BS; ++m)
#endif
        {
          UNROLL_FORCE(NN)
          for (int n = 0; n < NN; ++n) {
            const int ic = IDX(m + m0, n + j, task.offset_c, task.m, task.n);
            ACCUMULATE(&c_data[ic], alpha * TILE(m, n));
            TILE(m, n) = ZERO; /* reset */
          }
        }
      }

#if defined(TRACK_C)
      tc = task.offset_c; /* track change in C-offset */
#endif
    }
  }
}
