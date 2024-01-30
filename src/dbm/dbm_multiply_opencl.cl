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

#define IDX(I, J, K, M, N) ((I) * (N) + (J) + (K))
#define IDT(I, J, K, M, N) IDX(J, I, K, N, M)
#if !defined(NN)
#define NN 16
#endif
#if !defined(NK)
#define NK 4
#endif

kernel void dbm_multiply(double alpha, int m_max, int n_max, int itask,
                         int ntasks, global const dbm_task_t *tasks,
                         global const double *restrict a_data,
                         global const double *restrict b_data,
                         global const double *restrict c_data) {
  const int work_size = (int)get_global_size(0);
  const int i0 = (int)get_global_id(0) * ntasks;
  const int i1 = MIN(i0 + ntasks, work_size);
  double vec[NN] = {0};

  UNROLL_OUTER(1)
  for (int j = 0; j < n_max; j += NN) {
    const int nn = MIN(n_max - j, NN);
    int offset = -1;

    UNROLL_OUTER(1)
    for (int i = i0; i < i1; ++i) {
      const int tid = i / m_max, m = i - tid * m_max;
      const dbm_task_t task = tasks[tid + itask];

      if (m < task.m && j < task.n) { /* valid task */
        UNROLL(NK)
        for (int k = 0; k < task.k; ++k) {
          const int ia = IDX(m, k, task.offset_a, task.m, task.k);
          const double a = a_data[ia];

          UNROLL(NN)
          for (int n = 0; n < nn; ++n) {
            const int ib = IDT(k, n + j, task.offset_b, task.k, task.n);
            const double b = b_data[ib];
            vec[n] = MAD(a, b, vec[n]);
          }
        }
      }

      /* flush private accumulator to global memory using atomics */
      if ((0 <= offset && task.offset_c != offset) || i1 == (i + 1)) {
        UNROLL(NN)
        for (int n = 0; n < nn; ++n) {
          const int ic = IDX(m, n + j, task.offset_c, task.m, task.n);
          ACCUMULATE(&c_data[ic], alpha * vec[n]);
          vec[n] = ZERO; /* reset */
        }
      }

      /* track change in C-offset */
      offset = task.offset_c;
    }
  }
}
