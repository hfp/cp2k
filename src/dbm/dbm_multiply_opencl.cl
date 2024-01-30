/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "../../exts/dbcsr/src/acc/opencl/common/opencl_atomics.h"
#include "dbm_multiply_internal.h"

#define IDX(I, J, K, M, N) ((I) * (N) + (J) + (K))
#define IDT(I, J, K, M, N) IDX(J, I, K, N, M)

kernel void dbm_multiply(double alpha, int m_max, int itask, int ntasks,
                         global const dbm_task_t *tasks,
                         global const double *restrict a_data,
                         global const double *restrict b_data,
                         global const double *restrict c_data) {
  const int i0 = (int)get_global_id(0) * ntasks;
  const int i1 = i0 + ntasks;
  double vec[16];

  for (int i = i0; i < i1; ++i) {
    const int tid = i / m_max, m = i - tid * m_max;
    const dbm_task_t task = tasks[tid + itask];
    if (m < task.m) {
      for (int n = 0; n < task.n; ++n) {
        vec[n] = ZERO;
      }
      for (int k = 0; k < task.k; ++k) {
        const double a = a_data[IDX(m, k, task.offset_a, task.m, task.k)];
        for (int n = 0; n < task.n; ++n) {
          const double b = b_data[IDT(k, n, task.offset_b, task.k, task.n)];
          vec[n] = MAD(a, b, vec[n]);
        }
      }
      for (int n = 0; n < task.n; ++n) {
        ACCUMULATE(&c_data[IDX(m, n, task.offset_c, task.m, task.n)],
                   alpha * vec[n]);
        vec[n] = ZERO; /* reset */
      }
    }
  }
}
