/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "../../exts/dbcsr/src/acc/opencl/common/opencl_atomics.h"
#include "dbm_multiply_internal.h"

#if 1
#define IDX(I, J, K, M, N) ((I) * (N) + (J) + (K))
#define IDT(I, J, K, M, N) IDX(J, I, K, N, M)
#else
#define IDT(I, J, K, M, N) ((I) * (N) + (J) + (K))
#define IDX(I, J, K, M, N) IDT(J, I, K, N, M)
#endif

/**
 * A * B^T -> C
 * TEST: benchmark_multiply(2, 2, 3, 2, 2, 4, comm)
 */
kernel void process_batch_kernel(double alpha, int ntasks, int m_max,
                                 int task_offset,
                                 global const dbm_task_t *restrict tasks,
                                 global const double *restrict a_data,
                                 global const double *restrict b_data,
                                 global double *restrict c_data) {
  const int work_size = (int)get_global_size(0), idx = (int)get_global_id(0);
  const int batchsize = (ntasks * m_max + work_size - 1) / work_size;
  const int i0 = idx * batchsize, i1 = i0 + batchsize;
  double vec[16];

  for (int i = i0; i < i1; ++i) {
    const int tid = i % ntasks, m = i % m_max;
    const dbm_task_t task = tasks[task_offset + tid];
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
#if 0
      printf("idx=%i tid=%i i=%i m=%i\n", idx, tid, i, m);
#endif
    }
  }
}
