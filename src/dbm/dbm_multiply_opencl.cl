/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#if defined(DBM_MULTIPLY_OPENCL_GEN)
#include "dbm_multiply_opencl.ir.h"
#else
#include "../../exts/dbcsr/src/acc/opencl/common/opencl_atomics.h"
#include "dbm_multiply_internal.h"

#define IDX(I, J, OFFSET, M, N) ((I) * (N) + (J) + (OFFSET))
#define IDT(I, J, OFFSET, M, N) IDX(J, I, OFFSET, N, M)
#define X(T, I) (T)->I

#define BCAST_WG(A) work_group_broadcast(A, get_local_id(0))
#define BCAST_NO(A) (A)

#define DBM_MULTIPLY_KERNEL(TASK, AMAT, BMAT, CVEC, M, N0, N1, BROADCAST,      \
                            UNROLL_N, UNROLL_K)                                \
  do {                                                                         \
    UNROLL_K for (int k = 0; k < X(TASK, k); ++k) {                            \
      const int ia = IDT(M, k, X(TASK, offset_a), X(TASK, m), X(TASK, k));     \
      const double a = (AMAT)[ia];                                             \
      UNROLL_N for (int n = 0; n < (N1); ++n) {                                \
        const int tb = X(TASK, offset_b);                                      \
        const int ib = IDX(k, n + (N0), tb, X(TASK, k), X(TASK, n));           \
        const double b = (BMAT)[ib];                                           \
        (CVEC)[n] = MAD(a, BROADCAST(b), (CVEC)[n]);                           \
      }                                                                        \
    }                                                                          \
  } while (0)

#define DBM_MULTIPLY_ACCUMULATE(ALPHA, TASK, CMAT, CVEC, M, N0, BN)            \
  do { /* flush private accumulator to global memory using atomics */          \
    UNROLL_FORCE(BN) for (int n = 0; n < (BN); ++n) {                          \
      const int tc = X(TASK, offset_c);                                        \
      const int ic = IDT(M, n + (N0), tc, X(TASK, m), X(TASK, n));             \
      ACCUMULATE((CMAT) + ic, (ALPHA) * (CVEC)[n]);                            \
      (CVEC)[n] = ZERO; /* reset */                                            \
    }                                                                          \
  } while (0)

#if defined(WG) && (0 < WG)
__attribute__((reqd_work_group_size(WG, 1, 1)))
#endif
kernel void
dbm_multiply(double alpha, int itask, int ntasks,
             global const dbm_task_t *tasks, global const double *restrict amat,
             global const double *restrict bmat, global double *restrict cmat) {
  double cvec[BN];
  const int size = (int)get_global_size(0), i = (int)get_global_id(0);

  UNROLL_FORCE(BN) for (int n = 0; n < (BN); ++n) cvec[n] = 0; /* clear */
#if !defined(SPLIT)
  if (size != ntasks)
#endif
  { /* DBM_MULTIPLY_SPLIT */
    const int wgsize = (int)get_local_size(0);
    const int max_m = size / ntasks, tid = i / max_m;
    /* task can be taken by value or by pointer (adjust X-macro accordingly) */
    global const dbm_task_t *const task = &tasks[itask + min(tid, ntasks - 1)];
    const int m = i - tid * max_m;
    if (m < X(task, m)) {    /* valid task */
#if defined(BCAST) && defined(GPU) && (200 /*2.0*/ <= ACC_OPENCL_VERSION)
      if (max_m <= wgsize) { /* broadcast B-values */
        if ((BN) < X(task, n)) {
          UNROLL_AUTO for (int n0 = 0; n0 < X(task, n); n0 += (BN)) {
            const int n1 = min(BN, X(task, n) - n0);
            DBM_MULTIPLY_KERNEL(task, amat, bmat, cvec, m, n0, n1, BCAST_WG,
                                UNROLL_FORCE(BN), UNROLL_AUTO);
            DBM_MULTIPLY_ACCUMULATE(alpha, task, cmat, cvec, m, n0, BN);
          }
        } else { /* task.n <= BN */
          DBM_MULTIPLY_KERNEL(task, amat, bmat, cvec, m, 0, X(task, n),
                              BCAST_WG, UNROLL_FORCE(BN), UNROLL_FORCE(BN));
          DBM_MULTIPLY_ACCUMULATE(alpha, task, cmat, cvec, m, 0, BN);
        }
      } else
#endif
      {
        if ((BN) < X(task, n)) {
          UNROLL_AUTO for (int n0 = 0; n0 < X(task, n); n0 += (BN)) {
            const int n1 = min(BN, X(task, n) - n0);
            DBM_MULTIPLY_KERNEL(task, amat, bmat, cvec, m, n0, n1, BCAST_NO,
                                UNROLL_FORCE(BN), UNROLL_AUTO);
            DBM_MULTIPLY_ACCUMULATE(alpha, task, cmat, cvec, m, n0, BN);
          }
        } else { /* task.n <= BN */
          DBM_MULTIPLY_KERNEL(task, amat, bmat, cvec, m, 0, X(task, n),
                              BCAST_NO, UNROLL_FORCE(BN), UNROLL_FORCE(BN));
          DBM_MULTIPLY_ACCUMULATE(alpha, task, cmat, cvec, m, 0, BN);
        }
      }
    }
  }
#if !defined(SPLIT)
  else { /* full matrix multiplication */
    global const dbm_task_t *const task = &tasks[itask + i];
    UNROLL_OUTER(1) for (int m = 0; m < X(task, m); ++m) {
      UNROLL(1) for (int n0 = 0; n0 < X(task, n); n0 += (BN)) {
        const int n1 = min(BN, X(task, n) - n0);
        DBM_MULTIPLY_KERNEL(task, amat, bmat, cvec, m, n0, n1, BCAST_NO,
                            UNROLL_AUTO, UNROLL_AUTO);
        DBM_MULTIPLY_ACCUMULATE(alpha, task, cmat, cvec, m, n0, BN);
      }
    }
  }
#endif
}
#endif
