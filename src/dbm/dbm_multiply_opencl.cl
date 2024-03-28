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

#if defined(BCAST) && defined(GPU) && (200 /*2.0*/ <= ACC_OPENCL_VERSION) &&   \
    defined(WG) && (0 < WG) && (BN <= WG)
#if defined(SG) && (WG == SG)
#define BCST_WG(V, I) sub_group_broadcast(V, I)
#else
#define BCST_WG(V, I) work_group_broadcast(V, I)
#endif
#endif
#define BCST_NO(V, I) (V)

#define IDX(I, J, M, N) ((I) * (N) + (J))
#define IDT(I, J, M, N) IDX(J, I, N, M)
#define X(T, I) (T)->I
#define XM(T) X(T, m)
#define XN(T) X(T, n)
#define XK(T) X(T, k)

#define DBM_MULTIPLY_KERNEL(TASK, AMAT, BMAT, CVEC, M, N0, N1, BCST, UNROLL_N, \
                            UNROLL_K)                                          \
  do {                                                                         \
    UNROLL_K for (int k = 0; k < XK(TASK); ++k) {                              \
      const int ia = IDT(M, k, XM(TASK), XK(TASK));                            \
      const double a = (AMAT)[ia + X(TASK, offset_a)];                         \
      UNROLL_N for (int n = 0; n < (N1); ++n) {                                \
        const int ib = IDX(k, n + (N0), XK(TASK), XN(TASK));                   \
        const double b = (BMAT)[ib + X(TASK, offset_b)];                       \
        (CVEC)[n] = MAD(a, BCST(b, n), (CVEC)[n]);                             \
      }                                                                        \
    }                                                                          \
  } while (0)

#define DBM_MULTIPLY_ACCUMULATE(ALPHA, TASK, CMAT, CVEC, M, N0, BN)            \
  do { /* flush private accumulator to global memory using atomics */          \
    UNROLL_FORCE(BN) for (int n = 0; n < (BN); ++n) {                          \
      const int ic = IDT(M, n + (N0), XM(TASK), XN(TASK));                     \
      ACCUMULATE((CMAT) + ic + X(TASK, offset_c), (ALPHA) * (CVEC)[n]);        \
      (CVEC)[n] = ZERO; /* reset */                                            \
    }                                                                          \
  } while (0)

#if defined(WG) && (0 < WG)
__attribute__((reqd_work_group_size(WG, 1, 1)))
#if defined(SG) && (WG == SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
#endif
kernel void
dbm_multiply(double alpha, int itask, int ntasks, int size,
             global const dbm_task_t *tasks, global const double *restrict amat,
             global const double *restrict bmat, global double *restrict cmat) {
  double cvec[BN];
  const int i = (int)get_global_id(0);

  UNROLL_FORCE(BN) for (int n = 0; n < (BN); ++n) cvec[n] = 0; /* clear */
#if !defined(SPLIT)
  if (size != ntasks)
#endif
  { /* DBM_MULTIPLY_SPLIT */
    const int max_m = size / ntasks, tid = i / max_m;
    /* task can be taken by value or by pointer (adjust X-macro accordingly) */
    global const dbm_task_t *const task = &tasks[itask + min(tid, ntasks - 1)];
    const int m = i - tid * max_m;
    if (m < XM(task)
#if defined(BCST_WG)
        && i < size
#endif
    ) { /* valid task */
#if defined(BCST_WG) /* broadcast B-values */
      if (m < (WG)) {
        if ((BN) < XN(task)) {
          UNROLL_AUTO for (int n0 = 0; n0 < XN(task); n0 += (BN)) {
            const int n1 = min(BN, XN(task) - n0);
            DBM_MULTIPLY_KERNEL(task, amat, bmat, cvec, m, n0, n1, BCST_WG,
                                UNROLL_FORCE(BN), UNROLL_AUTO);
            DBM_MULTIPLY_ACCUMULATE(alpha, task, cmat, cvec, m, n0, BN);
          }
        } else { /* task.n <= BN */
          DBM_MULTIPLY_KERNEL(task, amat, bmat, cvec, m, 0, XN(task), BCST_WG,
                              UNROLL_FORCE(BN), UNROLL_FORCE(BN));
          DBM_MULTIPLY_ACCUMULATE(alpha, task, cmat, cvec, m, 0, BN);
        }
      } else
#endif
      {
        if ((BN) < XN(task)) {
          UNROLL_AUTO for (int n0 = 0; n0 < XN(task); n0 += (BN)) {
            const int n1 = min(BN, XN(task) - n0);
            DBM_MULTIPLY_KERNEL(task, amat, bmat, cvec, m, n0, n1, BCST_NO,
                                UNROLL_FORCE(BN), UNROLL_AUTO);
            DBM_MULTIPLY_ACCUMULATE(alpha, task, cmat, cvec, m, n0, BN);
          }
        } else { /* task.n <= BN */
          DBM_MULTIPLY_KERNEL(task, amat, bmat, cvec, m, 0, XN(task), BCST_NO,
                              UNROLL_FORCE(BN), UNROLL_FORCE(BN));
          DBM_MULTIPLY_ACCUMULATE(alpha, task, cmat, cvec, m, 0, BN);
        }
      }
    }
  }
#if !defined(SPLIT)
  else if (i < size) { /* full matrix multiplication */
    global const dbm_task_t *const task = &tasks[itask + i];
    UNROLL_OUTER(1) for (int m = 0; m < XM(task); ++m) {
      UNROLL_AUTO for (int n0 = 0; n0 < XN(task); n0 += (BN)) {
        const int n1 = min(BN, XN(task) - n0);
        DBM_MULTIPLY_KERNEL(task, amat, bmat, cvec, m, n0, n1, BCST_NO,
                            UNROLL_AUTO, UNROLL_AUTO);
        DBM_MULTIPLY_ACCUMULATE(alpha, task, cmat, cvec, m, n0, BN);
      }
    }
  }
#endif
}
#endif
