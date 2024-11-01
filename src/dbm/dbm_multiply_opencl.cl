/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#if defined(DBM_MULTIPLY_OPENCL_GEN)
#include "dbm_multiply_opencl.irh"
#else
#include "../../exts/dbcsr/src/acc/opencl/common/opencl_atomics.h"
#include "dbm_multiply_internal.h"

#if defined(GPU) && defined(WG) && (0 < WG) && (200 <= ACC_OPENCL_VERSION)
#if defined(SG) && (0 < SG)
#define BCST_WG(V) sub_group_broadcast(V, 0)
#else
#define BCST_WG(V) work_group_broadcast(V, 0)
#endif
#endif
#define BCST_NO(V) (V)

#define SINT short

#define X(T, I) (T)->I /* task can be taken by value or by pointer */
#define XA(T) X(T, offset_a)
#define XB(T) X(T, offset_b)
#define XC(T) X(T, offset_c)
#define XM(T) (SINT) X(T, m)
#define XN(T) (SINT) X(T, n)
#define XK(T) (SINT) X(T, k)

#define DBM_MULTIPLY_KERNEL(ALPHA, TASK, AMAT, BMAT, CMAT, CVEC, M, N0, N1,    \
                            BCST)                                              \
  UNROLL_AUTO for (SINT k = 0; k < XK(TASK); ++k) {                            \
    const double a = (AMAT)[XA(TASK) + IDT(M, k, XM(TASK), XK(TASK))];         \
    UNROLL_AUTO for (SINT n = 0; n < (N1); ++n) {                              \
      const SINT idx = IDX(k, n + (N0), XK(TASK), XN(TASK));                   \
      const double b = (BMAT)[XB(TASK) + idx];                                 \
      (CVEC)[n] = MAD(a, BCST(b), (CVEC)[n]);                                  \
    }                                                                          \
  }                                                                            \
  UNROLL_AUTO for (SINT n = 0; n < (N1); ++n) { /* flush to global */          \
    const int idx = IDT(M, n + (N0), XM(TASK), XN(TASK));                      \
    ACCUMULATE((CMAT) + XC(TASK) + idx, (ALPHA) * (CVEC)[n]);                  \
    (CVEC)[n] = ZERO; /* reset */                                              \
  }

#define DBM_MULTIPLY(ALPHA, TASK, AMAT, BMAT, CMAT, M, BN, BCST)               \
  do { /* DBM_MULTIPLY_KERNEL unrolled/specialized over N and K */             \
    double cvec[BN];                                                           \
    UNROLL_AUTO for (SINT n = 0; n < (BN); ++n) { cvec[n] = ZERO; }            \
    if (1 < XN(TASK)) {                                                        \
      SINT n0 = 0;                                                             \
      if ((BN) <= XN(TASK)) {                                                  \
        UNROLL_OUTER(1) for (; (n0 + (BN)) <= XN(TASK); n0 += (BN)) {          \
          DBM_MULTIPLY_KERNEL(ALPHA, TASK, AMAT, BMAT, CMAT, cvec, M, n0, BN,  \
                              BCST);                                           \
        }                                                                      \
      }                                                                        \
      DBM_MULTIPLY_KERNEL(ALPHA, TASK, AMAT, BMAT, CMAT, cvec, M, n0,          \
                          XN(TASK) - n0, BCST);                                \
    } else { /* N = 1 */                                                       \
      DBM_MULTIPLY_KERNEL(ALPHA, TASK, AMAT, BMAT, CMAT, cvec, M, 0, 1, BCST); \
    }                                                                          \
  } while (0)

#if defined(WG) && (0 < WG)
__attribute__((reqd_work_group_size(WG, 1, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
#endif
kernel void
dbm_multiply(double alpha, int itask, int ntasks, int size,
             global const dbm_task_t *tasks, global const double *restrict amat,
             global const double *restrict bmat, global double *restrict cmat) {
  const int i = (int)get_global_id(0);
#if defined(BCST_WG)
  if (i < size)
#endif
  {
    const int max_m = size / ntasks, tid = i / max_m;
    const SINT m = i - tid * max_m;
    global const dbm_task_t *const task = &tasks[itask + tid];
    if (m < XM(task)) { /* valid task */
#if defined(BCST_WG)
      if (XM(task) <= XN(task)) { /* BCST_WG to broadcast B-values */
        DBM_MULTIPLY(alpha, task, amat, bmat, cmat, m, BN, BCST_WG);
      } else
#endif
      {
        DBM_MULTIPLY(alpha, task, amat, bmat, cmat, m, BN, BCST_NO);
      }
    }
  }
}
#endif
