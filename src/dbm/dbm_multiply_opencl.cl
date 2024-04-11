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

#if defined(GPU) && defined(WG) && (0 < WG) && (200 <= ACC_OPENCL_VERSION)
#if defined(SG) && (0 < SG)
#define BCST_WG(V) sub_group_broadcast(V, 0)
#else
#define BCST_WG(V) work_group_broadcast(V, 0)
#endif
#endif
#define BCST_NO(V) (V)

#define DIVUP(A, B) (((A) + (B)-1) / (B))
#define NUP(N, UP) (DIVUP(N, UP) * (UP))
#define BLR(N, BN) (NUP(N, BN) - (N))

#define IDX(I, J, M, N) ((I) * (N) + (J))
#define IDT(I, J, M, N) IDX(J, I, N, M)
#define X(T, I) (T)->I /* task can be taken by value or by pointer */
#define XA(T) X(T, offset_a)
#define XB(T) X(T, offset_b)
#define XC(T) X(T, offset_c)
#define XM(T) X(T, m)
#define XN(T) X(T, n)
#define XK(T) X(T, k)

#define DBM_MULTIPLY_TASK(ALPHA, TASK, AMAT, BMAT, CMAT, SHM, WG, BM, BN)      \
  do {                                                                         \
    local double *restrict const ashm = (SHM);                                 \
    local double *restrict const bshm = (SHM) + (WG);                          \
    const short mk = XM(TASK) * XK(TASK), kn = XK(TASK) * XN(TASK);            \
    const short tid = (short)get_local_id(0);                                  \
    /* y/s can exceed BN/BM (up to BK), and x/t is fast index (up to BM/BN) */ \
    const short y = tid / (BM), x = tid - y * (BM), bk = (WG) / MAX(BM, BN);   \
    const short s = tid / (BN), t = tid - s * (BN);                            \
    for (short m0 = 0; m0 < XM(TASK); m0 += (BM)) {                            \
      for (short n0 = 0; n0 < XN(TASK); n0 += (BN)) {                          \
        double r = ZERO;                                                       \
        UNROLL_AUTO for (short k0 = 0; k0 < XK(TASK); k0 += bk) {              \
          if (x < (BM) && y < bk) { /* load A-tile */                          \
            const short idx = IDT(m0 + x, k0 + y, XM(TASK), XK(TASK));         \
            ashm[y * (BM) + x] = (idx < mk ? (AMAT)[XA(TASK) + idx] : ZERO);   \
          }                                                                    \
          if (s < bk && t < (BN)) { /* load B-tile */                          \
            const short idx = IDX(k0 + s, n0 + t, XK(TASK), XN(TASK));         \
            bshm[s * (BN) + t] = (idx < kn ? (BMAT)[XB(TASK) + idx] : ZERO);   \
          }                                                                    \
          BARRIER(CLK_LOCAL_MEM_FENCE);                                        \
          if (x < (BM) && y < (BN)) { /* multiply tiles */                     \
            UNROLL_AUTO for (short z = 0; z < bk; ++z) {                       \
              r = MAD(ashm[z * (BM) + x], bshm[z * (BN) + y], r);              \
            }                                                                  \
          }                                                                    \
          BARRIER(CLK_LOCAL_MEM_FENCE);                                        \
        }                                                                      \
        if (x < (BM) && y < (BN)) { /* flush to global */                      \
          const short m = m0 + x, n = n0 + y;                                  \
          if (m < XM(TASK) && n < XN(TASK)) {                                  \
            const short idx = IDT(m, n, XM(TASK), XN(TASK));                   \
            ACCUMULATE((CMAT) + XC(TASK) + idx, (ALPHA)*r);                    \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

#define DBM_MULTIPLY_KERNEL(ALPHA, TASK, AMAT, BMAT, CMAT, CVEC, M, N0, N1,    \
                            BN, BCST, UNROLL_N, UNROLL_K)                      \
  UNROLL_K for (int k = 0; k < XK(TASK); ++k) {                                \
    const int ia = IDT(M, k, XM(TASK), XK(TASK));                              \
    const double a = (AMAT)[XA(TASK) + ia];                                    \
    UNROLL_N for (int n = 0; n < (N1); ++n) {                                  \
      const int ib = IDX(k, n + (N0), XK(TASK), XN(TASK));                     \
      const double b = (BMAT)[XB(TASK) + ib];                                  \
      (CVEC)[n] = MAD(a, BCST(b), (CVEC)[n]);                                  \
    }                                                                          \
  }                                                                            \
  UNROLL_FORCE(BN) for (int n = 0; n < (BN); ++n) { /* flush to global */      \
    const int ic = IDT(M, n + (N0), XM(TASK), XN(TASK));                       \
    ACCUMULATE((CMAT) + XC(TASK) + ic, (ALPHA) * (CVEC)[n]);                   \
    (CVEC)[n] = ZERO; /* reset */                                              \
  }

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
#if defined(SPLIT) && (1 < SPLIT) && defined(WG) && (0 < WG)
  local double shm[WG * 2];
  global const dbm_task_t *const task = &tasks[itask + get_group_id(0)];
  const short rmin = MIN(XM(task), XN(task)), rmax = MAX(XM(task), XN(task));
  if ((rmax - rmin) <= BN) {
    if ((rmin * 4) < BN) {
      DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN / 4, BN / 4);
    } else if ((rmin * 2) < BN) {
      DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN / 2, BN / 2);
    } else {
      DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN, BN);
    }
  } else if (XM(task) <= XN(task)) {
    const short r1 = BLR(XM(task), BN);
    const short r2 = BLR(XM(task), BN / 2) * 2;
    const short r3 = BLR(XM(task), BN / 4) * 4;
    if (r1 <= r2) {
      if (r1 <= r3) {
        DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN, BN);
      } else {
        DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN / 4,
                          BN * 4);
      }
    } else if (r2 <= r3) {
      DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN / 2, BN * 2);
    } else {
      DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN / 4, BN * 4);
    }
  } else {
    const short r1 = BLR(XN(task), BN);
    const short r2 = BLR(XN(task), BN / 2) * 2;
    const short r3 = BLR(XN(task), BN / 4) * 4;
    if (r1 <= r2) {
      if (r1 <= r3) {
        DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN, BN);
      } else {
        DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN * 4,
                          BN / 4);
      }
    } else if (r2 <= r3) {
      DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN * 2, BN / 2);
    } else {
      DBM_MULTIPLY_TASK(alpha, task, amat, bmat, cmat, shm, WG, BN * 4, BN / 4);
    }
  }
#elif defined(SPLIT) && (0 != SPLIT)
  const int i = (int)get_global_id(0);
#if defined(BCST_WG)
  if (i < size)
#endif
  { /* DBM_MULTIPLY_SPLIT */
    const int max_m = size / ntasks, tid = i / max_m, m = i - tid * max_m;
    global const dbm_task_t *const task = &tasks[itask + tid];
    if (m < XM(task)) { /* valid task */
      double cvec[BN] = {ZERO};
#if defined(BCST_WG) /* broadcast B-values */
      if (XM(task) <= XN(task)) {
        if (BN < XN(task)) {
          UNROLL_AUTO for (int n0 = 0; n0 < XN(task); n0 += BN) {
            const int n1 = min(BN, XN(task) - n0);
            DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, n1,
                                BN, BCST_WG, UNROLL_FORCE(BN), UNROLL_AUTO);
          }
        } else { /* task.n <= BN */
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, 0,
                              XN(task), BN, BCST_WG, UNROLL_FORCE(BN),
                              UNROLL_FORCE(BN));
        }
      } else
#endif
      {
        int n0 = 0;
        switch (XN(task) / BN) {
        case 0: /* task.n < BN */
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, 0,
                              XN(task), BN, BCST_NO, UNROLL_AUTO, UNROLL_AUTO);
          n0 = BN;
          break;
        case 1:
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, 0, BN, BN,
                              BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          n0 = BN;
          break;
        case 2:
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, 0, BN, BN,
                              BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, BN, BN,
                              BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          n0 = BN + BN;
          break;
        case 3:
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, 0, BN, BN,
                              BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, BN, BN,
                              BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, BN + BN,
                              BN, BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          n0 = BN * 3;
          break;
        case 4:
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, 0, BN, BN,
                              BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, BN, BN,
                              BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, BN + BN,
                              BN, BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, BN * 3,
                              BN, BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          n0 = BN * 4;
          break;
        case 5:
          UNROLL_AUTO for (; n0 < (BN * 5); n0 += BN) {
            DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, BN,
                                BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          }
          break;
        case 6:
          UNROLL_AUTO for (; n0 < (BN * 6); n0 += BN) {
            DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, BN,
                                BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          }
          break;
        case 7:
          UNROLL_AUTO for (; n0 < (BN * 7); n0 += BN) {
            DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, BN,
                                BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          }
          break;
        case 8:
        case 9:
          UNROLL_AUTO for (; n0 < (BN * 8); n0 += BN) {
            DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, BN,
                                BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          }
          break;
        case 10:
        case 11:
          UNROLL_AUTO for (; n0 < (BN * 10); n0 += BN) {
            DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, BN,
                                BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          }
          break;
        case 12:
        case 13:
          UNROLL_AUTO for (; n0 < (BN * 12); n0 += BN) {
            DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, BN,
                                BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          }
          break;
        case 14:
        case 15:
          UNROLL_AUTO for (; n0 < (BN * 14); n0 += BN) {
            DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, BN,
                                BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          }
          break;
        default:
          UNROLL_AUTO for (; n0 < (BN * 16); n0 += BN) {
            DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, BN,
                                BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
          }
        }
        UNROLL_OUTER(1) for (; n0 < XN(task); n0 += BN) { /* remainder */
          const int n1 = min(BN, XN(task) - n0);
          DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, n1,
                              BN, BCST_NO, UNROLL_FORCE(BN), UNROLL_AUTO);
        }
      }
    }
  }
#else
#if defined(BCST_WG)
  if (get_global_id(0) < size)
#endif
  { /* full matrix multiplication */
    double cvec[BN] = {ZERO};
    global const dbm_task_t *const task = &tasks[itask + get_global_id(0)];
    UNROLL_OUTER(1) for (int m = 0; m < XM(task); ++m) {
      UNROLL_AUTO for (int n0 = 0; n0 < XN(task); n0 += BN) {
        const int n1 = min(BN, XN(task) - n0);
        DBM_MULTIPLY_KERNEL(alpha, task, amat, bmat, cmat, cvec, m, n0, n1, BN,
                            BCST_NO, UNROLL_AUTO, UNROLL_AUTO);
      }
    }
  }
#endif
}
#endif
