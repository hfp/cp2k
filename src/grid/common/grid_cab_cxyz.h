/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_CAB_CXYZ_H
#define GRID_CAB_CXYZ_H

// Transformation between the coefficients C_ab of a product of Cartesian
// Gaussians about the atoms A and B and the coefficients C_xyz of a polynomial
// about the center P of the product, in three separable 1D steps. Without
// variable-length arrays and with optional address space qualifiers, hence
// usable from C, CUDA/HIP, and OpenCL C. The caller provides the scratch.

#include "../common/grid_common.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GRID_CAB_CXYZ_DEVICE __device__
#else
#define GRID_CAB_CXYZ_DEVICE
#endif
// Address space qualifiers of the arrays (OpenCL C), empty by default.
#if !defined(GRID_ALPHA_QUAL)
#define GRID_ALPHA_QUAL
#endif
#if !defined(GRID_CAB_QUAL)
#define GRID_CAB_QUAL
#endif
#if !defined(GRID_CXYZ_QUAL)
#define GRID_CXYZ_QUAL
#endif
#if !defined(GRID_SCRATCH_QUAL)
#define GRID_SCRATCH_QUAL
#endif

/*******************************************************************************
 * \brief Index of alpha[i][lxb][lxa][k] in the flat array of size
 *        3 * (lb_max + 1) * (la_max + 1) * (la_max + lb_max + 1).
 ******************************************************************************/
GRID_CAB_CXYZ_DEVICE static inline int
grid_alpha_index(const int la_max, const int lb_max, const int i, const int lxb,
                 const int lxa, const int k) {
  return ((i * (lb_max + 1) + lxb) * (la_max + 1) + lxa) *
             (la_max + lb_max + 1) +
         k;
}

/*******************************************************************************
 * \brief Polynomial expansion coefficients of the 1D products:
 *        (x-a)**lxa (x-b)**lxb -> sum_k alpha[i][lxb][lxa][k] * (x-p)**k
 ******************************************************************************/
GRID_CAB_CXYZ_DEVICE static inline void
grid_cab_alpha(const int la_max, const int lb_max, const double ra[3],
               const double rb[3], const double rp[3],
               GRID_ALPHA_QUAL double *alpha) {
#define GRID_ALPHA(I, LXB, LXA, K)                                             \
  alpha[grid_alpha_index(la_max, lb_max, I, LXB, LXA, K)]
  for (int i = 0; i < 3; i++) {
    const double drpa = rp[i] - ra[i];
    const double drpb = rp[i] - rb[i];
    GRID_ALPHA(i, 0, 0, 0) = 1.0;
    for (int lxa = 1; lxa <= la_max; lxa++) {
      GRID_ALPHA(i, 0, lxa, 0) = drpa * GRID_ALPHA(i, 0, lxa - 1, 0);
      for (int k = 1; k < lxa; k++) {
        GRID_ALPHA(i, 0, lxa, k) = GRID_ALPHA(i, 0, lxa - 1, k - 1) +
                                   drpa * GRID_ALPHA(i, 0, lxa - 1, k);
      }
      GRID_ALPHA(i, 0, lxa, lxa) = 1.0;
    }
    for (int lxb = 1; lxb <= lb_max; lxb++) {
      for (int lxa = 0; lxa <= la_max; lxa++) {
        const int n = lxa + lxb;
        GRID_ALPHA(i, lxb, lxa, 0) = drpb * GRID_ALPHA(i, lxb - 1, lxa, 0);
        for (int k = 1; k < n; k++) {
          GRID_ALPHA(i, lxb, lxa, k) = GRID_ALPHA(i, lxb - 1, lxa, k - 1) +
                                       drpb * GRID_ALPHA(i, lxb - 1, lxa, k);
        }
        GRID_ALPHA(i, lxb, lxa, n) = 1.0;
      }
    }
  }
#undef GRID_ALPHA
}

/*******************************************************************************
 * \brief Transforms C_ab into C_xyz (collocate), adding to cxyz[lzp][lyp][lxp]
 *        of size (lp + 1)**3. Scratch: tx of size lp + 1, txy of size
 *        (lp + 1)**2, where lp = la_max + lb_max.
 ******************************************************************************/
GRID_CAB_CXYZ_DEVICE static inline void
grid_cab_to_cxyz(const int la_max, const int la_min, const int lb_max,
                 const int lb_min, const double prefactor,
                 const GRID_ALPHA_QUAL double *alpha,
                 const GRID_CAB_QUAL double *cab, GRID_CXYZ_QUAL double *cxyz,
                 GRID_SCRATCH_QUAL double *tx, GRID_SCRATCH_QUAL double *txy) {
#define GRID_ALPHA(I, LXB, LXA, K)                                             \
  alpha[grid_alpha_index(la_max, lb_max, I, LXB, LXA, K)]
  const int lp = la_max + lb_max;
  const int lp1 = lp + 1;
  const int ncoset_a = ncoset(la_max);
  for (int lzb = 0; lzb <= lb_max; lzb++) {
    for (int lza = 0; lza <= la_max; lza++) {
      const int lz_max = lza + lzb;
      const int lyp_max = lp - lz_max;
      for (int i = 0; i < (lyp_max + 1) * lp1; i++) {
        txy[i] = 0.0;
      }
      for (int lyb = 0; lyb <= lb_max - lzb; lyb++) {
        for (int lya = 0; lya <= la_max - lza; lya++) {
          const int ly_max = lya + lyb;
          const int lxb_min = imax(lb_min - lzb - lyb, 0);
          const int lxa_min = imax(la_min - lza - lya, 0);
          const int lxb_max = lb_max - lzb - lyb;
          const int lxa_max = la_max - lza - lya;
          const int lxp_max = lxa_max + lxb_max;
          for (int i = 0; i <= lxp_max; i++) {
            tx[i] = 0.0;
          }
          for (int lxb = lxb_min; lxb <= lxb_max; lxb++) {
            const int jco = coset(lxb, lyb, lzb);
            for (int lxa = lxa_min; lxa <= lxa_max; lxa++) {
              const int ico = coset(lxa, lya, lza);
              const int cab_index = jco * ncoset_a + ico; // [jco, ico]
              const double cab_val = cab[cab_index] * prefactor;
              for (int lxp = 0; lxp <= lxa + lxb; lxp++) {
                tx[lxp] += cab_val * GRID_ALPHA(0, lxb, lxa, lxp);
              }
            }
          }
          for (int lyp = 0; lyp <= ly_max; lyp++) {
            const double ay = GRID_ALPHA(1, lyb, lya, lyp);
            GRID_SCRATCH_QUAL double *txy_row = &txy[lyp * lp1];
            for (int lxp = 0; lxp <= lxp_max; lxp++) {
              txy_row[lxp] += tx[lxp] * ay;
            }
          }
        }
      }
      for (int lzp = 0; lzp <= lz_max; lzp++) {
        const double az = GRID_ALPHA(2, lzb, lza, lzp);
        GRID_CXYZ_QUAL double *cxyz_z = &cxyz[lzp * lp1 * lp1];
        for (int lyp = 0; lyp <= lyp_max; lyp++) {
          const GRID_SCRATCH_QUAL double *txy_row = &txy[lyp * lp1];
          GRID_CXYZ_QUAL double *cxyz_zy = &cxyz_z[lyp * lp1];
          for (int lxp = 0; lxp <= lyp_max - lyp; lxp++) {
            cxyz_zy[lxp] += txy_row[lxp] * az;
          }
        }
      }
    }
  }
#undef GRID_ALPHA
}

/*******************************************************************************
 * \brief Transforms C_xyz into C_ab (integrate), adding to cab[jco][ico].
 *        Scratch as for grid_cab_to_cxyz.
 ******************************************************************************/
GRID_CAB_CXYZ_DEVICE static inline void
grid_cxyz_to_cab(const int la_max, const int la_min, const int lb_max,
                 const int lb_min, const double prefactor,
                 const GRID_ALPHA_QUAL double *alpha,
                 const GRID_CXYZ_QUAL double *cxyz, GRID_CAB_QUAL double *cab,
                 GRID_SCRATCH_QUAL double *tx, GRID_SCRATCH_QUAL double *txy) {
#define GRID_ALPHA(I, LXB, LXA, K)                                             \
  alpha[grid_alpha_index(la_max, lb_max, I, LXB, LXA, K)]
  const int lp = la_max + lb_max;
  const int lp1 = lp + 1;
  const int ncoset_a = ncoset(la_max);
  for (int lzb = 0; lzb <= lb_max; lzb++) {
    for (int lza = 0; lza <= la_max; lza++) {
      const int lz_max = lza + lzb;
      const int lyp_max = lp - lz_max;
      for (int i = 0; i < (lyp_max + 1) * lp1; i++) {
        txy[i] = 0.0;
      }
      for (int lzp = 0; lzp <= lz_max; lzp++) {
        const double az = GRID_ALPHA(2, lzb, lza, lzp) * prefactor;
        const GRID_CXYZ_QUAL double *cxyz_z = &cxyz[lzp * lp1 * lp1];
        for (int lyp = 0; lyp <= lyp_max; lyp++) {
          GRID_SCRATCH_QUAL double *txy_row = &txy[lyp * lp1];
          const GRID_CXYZ_QUAL double *cxyz_zy = &cxyz_z[lyp * lp1];
          for (int lxp = 0; lxp <= lyp_max - lyp; lxp++) {
            txy_row[lxp] += cxyz_zy[lxp] * az;
          }
        }
      }
      for (int lyb = 0; lyb <= lb_max - lzb; lyb++) {
        for (int lya = 0; lya <= la_max - lza; lya++) {
          const int ly_max = lya + lyb;
          const int lxb_min = imax(lb_min - lzb - lyb, 0);
          const int lxa_min = imax(la_min - lza - lya, 0);
          const int lxb_max = lb_max - lzb - lyb;
          const int lxa_max = la_max - lza - lya;
          const int lxp_max = lxa_max + lxb_max;
          for (int i = 0; i <= lxp_max; i++) {
            tx[i] = 0.0;
          }
          for (int lyp = 0; lyp <= ly_max; lyp++) {
            const double ay = GRID_ALPHA(1, lyb, lya, lyp);
            const GRID_SCRATCH_QUAL double *txy_row = &txy[lyp * lp1];
            for (int lxp = 0; lxp <= lxp_max; lxp++) {
              tx[lxp] += txy_row[lxp] * ay;
            }
          }
          for (int lxb = lxb_min; lxb <= lxb_max; lxb++) {
            const int jco = coset(lxb, lyb, lzb);
            for (int lxa = lxa_min; lxa <= lxa_max; lxa++) {
              const int ico = coset(lxa, lya, lza);
              const int cab_index = jco * ncoset_a + ico; // [jco, ico]
              double sum_x = 0.0;
              for (int lxp = 0; lxp <= lxa + lxb; lxp++) {
                sum_x += tx[lxp] * GRID_ALPHA(0, lxb, lxa, lxp);
              }
              cab[cab_index] += sum_x;
            }
          }
        }
      }
    }
  }
#undef GRID_ALPHA
}

#endif // GRID_CAB_CXYZ_H

// EOF
