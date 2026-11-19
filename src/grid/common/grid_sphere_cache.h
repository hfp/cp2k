/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#ifndef GRID_SPHERE_CACHE_H
#define GRID_SPHERE_CACHE_H

/*******************************************************************************
 * \brief Struct holding the sphere cache for one grid as specified by dr[3].
 * \author Ole Schuett
 ******************************************************************************/
#include <math.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GRID_SPHERE_HOST_DEVICE __host__ __device__
#else
#define GRID_SPHERE_HOST_DEVICE
#endif

// Number of steps per grid spacing in which the cutoff radius is discretized.
#define GRID_SPHERE_STEPS 8

/*******************************************************************************
 * \brief Discretized cutoff radius in steps of drmin / GRID_SPHERE_STEPS: the
 *        radius plus half the smallest grid spacing drmin, rounded up. A task
 *        contributes to the grid points within this radius of the grid cell
 *        that holds its center (orthorhombic), or of its center otherwise; the
 *        margin of 1/2 to 5/8 grid spacing keeps the truncation error below
 *        the former rounding to whole spacings (or the bare radius).
 ******************************************************************************/
GRID_SPHERE_HOST_DEVICE static inline int
grid_sphere_index(const double radius, const double drmin) {
  return (int)ceil((radius + 0.5 * drmin) * GRID_SPHERE_STEPS / drmin);
}

/*******************************************************************************
 * \brief Cutoff radius of the given discretization index.
 ******************************************************************************/
GRID_SPHERE_HOST_DEVICE static inline double
grid_sphere_radius(const int index, const double drmin) {
  return index * drmin / GRID_SPHERE_STEPS;
}

/*******************************************************************************
 * \brief Smallest length of the lattice vectors dh[0..2] (rows of dh, given as
 *        nine values), the grid spacing of the cutoff margin.
 ******************************************************************************/
GRID_SPHERE_HOST_DEVICE static inline double
grid_sphere_spacing(const double *dh) {
  double h = 0.0;
  for (int i = 0; i < 3; i++) {
    const double n =
        sqrt(dh[3 * i] * dh[3 * i] + dh[3 * i + 1] * dh[3 * i + 1] +
             dh[3 * i + 2] * dh[3 * i + 2]);
    h = (0 == i || n < h) ? n : h;
  }
  return h;
}

/*******************************************************************************
 * \brief Cutoff radius for the given radius and grid spacing.
 ******************************************************************************/
GRID_SPHERE_HOST_DEVICE static inline double
grid_sphere_cutoff(const double radius, const double h) {
  return grid_sphere_radius(grid_sphere_index(radius, h), h);
}

typedef struct {
  double dr[3];
  double drmin;
  double drmin_inv;
  int max_imr;
  int *offsets;
  int *storage;
} grid_sphere_cache_entry;

/*******************************************************************************
 * \brief Struct holding the entire sphere cache, ie. for all grids.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  int size;
  int prev_match;
  grid_sphere_cache_entry *entries;
} grid_sphere_cache;

/*******************************************************************************
 * \brief Lookup the sphere bounds from the cache and compute them when missing.
 * \param radius        Non-discretized radius.
 * \param dh            Incremental grid matrix.
 * \param dh_inv        Inverse incremental grid matrix.
 * \param sphere_bounds Returned pointer to sphere bounds.
 * \param discr_radius  Returned discretized radius.
 * \author Ole Schuett
 ******************************************************************************/
void grid_sphere_cache_lookup(const double radius, const double dh[3][3],
                              const double dh_inv[3][3], int **sphere_bounds,
                              double *discretized_radius);

/*******************************************************************************
 * \brief Free the memory of the sphere cache.
 * \author Ole Schuett
 ******************************************************************************/
void grid_sphere_cache_free(grid_sphere_cache *cache);

#endif

// EOF
