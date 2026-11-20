/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../base/cp_assert.h"
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../offload/offload_buffer.h"
#include "common/grid_common.h"
#include "grid_replay.h"

#include "cpu/grid_cpu_collocate.h"
#include "cpu/grid_cpu_integrate.h"
#include "grid_task_list.h"
#include "grid_task_list_internal.h"

/*******************************************************************************
 * \brief Reads next line from given filehandle and handles errors.
 * \author Ole Schuett
 ******************************************************************************/
static void read_next_line(char line[], int length, FILE *fp) {
  if (fgets(line, length, fp) == NULL) {
    fprintf(stderr, "Error: Could not read line.\n");
    abort();
  }
}

/*******************************************************************************
 * \brief Shorthand for parsing a single integer value.
 * \author Ole Schuett
 ******************************************************************************/
static int parse_int(const char key[], FILE *fp) {
  int value;
  char line[100], format[100];
  read_next_line(line, sizeof(line), fp);
  snprintf(format, sizeof(format), "%s %%i", key);
  if (sscanf(line, format, &value) != 1) {
    cp_assert(!"parse_int failed");
  }
  return value;
}

/*******************************************************************************
 * \brief Shorthand for parsing a vector of three integer values.
 * \author Ole Schuett
 ******************************************************************************/
static void parse_int3(const char key[], FILE *fp, int vec[3]) {
  char line[100], format[100];
  read_next_line(line, sizeof(line), fp);
  snprintf(format, sizeof(format), "%s %%i %%i %%i", key);
  if (sscanf(line, format, &vec[0], &vec[1], &vec[2]) != 3) {
    cp_assert(!"parse_int3 failed");
  }
}

/*******************************************************************************
 * \brief Shorthand for parsing a single double value.
 * \author Ole Schuett
 ******************************************************************************/
static double parse_double(const char key[], FILE *fp) {
  double value;
  char line[100], format[100];
  read_next_line(line, sizeof(line), fp);
  snprintf(format, sizeof(format), "%s %%lf", key);
  if (sscanf(line, format, &value) != 1) {
    cp_assert(!"parse_double failed");
  }
  return value;
}

/*******************************************************************************
 * \brief Shorthand for parsing a vector of three double values.
 * \author Ole Schuett
 ******************************************************************************/
static void parse_double3(const char key[], FILE *fp, double vec[3]) {
  char line[100], format[100];
  read_next_line(line, sizeof(line), fp);
  snprintf(format, sizeof(format), "%s %%lf %%lf %%lf", key);
  if (sscanf(line, format, &vec[0], &vec[1], &vec[2]) != 3) {
    cp_assert(!"parse_double3 failed");
  }
}

/*******************************************************************************
 * \brief Shorthand for parsing a 3x3 matrix of doubles.
 * \author Ole Schuett
 ******************************************************************************/
static void parse_double3x3(const char key[], FILE *fp, double mat[3][3]) {
  char line[100], format[100];
  for (int i = 0; i < 3; i++) {
    read_next_line(line, sizeof(line), fp);
    snprintf(format, sizeof(format), "%s %i %%lf %%lf %%lf", key, i);
    if (sscanf(line, format, &mat[i][0], &mat[i][1], &mat[i][2]) != 3) {
      cp_assert(!"parse_double3x3 failed");
    }
  }
}

/*******************************************************************************
 * \brief Creates mock basis set using the identity as decontraction matrix.
 * \author Ole Schuett
 ******************************************************************************/
static void create_dummy_basis_set(const int size, const int lmin,
                                   const int lmax, const double zet,
                                   grid_basis_set **basis_set) {

  double sphi_mutable[size][size];
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      sphi_mutable[i][j] = (i == j) ? 1.0 : 0.0; // identity matrix
    }
  }
  const double (*sphi)[size] = (const double (*)[size])sphi_mutable;

  const int npgf = size / ncoset(lmax);
  cp_assert(size == npgf * ncoset(lmax));

  const int first_sgf[1] = {1};

  double zet_array_mutable[1][npgf];
  for (int i = 0; i < npgf; i++) {
    zet_array_mutable[0][i] = zet;
  }
  const double (*zet_array)[npgf] = (const double (*)[npgf])zet_array_mutable;

  grid_create_basis_set(/*nset=*/1,
                        /*nsgf=*/size,
                        /*maxco=*/size,
                        /*maxpgf=*/npgf,
                        /*lmin=*/&lmin,
                        /*lmax=*/&lmax,
                        /*npgf=*/&npgf,
                        /*nsgf_set=*/&size,
                        /*first_sgf=*/first_sgf,
                        /*sphi=*/sphi,
                        /*zet=*/zet_array, basis_set);
}

/*******************************************************************************
 * \brief Compares n results with their reference times scale, prints the ones
 *        beyond the tolerance if named, and returns the maximal relative
 *        difference times weight (infinity for a result that is not finite).
 * \author Hans Pabst
 ******************************************************************************/
static double compare(const char *name, const double *ref, const double scale,
                      const double *test, const int64_t n, const double weight,
                      const double tolerance, double *max_value) {
  double result = 0.0;
  for (int64_t i = 0; i < n; i++) {
    const double ref_value = scale * ref[i];
    const double diff = fabs(test[i] - ref_value);
    const double rel_diff = isfinite(test[i])
                                ? weight * diff / fmax(1.0, fabs(ref_value))
                                : INFINITY;
    if (name != NULL && rel_diff > tolerance) {
      printf("%s[%li] ref: %le test: %le diff: %le rel_diff: %le\n", name,
             (long)i, ref_value, test[i], diff, rel_diff);
    }
    result = fmax(result, rel_diff);
    if (max_value != NULL) {
      *max_value = fmax(*max_value, fabs(test[i]));
    }
  }
  return result;
}

/*******************************************************************************
 * \brief Returns the number of blocks of the mock task list.
 * \author Ole Schuett
 ******************************************************************************/
static int dummy_nblocks(const int cycles, const int cycles_per_block) {
  /* we can not have more blocks than the number of tasks */
  return (cycles == 1) ? 1 : (cycles / cycles_per_block + 1);
}

/*******************************************************************************
 * \brief Creates mock task list with one task per cycle. Each block has its own
 *        copy of the data, since blocks are processed concurrently and
 *        integrate writes into them.
 * \author Ole Schuett
 ******************************************************************************/
static void create_dummy_task_list(
    const bool orthorhombic, const int border_mask, const double ra[3],
    const double rab[3], const double radius, const grid_basis_set *basis_set_a,
    const grid_basis_set *basis_set_b, const int o1, const int o2,
    const int la_max, const int lb_max, const int cycles,
    const int cycles_per_block, const int block_size,
    const int npts_global[][3], const int npts_local[][3],
    const int shift_local[][3], const int border_width[][3],
    const double dh[][3][3], const double dh_inv[][3][3],
    grid_task_list **task_list) {

  const int ntasks = cycles;
  const int nlevels = 1;
  const int natoms = 2;
  const int nkinds = 2;
  const int nblocks = dummy_nblocks(cycles, cycles_per_block);

  int block_offsets[nblocks];
  for (int i = 0; i < nblocks; i++) {
    block_offsets[i] = i * block_size;
  }
  const double atom_positions[2][3] = {
      {ra[0], ra[1], ra[2]}, {rab[0] + ra[0], rab[1] + ra[1], rab[2] + ra[2]}};
  const int atom_kinds[2] = {1, 2};
  const grid_basis_set *basis_sets[2] = {basis_set_a, basis_set_b};
  const int ipgf = o1 / ncoset(la_max) + 1;
  const int jpgf = o2 / ncoset(lb_max) + 1;
  cp_assert(o1 == (ipgf - 1) * ncoset(la_max));
  cp_assert(o2 == (jpgf - 1) * ncoset(lb_max));

  int level_list[ntasks], iatom_list[ntasks], jatom_list[ntasks];
  int iset_list[ntasks], jset_list[ntasks], ipgf_list[ntasks],
      jpgf_list[ntasks];
  int border_mask_list[ntasks], block_num_list[ntasks];
  double radius_list[ntasks], rab_list_mutable[ntasks][3];
  for (int i = 0; i < cycles; i++) {
    level_list[i] = 1;
    iatom_list[i] = 1;
    jatom_list[i] = 2;
    iset_list[i] = 1;
    jset_list[i] = 1;
    ipgf_list[i] = ipgf;
    jpgf_list[i] = jpgf;
    border_mask_list[i] = border_mask;
    block_num_list[i] = i / cycles_per_block + 1;
    radius_list[i] = radius;
    rab_list_mutable[i][0] = rab[0];
    rab_list_mutable[i][1] = rab[1];
    rab_list_mutable[i][2] = rab[2];
  }
  const double (*rab_list)[3] = (const double (*)[3])rab_list_mutable;

  grid_create_task_list(
      orthorhombic, ntasks, nlevels, natoms, nkinds, nblocks, block_offsets,
      atom_positions, atom_kinds, basis_sets, level_list, iatom_list,
      jatom_list, iset_list, jset_list, ipgf_list, jpgf_list, border_mask_list,
      block_num_list, radius_list, rab_list, npts_global, npts_local,
      shift_local, border_width, dh, dh_inv, task_list);
}

/*******************************************************************************
 * \brief Reads a .task file, collocates/integrates it, and compares results.
 *        See grid_replay.h for details.
 * \author Ole Schuett
 ******************************************************************************/
bool grid_replay(const char *filename, const int cycles, const bool collocate,
                 const bool batch, const int cycles_per_block,
                 const double tolerance) {

  if (cycles < 1) {
    fprintf(stderr, "Error: Cycles have to be greater than zero.\n");
    exit(1);
  }

  if (cycles_per_block < 1 || cycles_per_block > cycles) {
    fprintf(stderr,
            "Error: Cycles per block has to be between 1 and cycles.\n");
    exit(1);
  }

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Could not open task file: %s\n", filename);
    exit(1);
  }

  char header_line[100];
  read_next_line(header_line, sizeof(header_line), fp);
  if (strcmp(header_line, "#Grid task v10\n") != 0) {
    fprintf(stderr, "Error: Wrong file header.\n");
    abort();
  }

  const bool orthorhombic = parse_int("orthorhombic", fp);
  const int border_mask = parse_int("border_mask", fp);
  const enum grid_func func = (enum grid_func)parse_int("func", fp);
  const bool compute_tau = (func == GRID_FUNC_DADB);
  const int la_max = parse_int("la_max", fp);
  const int la_min = parse_int("la_min", fp);
  const int lb_max = parse_int("lb_max", fp);
  const int lb_min = parse_int("lb_min", fp);
  const double zeta = parse_double("zeta", fp);
  const double zetb = parse_double("zetb", fp);
  const double rscale = parse_double("rscale", fp);

  double dh_mutable[3][3], dh_inv_mutable[3][3], ra[3], rab[3];
  parse_double3x3("dh", fp, dh_mutable);
  parse_double3x3("dh_inv", fp, dh_inv_mutable);
  parse_double3("ra", fp, ra);
  parse_double3("rab", fp, rab);
  const double (*dh)[3] = (const double (*)[3])dh_mutable;
  const double (*dh_inv)[3] = (const double (*)[3])dh_inv_mutable;

  int npts_global[3], npts_local[3], shift_local[3], border_width[3];
  parse_int3("npts_global", fp, npts_global);
  parse_int3("npts_local", fp, npts_local);
  parse_int3("shift_local", fp, shift_local);
  parse_int3("border_width", fp, border_width);

  const double radius = parse_double("radius", fp);
  const int o1 = parse_int("o1", fp);
  const int o2 = parse_int("o2", fp);
  const int n1 = parse_int("n1", fp);
  const int n2 = parse_int("n2", fp);
  const size_t n12 = (size_t)n1 * (size_t)n2;

  double *pab_storage = malloc(n12 * sizeof(double));
  if (pab_storage == NULL) {
    fprintf(stderr, "Error: Could not allocate pab buffer.\n");
    abort();
  }
  double (*pab)[n1] = (double (*)[n1])pab_storage;
  char line[100], format[100];
  for (int i = 0; i < n2; i++) {
    for (int j = 0; j < n1; j++) {
      read_next_line(line, sizeof(line), fp);
      snprintf(format, sizeof(format), "pab %i %i %%lf", i, j);
      if (sscanf(line, format, &pab[i][j]) != 1) {
        cp_assert(!"parse_pab failed");
      }
    }
  }

  const int npts_local_total = npts_local[0] * npts_local[1] * npts_local[2];
  offload_buffer *grid_ref = NULL;
  offload_create_buffer(npts_local_total, &grid_ref);
  memset(grid_ref->host_buffer, 0, npts_local_total * sizeof(double));

  const int ngrid_nonzero = parse_int("ngrid_nonzero", fp);
  for (int n = 0; n < ngrid_nonzero; n++) {
    int i, j, k;
    double value;
    read_next_line(line, sizeof(line), fp);
    if (sscanf(line, "grid %i %i %i %le", &i, &j, &k, &value) != 4) {
      cp_assert(!"parse_grid failed");
    }
    grid_ref->host_buffer[k * npts_local[1] * npts_local[0] +
                          j * npts_local[0] + i] = value;
  }

  double *hab_ref_storage = calloc(n12, sizeof(double));
  if (hab_ref_storage == NULL) {
    fprintf(stderr, "Error: Could not allocate hab_ref buffer.\n");
    abort();
  }
  double (*hab_ref)[n1] = (double (*)[n1])hab_ref_storage;
  for (int i = o2; i < ncoset(lb_max) + o2; i++) {
    for (int j = o1; j < ncoset(la_max) + o1; j++) {
      read_next_line(line, sizeof(line), fp);
      snprintf(format, sizeof(format), "hab %i %i %%lf", i, j);
      if (sscanf(line, format, &hab_ref[i][j]) != 1) {
        cp_assert(!"parse_hab failed");
      }
    }
  }

  double forces_ref[2][3];
  parse_double3("force_a", fp, forces_ref[0]);
  parse_double3("force_b", fp, forces_ref[1]);

  double virial_ref[3][3];
  parse_double3x3("virial", fp, virial_ref);

  char footer_line[100];
  read_next_line(footer_line, sizeof(footer_line), fp);
  if (strcmp(footer_line, "#THE_END\n") != 0) {
    fprintf(stderr, "Error: Wrong footer line.\n");
    abort();
  }

  if (fclose(fp) != 0) {
    fprintf(stderr, "Could not close task file: %s\n", filename);
    abort();
  }

  offload_buffer *grid_test = NULL;
  offload_create_buffer(npts_local_total, &grid_test);
  double *hab_test_storage = calloc(n12, sizeof(double));
  if (hab_test_storage == NULL) {
    fprintf(stderr, "Error: Could not allocate hab_test buffer.\n");
    abort();
  }
  double (*hab_test)[n1] = (double (*)[n1])hab_test_storage;
  double forces_test[2][3];
  double virial_test[3][3];
  double start_time, end_time;

  if (batch) {
    grid_basis_set *basisa = NULL, *basisb = NULL;
    create_dummy_basis_set(n1, la_min, la_max, zeta, &basisa);
    create_dummy_basis_set(n2, lb_min, lb_max, zetb, &basisb);
    grid_task_list *task_list = NULL;
    create_dummy_task_list(
        orthorhombic, border_mask, ra, rab, radius, basisa, basisb, o1, o2,
        la_max, lb_max, cycles, cycles_per_block, n1 * n2,
        (const int (*)[3])npts_global, (const int (*)[3])npts_local,
        (const int (*)[3])shift_local, (const int (*)[3])border_width,
        (const double (*)[3][3])dh, (const double (*)[3][3])dh_inv, &task_list);
    const int nblocks = dummy_nblocks(cycles, cycles_per_block);
    offload_buffer *pab_blocks = NULL, *hab_blocks = NULL;
    offload_create_buffer(nblocks * n1 * n2, &pab_blocks);
    offload_create_buffer(nblocks * n1 * n2, &hab_blocks);
    const double f = (collocate) ? rscale : 1.0;
    for (int b = 0; b < nblocks; b++) {
      for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
          pab_blocks->host_buffer[b * n1 * n2 + j * n1 + i] =
              0.5 * f * pab[j][i];
        }
      }
    }
    start_time = omp_get_wtime();
    const int nlevels = 1;
    const int natoms = 2;
    if (collocate) {
      // collocate
      offload_buffer *grids[1] = {grid_test};
      grid_collocate_task_list(task_list, func, nlevels,
                               (const int (*)[3])npts_local, pab_blocks, grids);
    } else {
      // integrate
      const offload_buffer *grids[1] = {grid_ref};
      grid_integrate_task_list(task_list, compute_tau, natoms, nlevels,
                               (const int (*)[3])npts_local, pab_blocks, grids,
                               hab_blocks, forces_test, virial_test);
      for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n1; j++) {
          hab_test[i][j] = 0.0;
          for (int b = 0; b < nblocks; b++) {
            hab_test[i][j] += hab_blocks->host_buffer[b * n1 * n2 + i * n1 + j];
          }
        }
      }
    }
    end_time = omp_get_wtime();
    grid_free_basis_set(basisa);
    grid_free_basis_set(basisb);
    grid_free_task_list(task_list);
    offload_free_buffer(pab_blocks);
    offload_free_buffer(hab_blocks);
  } else {
    start_time = omp_get_wtime();
    if (collocate) {
      // collocate
      memset(grid_test->host_buffer, 0, npts_local_total * sizeof(double));
      for (int i = 0; i < cycles; i++) {
        grid_cpu_collocate_pgf_product(
            orthorhombic, border_mask, func, la_max, la_min, lb_max, lb_min,
            zeta, zetb, rscale, dh, dh_inv, ra, rab, npts_global, npts_local,
            shift_local, border_width, radius, o1, o2, n1, n2, pab,
            grid_test->host_buffer);
      }
    } else {
      // integrate
      memset(hab_test_storage, 0, n12 * sizeof(double));
      memset(forces_test, 0, 2 * 3 * sizeof(double));
      double virials_test[2][3][3] = {0};
      for (int i = 0; i < cycles; i++) {
        grid_cpu_integrate_pgf_product(
            orthorhombic, compute_tau, border_mask, la_max, la_min, lb_max,
            lb_min, zeta, zetb, dh, dh_inv, ra, rab, npts_global, npts_local,
            shift_local, border_width, radius, o1, o2, n1, n2,
            grid_ref->host_buffer, hab_test, pab, forces_test, virials_test,
            NULL, NULL, NULL);
      }
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          virial_test[i][j] = virials_test[0][i][j] + virials_test[1][i][j];
        }
      }
    }
    end_time = omp_get_wtime();
  }

  double max_value = 0.0, max_rel_diff = 0.0;
  const double derivatives_precision = 1e-4; // account for higher numeric noise
  if (collocate) {
    max_rel_diff =
        compare(NULL, grid_ref->host_buffer, cycles, grid_test->host_buffer,
                npts_local_total, 1.0, tolerance, &max_value);
  } else {
    max_rel_diff = compare("hab", hab_ref_storage, cycles, hab_test_storage,
                           n12, 1.0, tolerance, &max_value);
    max_rel_diff =
        fmax(max_rel_diff,
             compare("forces", &forces_ref[0][0], cycles, &forces_test[0][0], 6,
                     derivatives_precision, tolerance, NULL));
    max_rel_diff =
        fmax(max_rel_diff,
             compare("virial", &virial_ref[0][0], cycles, &virial_test[0][0], 9,
                     derivatives_precision, tolerance, NULL));
  }
  printf("Task: %-55s   %9s %-7s   Cycles: %e   Max value: %le   "
         "Max rel diff: %le   Time: %le sec\n",
         filename, collocate ? "Collocate" : "Integrate",
         batch ? "Batched" : "PGF-CPU", (float)cycles, max_value, max_rel_diff,
         end_time - start_time);

  offload_free_buffer(grid_ref);
  offload_free_buffer(grid_test);
  free(pab_storage);
  free(hab_ref_storage);
  free(hab_test_storage);

  // Check floating point exceptions.
  if (fetestexcept(FE_DIVBYZERO) != 0) {
    fprintf(stderr, "Error: Floating point exception FE_DIVBYZERO.\n");
    exit(1);
  }
  if (fetestexcept(FE_OVERFLOW) != 0) {
    fprintf(stderr, "Error: Floating point exception FE_OVERFLOW.\n");
    exit(1);
  }

  return max_rel_diff < tolerance;
}

/*******************************************************************************
 * \brief Reads an array preceded by its number of elements from a task list
 *        file. A non-negative expected count must match. The array is
 *        allocated and has to be freed by the caller.
 * \author Hans Pabst
 ******************************************************************************/
static void *read_array(FILE *fp, const size_t elem_size,
                        const int64_t expected, int64_t *count) {
  int64_t n = 0;
  if (fread(&n, sizeof(int64_t), 1, fp) != 1 || n < 0 ||
      (expected >= 0 && n != expected)) {
    fprintf(stderr, "Error: Malformed task list file.\n");
    abort();
  }
  void *array = malloc((n > 0 ? (size_t)n : 1) * elem_size);
  if (array == NULL) {
    fprintf(stderr, "Error: Could not allocate task list array.\n");
    abort();
  }
  if (n > 0 && fread(array, elem_size, (size_t)n, fp) != (size_t)n) {
    fprintf(stderr, "Error: Truncated task list file.\n");
    abort();
  }
  if (count != NULL) {
    *count = n;
  }
  return array;
}

/*******************************************************************************
 * \brief Tells whether the given file is a task list (grid_replay_task_list).
 * \author Hans Pabst
 ******************************************************************************/
bool grid_replay_is_task_list(const char *filename) {
  bool result = false;
  FILE *fp = fopen(filename, "rb");
  if (fp != NULL) {
    char header_line[100];
    result = (fgets(header_line, sizeof(header_line), fp) != NULL &&
              strcmp(header_line, GRID_TASK_LIST_HEADER) == 0);
    fclose(fp);
  }
  return result;
}

/*******************************************************************************
 * \brief Replays a task list written by grid_collocate_task_list or
 *        grid_integrate_task_list. See grid_replay.h for details.
 * \author Hans Pabst
 ******************************************************************************/
bool grid_replay_task_list(const char *filename, const int cycles,
                           const double tolerance) {
  if (cycles < 1) {
    fprintf(stderr, "Error: Cycles have to be greater than zero.\n");
    exit(1);
  }
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    fprintf(stderr, "Could not open task list file: %s\n", filename);
    exit(1);
  }
  char header_line[100];
  read_next_line(header_line, sizeof(header_line), fp);
  if (strcmp(header_line, GRID_TASK_LIST_HEADER) != 0) {
    fprintf(stderr, "Error: Wrong file header.\n");
    abort();
  }
  int *probe_int = read_array(fp, sizeof(int), 1, NULL);
  double *probe_double = read_array(fp, sizeof(double), 1, NULL);
  if (*probe_int != 1 || *probe_double != 1.0) {
    fprintf(stderr, "Error: Task list written with another byte order.\n");
    abort();
  }
  free(probe_int);
  free(probe_double);
  int *kind = read_array(fp, sizeof(int), 1, NULL);
  const bool collocate = (*kind == 0);
  free(kind);

  // Arguments of grid_create_task_list.
  int *scalars = read_array(fp, sizeof(int), 6, NULL);
  const bool orthorhombic = scalars[0];
  const int ntasks = scalars[1], nlevels = scalars[2], natoms = scalars[3];
  const int nkinds = scalars[4], nblocks = scalars[5];
  free(scalars);
  int *block_offsets = read_array(fp, sizeof(int), nblocks, NULL);
  double *atom_positions = read_array(fp, sizeof(double), 3 * natoms, NULL);
  int *atom_kinds = read_array(fp, sizeof(int), natoms, NULL);
  grid_basis_set **basis_sets = calloc(nkinds, sizeof(grid_basis_set *));
  cp_assert(basis_sets != NULL);
  for (int ikind = 0; ikind < nkinds; ikind++) {
    int *sizes = read_array(fp, sizeof(int), 4, NULL);
    const int nset = sizes[0], nsgf = sizes[1], maxco = sizes[2];
    const int maxpgf = sizes[3];
    free(sizes);
    int *lmin = read_array(fp, sizeof(int), nset, NULL);
    int *lmax = read_array(fp, sizeof(int), nset, NULL);
    int *npgf = read_array(fp, sizeof(int), nset, NULL);
    int *nsgf_set = read_array(fp, sizeof(int), nset, NULL);
    int *first_sgf = read_array(fp, sizeof(int), nset, NULL);
    double *sphi = read_array(fp, sizeof(double), (int64_t)nsgf * maxco, NULL);
    double *zet = read_array(fp, sizeof(double), (int64_t)nset * maxpgf, NULL);
    grid_create_basis_set(nset, nsgf, maxco, maxpgf, lmin, lmax, npgf, nsgf_set,
                          first_sgf, (const double (*)[maxco])sphi,
                          (const double (*)[maxpgf])zet, &basis_sets[ikind]);
    free(lmin);
    free(lmax);
    free(npgf);
    free(nsgf_set);
    free(first_sgf);
    free(sphi);
    free(zet);
  }
  int *task_ints[9];
  for (int i = 0; i < 9; i++) {
    task_ints[i] = read_array(fp, sizeof(int), ntasks, NULL);
  }
  double *radius_list = read_array(fp, sizeof(double), ntasks, NULL);
  double *rab_list = read_array(fp, sizeof(double), 3 * (int64_t)ntasks, NULL);
  int *npts_global = read_array(fp, sizeof(int), 3 * nlevels, NULL);
  int *npts_local = read_array(fp, sizeof(int), 3 * nlevels, NULL);
  int *shift_local = read_array(fp, sizeof(int), 3 * nlevels, NULL);
  int *border_width = read_array(fp, sizeof(int), 3 * nlevels, NULL);
  double *dh = read_array(fp, sizeof(double), 9 * nlevels, NULL);
  double *dh_inv = read_array(fp, sizeof(double), 9 * nlevels, NULL);
  const int (*npts)[3] = (const int (*)[3])npts_local;

  grid_task_list *task_list = NULL;
  const double create_start = omp_get_wtime();
  grid_create_task_list(
      orthorhombic, ntasks, nlevels, natoms, nkinds, nblocks, block_offsets,
      (const double (*)[3])atom_positions, atom_kinds,
      (const grid_basis_set **)basis_sets, task_ints[0], task_ints[1],
      task_ints[2], task_ints[3], task_ints[4], task_ints[5], task_ints[6],
      task_ints[7], task_ints[8], radius_list, (const double (*)[3])rab_list,
      (const int (*)[3])npts_global, npts, (const int (*)[3])shift_local,
      (const int (*)[3])border_width, (const double (*)[3][3])dh,
      (const double (*)[3][3])dh_inv, &task_list);
  const double create_time = omp_get_wtime() - create_start;

  // Arguments of the call, its results, and the replay.
  offload_buffer *pab_blocks = NULL;
  offload_buffer *grids[nlevels];
  double *grids_ref[nlevels];
  for (int level = 0; level < nlevels; level++) {
    grids[level] = NULL;
    grids_ref[level] = NULL;
  }
  const double derivatives_precision = 1e-4; // as in grid_replay
  double max_value = 0.0, max_rel_diff = 0.0, start_time, end_time;
  int64_t npab = 0;
  if (collocate) {
    int *func = read_array(fp, sizeof(int), 1, NULL);
    double *pab = read_array(fp, sizeof(double), -1, &npab);
    offload_create_buffer((int)npab, &pab_blocks);
    memcpy(pab_blocks->host_buffer, pab, npab * sizeof(double));
    free(pab);
    for (int level = 0; level < nlevels; level++) {
      const int64_t n =
          (int64_t)npts[level][0] * npts[level][1] * npts[level][2];
      grids_ref[level] = read_array(fp, sizeof(double), n, NULL);
      offload_create_buffer((int)n, &grids[level]);
    }
    start_time = omp_get_wtime();
    for (int i = 0; i < cycles; i++) {
      grid_collocate_task_list(task_list, (enum grid_func) * func, nlevels,
                               npts, pab_blocks, grids);
    }
    end_time = omp_get_wtime();
    free(func);
    for (int level = 0; level < nlevels; level++) {
      const int64_t n =
          (int64_t)npts[level][0] * npts[level][1] * npts[level][2];
      max_rel_diff = fmax(max_rel_diff, compare(NULL, grids_ref[level], 1.0,
                                                grids[level]->host_buffer, n,
                                                1.0, tolerance, &max_value));
    }
  } else {
    int *flags = read_array(fp, sizeof(int), 4, NULL);
    const bool compute_tau = flags[0], has_pab = flags[1];
    const bool has_forces = flags[2], has_virial = flags[3];
    free(flags);
    double *pab = read_array(fp, sizeof(double), -1, &npab);
    if (has_pab) {
      offload_create_buffer((int)npab, &pab_blocks);
      memcpy(pab_blocks->host_buffer, pab, npab * sizeof(double));
    }
    free(pab);
    for (int level = 0; level < nlevels; level++) {
      const int64_t n =
          (int64_t)npts[level][0] * npts[level][1] * npts[level][2];
      double *grid = read_array(fp, sizeof(double), n, NULL);
      offload_create_buffer((int)n, &grids[level]);
      memcpy(grids[level]->host_buffer, grid, n * sizeof(double));
      free(grid);
    }
    int64_t nhab = 0;
    double *hab_ref = read_array(fp, sizeof(double), -1, &nhab);
    double *forces_ref =
        read_array(fp, sizeof(double), has_forces ? 3 * natoms : 0, NULL);
    double *virial_ref =
        read_array(fp, sizeof(double), has_virial ? 9 : 0, NULL);
    offload_buffer *hab_blocks = NULL;
    offload_create_buffer((int)nhab, &hab_blocks);
    double *forces = calloc(3 * (size_t)natoms + 1, sizeof(double));
    double virial[3][3];
    cp_assert(forces != NULL);
    start_time = omp_get_wtime();
    for (int i = 0; i < cycles; i++) {
      grid_integrate_task_list(task_list, compute_tau, natoms, nlevels, npts,
                               pab_blocks, (const offload_buffer **)grids,
                               hab_blocks,
                               has_forces ? (double (*)[3])forces : NULL,
                               has_virial ? virial : NULL);
    }
    end_time = omp_get_wtime();
    max_rel_diff = compare("hab", hab_ref, 1.0, hab_blocks->host_buffer, nhab,
                           1.0, tolerance, &max_value);
    max_rel_diff =
        fmax(max_rel_diff, compare("forces", forces_ref, 1.0, forces,
                                   has_forces ? 3 * natoms : 0,
                                   derivatives_precision, tolerance, NULL));
    max_rel_diff =
        fmax(max_rel_diff, compare("virial", virial_ref, 1.0, &virial[0][0],
                                   has_virial ? 9 : 0, derivatives_precision,
                                   tolerance, NULL));
    offload_free_buffer(hab_blocks);
    free(hab_ref);
    free(forces_ref);
    free(virial_ref);
    free(forces);
  }
  if (fclose(fp) != 0) {
    fprintf(stderr, "Could not close task list file: %s\n", filename);
    abort();
  }

  static const char *const backends[] = {"AUTO", "REF", "CPU", "DGEMM", "GPU"};
  const int backend = ((const grid_task_list_internal *)task_list)->backend;
  const int ibackend = backend - GRID_BACKEND_AUTO;
  printf("Task list: %-48s %9s %-5s   Tasks: %i   Levels: %i   Cycles: %i   "
         "Max value: %le   Max rel diff: %le   Create: %le sec   "
         "Time: %le sec\n",
         filename, collocate ? "Collocate" : "Integrate",
         (0 <= ibackend && ibackend < 5) ? backends[ibackend] : "?", ntasks,
         nlevels, cycles, max_value, max_rel_diff, create_time,
         end_time - start_time);

  grid_free_task_list(task_list);
  for (int level = 0; level < nlevels; level++) {
    offload_free_buffer(grids[level]);
    free(grids_ref[level]);
  }
  if (pab_blocks != NULL) {
    offload_free_buffer(pab_blocks);
  }
  for (int ikind = 0; ikind < nkinds; ikind++) {
    grid_free_basis_set(basis_sets[ikind]);
  }
  for (int i = 0; i < 9; i++) {
    free(task_ints[i]);
  }
  free(basis_sets);
  free(block_offsets);
  free(atom_positions);
  free(atom_kinds);
  free(radius_list);
  free(rab_list);
  free(npts_global);
  free(npts_local);
  free(shift_local);
  free(border_width);
  free(dh);
  free(dh_inv);

  return max_rel_diff < tolerance;
}

// EOF
