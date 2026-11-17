/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mpiwrap/cp_mpi.h"
#include "grid_task_list.h"
#include "grid_task_list_internal.h"

// Number of collocate (0) and integrate (1) calls written to file so far.
static int dump_count[2] = {0, 0};

/*******************************************************************************
 * \brief Appends an array preceded by its number of elements to a buffer.
 * \author Hans Pabst
 ******************************************************************************/
static void dump_append(char **data, size_t *size, const void *array,
                        const int64_t count, const size_t elem_size) {
  const size_t bytes = (size_t)count * elem_size;
  *data = realloc(*data, *size + sizeof(int64_t) + bytes);
  assert(*data != NULL);
  memcpy(*data + *size, &count, sizeof(int64_t));
  if (bytes > 0) {
    memcpy(*data + *size + sizeof(int64_t), array, bytes);
  }
  *size += sizeof(int64_t) + bytes;
}

/*******************************************************************************
 * \brief Writes an array preceded by its number of elements to a file.
 * \author Hans Pabst
 ******************************************************************************/
static void dump_write(FILE *fp, const void *array, const int64_t count,
                       const size_t elem_size) {
  bool ok = fwrite(&count, sizeof(int64_t), 1, fp) == 1;
  if (count > 0) {
    ok = ok && fwrite(array, elem_size, (size_t)count, fp) == (size_t)count;
  }
  if (!ok) {
    fprintf(stderr, "Error: Could not write grid task list.\n");
    abort();
  }
}

/*******************************************************************************
 * \brief Opens the next task list file of the given kind (0: collocate, 1:
 *        integrate) and writes the task list, or returns NULL if not wanted.
 * \author Hans Pabst
 ******************************************************************************/
static FILE *dump_begin(const grid_task_list_internal *task_list,
                        const int kind) {
  if (task_list->dump == NULL ||
      dump_count[kind] >= grid_library_get_config().dump) {
    return NULL;
  }
  const int rank = cp_mpi_comm_rank(cp_mpi_get_comm_world());
  char filename[64];
  snprintf(filename, sizeof(filename), "grid_%s_%i_%i.tasklist",
           0 == kind ? "collocate" : "integrate", rank, dump_count[kind]++);
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    fprintf(stderr, "Error: Could not open %s.\n", filename);
    abort();
  }
  // Values of known bits catch a reader with another byte order.
  const int probe_int = 1;
  const double probe_double = 1.0;
  const bool ok = fputs(GRID_TASK_LIST_HEADER, fp) >= 0;
  if (!ok) {
    fprintf(stderr, "Error: Could not write %s.\n", filename);
    abort();
  }
  dump_write(fp, &probe_int, 1, sizeof(int));
  dump_write(fp, &probe_double, 1, sizeof(double));
  dump_write(fp, &kind, 1, sizeof(int));
  if (fwrite(task_list->dump, 1, task_list->dump_size, fp) !=
      task_list->dump_size) {
    fprintf(stderr, "Error: Could not write %s.\n", filename);
    abort();
  }
  return fp;
}

/*******************************************************************************
 * \brief Closes a task list file.
 * \author Hans Pabst
 ******************************************************************************/
static void dump_end(FILE *fp) {
  if (fclose(fp) != 0) {
    fprintf(stderr, "Error: Could not close grid task list.\n");
    abort();
  }
}

/*******************************************************************************
 * \brief Allocates a task list which can be passed to grid_collocate_task_list.
 *        See grid_task_list.h for details.
 * \author Ole Schuett
 ******************************************************************************/
void grid_create_task_list(
    const bool orthorhombic, const int ntasks, const int nlevels,
    const int natoms, const int nkinds, const int nblocks,
    const int block_offsets[nblocks], const double atom_positions[natoms][3],
    const int atom_kinds[natoms], const grid_basis_set *basis_sets[nkinds],
    const int level_list[ntasks], const int iatom_list[ntasks],
    const int jatom_list[ntasks], const int iset_list[ntasks],
    const int jset_list[ntasks], const int ipgf_list[ntasks],
    const int jpgf_list[ntasks], const int border_mask_list[ntasks],
    const int block_num_list[ntasks], const double radius_list[ntasks],
    const double rab_list[ntasks][3], const int npts_global[nlevels][3],
    const int npts_local[nlevels][3], const int shift_local[nlevels][3],
    const int border_width[nlevels][3], const double dh[nlevels][3][3],
    const double dh_inv[nlevels][3][3], grid_task_list **task_list_out) {

  const grid_library_config config = grid_library_get_config();

  grid_task_list_internal *task_list = NULL;

  if (*task_list_out == NULL) {
    task_list = malloc(sizeof(grid_task_list_internal));
    // not a proper handling of errors. assert is a debug tool
    assert(task_list != NULL);
    memset(task_list, 0, sizeof(grid_task_list_internal));
    *task_list_out = task_list;

    // Resolve AUTO to a concrete backend.
    if (config.backend == GRID_BACKEND_AUTO) {
#if (defined(__OFFLOAD_CUDA) || defined(__OFFLOAD_HIP)) &&                     \
    !defined(__NO_OFFLOAD_GRID)
      task_list->backend = GRID_BACKEND_GPU;
#else
      task_list->backend = GRID_BACKEND_CPU;
#endif
    } else {
      task_list->backend = config.backend;
    }
  } else {
    // Reuse existing task list.
    task_list = (grid_task_list_internal *)*task_list_out;
  }

  free(task_list->dump);
  task_list->dump = NULL;
  task_list->dump_size = 0;

  if ((nblocks == 0) || (ntasks == 0) || (nlevels == 0)) {
    task_list->empty = true;
    return;
  } else {
    task_list->empty = false;
  }

  // Keep the arguments as long as calls are to be written to file.
  if (dump_count[0] < config.dump || dump_count[1] < config.dump) {
    char **data = &task_list->dump;
    size_t *size = &task_list->dump_size;
    const int scalars[6] = {orthorhombic, ntasks, nlevels,
                            natoms,       nkinds, nblocks};
    dump_append(data, size, scalars, 6, sizeof(int));
    dump_append(data, size, block_offsets, nblocks, sizeof(int));
    dump_append(data, size, atom_positions, 3 * natoms, sizeof(double));
    dump_append(data, size, atom_kinds, natoms, sizeof(int));
    for (int ikind = 0; ikind < nkinds; ikind++) {
      const grid_basis_set *basis = basis_sets[ikind];
      const int sizes[4] = {basis->nset, basis->nsgf, basis->maxco,
                            basis->maxpgf};
      dump_append(data, size, sizes, 4, sizeof(int));
      dump_append(data, size, basis->lmin, basis->nset, sizeof(int));
      dump_append(data, size, basis->lmax, basis->nset, sizeof(int));
      dump_append(data, size, basis->npgf, basis->nset, sizeof(int));
      dump_append(data, size, basis->nsgf_set, basis->nset, sizeof(int));
      dump_append(data, size, basis->first_sgf, basis->nset, sizeof(int));
      dump_append(data, size, basis->sphi, (int64_t)basis->nsgf * basis->maxco,
                  sizeof(double));
      dump_append(data, size, basis->zet, (int64_t)basis->nset * basis->maxpgf,
                  sizeof(double));
    }
    const int *task_ints[9] = {level_list, iatom_list,       jatom_list,
                               iset_list,  jset_list,        ipgf_list,
                               jpgf_list,  border_mask_list, block_num_list};
    for (int i = 0; i < 9; i++) {
      dump_append(data, size, task_ints[i], ntasks, sizeof(int));
    }
    dump_append(data, size, radius_list, ntasks, sizeof(double));
    dump_append(data, size, rab_list, 3 * (int64_t)ntasks, sizeof(double));
    dump_append(data, size, npts_global, 3 * nlevels, sizeof(int));
    dump_append(data, size, npts_local, 3 * nlevels, sizeof(int));
    dump_append(data, size, shift_local, 3 * nlevels, sizeof(int));
    dump_append(data, size, border_width, 3 * nlevels, sizeof(int));
    dump_append(data, size, dh, 9 * nlevels, sizeof(double));
    dump_append(data, size, dh_inv, 9 * nlevels, sizeof(double));
  }

  size_t size = (size_t)nlevels * 3 * sizeof(int);

  if (task_list->nlevels < nlevels) {
    free(task_list->npts_local);
    task_list->npts_local = malloc(size);
  }

  // Store npts_local for bounds checking and validation.
  task_list->nlevels = nlevels;
  assert(task_list->npts_local != NULL);
  memcpy(task_list->npts_local, npts_local, size);

  // Always create reference backend because it might be needed for validation.
  grid_ref_create_task_list(
      orthorhombic, ntasks, nlevels, natoms, nkinds, nblocks, block_offsets,
      atom_positions, atom_kinds, basis_sets, level_list, iatom_list,
      jatom_list, iset_list, jset_list, ipgf_list, jpgf_list, border_mask_list,
      block_num_list, radius_list, rab_list, npts_global, npts_local,
      shift_local, border_width, dh, dh_inv, &task_list->ref);

  // Create other backend, if selected.
  switch (task_list->backend) {
  case GRID_BACKEND_REF:
    break; // was already created above
  case GRID_BACKEND_CPU:
    grid_cpu_create_task_list(
        orthorhombic, ntasks, nlevels, natoms, nkinds, nblocks, block_offsets,
        atom_positions, atom_kinds, basis_sets, level_list, iatom_list,
        jatom_list, iset_list, jset_list, ipgf_list, jpgf_list,
        border_mask_list, block_num_list, radius_list, rab_list, npts_global,
        npts_local, shift_local, border_width, dh, dh_inv, &task_list->cpu);
    break;
  case GRID_BACKEND_DGEMM:
    grid_dgemm_create_task_list(
        orthorhombic, ntasks, nlevels, natoms, nkinds, nblocks, block_offsets,
        atom_positions, atom_kinds, basis_sets, level_list, iatom_list,
        jatom_list, iset_list, jset_list, ipgf_list, jpgf_list,
        border_mask_list, block_num_list, radius_list, rab_list, npts_global,
        npts_local, shift_local, border_width, dh, dh_inv, &task_list->dgemm);
    break;

  case GRID_BACKEND_GPU:
#if (defined(__OFFLOAD_CUDA) || defined(__OFFLOAD_HIP)) &&                     \
    !defined(__NO_OFFLOAD_GRID)
    grid_gpu_create_task_list(
        orthorhombic, ntasks, nlevels, natoms, nkinds, nblocks, block_offsets,
        &atom_positions[0][0], atom_kinds, basis_sets, level_list, iatom_list,
        jatom_list, iset_list, jset_list, ipgf_list, jpgf_list,
        border_mask_list, block_num_list, radius_list, &rab_list[0][0],
        &npts_global[0][0], &npts_local[0][0], &shift_local[0][0],
        &border_width[0][0], &dh[0][0][0], &dh_inv[0][0][0], &task_list->gpu);
#else
    fprintf(stderr, "Error: The GPU grid backend is not available. "
                    "Please re-compile with -D__OFFLOAD");
    abort();
#endif
    break;

  default:
    printf("Error: Unknown grid backend: %i.\n", config.backend);
    abort();
    break;
  }
}

/*******************************************************************************
 * \brief Deallocates given task list, basis_sets have to be freed separately.
 * \author Ole Schuett
 ******************************************************************************/
void grid_free_task_list(grid_task_list *ptr) {
  if (ptr == NULL)
    return;

  grid_task_list_internal *task_list = (grid_task_list_internal *)ptr;

  if (task_list->ref != NULL) {
    grid_ref_free_task_list(task_list->ref);
    task_list->ref = NULL;
  }
  if (task_list->cpu != NULL) {
    grid_cpu_free_task_list(task_list->cpu);
    task_list->cpu = NULL;
  }
  if (task_list->dgemm != NULL) {
    grid_dgemm_free_task_list(task_list->dgemm);
    task_list->dgemm = NULL;
  }
#if (defined(__OFFLOAD_CUDA) || defined(__OFFLOAD_HIP)) &&                     \
    !defined(__NO_OFFLOAD_GRID)
  if (task_list->gpu != NULL) {
    grid_gpu_free_task_list(task_list->gpu);
    task_list->gpu = NULL;
  }
#endif

  free(task_list->npts_local);
  free(task_list->dump);
  free(task_list);
}

/*******************************************************************************
 * \brief Collocate all tasks of in given list onto given grids.
 *        See grid_task_list.h for details.
 * \author Ole Schuett
 ******************************************************************************/
void grid_collocate_task_list(const grid_task_list *ptr,
                              const enum grid_func func, const int nlevels,
                              const int npts_local[nlevels][3],
                              const offload_buffer *pab_blocks,
                              offload_buffer *grids[nlevels]) {
  if (ptr == NULL)
    return;

  grid_task_list_internal *task_list = (grid_task_list_internal *)ptr;

  if (task_list->empty) {
    for (int level = 0; level < nlevels; level++)
      memset(grids[level]->host_buffer, 0, grids[level]->size);
    return;
  }

  // Bounds check.
  assert(task_list->nlevels == nlevels);
  for (int ilevel = 0; ilevel < nlevels; ilevel++) {
    assert(task_list->npts_local[ilevel][0] == npts_local[ilevel][0]);
    assert(task_list->npts_local[ilevel][1] == npts_local[ilevel][1]);
    assert(task_list->npts_local[ilevel][2] == npts_local[ilevel][2]);
  }

  switch (task_list->backend) {
  case GRID_BACKEND_REF:
    grid_ref_collocate_task_list(task_list->ref, func, nlevels, pab_blocks,
                                 grids);
    break;
  case GRID_BACKEND_CPU:
    grid_cpu_collocate_task_list(task_list->cpu, func, nlevels, pab_blocks,
                                 grids);
    break;
  case GRID_BACKEND_DGEMM:
    grid_dgemm_collocate_task_list(task_list->dgemm, func, nlevels, pab_blocks,
                                   grids);
    break;
#if (defined(__OFFLOAD_CUDA) || defined(__OFFLOAD_HIP)) &&                     \
    !defined(__NO_OFFLOAD_GRID)
  case GRID_BACKEND_GPU:
    grid_gpu_collocate_task_list(task_list->gpu, func, nlevels, pab_blocks,
                                 grids);
    break;
#endif
  default:
    printf("Error: Unknown grid backend: %i.\n", task_list->backend);
    abort();
    break;
  }

  // Write task list, arguments, and results to file if requested.
  FILE *const fp = dump_begin(task_list, 0);
  if (fp != NULL) {
    const int ifunc = func;
    dump_write(fp, &ifunc, 1, sizeof(int));
    dump_write(fp, pab_blocks->host_buffer,
               (int64_t)(pab_blocks->size / sizeof(double)), sizeof(double));
    for (int level = 0; level < nlevels; level++) {
      dump_write(fp, grids[level]->host_buffer,
                 (int64_t)npts_local[level][0] * npts_local[level][1] *
                     npts_local[level][2],
                 sizeof(double));
    }
    dump_end(fp);
  }

  // Perform validation if enabled.
  if (grid_library_get_config().validate) {
    // Allocate space for reference results.
    offload_buffer *grids_ref[nlevels];
    for (int level = 0; level < nlevels; level++) {
      const int npts_local_total =
          npts_local[level][0] * npts_local[level][1] * npts_local[level][2];
      grids_ref[level] = NULL;
      offload_create_buffer(npts_local_total, &grids_ref[level]);
    }

    // Call reference implementation.
    grid_ref_collocate_task_list(task_list->ref, func, nlevels, pab_blocks,
                                 grids_ref);

    // Compare results.
    const double tolerance = 1e-12;
    double max_rel_diff = 0.0;
    for (int level = 0; level < nlevels; level++) {
      for (int i = 0; i < npts_local[level][0]; i++) {
        for (int j = 0; j < npts_local[level][1]; j++) {
          for (int k = 0; k < npts_local[level][2]; k++) {
            const int idx = k * npts_local[level][1] * npts_local[level][0] +
                            j * npts_local[level][0] + i;
            const double ref_value = grids_ref[level]->host_buffer[idx];
            const double test_value = grids[level]->host_buffer[idx];
            const double diff = fabs(test_value - ref_value);
            const double rel_diff = diff / fmax(1.0, fabs(ref_value));
            max_rel_diff = fmax(max_rel_diff, rel_diff);
            if (rel_diff > tolerance) {
              fprintf(stderr, "Error: Validation failure in grid collocate\n");
              fprintf(stderr, "   diff:     %le\n", diff);
              fprintf(stderr, "   rel_diff: %le\n", rel_diff);
              fprintf(stderr, "   value:    %le\n", ref_value);
              fprintf(stderr, "   level:    %i\n", level);
              fprintf(stderr, "   ijk:      %i  %i  %i\n", i, j, k);
              abort();
            }
          }
        }
      }
      offload_free_buffer(grids_ref[level]);
      printf("Validated grid collocate, max rel. diff: %le\n", max_rel_diff);
    }
  }
}

/*******************************************************************************
 * \brief Integrate all tasks of in given list from given grids.
 *        See grid_task_list.h for details.
 * \author Ole Schuett
 ******************************************************************************/
void grid_integrate_task_list(const grid_task_list *ptr, const bool compute_tau,
                              const int natoms, const int nlevels,
                              const int npts_local[nlevels][3],
                              const offload_buffer *pab_blocks,
                              const offload_buffer *grids[nlevels],
                              offload_buffer *hab_blocks,
                              double forces[natoms][3], double virial[3][3]) {

  if (ptr == NULL)
    return;

  grid_task_list_internal *task_list = (grid_task_list_internal *)ptr;

  if (task_list->empty) {
    memset(hab_blocks->host_buffer, 0, hab_blocks->size);
    if (virial) {
      virial[0][0] = 0.0;
      virial[0][1] = 0.0;
      virial[0][2] = 0.0;
      virial[1][0] = 0.0;
      virial[1][1] = 0.0;
      virial[1][2] = 0.0;
      virial[2][0] = 0.0;
      virial[2][1] = 0.0;
      virial[2][2] = 0.0;
    }
    if (forces) {
      for (int atom = 0; atom < natoms; atom++) {
        forces[atom][0] = 0.0;
        forces[atom][1] = 0.0;
        forces[atom][2] = 0.0;
      }
    }
    return;
  }

  // Bounds check.
  assert(task_list->nlevels == nlevels);
  for (int ilevel = 0; ilevel < nlevels; ilevel++) {
    assert(task_list->npts_local[ilevel][0] == npts_local[ilevel][0]);
    assert(task_list->npts_local[ilevel][1] == npts_local[ilevel][1]);
    assert(task_list->npts_local[ilevel][2] == npts_local[ilevel][2]);
  }

  assert(forces == NULL || pab_blocks != NULL);
  assert(virial == NULL || pab_blocks != NULL);

  switch (task_list->backend) {
#if (defined(__OFFLOAD_CUDA) || defined(__OFFLOAD_HIP)) &&                     \
    !defined(__NO_OFFLOAD_GRID)
  case GRID_BACKEND_GPU:
    grid_gpu_integrate_task_list(task_list->gpu, compute_tau, nlevels,
                                 pab_blocks, grids, hab_blocks, &forces[0][0],
                                 &virial[0][0]);
    break;
#endif
  case GRID_BACKEND_DGEMM:
    grid_dgemm_integrate_task_list(task_list->dgemm, compute_tau, natoms,
                                   nlevels, pab_blocks, grids, hab_blocks,
                                   forces, virial);
    break;
  case GRID_BACKEND_CPU:
    grid_cpu_integrate_task_list(task_list->cpu, compute_tau, natoms, nlevels,
                                 pab_blocks, grids, hab_blocks, forces, virial);
    break;
  case GRID_BACKEND_REF:
    grid_ref_integrate_task_list(task_list->ref, compute_tau, natoms, nlevels,
                                 pab_blocks, grids, hab_blocks, forces, virial);
    break;
  default:
    printf("Error: Unknown grid backend: %i.\n", task_list->backend);
    abort();
    break;
  }

  // Write task list, arguments, and results to file if requested.
  FILE *const fp = dump_begin(task_list, 1);
  if (fp != NULL) {
    const int flags[4] = {compute_tau, pab_blocks != NULL, forces != NULL,
                          virial != NULL};
    dump_write(fp, flags, 4, sizeof(int));
    dump_write(fp, pab_blocks != NULL ? pab_blocks->host_buffer : NULL,
               pab_blocks != NULL ? (int64_t)(pab_blocks->size / sizeof(double))
                                  : 0,
               sizeof(double));
    for (int level = 0; level < nlevels; level++) {
      dump_write(fp, grids[level]->host_buffer,
                 (int64_t)npts_local[level][0] * npts_local[level][1] *
                     npts_local[level][2],
                 sizeof(double));
    }
    dump_write(fp, hab_blocks->host_buffer,
               (int64_t)(hab_blocks->size / sizeof(double)), sizeof(double));
    dump_write(fp, forces, forces != NULL ? 3 * natoms : 0, sizeof(double));
    dump_write(fp, virial, virial != NULL ? 9 : 0, sizeof(double));
    dump_end(fp);
  }

  // Perform validation if enabled.
  if (grid_library_get_config().validate) {
    // Allocate space for reference results.
    const int hab_length = hab_blocks->size / sizeof(double);
    offload_buffer *hab_blocks_ref = NULL;
    offload_create_buffer(hab_length, &hab_blocks_ref);
    double forces_ref[natoms][3], virial_ref[3][3];

    // Call reference implementation.
    grid_ref_integrate_task_list(task_list->ref, compute_tau, natoms, nlevels,
                                 pab_blocks, grids, hab_blocks_ref,
                                 (forces != NULL) ? forces_ref : NULL,
                                 (virial != NULL) ? virial_ref : NULL);

    // Compare hab.
    const double hab_tolerance = 1e-12;
    double hab_max_rel_diff = 0.0;
    for (int i = 0; i < hab_length; i++) {
      const double ref_value = hab_blocks_ref->host_buffer[i];
      const double test_value = hab_blocks->host_buffer[i];
      const double diff = fabs(test_value - ref_value);
      const double rel_diff = diff / fmax(1.0, fabs(ref_value));
      hab_max_rel_diff = fmax(hab_max_rel_diff, rel_diff);
      if (rel_diff > hab_tolerance) {
        fprintf(stderr, "Error: Validation failure in grid integrate\n");
        fprintf(stderr, "   hab diff:     %le\n", diff);
        fprintf(stderr, "   hab rel_diff: %le\n", rel_diff);
        fprintf(stderr, "   hab value:    %le\n", ref_value);
        fprintf(stderr, "   hab i:        %i\n", i);
        abort();
      }
    }

    // Compare forces.
    const double forces_tolerance = 1e-8; // account for higher numeric noise
    double forces_max_rel_diff = 0.0;
    if (forces != NULL) {
      for (int iatom = 0; iatom < natoms; iatom++) {
        for (int idir = 0; idir < 3; idir++) {
          const double ref_value = forces_ref[iatom][idir];
          const double test_value = forces[iatom][idir];
          const double diff = fabs(test_value - ref_value);
          const double rel_diff = diff / fmax(1.0, fabs(ref_value));
          forces_max_rel_diff = fmax(forces_max_rel_diff, rel_diff);
          if (rel_diff > forces_tolerance) {
            fprintf(stderr, "Error: Validation failure in grid integrate\n");
            fprintf(stderr, "   forces diff:     %le\n", diff);
            fprintf(stderr, "   forces rel_diff: %le\n", rel_diff);
            fprintf(stderr, "   forces value:    %le\n", ref_value);
            fprintf(stderr, "   forces atom:     %i\n", iatom);
            fprintf(stderr, "   forces dir:      %i\n", idir);
            abort();
          }
        }
      }
    }

    // Compare virial.
    const double virial_tolerance = 1e-8; // account for higher numeric noise
    double virial_max_rel_diff = 0.0;
    if (virial != NULL) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          const double ref_value = virial_ref[i][j];
          const double test_value = virial[i][j];
          const double diff = fabs(test_value - ref_value);
          const double rel_diff = diff / fmax(1.0, fabs(ref_value));
          virial_max_rel_diff = fmax(virial_max_rel_diff, rel_diff);
          if (rel_diff > virial_tolerance) {
            fprintf(stderr, "Error: Validation failure in grid integrate\n");
            fprintf(stderr, "   virial diff:     %le\n", diff);
            fprintf(stderr, "   virial rel_diff: %le\n", rel_diff);
            fprintf(stderr, "   virial value:    %le\n", ref_value);
            fprintf(stderr, "   virial ij:       %i  %i\n", i, j);
            abort();
          }
        }
      }
    }

    printf("Validated grid_integrate, max rel. diff: %le %le %le\n",
           hab_max_rel_diff, forces_max_rel_diff, virial_max_rel_diff);
    offload_free_buffer(hab_blocks_ref);
  }
}

// EOF
