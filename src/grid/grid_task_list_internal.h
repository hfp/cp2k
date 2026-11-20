/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#ifndef GRID_TASK_LIST_INTERNAL_H
#define GRID_TASK_LIST_INTERNAL_H

#include "common/grid_basis_set.h"
#include "common/grid_common.h"
#include "common/grid_constants.h"
#include "common/grid_library.h"
#include "cpu/grid_cpu_task_list.h"
#include "dgemm/grid_dgemm_task_list.h"
#include "gpu/grid_gpu_task_list.h"
#include "ref/grid_ref_task_list.h"

#include "../offload/offload_runtime.h"

/*******************************************************************************
 * \brief Internal representation of a task list, abstracting various backends.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  bool empty;
  int backend;
  int nlevels;
  int (*npts_local)[3];
  grid_ref_task_list *ref;
  grid_cpu_task_list *cpu;
  grid_dgemm_task_list *dgemm;
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_GRID)
  grid_gpu_task_list *gpu;
#endif
  // more backends to be added here
  char *dump;       // serialized arguments of grid_create_task_list
  size_t dump_size; // when the config asks for task lists written to file
} grid_task_list_internal;

// First line of a task list file, followed by binary records (count, data).
#define GRID_TASK_LIST_HEADER "#Grid task list v1\n"

#endif
