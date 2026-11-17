/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "../mpiwrap/cp_mpi.h"
#include "../offload/offload_library.h"
#include "../offload/offload_mempool.h"
#include "common/grid_library.h"
#include "grid_replay.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Wrapper for printf, passed to grid_library_print_stats.
 * \author Ole Schuett
 ******************************************************************************/
static void print_func(const char *msg, int msglen, int output_unit) {
  (void)msglen;           // mark used
  if (output_unit == 0) { // i.e. my_rank == 0
    printf("%s", msg);
  }
}

/*******************************************************************************
 * \brief Stand-alone miniapp for running .task files and task lists.
 * \author Ole Schuett
 ******************************************************************************/
int main(int argc, char *argv[]) {
  // Parsing of optional args.
  int iarg = 1;
  int nrequired_args = 2;
  bool collocate = true;
  bool batch = false;
  int cycles_per_block = 1;
  grid_library_config config = grid_library_get_config();
  bool valid = true;
  while (valid && iarg < argc && strncmp(argv[iarg], "--", 2) == 0) {
    const char *const option = argv[iarg++];
    if (strcmp(option, "--integrate") == 0) {
      collocate = false;
    } else if (strcmp(option, "--batch") == 0) {
      batch = true;
      valid =
          (iarg < argc && sscanf(argv[iarg++], "%i", &cycles_per_block) == 1);
    } else if (strcmp(option, "--backend") == 0 && iarg < argc) {
      const char *const name = argv[iarg++];
      if (strcmp(name, "auto") == 0) {
        config.backend = GRID_BACKEND_AUTO;
      } else if (strcmp(name, "ref") == 0) {
        config.backend = GRID_BACKEND_REF;
      } else if (strcmp(name, "cpu") == 0) {
        config.backend = GRID_BACKEND_CPU;
      } else if (strcmp(name, "dgemm") == 0) {
        config.backend = GRID_BACKEND_DGEMM;
      } else if (strcmp(name, "gpu") == 0) {
        config.backend = GRID_BACKEND_GPU;
      } else {
        valid = false;
      }
    } else if (strcmp(option, "--cutoff") == 0) {
      config.apply_cutoff = true; // as GLOBAL%GRID%APPLY_CUTOFF (default)
    } else if (strcmp(option, "--dump") == 0) {
      valid = (iarg < argc && sscanf(argv[iarg++], "%i", &config.dump) == 1);
    } else {
      valid = false;
    }
  }

  // All optional args have been parsed.
  if (!valid || argc - iarg != nrequired_args) {
    fprintf(stderr,
            "Usage: grid_miniapp.x [--backend <auto|ref|cpu|dgemm|gpu>] "
            "[--cutoff] [--dump <calls>] [--integrate] "
            "[--batch <cycles-per-block>] "
            "<cycles> <task-file>\n"
            "       A task list (GLOBAL%%GRID%%DUMP_TASK_LISTS) is replayed as "
            "recorded, i.e., without --integrate and --batch.\n");
    return 1;
  }

  int cycles;
  if (sscanf(argv[iarg++], "%i", &cycles) != 1) {
    fprintf(stderr, "Error: Could not parse cycles.\n");
    return 1;
  }
  const char *const filename = argv[iarg++];
  const bool task_list = grid_replay_is_task_list(filename);
  if (task_list && (batch || !collocate)) {
    fprintf(stderr, "Error: A task list is replayed as recorded.\n");
    return 1;
  }

  cp_mpi_init(&argc, &argv);
  offload_set_chosen_device(0);
  grid_library_init();
  grid_library_set_config(config.backend, config.validate, config.apply_cutoff,
                          config.dump);

  // Results of a task list do not accumulate over cycles.
  const double tolerance = 1e-12 * (task_list ? 1 : cycles);
  const bool success = task_list
                           ? grid_replay_task_list(filename, cycles, tolerance)
                           : grid_replay(filename, cycles, collocate, batch,
                                         cycles_per_block, tolerance);
  if (success) {
    grid_library_print_stats(0 /*fortran_comm*/, &print_func, 0 /*rank*/);
    offload_mempool_stats_print(0 /*fortran_comm*/, &print_func, 0 /*rank*/);
  } else {
    fprintf(stderr, "Error: Maximal difference is too large.\n");
  }
  grid_library_finalize();
  cp_mpi_finalize();
  return success ? 0 : 2;
}

// EOF
