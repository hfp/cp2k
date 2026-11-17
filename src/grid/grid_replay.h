/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#ifndef GRID_REPLAY_H
#define GRID_REPLAY_H

#include <stdbool.h>

/*******************************************************************************
 * \brief Reads a .task file, collocates/integrates it, and compares results.
 *
 * \param filename          Name of the task file.
 * \param cycles            Number of times the task should be collocated.
 * \param collocate         When true collocate is called otherwise integrate.
 * \param batch             When false grid_ref_collocate_pgf_product is called.
 *                          When true grid_collocate_task_list is called.
 * \param cycles_per_block  Number of cycles per matrix block decontraction.
 * \param tolerance         Tolerance for comparing floating point results.
 * \returns                 Returns true iff the test passed.
 *
 * \author Ole Schuett
 ******************************************************************************/
bool grid_replay(const char *filename, const int cycles, const bool collocate,
                 const bool batch, const int cycles_per_block,
                 const double tolerance);

/*******************************************************************************
 * \brief Tells whether the given file is a task list, i.e., written by
 *        grid_collocate_task_list or grid_integrate_task_list when the grid
 *        library is configured to dump calls (GLOBAL%GRID%DUMP_TASK_LISTS).
 * \author Hans Pabst
 ******************************************************************************/
bool grid_replay_is_task_list(const char *filename);

/*******************************************************************************
 * \brief Reads a task list, replays the recorded call with the configured
 *        backend, and compares the results with the recorded ones.
 *
 * \param filename          Name of the task list file.
 * \param cycles            Number of times the call is repeated (timed).
 * \param tolerance         Tolerance for comparing floating point results.
 * \returns                 Returns true iff the test passed.
 *
 * \author Hans Pabst
 ******************************************************************************/
bool grid_replay_task_list(const char *filename, const int cycles,
                           const double tolerance);

#endif

// EOF
