/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef DBM_LIBRARY_H
#define DBM_LIBRARY_H

#include "dbm_multiply.h"

#include <stdint.h>

/*******************************************************************************
 * \brief Initializes the DBM library.
 * \author Ole Schuett
 ******************************************************************************/
void dbm_library_init(void);

/*******************************************************************************
 * \brief Finalizes the DBM library.
 * \author Ole Schuett
 ******************************************************************************/
void dbm_library_finalize(void);

#define DBM_NUM_COUNTERS 64

/*******************************************************************************
 * \brief Internal routine for the stats counter of a block multiplication.
 * \author Ole Schuett
 ******************************************************************************/
static inline int dbm_library_counter_index(const int m, const int n,
                                            const int k) {
  const int lm = (100 <= m ? (1000 <= m ? 3 : 2) : (10 <= m ? 1 : 0));
  const int ln = (100 <= n ? (1000 <= n ? 3 : 2) : (10 <= n ? 1 : 0));
  const int lk = (100 <= k ? (1000 <= k ? 3 : 2) : (10 <= k ? 1 : 0));
  return 16 * lm + 4 * ln + lk;
}

/*******************************************************************************
 * \brief Add the calling thread's counts (dbm_library_counter_index) to the
 *        stats. Counting locally first avoids a call per block multiplication.
 *        This routine is thread-safe.
 * \author Hans Pabst
 ******************************************************************************/
void dbm_library_counters_add(const int64_t counters[DBM_NUM_COUNTERS]);

/*******************************************************************************
 * \brief Phases of dbm_multiply whose duration the stats report per rank.
 * \author Hans Pabst
 ******************************************************************************/
enum dbm_library_phase {
  DBM_PHASE_SETUP = 0,        // scaling, filter thresholds, backend, filter
  DBM_PHASE_REDISTRIBUTE = 1, // packing A and B, all-to-all exchange
  DBM_PHASE_SHIFT = 2,        // exchanging packs per tick (sendrecv)
  DBM_PHASE_UPLOAD = 3,       // handing packs to the backend
  DBM_PHASE_MULTIPLY = 4,     // generating and processing batches
  DBM_PHASE_FINISH = 5,       // backend stop, i.e., waiting for the results
  DBM_NUM_PHASES = 6
};

/*******************************************************************************
 * \brief Add the durations (seconds) of a multiplication's phases to the
 *        stats. Called once per dbm_multiply outside of parallel regions.
 * \author Hans Pabst
 ******************************************************************************/
void dbm_library_phases_add(const double seconds[DBM_NUM_PHASES]);

/*******************************************************************************
 * \brief Prints statistics gathered by the DBM library.
 * \author Ole Schuett
 ******************************************************************************/
void dbm_library_print_stats(const int fortran_comm,
                             void (*print_func)(const char *, int, int),
                             const int output_unit);

#endif

// EOF
