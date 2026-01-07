/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "offload_mempool.h"
#include "../mpiwrap/cp_mpi.h"
#include "offload_library.h"
#include "offload_runtime.h"

#include <assert.h>
#include <inttypes.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__parallel)
#include <mpi.h>
#endif

#define OFFLOAD_MEMPOOL_PRINT(FN, MSG, OUTPUT_UNIT)                            \
  ((FN)(MSG, (int)strlen(MSG), OUTPUT_UNIT))
#define OFFLOAD_MEMPOOL_OMPALLOC 1
#define OFFLOAD_MEMPOOL_COUNTER int
#define OFFLOAD_MEMPOOL_UPSIZE (2 << 20) // permit slack size when reuse

/*******************************************************************************
 * \brief Private struct for storing a chunk of memory.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct offload_memchunk {
  void *mem; // first: allows to cast memchunk into mem-ptr...
  struct offload_memchunk *next;
  size_t size, used;
} offload_memchunk_t;

/*******************************************************************************
 * \brief Private struct for storing a memory pool.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct offload_mempool {
  offload_memchunk_t *available_head, *allocated_head; // single-linked lists
} offload_mempool_t;

/*******************************************************************************
 * \brief Private pools for host and device memory.
 * \author Ole Schuett
 ******************************************************************************/
static offload_mempool_t mempool_host = {0}, mempool_device = {0};

/*******************************************************************************
 * \brief Private some counters for statistics.
 * \author Hans Pabst
 ******************************************************************************/
static struct {
  uint64_t mallocs, mempeak;
} host_stats = {0, 0}, device_stats = {0, 0};

/*******************************************************************************
 * \brief Private routine for actually allocating system memory.
 * \author Ole Schuett
 ******************************************************************************/
static void *actual_malloc(const size_t size, const bool on_device) {
  if (size == 0) {
    return NULL;
  }

  void *memory = NULL;
#if defined(__OFFLOAD)
  if (on_device) {
    offload_activate_chosen_device();
    offloadMalloc(&memory, size);
  } else {
    offload_activate_chosen_device();
    offloadMallocHost(&memory, size);
  }
#elif OFFLOAD_MEMPOOL_OMPALLOC && (201811 /*v5.0*/ <= _OPENMP)
  memory = omp_alloc(size, omp_null_allocator);
#elif defined(__parallel) && !OFFLOAD_MEMPOOL_OMPALLOC
  if (MPI_SUCCESS != MPI_Alloc_mem((MPI_Aint)size, MPI_INFO_NULL, &memory)) {
    fprintf(stderr, "ERROR: MPI_Alloc_mem failed at %s:%i\n", name, __FILE__,
            __LINE__);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
#else
  memory = malloc(size);
#endif

  // Update statistics.
  if (on_device) {
#pragma omp atomic
    ++device_stats.mallocs;
  } else {
#pragma omp atomic
    ++host_stats.mallocs;
  }

  assert(memory != NULL);
  return memory;
}

/*******************************************************************************
 * \brief Private routine for actually freeing system memory.
 * \author Ole Schuett
 ******************************************************************************/
static void actual_free(void *memory, const bool on_device) {
  if (NULL == memory) {
    return;
  }

#if defined(__OFFLOAD)
  if (on_device) {
    offload_activate_chosen_device();
    offloadFree(memory);
  } else {
    offload_activate_chosen_device();
    offloadFreeHost(memory);
  }
#elif OFFLOAD_MEMPOOL_OMPALLOC && (201811 /*v5.0*/ <= _OPENMP)
  (void)on_device; // mark used
  omp_free(memory, omp_null_allocator);
#elif defined(__parallel) && !OFFLOAD_MEMPOOL_OMPALLOC
  (void)on_device; // mark used
  if (MPI_SUCCESS != MPI_Free_mem(memory)) {
    fprintf(stderr, "ERROR: MPI_Free_mem failed at %s:%i\n", name, __FILE__,
            __LINE__);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
#else
  (void)on_device; // mark used
  free(memory);
#endif
}

/*******************************************************************************
 * \brief Private routine for allocating host or device memory from the pool.
 * \author Ole Schuett and Hans Pabst
 ******************************************************************************/
static void *internal_mempool_malloc(offload_mempool_t *pool,
                                     const size_t size) {
  if (size == 0) {
    return NULL;
  }
  offload_memchunk_t *chunk = NULL;
  const bool on_device = (pool == &mempool_device);
  assert(on_device || pool == &mempool_host);
#pragma omp critical(offload_mempool_modify)
  {
    // Find a possible chunk to reuse or reclaim in available list.
    offload_memchunk_t **reuse = NULL, **reclaim = NULL, **reclaim0 = NULL;
    offload_memchunk_t **indirect = &pool->available_head;
    while (*indirect != NULL) {
      const size_t s = (*indirect)->size;
#if defined(OFFLOAD_MEMPOOL_COUNTER)
      OFFLOAD_MEMPOOL_COUNTER *counter = NULL;
      if (!on_device && sizeof(OFFLOAD_MEMPOOL_COUNTER) <= s) {
        counter = (OFFLOAD_MEMPOOL_COUNTER *)(*indirect)->mem;
        assert(NULL != counter);
        if (0 == (*indirect)->used) {
          ++*counter;
        } else {
          (*indirect)->used = 0;
          *counter = 0;
        }
      }
#endif
      if (size <= s) {
        if (s <= (size + OFFLOAD_MEMPOOL_UPSIZE)) {
          reuse = indirect;
          break; // almost perfect, exit early
        } else if (reuse != NULL) {
          if (s < (*reuse)->size) {
            reuse = indirect;
          }
        } else {
          reuse = indirect;
        }
      } else if (reclaim != NULL) {
        if ((*reclaim)->size < s) {
#if defined(OFFLOAD_MEMPOOL_COUNTER)
          if (counter == NULL ||
              *(OFFLOAD_MEMPOOL_COUNTER *)(*reclaim)->mem < *counter)
#endif
          {
            reclaim = indirect;
          }
        }
      } else {
#if defined(OFFLOAD_MEMPOOL_COUNTER)
        if (counter != NULL) {
          reclaim0 = indirect;
        }
#endif
        reclaim = indirect;
      }
      indirect = &(*indirect)->next;
    } // finished searching chunk for reuse/reclaim

    // Prefer reusing chunk/memory over reclaim (only struct).
    if (reuse != NULL) {
      chunk = *reuse;
      *reuse = chunk->next; // remove chunk from list
    }
    // Reclaim a chunk (resize outside of crit. region).
    else if (reclaim != reclaim0) {
      chunk = *reclaim;
      assert(*reclaim != NULL);
      *reclaim = chunk->next; // remove chunk from list
    }
  } // end of critical section

  // Resize/allocate chunk outside of critical region.
  if (chunk == NULL) {
    chunk = malloc(sizeof(offload_memchunk_t));
    assert(chunk != NULL);
    chunk->mem = actual_malloc(size, on_device);
    chunk->size = size;
  }
  // Free memory and allocate with growth.
  else if (chunk->size < size) {
    actual_free(chunk->mem, on_device);
    chunk->mem = actual_malloc(size, on_device);
    chunk->size = size;
  }
  // For statistics.
  chunk->used = size;

  // Insert chunk into allocated list.
#pragma omp critical(offload_mempool_modify)
  {
    chunk->next = pool->allocated_head;
    pool->allocated_head = chunk;
  }

  return chunk->mem;
}

/*******************************************************************************
 * \brief Internal routine for allocating host memory from the pool.
 * \author Ole Schuett
 ******************************************************************************/
void *offload_mempool_host_malloc(const size_t size) {
  return internal_mempool_malloc(&mempool_host, size);
}

/*******************************************************************************
 * \brief Internal routine for allocating device memory from the pool
 * \author Ole Schuett
 ******************************************************************************/
void *offload_mempool_device_malloc(const size_t size) {
  return internal_mempool_malloc(&mempool_device, size);
}

/*******************************************************************************
 * \brief Private routine for freeing all memory in the pool (not locked!).
 * \author Ole Schuett and Hans Pabst
 ******************************************************************************/
static void internal_mempool_clear(offload_mempool_t *pool, uint64_t mempeak) {
  const bool on_device = (pool == &mempool_device);
  assert(on_device || pool == &mempool_host);
  // Free all chunks in available list.
  while (pool->available_head != NULL) {
    offload_memchunk_t *chunk = pool->available_head;
    pool->available_head = chunk->next; // remove chunk
    actual_free(chunk->mem, on_device);
    free(chunk);
  }
  // Record peak size
  if (on_device) {
    if (device_stats.mempeak < mempeak) {
      device_stats.mempeak = mempeak;
    }
  } else {
    if (host_stats.mempeak < mempeak) {
      host_stats.mempeak = mempeak;
    }
  }
}

/*******************************************************************************
 * \brief Private routine for summing alloc sizes of all chunks in given list.
 * \author Ole Schuett and Hans Pabst
 ******************************************************************************/
static uint64_t sum_chunks_size(const offload_memchunk_t *head, size_t offset) {
  uint64_t result = 0;
  for (const offload_memchunk_t *chunk = head; chunk != NULL;
       chunk = chunk->next) {
    result += *(const size_t *)((const char *)chunk + offset);
  }
  return result;
}

/*******************************************************************************
 * \brief Private routine to query statistics (not locked!).
 * \author Hans Pabst
 ******************************************************************************/
void stats_sizes_get(const offload_mempool_t *pool, uint64_t *size,
                     uint64_t *used) {
  const size_t offset_size = offsetof(offload_memchunk_t, size);
  if (NULL != size) {
    *size = sum_chunks_size(pool->allocated_head, offset_size) +
            sum_chunks_size(pool->available_head, offset_size);
  }
  if (NULL != used) {
    const size_t offset_used = offsetof(offload_memchunk_t, used);
    *used = sum_chunks_size(pool->allocated_head, offset_used) +
#if defined(OFFLOAD_MEMPOOL_COUNTER)
            sum_chunks_size(pool->available_head, offset_size);
#else
            sum_chunks_size(pool->available_head, offset_used);
#endif
    assert(NULL == size || *used <= *size); // sanity
  }
}

/*******************************************************************************
 * \brief Private routine for releasing memory back to the pool.
 * \author Ole Schuett
 ******************************************************************************/
static void internal_mempool_free(offload_mempool_t *pool, const void *mem) {
  if (mem == NULL) {
    return;
  }

#pragma omp critical(offload_mempool_modify)
  {
    // Find chunk in allocated list.
    offload_memchunk_t **indirect = &pool->allocated_head;
    while (*indirect != NULL && (*indirect)->mem != mem) {
      indirect = &(*indirect)->next;
    }
    offload_memchunk_t *chunk = *indirect;
    assert(chunk != NULL && chunk->mem == mem);

    // Remove chunk from allocated list.
    *indirect = chunk->next;

    // Add chunk to available list.
    chunk->next = pool->available_head;
    pool->available_head = chunk;
  }
}

/*******************************************************************************
 * \brief Internal routine for releasing memory back to the pool.
 * \author Ole Schuett
 ******************************************************************************/
void offload_mempool_host_free(const void *memory) {
  internal_mempool_free(&mempool_host, memory);
}

/*******************************************************************************
 * \brief Internal routine for releasing memory back to the pool.
 * \author Ole Schuett
 ******************************************************************************/
void offload_mempool_device_free(const void *memory) {
  internal_mempool_free(&mempool_device, memory);
}

/*******************************************************************************
 * \brief Internal routine for freeing all memory in the pool.
 * \author Ole Schuett
 ******************************************************************************/
void offload_mempool_clear(void) {
  // TODO: check for leaks like assert(pool->allocated_head == NULL).
#pragma omp critical(offload_mempool_modify)
  {
    uint64_t size = 0;
    stats_sizes_get(&mempool_host, &size, NULL);
    internal_mempool_clear(&mempool_host, size);
    stats_sizes_get(&mempool_device, &size, NULL);
    internal_mempool_clear(&mempool_device, size);
  }
}

/*******************************************************************************
 * \brief Internal routine to query statistics.
 * \author Hans Pabst
 ******************************************************************************/
void offload_mempool_stats_get(offload_mempool_stats_t *memstats) {
  assert(NULL != memstats);
#pragma omp critical(offload_mempool_modify)
  {
    memstats->host_mallocs = host_stats.mallocs;
    stats_sizes_get(&mempool_host, &memstats->host_size, &memstats->host_used);
    memstats->host_peak = memstats->host_size < host_stats.mempeak
                              ? host_stats.mempeak
                              : memstats->host_size;
    memstats->device_mallocs = device_stats.mallocs;
    stats_sizes_get(&mempool_device, &memstats->device_size,
                    &memstats->device_used);
    memstats->device_peak = memstats->device_size < device_stats.mempeak
                                ? device_stats.mempeak
                                : memstats->device_size;
  }
}

/*******************************************************************************
 * \brief Print allocation statistics..
 * \author Hans Pabst
 ******************************************************************************/
void offload_mempool_stats_print(int fortran_comm,
                                 void (*print_func)(const char *, int, int),
                                 int output_unit) {
  assert(omp_get_num_threads() == 1);

  char buffer[100];
  const cp_mpi_comm_t comm = cp_mpi_comm_f2c(fortran_comm);
  offload_mempool_stats_t memstats;
  offload_mempool_stats_get(&memstats);
  cp_mpi_max_uint64(&memstats.device_mallocs, 1, comm);
  cp_mpi_max_uint64(&memstats.host_mallocs, 1, comm);

  if (0 != memstats.device_mallocs || 0 != memstats.host_mallocs) {
    OFFLOAD_MEMPOOL_PRINT(print_func, "\n", output_unit);
    OFFLOAD_MEMPOOL_PRINT(
        print_func,
        " ----------------------------------------------------------------"
        "---------------\n",
        output_unit);
    OFFLOAD_MEMPOOL_PRINT(
        print_func,
        " -                                                               "
        "              -\n",
        output_unit);

    OFFLOAD_MEMPOOL_PRINT(
        print_func,
        " -                          OFFLOAD MEMPOOL STATISTICS           "
        "              -\n",
        output_unit);
    OFFLOAD_MEMPOOL_PRINT(
        print_func,
        " -                                                               "
        "              -\n",
        output_unit);
    OFFLOAD_MEMPOOL_PRINT(
        print_func,
        " ----------------------------------------------------------------"
        "---------------\n",
        output_unit);
    OFFLOAD_MEMPOOL_PRINT(print_func,
                          " Memory consumption               "
                          " Number of allocations  Used [MiB]  Size [MiB]\n",
                          output_unit);
  }
  if (0 < memstats.device_mallocs) {
    cp_mpi_max_uint64(&memstats.device_peak, 1, comm);
    snprintf(buffer, sizeof(buffer),
             " Device                            "
             " %20" PRIuPTR "  %10" PRIuPTR "  %10" PRIuPTR "\n",
             (uintptr_t)memstats.device_mallocs,
             (uintptr_t)((memstats.device_used + (512U << 10)) >> 20),
             (uintptr_t)((memstats.device_peak + (512U << 10)) >> 20));
    OFFLOAD_MEMPOOL_PRINT(print_func, buffer, output_unit);
  }
  if (0 < memstats.host_mallocs) {
    cp_mpi_max_uint64(&memstats.host_peak, 1, comm);
    snprintf(buffer, sizeof(buffer),
             " Host                              "
             " %20" PRIuPTR "  %10" PRIuPTR "  %10" PRIuPTR "\n",
             (uintptr_t)memstats.host_mallocs,
             (uintptr_t)((memstats.host_used + (512U << 10)) >> 20),
             (uintptr_t)((memstats.host_peak + (512U << 10)) >> 20));
    OFFLOAD_MEMPOOL_PRINT(print_func, buffer, output_unit);
  }
  if (0 < memstats.device_mallocs || 0 < memstats.host_mallocs) {
    OFFLOAD_MEMPOOL_PRINT(
        print_func,
        " ----------------------------------------------------------------"
        "---------------\n",
        output_unit);
  }
}

// EOF
