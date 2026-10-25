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

// Private toggle: LIBXS pools on top of the same system allocations.
#if !defined(OFFLOAD_MEMPOOL_LIBXS) && defined(__LIBXS) && 0
#define OFFLOAD_MEMPOOL_LIBXS
#endif
#if defined(OFFLOAD_MEMPOOL_LIBXS)
#include <libxs/libxs_malloc.h>
#endif

#define OFFLOAD_MEMPOOL_PRINT(FN, MSG, OUTPUT_UNIT)                            \
  ((FN)(MSG, (int)strlen(MSG), OUTPUT_UNIT))
#define OFFLOAD_MEMPOOL_OMPALLOC 1
// A chunk is reused for at most this multiple of its size (larger ones are
// resized), and the available chunks are released down to the size in use.
#define OFFLOAD_MEMPOOL_FIT 2

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
  uint64_t available_size, allocated_size; // sums of chunk sizes in the lists
  uint64_t peak_size;                      // for statistics
} offload_mempool_t;

/*******************************************************************************
 * \brief Private pools for host and device memory.
 * \author Ole Schuett
 ******************************************************************************/
static offload_mempool_t mempool_host = {0}, mempool_device = {0};

/*******************************************************************************
 * \brief Private counters for statistics.
 * \author Hans Pabst
 ******************************************************************************/
static uint64_t host_malloc_counter = 0, device_malloc_counter = 0;

/*******************************************************************************
 * \brief Returns the larger of two given integer (missing from the C standard)
 * \author Ole Schuett
 ******************************************************************************/
static inline uint64_t imax(uint64_t x, uint64_t y) { return (x > y ? x : y); }

/*******************************************************************************
 * \brief Private routine for actually allocating system memory.
 * \author Ole Schuett
 ******************************************************************************/
static void *actual_malloc(const size_t size, const bool on_device) {
  if (size == 0) {
    return NULL;
  }

  void *memory = NULL;

// This pool must not sit on top of LIBXSTREAM's pool (double caching).
#if defined(__OFFLOAD) && defined(__LIBXSTREAM)
  offload_activate_chosen_device();
  if (on_device) {
    OFFLOAD_CHECK(libxstream_mem_dev_allocate_hint(&memory, size,
                                                   LIBXSTREAM_MALLOC_CP2K));
  } else {
    OFFLOAD_CHECK(libxstream_mem_host_allocate_hint(&memory, size,
                                                    LIBXSTREAM_MALLOC_CP2K));
  }
#elif defined(__OFFLOAD)
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
    ++device_malloc_counter;
  } else {
#pragma omp atomic
    ++host_malloc_counter;
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

// This pool must not sit on top of LIBXSTREAM's pool (double caching).
#if defined(__OFFLOAD) && defined(__LIBXSTREAM)
  offload_activate_chosen_device();
  if (on_device) {
    OFFLOAD_CHECK(
        libxstream_mem_dev_deallocate_hint(memory, LIBXSTREAM_MALLOC_CP2K));
  } else {
    OFFLOAD_CHECK(
        libxstream_mem_host_deallocate_hint(memory, LIBXSTREAM_MALLOC_CP2K));
  }
#elif defined(__OFFLOAD)
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

#if defined(OFFLOAD_MEMPOOL_LIBXS)
/*******************************************************************************
 * \brief Private LIBXS pools (created on first use) and their allocators.
 ******************************************************************************/
static libxs_malloc_pool_t *libxs_pool_host = NULL, *libxs_pool_device = NULL;
static void *libxs_host_malloc(size_t size) {
  return actual_malloc(size, false);
}
static void libxs_host_free(void *memory) { actual_free(memory, false); }
static void *libxs_device_malloc(size_t size) {
  return actual_malloc(size, true);
}
static void libxs_device_free(void *memory) { actual_free(memory, true); }

/*******************************************************************************
 * \brief Private routine for allocating from a LIBXS pool (native pointers,
 *        as device memory must not carry inline metadata).
 ******************************************************************************/
static void *internal_libxs_malloc(libxs_malloc_pool_t **pool,
                                   const size_t size, const bool on_device) {
  void *result = NULL;
  if (size != 0) {
    libxs_malloc_pool_t *p;
#pragma omp critical(offload_mempool_modify)
    {
      if (*pool == NULL) {
        *pool = (on_device
                     ? libxs_malloc_pool(libxs_device_malloc, libxs_device_free)
                     : libxs_malloc_pool(libxs_host_malloc, libxs_host_free));
      }
      p = *pool;
    }
    assert(p != NULL);
    result = libxs_malloc(p, size, LIBXS_MALLOC_NATIVE);
    assert(result != NULL);
  }
  return result;
}
#endif

/*******************************************************************************
 * \brief Private routine for allocating host or device memory from the pool.
 * \author Ole Schuett and Hans Pabst
 ******************************************************************************/
static void *internal_mempool_malloc(offload_mempool_t *pool, const size_t size,
                                     const bool on_device) {
  if (size == 0) {
    return NULL;
  }

  offload_memchunk_t *chunk;

#pragma omp critical(offload_mempool_modify)
  {
    // Find a possible chunk to reuse or reclaim in available list.
    offload_memchunk_t **reuse = NULL,
                       **reclaim = NULL; // ** for easy list removal
    offload_memchunk_t **indirect = &pool->available_head;
    while (*indirect != NULL) {
      const size_t s = (*indirect)->size;
      if (size <= s && s / OFFLOAD_MEMPOOL_FIT <= size &&
          (reuse == NULL || s < (*reuse)->size)) {
        reuse = indirect; // reuse smallest suitable chunk
        if (s == size) {
          break; // perfect match, exit early
        }
      } else if (reclaim == NULL || (*reclaim)->size < s) {
        reclaim = indirect; // reclaim largest unsuitable chunk (resized)
      }
      indirect = &(*indirect)->next;
    }

    // Select an existing chunk or allocate a new one.
    if (reuse != NULL) {
      // Reusing an exising chunk that's already large enough.
      chunk = *reuse;
      *reuse = chunk->next; // remove chunk from available list.
      pool->available_size -= chunk->size;
    } else if (reclaim != NULL) {
      // Reclaiming an existing chunk (resize will happen outside crit. region).
      chunk = *reclaim;
      *reclaim = chunk->next; // remove chunk from available list.
      pool->available_size -= chunk->size;
    } else {
      // Found no available chunk, allocate a new one.
      chunk = calloc(1, sizeof(offload_memchunk_t));
      assert(chunk != NULL);
    }
  }

  // Resize chunk outside of critical region before adding it to allocated list.
  if (chunk->size < size || chunk->size / OFFLOAD_MEMPOOL_FIT > size) {
    actual_free(chunk->mem, on_device);
    chunk->mem = actual_malloc(size, on_device);
    chunk->size = size;
  }

  chunk->used = size; // for statistics

  // Insert chunk into allocated list.
#pragma omp critical(offload_mempool_modify)
  {
    chunk->next = pool->allocated_head;
    pool->allocated_head = chunk;
    pool->allocated_size += chunk->size;
    pool->peak_size = imax(pool->peak_size,
                           pool->available_size + pool->allocated_size);
  }

  return chunk->mem;
}

/*******************************************************************************
 * \brief Internal routine for allocating host memory from the pool.
 * \author Ole Schuett
 ******************************************************************************/
void *offload_mempool_host_malloc(const size_t size) {
#if defined(OFFLOAD_MEMPOOL_LIBXS)
  return internal_libxs_malloc(&libxs_pool_host, size, false);
#else
  return internal_mempool_malloc(&mempool_host, size, false);
#endif
}

/*******************************************************************************
 * \brief Internal routine for allocating device memory from the pool
 * \author Ole Schuett
 ******************************************************************************/
void *offload_mempool_device_malloc(const size_t size) {
#if defined(OFFLOAD_MEMPOOL_LIBXS)
  return internal_libxs_malloc(&libxs_pool_device, size, true);
#else
  return internal_mempool_malloc(&mempool_device, size, true);
#endif
}

/*******************************************************************************
 * \brief Private routine for releasing memory back to the pool.
 * \author Ole Schuett
 ******************************************************************************/
static void internal_mempool_free(offload_mempool_t *pool, const void *mem,
                                  const bool on_device) {
  if (mem == NULL) {
    return;
  }

  offload_memchunk_t *release = NULL; // chunks to give back to the system

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
    pool->allocated_size -= chunk->size;

    // Add chunk to available list.
    chunk->next = pool->available_head;
    pool->available_head = chunk;
    pool->available_size += chunk->size;

    // Release the largest available chunks while they exceed what is in use.
    while (pool->allocated_size < pool->available_size) {
      offload_memchunk_t **largest = &pool->available_head;
      for (indirect = &(*largest)->next; *indirect != NULL;
           indirect = &(*indirect)->next) {
        if ((*largest)->size < (*indirect)->size) {
          largest = indirect;
        }
      }
      chunk = *largest;
      *largest = chunk->next; // remove chunk from available list
      pool->available_size -= chunk->size;
      chunk->next = release;
      release = chunk;
    }
  }

  // Free released chunks outside of critical region.
  while (release != NULL) {
    offload_memchunk_t *chunk = release;
    release = chunk->next;
    actual_free(chunk->mem, on_device);
    free(chunk);
  }
}

/*******************************************************************************
 * \brief Internal routine for releasing memory back to the pool.
 * \author Ole Schuett
 ******************************************************************************/
void offload_mempool_host_free(const void *memory) {
#if defined(OFFLOAD_MEMPOOL_LIBXS)
  libxs_free((void *)memory);
#else
  internal_mempool_free(&mempool_host, memory, false);
#endif
}

/*******************************************************************************
 * \brief Internal routine for releasing memory back to the pool.
 * \author Ole Schuett
 ******************************************************************************/
void offload_mempool_device_free(const void *memory) {
#if defined(OFFLOAD_MEMPOOL_LIBXS)
  libxs_free((void *)memory);
#else
  internal_mempool_free(&mempool_device, memory, true);
#endif
}

/*******************************************************************************
 * \brief Private routine for freeing all memory in the pool.
 * \author Ole Schuett
 ******************************************************************************/
static void internal_mempool_clear(offload_mempool_t *pool,
                                   const bool on_device) {

#pragma omp critical(offload_mempool_modify)
  {
    uint64_t pool_size = 0;

    // Check for leaks, i.e. that the allocated list is empty.
    assert(pool->allocated_head == NULL);

    // Free all chunks in available list.
    while (pool->available_head != NULL) {
      offload_memchunk_t *chunk = pool->available_head;
      pool->available_head = chunk->next; // remove chunk
      actual_free(chunk->mem, on_device);
      pool_size += chunk->size;
      free(chunk);
    }
    pool->available_size = 0;

    // Update stats.
    pool->peak_size = imax(pool->peak_size, pool_size);
  }
}

/*******************************************************************************
 * \brief Internal routine for freeing all memory in the pool.
 * \author Ole Schuett and Hans Pabst
 ******************************************************************************/
void offload_mempool_clear(void) {
#if defined(OFFLOAD_MEMPOOL_LIBXS)
#pragma omp critical(offload_mempool_modify)
  {
    libxs_free_pool(libxs_pool_host);
    libxs_free_pool(libxs_pool_device);
    libxs_pool_host = libxs_pool_device = NULL;
  }
#endif
  internal_mempool_clear(&mempool_host, false);
  internal_mempool_clear(&mempool_device, true);
}

/*******************************************************************************
 * \brief Private routine for summing alloc sizes of all chunks in given list.
 * \author Ole Schuett
 ******************************************************************************/
static uint64_t sum_chunks_size(const offload_memchunk_t *head) {
  uint64_t size_sum = 0;
  for (const offload_memchunk_t *chunk = head; chunk != NULL;
       chunk = chunk->next) {
    size_sum += chunk->size;
  }
  return size_sum;
}

/*******************************************************************************
 * \brief Private routine for summing used sizes of all chunks in given list.
 * \author Ole Schuett
 ******************************************************************************/
static uint64_t sum_chunks_used(const offload_memchunk_t *head) {
  uint64_t used_sum = 0;
  for (const offload_memchunk_t *chunk = head; chunk != NULL;
       chunk = chunk->next) {
    used_sum += chunk->used;
  }
  return used_sum;
}

/*******************************************************************************
 * \brief Internal routine to query statistics.
 * \author Hans Pabst
 ******************************************************************************/
void offload_mempool_stats_get(offload_mempool_stats_t *memstats) {
  assert(NULL != memstats);
#pragma omp critical(offload_mempool_modify)
  {
    memstats->host_mallocs = host_malloc_counter;
    memstats->host_used = sum_chunks_used(mempool_host.available_head) +
                          sum_chunks_used(mempool_host.allocated_head);
    memstats->host_size = sum_chunks_size(mempool_host.available_head) +
                          sum_chunks_size(mempool_host.allocated_head);
    memstats->host_peak = imax(mempool_host.peak_size, memstats->host_size);

    memstats->device_mallocs = device_malloc_counter;
    memstats->device_used = sum_chunks_used(mempool_device.available_head) +
                            sum_chunks_used(mempool_device.allocated_head);
    memstats->device_size = sum_chunks_size(mempool_device.available_head) +
                            sum_chunks_size(mempool_device.allocated_head);
    memstats->device_peak =
        imax(mempool_device.peak_size, memstats->device_size);
#if defined(OFFLOAD_MEMPOOL_LIBXS)
    libxs_malloc_pool_info_t info;
    if (libxs_pool_host != NULL &&
        EXIT_SUCCESS == libxs_malloc_pool_info(libxs_pool_host, &info)) {
      memstats->host_used = info.used;
      memstats->host_size = info.size;
      memstats->host_peak = info.peak;
    }
    if (libxs_pool_device != NULL &&
        EXIT_SUCCESS == libxs_malloc_pool_info(libxs_pool_device, &info)) {
      memstats->device_used = info.used;
      memstats->device_size = info.size;
      memstats->device_peak = info.peak;
    }
#endif
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
#if defined(__OFFLOAD)
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
#endif
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
