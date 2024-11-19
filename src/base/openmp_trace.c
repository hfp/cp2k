/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

static int openmp_trace_issues_n;
static int openmp_trace_level;

int openmp_trace_issues(void);
int openmp_trace_issues(void) { /* routine is exposed in Fortran interface */
  return 0 != openmp_trace_level ? openmp_trace_issues_n : -1 /*disabled*/;
}

#if defined(_OPENMP)
/* #include <omp.h>: avoid functionality being traced */
#include <assert.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Simple compile-time check if OMPT is available (omp/iomp, not gomp).
 * __clang__: omp and iomp/icx, __INTEL_COMPILER: iomp/icc
 * __INTEL_LLVM_COMPILER: already covered by __clang__
 */
#if (defined(__clang__) || defined(__INTEL_COMPILER))
#include <omp-tools.h>
#else
typedef struct ompt_frame_t ompt_frame_t;
typedef void *ompt_initialize_t;
typedef void *ompt_finalize_t;
typedef void *ompt_callback_t;
typedef union ompt_data_t {
  uint64_t value;
  void *ptr;
} ompt_data_t;
typedef struct ompt_start_tool_result_t {
  ompt_initialize_t initialize;
  ompt_finalize_t finalize;
  ompt_data_t tool_data;
} ompt_start_tool_result_t;
typedef enum ompt_scope_endpoint_t {
  ompt_scope_begin = 1,
  ompt_scope_end
} ompt_scope_endpoint_t;
typedef enum ompt_set_result_t {
  ompt_set_never = 1,
} ompt_set_result_t;
typedef enum ompt_callbacks_t {
  ompt_callback_parallel_begin = 3,
  ompt_callback_parallel_end = 4,
  ompt_callback_master = 21,
  ompt_callback_sync_region = 23,
} ompt_callbacks_t;
typedef enum ompt_parallel_flag_t {
  ompt_parallel_team = 0x80000000
} ompt_parallel_flag_t;
typedef enum ompt_sync_region_t {
  ompt_sync_region_barrier = 1,
  ompt_sync_region_barrier_implicit,
  ompt_sync_region_barrier_explicit,
  ompt_sync_region_barrier_implementation
} ompt_sync_region_t;

typedef void (*ompt_interface_fn_t)(void);
typedef int (*ompt_get_parallel_info_t)(int, ompt_data_t **, int *);
typedef ompt_interface_fn_t (*ompt_function_lookup_t)(const char *);
typedef ompt_set_result_t (*ompt_set_callback_t)(ompt_callbacks_t,
                                                 ompt_callback_t);
#endif

#if !defined(_WIN32) && !defined(__CYGWIN__) && !defined(OPENMP_TRACE_SYMBOL)
#define OPENMP_TRACE_SYMBOL
#include <execinfo.h>
#include <unistd.h>
#endif

#define OPENMP_TRACE_UNUSED(VAR) (void)VAR

#define OPENMP_TRACE_SET_CALLBACK(PREFIX, NAME)                                \
  if (ompt_set_never ==                                                        \
      set_callback(ompt_callback_##NAME, (ompt_callback_t)PREFIX##_##NAME)) {  \
    ++openmp_trace_issues_n;                                                   \
  }

static int openmp_trace_parallel_n;
static int openmp_trace_sync_kind;
static int openmp_trace_sync_n;

static const void *openmp_trace_parallel;
static ompt_data_t *openmp_trace_sync;

static ompt_get_parallel_info_t openmp_trace_get_parallel_info;

/* attempt to translate symbol/address to character string */
static void openmp_trace_symbol(const void *symbol, char *str, size_t size,
                                int cleanup) {
  if (NULL != str && 0 < size) {
#if !defined(OPENMP_TRACE_SYMBOL)
    OPENMP_TRACE_UNUSED(symbol);
#else
    int pipefd[2];
    if (NULL != symbol && 0 == pipe(pipefd)) {
      void *const backtrace[] = {(void *)symbol};
      backtrace_symbols_fd(backtrace, 1, pipefd[1]);
      close(pipefd[1]);
      if (0 < read(pipefd[0], str, size)) {
        char *s = (char *)(0 != cleanup ? memchr(str, '(', size) : NULL);
        char *t =
            (char *)(NULL != s ? memchr(s + 1, '+', size - (s - str)) : NULL);
        if (NULL != t) {
          *t = '\0';
          memmove(str, s + 1, t - s);
        }
        s = (char *)memchr(str, '\n', size);
        if (NULL != s) {
          *s = '\0';
        }
        for (s = str; s < (str + size) && '\0' != *s; ++s) {
          if (0 == isprint(*s)) {
            *str = '\0';
            break;
          }
        }
      } else {
        *str = '\0';
      }
      close(pipefd[0]);
    } else
#endif
    { *str = '\0'; }
  }
}

/* ancestor level of given data relative to current parallel region */
static int openmp_trace_parallel_ancestor(const ompt_data_t *ancestor_data) {
  if (NULL != ancestor_data) {
    ompt_data_t *info = NULL;
    int level = 1;
    for (; 0 != openmp_trace_get_parallel_info(level, &info, NULL); ++level) {
      if (info == ancestor_data) {
        break;
      }
    }
    return parallel_data != ancestor_data ? (0 < openmp_trace_level ? 1 : 0)
                                          : level;
  }
  return 0;
}

/* https://www.openmp.org/spec-html/5.0/openmpsu187.html */
static void openmp_trace_parallel_begin(
    ompt_data_t *encountering_task_data,
    const ompt_frame_t *encountering_task_frame, ompt_data_t *parallel_data,
    unsigned int requested_parallelism, int flags, const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(encountering_task_data);
  OPENMP_TRACE_UNUSED(encountering_task_frame);
  OPENMP_TRACE_UNUSED(parallel_data);
  OPENMP_TRACE_UNUSED(requested_parallelism);
  if (0 != (ompt_parallel_team & flags) &&
      0 != openmp_trace_parallel_ancestor(openmp_trace_sync)) {
    ++openmp_trace_issues_n;
    if (2 <= openmp_trace_level || 0 > openmp_trace_level) {

      ompt_sync_region_barrier = 1, ompt_sync_region_barrier_implicit,
      ompt_sync_region_barrier_explicit,
      ompt_sync_region_barrier_implementation

          const char *
          kinds[] = {"master", "barrier", "implicit barrier", "barrier",
                     "barrier"};
      const char *const kind =
          (openmp_trace_sync_kind * sizeof(*kinds)) < sizeof(kinds)
              ? kinds[openmp_trace_sync_kind]
              : "synchronization";
      char sym_sync[1024], sym_parallel[1024];
      openmp_trace_symbol(openmp_trace_sync->ptr, sym_sync, sizeof(sym_sync),
                          1 /*cleanup*/);
      openmp_trace_symbol(codeptr_ra, sym_parallel, sizeof(sym_parallel),
                          1 /*cleanup*/);
      assert(NULL != kind);
      if ('\0' != *sym_parallel) {
        fprintf(stderr,
                "OMP/TRACE ERROR: parallel region \"%s\""
                " opened in %s construct",
                sym_parallel, kind);
      } else {
        fprintf(stderr,
                "OMP/TRACE ERROR: parallel region opened in %s construct",
                kind);
      }
      if ('\0' != *sym_sync) {
        fprintf(stderr, " \"%s\"\n", sym_sync);
      } else {
        fprintf(stderr, "\n");
      }
    } else {
      assert(0);
    }
  }
}

/* https://www.openmp.org/spec-html/5.0/openmpsu187.html */
static void openmp_trace_parallel_end(ompt_data_t *parallel_data,
                                      ompt_data_t *encountering_task_data,
                                      int flags, const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(parallel_data);
  OPENMP_TRACE_UNUSED(encountering_task_data);
  if (0 != (ompt_parallel_team & flags) &&
      0 != openmp_trace_get_parallel_info(openmp_trace_parallel_n + 1, NULL,
                                          NULL)) {
    openmp_trace_parallel = codeptr_ra;
    ++openmp_trace_parallel_n;
  }
}

/* https://www.openmp.org/spec-html/5.0/openmpsu187.html */
static void openmp_trace_master(ompt_scope_endpoint_t endpoint,
                                ompt_data_t *parallel_data,
                                ompt_data_t *task_data,
                                const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(task_data);
  if (NULL != parallel_data) {
    switch (endpoint) {
    case ompt_scope_begin: {
      if (0 == openmp_trace_sync_n++) {
        parallel_data->ptr = (void *)codeptr_ra;
        openmp_trace_sync = parallel_data;
        openmp_trace_sync_kind = 0;
      }
    } break;
    case ompt_scope_end: {
      if (0 == --openmp_trace_sync_n) {
        openmp_trace_sync = NULL;
      }
    } break;
    default:; /* ompt_scope_beginend */
    }
  }
}

/* https://www.openmp.org/spec-html/5.0/openmpsu187.html */
void openmp_trace_sync_region(ompt_sync_region_t kind,
                              ompt_scope_endpoint_t endpoint,
                              ompt_data_t *parallel_data,
                              ompt_data_t *task_data, const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(task_data);
  assert(0 < kind);
  if (NULL != parallel_data &&
      ompt_sync_region_barrier_implementation >= kind) {
    switch (endpoint) {
    case ompt_scope_begin: {
      if (0 == openmp_trace_sync_n++) {
        parallel_data->ptr = (void *)codeptr_ra;
        openmp_trace_sync = parallel_data;
        openmp_trace_sync_kind = kind;
      }
    } break;
    case ompt_scope_end: {
      if (0 == --openmp_trace_sync_n) {
        openmp_trace_sync = NULL;
      }
    } break;
    default:; /* ompt_scope_beginend */
    }
  }
}

/* initially, events of interest are registered */
static int openmp_trace_initialize(ompt_function_lookup_t lookup,
                                   int initial_device_num,
                                   ompt_data_t *tool_data) {
  const ompt_set_callback_t set_callback =
      (ompt_set_callback_t)lookup("ompt_set_callback");
  openmp_trace_get_parallel_info =
      (ompt_get_parallel_info_t)lookup("ompt_get_parallel_info");
  OPENMP_TRACE_UNUSED(initial_device_num);
  OPENMP_TRACE_UNUSED(tool_data);
  OPENMP_TRACE_SET_CALLBACK(openmp_trace, parallel_begin);
  OPENMP_TRACE_SET_CALLBACK(openmp_trace, parallel_end);
  OPENMP_TRACE_SET_CALLBACK(openmp_trace, sync_region);
  OPENMP_TRACE_SET_CALLBACK(openmp_trace, master);
  assert(NULL != openmp_trace_get_parallel_info);
  return 0 == openmp_trace_issues();
}

/* here tool_data might be freed and analysis concludes */
static void openmp_trace_finalize(ompt_data_t *tool_data) {
  OPENMP_TRACE_UNUSED(tool_data);
  if (3 <= openmp_trace_level || 0 > openmp_trace_level) {
    if (1 < openmp_trace_parallel_n) { /* nested */
      char sym_parallel[1024];
      openmp_trace_symbol(openmp_trace_parallel, sym_parallel,
                          sizeof(sym_parallel), 1 /*cleanup*/);
      if ('\0' != *sym_parallel) {
        fprintf(stderr,
                "OMP/TRACE INFO: parallelism "
                "in \"%s\" is nested (%i)\n",
                sym_parallel, openmp_trace_parallel_n);
      } else {
        fprintf(stderr, "OMP/TRACE INFO: parallelism is nested (%i)\n",
                openmp_trace_parallel_n);
      }
    }
  }
}

/* entry point which is automatically called by the OpenMP runtime */
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t openmp_start_tool;
  const char *const enabled_env = getenv("CP2K_OMP_TRACE");
  ompt_start_tool_result_t *result = NULL;
  openmp_trace_level = (NULL == enabled_env ? 0 : atoi(enabled_env));
  OPENMP_TRACE_UNUSED(omp_version);
  OPENMP_TRACE_UNUSED(runtime_version);
  if (0 != openmp_trace_level) { /* trace OpenMP constructs */
    openmp_start_tool.initialize = (ompt_initialize_t)openmp_trace_initialize;
    openmp_start_tool.finalize = (ompt_finalize_t)openmp_trace_finalize;
    openmp_start_tool.tool_data.ptr = NULL;
    result = &openmp_start_tool;
  }
  return result;
}

#endif
