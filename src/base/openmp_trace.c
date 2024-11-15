/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

static int openmp_trace_level;
static int openmp_trace_parallel_n;
static int openmp_trace_parallel_nmax;
static int openmp_trace_master_n;
static int openmp_trace_issues_n;

static const void *openmp_trace_parallel_codeptr;
static const void *openmp_trace_master_codeptr;

int openmp_trace_issues(void);
int openmp_trace_issues(void) { /* routine is exposed in Fortran interface */
  return 0 != openmp_trace_level ? openmp_trace_issues_n : -1 /*disabled*/;
}

#if defined(_OPENMP)
#include <assert.h>
/* careful: avoid functionality being traced */
#include <omp.h>
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
  ompt_callback_master = 21
} ompt_callbacks_t;
typedef void (*ompt_interface_fn_t)(void);
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
    ++openmp_trace_issues_n; /* sequential (no atomics needed) */              \
  }

/* attempt to translate symbol/address to character string */
static void openmp_trace_symbol(const void *symbol, char *buffer, size_t size,
                                int cleanup) {
#if !defined(OPENMP_TRACE_SYMBOL)
  OPENMP_TRACE_UNUSED(symbol);
  if (0 < size) {
    buffer[0] = '\0';
  }
#else
  int pipefd[2];
  if (NULL != symbol && NULL != buffer && 0 < size && 0 == pipe(pipefd)) {
    void *const backtrace[] = {(void *)symbol};
    backtrace_symbols_fd(backtrace, 1, pipefd[1]);
    close(pipefd[1]);
    if (0 < read(pipefd[0], buffer, size)) {
      if (0 != cleanup) {
        char *str = memchr(buffer, '(', size);
        char *end =
            (NULL != str ? memchr(str + 1, '+', size - (str - buffer)) : NULL);
        if (NULL != end) {
          *end = '\0';
          memmove(buffer, str + 1, end - str);
        }
      } else {
        char *str = memchr(buffer, '\n', size);
        if (NULL != str) {
          *str = '\0';
        }
      }
    } else {
      *buffer = '\0';
    }
    close(pipefd[0]);
  }
#endif
}

/* https://www.openmp.org/spec-html/5.0/openmpsu187.html */
static void openmp_trace_parallel_begin(
    ompt_data_t *encountering_task_data,
    const ompt_frame_t *encountering_task_frame, ompt_data_t *parallel_data,
    unsigned int requested_parallelism, int flags, const void *codeptr_ra) {
  const void *master_codeptr;
  OPENMP_TRACE_UNUSED(encountering_task_data);
  OPENMP_TRACE_UNUSED(encountering_task_frame);
  OPENMP_TRACE_UNUSED(parallel_data);
  OPENMP_TRACE_UNUSED(requested_parallelism);
  OPENMP_TRACE_UNUSED(flags);
#pragma omp critical(openmp_trace_parallel_begin)
  {
    ++openmp_trace_parallel_n;
    if (openmp_trace_parallel_nmax < openmp_trace_parallel_n) {
      openmp_trace_parallel_nmax = openmp_trace_parallel_n;
      openmp_trace_parallel_codeptr = codeptr_ra;
    }
    master_codeptr = openmp_trace_master_codeptr;
  }
  if (NULL != master_codeptr) {
#pragma omp atomic
    ++openmp_trace_issues_n;
    if (2 <= openmp_trace_level || 0 > openmp_trace_level) {
      char sym_master[1024], sym_parallel[1024];
      openmp_trace_symbol(master_codeptr, sym_master, sizeof(sym_master),
                          1 /*cleanup*/);
      openmp_trace_symbol(codeptr_ra, sym_parallel, sizeof(sym_parallel),
                          1 /*cleanup*/);
      if ('\0' != *sym_master && '\0' != *sym_parallel) {
        fprintf(stderr,
                "OMP/TRACE ERROR: parallel region \"%s\""
                " opened in master section \"%s\"\n",
                sym_parallel, sym_master);
      } else {
        fprintf(stderr,
                "OMP/TRACE ERROR: parallel region opened in master section\n");
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
  OPENMP_TRACE_UNUSED(flags);
  OPENMP_TRACE_UNUSED(codeptr_ra);
  if (0 < openmp_trace_parallel_n) {
#pragma omp atomic
    --openmp_trace_parallel_n;
  }
}

/* https://www.openmp.org/spec-html/5.0/openmpsu187.html */
static void openmp_trace_master(ompt_scope_endpoint_t endpoint,
                                ompt_data_t *parallel_data,
                                ompt_data_t *task_data,
                                const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(parallel_data);
  OPENMP_TRACE_UNUSED(task_data);
  if (0 < omp_get_active_level()) {
    int master_n;
    switch (endpoint) {
    case ompt_scope_begin: {
#pragma omp atomic capture
      master_n = openmp_trace_master_n++;
      if (0 == master_n) {
        openmp_trace_master_codeptr = codeptr_ra;
      }
    } break;
    case ompt_scope_end: {
#pragma omp atomic capture
      master_n = --openmp_trace_master_n;
      if (0 == master_n) {
        openmp_trace_master_codeptr = NULL;
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
  OPENMP_TRACE_UNUSED(initial_device_num);
  OPENMP_TRACE_UNUSED(tool_data);
  OPENMP_TRACE_SET_CALLBACK(openmp_trace, parallel_begin);
  OPENMP_TRACE_SET_CALLBACK(openmp_trace, parallel_end);
  OPENMP_TRACE_SET_CALLBACK(openmp_trace, master);
  return 0 == openmp_trace_issues();
}

/* here tool_data might be freed and analysis concludes */
static void openmp_trace_finalize(ompt_data_t *tool_data) {
  OPENMP_TRACE_UNUSED(tool_data);
  if (3 <= openmp_trace_level || 0 > openmp_trace_level) {
    if (1 < openmp_trace_parallel_nmax) { /* nested */
      char sym_parallel[1024];
      openmp_trace_symbol(openmp_trace_parallel_codeptr, sym_parallel,
                          sizeof(sym_parallel), 1 /*cleanup*/);
      if ('\0' != *sym_parallel) {
        fprintf(stderr,
                "OMP/TRACE INFO: parallelism "
                "in \"%s\" is nested (%i)\n",
                sym_parallel, openmp_trace_parallel_nmax);
      } else {
        fprintf(stderr, "OMP/TRACE INFO: parallelism is nested (%i)\n",
                openmp_trace_parallel_nmax);
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
    openmp_start_tool.initialize = openmp_trace_initialize;
    openmp_start_tool.finalize = openmp_trace_finalize;
    openmp_start_tool.tool_data.ptr = NULL;
    result = &openmp_start_tool;
  }
  return result;
}

#endif
