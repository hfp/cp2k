/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/
#define OPENMP_TRACE_DISABLED ((unsigned int)-1)

/* routine is exposed in Fortran, hence must be present */
int openmp_trace_issues(void);

/**
 * Simple compile-time check if OMPT is available (omp/iomp, not gomp).
 * __clang__: omp and iomp/icx, __INTEL_COMPILER: iomp/icc
 * __INTEL_LLVM_COMPILER: already covered by __clang__
 */
#if defined(_OPENMP) && (defined(__clang__) || defined(__INTEL_COMPILER))

#include <assert.h>
#include <omp-tools.h>
#include <stdlib.h>

#if !defined(_WIN32) && !defined(__CYGWIN__) && !defined(OPENMP_TRACE_SYMBOL)
#define OPENMP_TRACE_SYMBOL
#include <execinfo.h>
#include <unistd.h>
#endif

#define OPENMP_TRACE_UNUSED(VAR) (void)VAR

#define OPENMP_TRACE_SET_CALLBACK(PREFIX, NAME)                                \
  if (ompt_set_never ==                                                        \
      set_callback(ompt_callback_##NAME, (ompt_callback_t)PREFIX##_##NAME)) {  \
    ++openmp_trace_issues_count;                                               \
  }

static unsigned int openmp_trace_issues_count;
static unsigned int openmp_trace_parallel_count;
static unsigned int openmp_trace_parallel_count_max;
static unsigned int openmp_trace_master_count;

static char openmp_trace_master_codeptr[1024];

int openmp_trace_issues(void) { return (int)openmp_trace_issues_count; }

/* attempt to translate symbol/address to character string */
static void openmp_trace_symbol(const void *symbol, char *buffer, size_t size) {
#if !defined(OPENMP_TRACE_SYMBOL)
  OPENMP_TRACE_UNUSED(symbol);
  OPENMP_TRACE_UNUSED(buffer);
  OPENMP_TRACE_UNUSED(size);
#else
  int pipefd[2];
  if (NULL != symbol && NULL != buffer && 0 < size && 0 == pipe(pipefd)) {
    void *const backtrace[] = {(void *)symbol};
    backtrace_symbols_fd(backtrace, 1, pipefd[1]);
    close(pipefd[1]);
    read(pipefd[0], buffer, size);
    close(pipefd[0]);
  }
#endif
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
  OPENMP_TRACE_UNUSED(flags);
  OPENMP_TRACE_UNUSED(codeptr_ra);
  if (0 != openmp_trace_master_count) {
    ++openmp_trace_issues_count;
    assert(0);
  }
  ++openmp_trace_parallel_count;
}

/* https://www.openmp.org/spec-html/5.0/openmpsu187.html */
static void openmp_trace_parallel_end(ompt_data_t *parallel_data,
                                      ompt_data_t *encountering_task_data,
                                      int flags, const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(parallel_data);
  OPENMP_TRACE_UNUSED(encountering_task_data);
  OPENMP_TRACE_UNUSED(flags);
  OPENMP_TRACE_UNUSED(codeptr_ra);
  --openmp_trace_parallel_count;
  if (openmp_trace_parallel_count_max < openmp_trace_parallel_count) {
    openmp_trace_parallel_count_max = openmp_trace_parallel_count;
  }
}

/* https://www.openmp.org/spec-html/5.0/openmpsu187.html */
static void openmp_trace_master(ompt_scope_endpoint_t endpoint,
                                ompt_data_t *parallel_data,
                                ompt_data_t *task_data,
                                const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(parallel_data);
  OPENMP_TRACE_UNUSED(task_data);
  OPENMP_TRACE_UNUSED(codeptr_ra);
  switch (endpoint) {
  case ompt_scope_begin: {
    openmp_trace_symbol(codeptr_ra, openmp_trace_master_codeptr,
                        sizeof(openmp_trace_master_codeptr));
    ++openmp_trace_master_count;
  } break;
  case ompt_scope_end: {
    --openmp_trace_master_count;
  } break;
  default:; /* ompt_scope_beginend */
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
}

/* entry point which is automatically called by the OpenMP runtime */
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t openmp_start_tool = {
      openmp_trace_initialize, openmp_trace_finalize, {0}};
  const char *const enabled_env = getenv("CP2K_OMP_TRACE");
  const int enabled = (NULL == enabled_env ? 0 : atoi(enabled_env));
  ompt_start_tool_result_t *result = NULL;
  OPENMP_TRACE_UNUSED(omp_version);
  OPENMP_TRACE_UNUSED(runtime_version);
  if (0 == enabled) { /* not enabled */
    openmp_trace_issues_count = OPENMP_TRACE_DISABLED;
    assert(NULL == result);
  } else { /* trace OpenMP constructs */
    assert(0 == openmp_trace_issues_count);
    result = &openmp_start_tool;
  }
  return result;
}

#else

int openmp_trace_issues(void) { return OPENMP_TRACE_DISABLED; }

#endif
