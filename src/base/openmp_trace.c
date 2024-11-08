/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#if defined(_OPENMP) && defined(__clang__)

#include <assert.h>
#include <omp-tools.h>
#include <stdlib.h>

#define OPENMP_TRACE_UNUSED(VAR) (void)VAR

#define OPENMP_TRACE_SET_CALLBACK(PREFIX, NAME)                                \
  if (ompt_set_never ==                                                        \
      set_callback(ompt_callback_##NAME, (ompt_callback_t)PREFIX##_##NAME)) {  \
    ++openmp_trace_nissues;                                                    \
  }

static unsigned int openmp_trace_nissues;
static unsigned int openmp_trace_nparallel;
static unsigned int openmp_trace_nmaster;

int openmp_trace_issues(void);
int openmp_trace_issues(void) { return (int)openmp_trace_nissues; }

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
  if (0 != openmp_trace_nmaster) {
    ++openmp_trace_nissues;
    assert(0);
  }
  ++openmp_trace_nparallel;
}

static void openmp_trace_parallel_end(ompt_data_t *parallel_data,
                                      ompt_data_t *encountering_task_data,
                                      int flags, const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(parallel_data);
  OPENMP_TRACE_UNUSED(encountering_task_data);
  OPENMP_TRACE_UNUSED(flags);
  OPENMP_TRACE_UNUSED(codeptr_ra);
  --openmp_trace_nparallel;
}

static void openmp_trace_master(ompt_scope_endpoint_t endpoint,
                                ompt_data_t *parallel_data,
                                ompt_data_t *task_data,
                                const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(parallel_data);
  OPENMP_TRACE_UNUSED(task_data);
  OPENMP_TRACE_UNUSED(codeptr_ra);
  switch (endpoint) {
  case ompt_scope_begin:
    ++openmp_trace_nmaster;
    break;
  case ompt_scope_end:
    --openmp_trace_nmaster;
    break;
  default:; /* ompt_scope_beginend */
  }
}

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
  return 0 == openmp_trace_nissues;
}

static void openmp_trace_finalize(ompt_data_t *tool_data) {
  OPENMP_TRACE_UNUSED(tool_data);
}

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t openmp_start_tool = {
      openmp_trace_initialize, openmp_trace_finalize, {0}};
  const char *const enabled_env = getenv("CP2K_OMP_TRACE");
  const int enabled = (NULL == enabled_env ? 0 : atoi(enabled_env));
  ompt_start_tool_result_t *result = NULL;
  OPENMP_TRACE_UNUSED(omp_version);
  OPENMP_TRACE_UNUSED(runtime_version);
  if (0 == enabled) {
    openmp_trace_nissues = (unsigned int)-1;
    assert(NULL == result);
  } else {
    assert(0 == openmp_trace_nissues);
    result = &openmp_start_tool;
  }
  return result;
}

#endif /*defined(_OPENMP) && defined(__clang__)*/
