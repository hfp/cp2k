/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

/**
 * Development tool checking the nesting of OpenMP constructs at runtime using
 * OMPT, i.e., the OpenMP runtime must be lib(i)omp (also when GNU-compiled
 * code uses it at runtime), but not libgomp. CP2K_OMP_TRACE=<level>:
 *   1: abort at the first error, i.e., non-conforming code (deadlock risk)
 *   2: print errors (once per code location) and a summary at exit
 *   3: print warnings as well, i.e., valid but likely unwanted code
 *   4: print information as well, e.g., nested parallel regions
 * A negative level selects the most verbose level. A code location is printed
 * as function name (exported symbols only, e.g., -rdynamic) or as module with
 * offset or address (addr2line -f -e module offset). If the executable is not
 * linked against lib(i)omp, ompt_start_tool must be exported explicitly, e.g.,
 * -Wl,--export-dynamic-symbol=ompt_start_tool (or -rdynamic).
 */

int openmp_trace_issues(void); /* exposed in Fortran interface */

#if !defined(_OPENMP)
int openmp_trace_issues(void) { return -1 /*disabled*/; }
#else
/* no #include <omp.h> since functionality called by the tool would be traced */
#include <ctype.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__has_include)
#if __has_include(<omp-tools.h>)
#define OPENMP_TRACE_OMPT
#endif
#if __has_include(<execinfo.h>) && __has_include(<unistd.h>)
#define OPENMP_TRACE_SYMBOL
#endif
#else
#if defined(__clang__) || defined(__INTEL_COMPILER)
#define OPENMP_TRACE_OMPT
#endif
#if !defined(_WIN32) && !defined(__CYGWIN__)
#define OPENMP_TRACE_SYMBOL
#endif
#endif

#if defined(OPENMP_TRACE_OMPT)
#include <omp-tools.h>
#else /* e.g., GNU compiler without omp-tools.h but linked with lib(i)omp */
typedef int ompt_callbacks_t, ompt_set_result_t, ompt_scope_endpoint_t;
typedef int ompt_sync_region_t, ompt_work_t;
typedef struct ompt_frame_t ompt_frame_t;
typedef union ompt_data_t {
  uint64_t value;
  void *ptr;
} ompt_data_t;
typedef void (*ompt_interface_fn_t)(void);
typedef void (*ompt_callback_t)(void);
typedef ompt_interface_fn_t (*ompt_function_lookup_t)(const char *);
typedef ompt_set_result_t (*ompt_set_callback_t)(ompt_callbacks_t,
                                                 ompt_callback_t);
typedef int (*ompt_get_parallel_info_t)(int, ompt_data_t **, int *);
typedef int (*ompt_initialize_t)(ompt_function_lookup_t, int, ompt_data_t *);
typedef void (*ompt_finalize_t)(ompt_data_t *);
typedef struct ompt_start_tool_result_t {
  ompt_initialize_t initialize;
  ompt_finalize_t finalize;
  ompt_data_t tool_data;
} ompt_start_tool_result_t;
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version);
#endif

#if defined(OPENMP_TRACE_SYMBOL)
#include <execinfo.h>
#include <unistd.h>
#endif

/* task slot: count of constructs (4 bits), kind (4 bits), address (56 bits) */
#define OPENMP_TRACE_PTR(PTR) ((uint64_t)(uintptr_t)(PTR))
#define OPENMP_TRACE_ADDR(VALUE) (UINT64_C(0x00FFFFFFFFFFFFFF) & (VALUE))
#define OPENMP_TRACE_KIND(VALUE) ((int)(0xF & ((VALUE) >> 56)))
#define OPENMP_TRACE_COUNT(VALUE) ((int)((VALUE) >> 60))
#define OPENMP_TRACE_SLOT(COUNT, KIND, ADDR)                                   \
  (((uint64_t)(COUNT) << 60) | ((uint64_t)(KIND) << 56) |                      \
   OPENMP_TRACE_ADDR(ADDR))
/* parallel_data: nesting level (upper bits) and runtime-invoked (bit 0) */
#define OPENMP_TRACE_RUNTIME 1
#define OPENMP_TRACE_TEAM 0x80000000u
#define OPENMP_TRACE_INVOKER_RUNTIME 0x2u
#define OPENMP_TRACE_SEEN 256
#define OPENMP_TRACE_UNUSED(VAR) (void)VAR

/* values of OpenMP 5.1 (independent of the version of omp-tools.h) */
enum {
  openmp_trace_scope_begin = 1,
  openmp_trace_scope_end = 2,
  openmp_trace_scope_beginend = 3,
  openmp_trace_set_never = 1,
  openmp_trace_callback_parallel_begin = 3,
  openmp_trace_callback_implicit_task = 7,
  openmp_trace_callback_work = 20,
  openmp_trace_callback_masked = 21,
  openmp_trace_callback_sync_region = 23
};
enum {
  openmp_trace_sync_barrier = 1,
  openmp_trace_sync_barrier_implicit = 2,
  openmp_trace_sync_barrier_explicit = 3,
  openmp_trace_sync_barrier_implementation = 4,
  openmp_trace_sync_barrier_implicit_workshare = 8
};
enum {
  openmp_trace_work_loop = 1,
  openmp_trace_work_single_executor = 3,
  openmp_trace_work_workshare = 5,
  openmp_trace_work_scope = 8,
  openmp_trace_work_loop_static = 10,
  openmp_trace_work_loop_dynamic = 11,
  openmp_trace_work_loop_guided = 12,
  openmp_trace_work_loop_other = 13
};

enum {
  openmp_trace_level_abort = 1,
  openmp_trace_level_error,
  openmp_trace_level_warn,
  openmp_trace_level_info
};
enum { openmp_trace_error = 1, openmp_trace_warn, openmp_trace_info };

/* kinds of constructs (must fit into 4 bits) */
enum {
  openmp_trace_none,
  openmp_trace_master,
  openmp_trace_single,
  openmp_trace_loop,
  openmp_trace_workshare,
  openmp_trace_scope,
  openmp_trace_barrier,
  openmp_trace_parallel
};
static const char *const openmp_trace_names[] = {
    "construct", "master", "single",  "loop",
    "workshare", "scope",  "barrier", "parallel region"};

static int openmp_trace_level;
static atomic_int openmp_trace_errors;
static atomic_int openmp_trace_warnings;
static atomic_int openmp_trace_nested;
static _Atomic(uint64_t) openmp_trace_deepest;
static _Atomic(uint64_t) openmp_trace_seen[OPENMP_TRACE_SEEN];
static ompt_get_parallel_info_t openmp_trace_get_parallel_info;

int openmp_trace_issues(void) {
  return 0 != openmp_trace_level ? atomic_load(&openmp_trace_errors) : -1;
}

/* nonzero if key is seen the first time (or if the table is full) */
static int openmp_trace_unique(uint64_t key) {
  size_t i = (size_t)((key * UINT64_C(0x9E3779B97F4A7C15)) >> 56);
  int result = -1, n;
  for (n = 0; 0 > result && n < OPENMP_TRACE_SEEN; ++n) {
    uint64_t expected = 0;
    if (atomic_compare_exchange_strong(&openmp_trace_seen[i], &expected, key)) {
      result = 1;
    } else if (key == expected) {
      result = 0;
    } else {
      i = (i + 1) % OPENMP_TRACE_SEEN;
    }
  }
  return 0 != result;
}

/* translate code address to function name or module(+offset) */
static void openmp_trace_symbol(uint64_t addr, char *str, size_t size) {
  *str = '\0';
#if defined(OPENMP_TRACE_SYMBOL)
  if (0 != addr && 1 < size) {
    int pipefd[2];
    if (0 == pipe(pipefd)) { /* backtrace_symbols_fd avoids malloc */
      void *const frame[] = {(void *)(uintptr_t)addr};
      ssize_t n, i = 0;
      backtrace_symbols_fd(frame, 1, pipefd[1]);
      close(pipefd[1]);
      n = read(pipefd[0], str, size - 1);
      close(pipefd[0]);
      while (i < n && 0 != isprint((unsigned char)str[i])) {
        ++i;
      }
      str[i] = '\0';
      if (0 < i) { /* module(symbol+offset) [address]: keep all if no symbol */
        char *const s = strchr(str, '(');
        char *const t = (NULL != s ? strchr(s + 1, '+') : NULL);
        if (NULL != t && s + 1 < t) {
          *t = '\0';
          memmove(str, s + 1, (size_t)(t - s));
        }
      }
    }
  }
#else
  OPENMP_TRACE_UNUSED(addr);
  OPENMP_TRACE_UNUSED(size);
#endif
}

/* count issue, print it once per location, and abort if requested */
static void openmp_trace_issue(int category, int what, const void *codeptr,
                               uint64_t outer) {
  const uint64_t addr = OPENMP_TRACE_PTR(codeptr);
  if (openmp_trace_error == category) {
    atomic_fetch_add_explicit(&openmp_trace_errors, 1, memory_order_relaxed);
  } else {
    atomic_fetch_add_explicit(&openmp_trace_warnings, 1, memory_order_relaxed);
  }
  if ((openmp_trace_error == category ||
       openmp_trace_level_warn <= openmp_trace_level) &&
      (openmp_trace_level_abort == openmp_trace_level ||
       0 != openmp_trace_unique(OPENMP_TRACE_SLOT(category, what, addr)))) {
    const char *const type =
        (openmp_trace_error == category ? "ERROR" : "WARN");
    char a[1024], b[1024];
    openmp_trace_symbol(OPENMP_TRACE_ADDR(addr), a, sizeof(a));
    openmp_trace_symbol(OPENMP_TRACE_ADDR(outer), b, sizeof(b));
    fprintf(stderr, "OMP/TRACE %s: %s%s%s%s nested in %s%s%s%s\n", type,
            openmp_trace_names[what], '\0' != *a ? " \"" : "", a,
            '\0' != *a ? "\"" : "",
            openmp_trace_names[OPENMP_TRACE_KIND(outer)],
            '\0' != *b ? " \"" : "", b, '\0' != *b ? "\"" : "");
  }
  if (openmp_trace_error == category &&
      openmp_trace_level_abort == openmp_trace_level) {
    abort();
  }
}

/* enter or leave a construct binding to the (implicit) task */
static void openmp_trace_construct(ompt_scope_endpoint_t endpoint,
                                   ompt_data_t *task_data, int what,
                                   const void *codeptr) {
  const uint64_t slot = task_data->value;
  const int count = OPENMP_TRACE_COUNT(slot);
  if (openmp_trace_scope_end != (int)endpoint) {
    if (0 == count) {
      task_data->value = OPENMP_TRACE_SLOT(1, what, OPENMP_TRACE_PTR(codeptr));
    } else { /* master nested in master is valid */
      if (openmp_trace_master != what ||
          openmp_trace_master != OPENMP_TRACE_KIND(slot)) {
        openmp_trace_issue(openmp_trace_error, what, codeptr, slot);
      }
      if (0xF > count) {
        task_data->value = slot + OPENMP_TRACE_SLOT(1, 0, 0);
      }
    }
  }
  if (openmp_trace_scope_begin != (int)endpoint) {
    const uint64_t value = task_data->value;
    task_data->value =
        (1 < OPENMP_TRACE_COUNT(value) ? (value - OPENMP_TRACE_SLOT(1, 0, 0))
                                       : 0);
  }
}

/* OMPT callback (signature as in omp-tools.h) */
static void openmp_trace_parallel_begin(
    ompt_data_t *encountering_task_data,
    const ompt_frame_t *encountering_task_frame, ompt_data_t *parallel_data,
    unsigned int requested_parallelism, int flags, const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(encountering_task_frame);
  OPENMP_TRACE_UNUSED(requested_parallelism);
  if (0 != (OPENMP_TRACE_TEAM & (unsigned int)flags) && NULL != parallel_data) {
    ompt_data_t *outer = NULL;
    int level = 1;
    if (0 != openmp_trace_get_parallel_info(0, &outer, NULL) && NULL != outer) {
      level += (int)(outer->value >> 1);
    }
    /* GNU-compiled code (GOMP-interface) is invoked by the program */
    parallel_data->value =
        ((uint64_t)level << 1) |
        (0 != (OPENMP_TRACE_INVOKER_RUNTIME & (unsigned int)flags)
             ? OPENMP_TRACE_RUNTIME
             : 0);
    if (NULL != encountering_task_data && 0 != encountering_task_data->value) {
      openmp_trace_issue(openmp_trace_warn, openmp_trace_parallel, codeptr_ra,
                         encountering_task_data->value);
    }
    if (1 < level) {
      const uint64_t addr = OPENMP_TRACE_PTR(codeptr_ra);
      const uint64_t value = ((uint64_t)(0xFF < level ? 0xFF : level) << 56) |
                             OPENMP_TRACE_ADDR(addr);
      uint64_t deepest = atomic_load(&openmp_trace_deepest);
      while ((deepest >> 56) < (value >> 56) &&
             0 == atomic_compare_exchange_weak(&openmp_trace_deepest, &deepest,
                                               value)) {
      }
      atomic_fetch_add_explicit(&openmp_trace_nested, 1, memory_order_relaxed);
      if (openmp_trace_level_info <= openmp_trace_level &&
          0 != openmp_trace_unique(OPENMP_TRACE_SLOT(
                   openmp_trace_info, openmp_trace_parallel, addr))) {
        char a[1024];
        openmp_trace_symbol(OPENMP_TRACE_ADDR(addr), a, sizeof(a));
        fprintf(stderr, "OMP/TRACE INFO: parallel region%s%s%s at level %i\n",
                '\0' != *a ? " \"" : "", a, '\0' != *a ? "\"" : "", level);
      }
    }
  }
}

/* OMPT callback (signature as in omp-tools.h) */
static void openmp_trace_implicit_task(ompt_scope_endpoint_t endpoint,
                                       ompt_data_t *parallel_data,
                                       ompt_data_t *task_data,
                                       unsigned int actual_parallelism,
                                       unsigned int index, int flags) {
  OPENMP_TRACE_UNUSED(endpoint);
  OPENMP_TRACE_UNUSED(parallel_data);
  OPENMP_TRACE_UNUSED(actual_parallelism);
  OPENMP_TRACE_UNUSED(index);
  OPENMP_TRACE_UNUSED(flags);
  if (NULL != task_data) { /* implicit tasks of hot teams are reused */
    task_data->value = 0;
  }
}

/* OMPT callback (signature as in omp-tools.h) */
static void openmp_trace_masked(ompt_scope_endpoint_t endpoint,
                                ompt_data_t *parallel_data,
                                ompt_data_t *task_data,
                                const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(parallel_data);
  if (NULL != task_data) {
    openmp_trace_construct(endpoint, task_data, openmp_trace_master,
                           codeptr_ra);
  }
}

/* OMPT callback (signature as in omp-tools.h) */
static void openmp_trace_work(ompt_work_t work_type,
                              ompt_scope_endpoint_t endpoint,
                              ompt_data_t *parallel_data,
                              ompt_data_t *task_data, uint64_t count,
                              const void *codeptr_ra) {
  int what = openmp_trace_none;
  OPENMP_TRACE_UNUSED(count);
  switch ((int)work_type) {
  case openmp_trace_work_loop:
  case openmp_trace_work_loop_static:
  case openmp_trace_work_loop_dynamic:
  case openmp_trace_work_loop_guided:
  case openmp_trace_work_loop_other:
    what = openmp_trace_loop;
    break;
  case openmp_trace_work_workshare:
    what = openmp_trace_workshare;
    break;
  case openmp_trace_work_scope:
    what = openmp_trace_scope;
    break;
  case openmp_trace_work_single_executor:
    /* GOMP_single_start signals no end of the region */
    if (NULL != parallel_data &&
        0 != (OPENMP_TRACE_RUNTIME & parallel_data->value)) {
      what = openmp_trace_single;
    }
    break;
  default: /* sections: active regions signal no end */
    break;
  }
  if (openmp_trace_none != what && NULL != task_data) {
    openmp_trace_construct(endpoint, task_data, what, codeptr_ra);
  }
}

/* OMPT callback (signature as in omp-tools.h) */
static void openmp_trace_sync_region(ompt_sync_region_t kind,
                                     ompt_scope_endpoint_t endpoint,
                                     ompt_data_t *parallel_data,
                                     ompt_data_t *task_data,
                                     const void *codeptr_ra) {
  OPENMP_TRACE_UNUSED(parallel_data);
  if (NULL != task_data && 0 != task_data->value &&
      openmp_trace_scope_end != (int)endpoint) {
    const uint64_t slot = task_data->value;
    switch ((int)kind) {
    case openmp_trace_sync_barrier:
    case openmp_trace_sync_barrier_explicit:
    case openmp_trace_sync_barrier_implementation: /* GOMP_barrier */
      openmp_trace_issue(openmp_trace_error, openmp_trace_barrier, codeptr_ra,
                         slot);
      break;
    case openmp_trace_sync_barrier_implicit:
    case openmp_trace_sync_barrier_implicit_workshare:
      /* concludes a worksharing construct which signaled no end */
      if (1 == OPENMP_TRACE_COUNT(slot) &&
          openmp_trace_master != OPENMP_TRACE_KIND(slot)) {
        task_data->value = 0;
      }
      break;
    default: /* taskwait, taskgroup, reduction, end of parallel or teams */
      break;
    }
  }
}

/* register callback; nonzero if the runtime can dispatch the event */
static int openmp_trace_register(ompt_set_callback_t set_callback, int event,
                                 ompt_callback_t callback, const char *name) {
  const int result =
      (openmp_trace_set_never <
               (int)set_callback((ompt_callbacks_t)event, callback)
           ? 1
           : 0);
  if (0 == result && openmp_trace_level_info <= openmp_trace_level) {
    fprintf(stderr, "OMP/TRACE INFO: %s events are not supported\n", name);
  }
  return result;
}

/* initially, events of interest are registered */
static int openmp_trace_initialize(ompt_function_lookup_t lookup,
                                   int initial_device_num,
                                   ompt_data_t *tool_data) {
  const ompt_set_callback_t set_callback =
      (ompt_set_callback_t)lookup("ompt_set_callback");
  int result = 0;
  OPENMP_TRACE_UNUSED(initial_device_num);
  OPENMP_TRACE_UNUSED(tool_data);
  openmp_trace_get_parallel_info =
      (ompt_get_parallel_info_t)lookup("ompt_get_parallel_info");
  if (NULL != set_callback && NULL != openmp_trace_get_parallel_info &&
      0 != openmp_trace_register(
               set_callback, openmp_trace_callback_parallel_begin,
               (ompt_callback_t)openmp_trace_parallel_begin, "parallel") &&
      0 != openmp_trace_register(
               set_callback, openmp_trace_callback_implicit_task,
               (ompt_callback_t)openmp_trace_implicit_task, "implicit task")) {
    openmp_trace_register(set_callback, openmp_trace_callback_masked,
                          (ompt_callback_t)openmp_trace_masked, "master");
    openmp_trace_register(set_callback, openmp_trace_callback_work,
                          (ompt_callback_t)openmp_trace_work, "work");
    openmp_trace_register(set_callback, openmp_trace_callback_sync_region,
                          (ompt_callback_t)openmp_trace_sync_region, "sync");
    result = 1;
  } else { /* tool is not activated */
    openmp_trace_level = 0;
  }
  return result;
}

/* summary when the runtime shuts down */
static void openmp_trace_finalize(ompt_data_t *tool_data) {
  const int errors = atomic_load(&openmp_trace_errors);
  const int warnings = atomic_load(&openmp_trace_warnings);
  const int nested = atomic_load(&openmp_trace_nested);
  OPENMP_TRACE_UNUSED(tool_data);
  if (openmp_trace_level_error <= openmp_trace_level &&
      (0 != errors || 0 != warnings)) {
    fprintf(stderr, "OMP/TRACE: %i error(s), %i warning(s)\n", errors,
            warnings);
  }
  if (openmp_trace_level_info <= openmp_trace_level && 0 != nested) {
    const uint64_t deepest = atomic_load(&openmp_trace_deepest);
    char a[1024];
    openmp_trace_symbol(OPENMP_TRACE_ADDR(deepest), a, sizeof(a));
    fprintf(stderr,
            "OMP/TRACE INFO: %i nested parallel region(s), deepest at level "
            "%i%s%s%s\n",
            nested, (int)(deepest >> 56), '\0' != *a ? " in \"" : "", a,
            '\0' != *a ? "\"" : "");
  }
}

/* entry point which is automatically called by the OpenMP runtime */
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t openmp_start_tool;
  const char *const enabled_env = getenv("CP2K_OMP_TRACE");
  const int level = (NULL == enabled_env ? 0 : atoi(enabled_env));
  ompt_start_tool_result_t *result = NULL;
  OPENMP_TRACE_UNUSED(omp_version);
  OPENMP_TRACE_UNUSED(runtime_version);
  openmp_trace_level = (0 <= level ? level : openmp_trace_level_info);
  if (0 != openmp_trace_level) {
    openmp_start_tool.initialize = openmp_trace_initialize;
    openmp_start_tool.finalize = openmp_trace_finalize;
    openmp_start_tool.tool_data.ptr = NULL;
    result = &openmp_start_tool;
  }
  return result;
}

#endif
