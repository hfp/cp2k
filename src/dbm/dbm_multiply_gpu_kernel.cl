/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "../../exts/dbcsr/src/acc/opencl/common/opencl_atomics.h"
#include "dbm_multiply_internal.h"

kernel void process_batch_kernel(int m_max, int n_max, double alpha, size_t offset,
                                 global const dbm_task_t *restrict batch,
                                 global const double *restrict pack_a_data,
                                 global const double *restrict pack_b_data,
                                 global double *restrict shard_c_data) {
  /* TODO */
}