/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2023 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../offload/offload_runtime.h"
#if defined(__OFFLOAD_OPENCL) && !defined(__NO_OFFLOAD_DBM)

#include "dbm_multiply_gpu_kernel.h"
#include "dbm_multiply_gpu_kernel.cl.h"


void dbm_multiply_gpu_launch_kernel(const offloadStream_t stream,
                                    const double alpha, const int ntasks,
                                    const dbm_task_t *batch,
                                    const double *pack_a_data,
                                    const double *pack_b_data,
                                    double *shard_c_data)
{
}

#endif // defined(__OFFLOAD_OPENCL) && !defined(__NO_OFFLOAD_DBM)

// EOF
