/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef DBM_MULTIPLY_GPU_H
#define DBM_MULTIPLY_GPU_H

#include "../offload/offload_runtime.h"
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_DBM)

#include "dbm_internal.h"
#include "dbm_shard.h"

/*******************************************************************************
 * \brief Internal struct for storing per shard gpu objects.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  double *data; // on the device
  int data_size;
  int data_allocated; // beyond data_size, the device data is zero
  offloadStream_t stream;
  offloadEvent_t event;
  offloadEvent_t done; // recorded after each kernel
  double *host_data;   // contributions computed on the host (hybrid)
  int host_size;
  int host_allocated;
} dbm_shard_gpu_t;

// Separate from dbm_pack_t because device allocation lifetime differs from the
// host pack (which is reallocated each tick); tracks device capacity
// explicitly.
typedef struct {
  double *data; // on the device
  int data_allocated;
} dbm_pack_gpu_t;

/*******************************************************************************
 * \brief Internal struct for storing a thread's pair of host batches, which
 *        kernels read directly (unified): one is filled while the other is
 *read.
 * \author Hans Pabst
 ******************************************************************************/
typedef struct {
  dbm_task_t *batch[2];
  offloadEvent_t done[2]; // recorded after the kernel reading the batch
  bool pending[2];
} dbm_batch_gpu_t;

/*******************************************************************************
 * \brief Internal struct for storing the gpu backend's context.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  offloadStream_t main_stream;
  offloadEvent_t upload_event;

  int nshards;
  dbm_shard_gpu_t *shards_c_dev;

  dbm_pack_gpu_t pack_a_dev;
  dbm_pack_gpu_t pack_b_dev;

  int max_batch_size;
  dbm_task_t *batches_dev;

  dbm_shard_t *shards_c_host;
  bool unified; // kernels read host batches (DBM_MULTIPLY_UNIFIED)
  int nthreads; // number of batches_host
  dbm_batch_gpu_t *batches_host;
  int hybrid;        // max. number of threads computing on the host
  int hybrid_active; // number of threads computing on the host
  int64_t flops[2];  // computed on the GPU and on the host
} dbm_multiply_gpu_context_t;

/*******************************************************************************
 * \brief Internal routine for initializing the gpu backend.
 * \author Ole Schuett
 ******************************************************************************/
void dbm_multiply_gpu_start(const int max_batch_size, const int nshards,
                            dbm_shard_t *shards_c_host,
                            dbm_multiply_gpu_context_t *ctx);

/*******************************************************************************
 * \brief Internal routine for uploading newly arrived packs onto the device.
 * \author Ole Schuett and Hans Pabst
 ******************************************************************************/
bool dbm_multiply_gpu_upload_packs(const dbm_pack_t *pack_a,
                                   const dbm_pack_t *pack_b,
                                   dbm_multiply_gpu_context_t *ctx);

/*******************************************************************************
 * \brief Internal routine for obtaining the calling thread's host batch.
 * \author Hans Pabst
 ******************************************************************************/
dbm_task_t *dbm_multiply_gpu_batch_acquire(dbm_multiply_gpu_context_t *ctx);

/*******************************************************************************
 * \brief Internal routine for returning the calling thread's host batch once
 *        no kernel reads it anymore.
 * \author Hans Pabst
 ******************************************************************************/
void dbm_multiply_gpu_batch_release(dbm_task_t *batch,
                                    dbm_multiply_gpu_context_t *ctx);

/*******************************************************************************
 * \brief Internal routine for executing the tasks in given batch on the GPU.
 *        Returns the batch to be filled next, i.e., the given one unless a
 *        kernel reads it directly (unified).
 * \author Ole Schuett
 ******************************************************************************/
dbm_task_t *dbm_multiply_gpu_process_batch(const int ntasks, dbm_task_t *batch,
                                           const dbm_batch_shape_t *shape,
                                           const double alpha,
                                           dbm_shard_t *shard_c,
                                           const int kshard, const bool finish,
                                           dbm_multiply_gpu_context_t *ctx);

/*******************************************************************************
 * \brief Internal routine for executing the tasks in given batch on the host
 *        while the GPU is busy with the shard's previous batch (hybrid).
 *        Returns false if the GPU shall process the batch instead.
 * \author Hans Pabst
 ******************************************************************************/
bool dbm_multiply_gpu_process_batch_host(
    const int ntasks, const dbm_task_t *batch, const dbm_batch_shape_t *shape,
    const double alpha, const dbm_pack_t *pack_a, const dbm_pack_t *pack_b,
    dbm_shard_t *shard_c, const int kshard, const bool finish,
    const int cpu_options, dbm_multiply_gpu_context_t *ctx);

/*******************************************************************************
 * \brief Internal routine for shutting down the gpu backend.
 * \author Ole Schuett
 ******************************************************************************/
void dbm_multiply_gpu_stop(dbm_multiply_gpu_context_t *ctx);

#endif // defined(__OFFLOAD) && !defined(__NO_OFFLOAD_DBM)
#endif

// EOF
