/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../offload/offload_runtime.h"
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_DBM)

#include "../offload/offload_library.h"
#include "../offload/offload_mempool.h"
#include "dbm_hyperparams.h"
#include "dbm_multiply_cpu.h"
#include "dbm_multiply_gpu.h"
#include "dbm_multiply_gpu_kernel.h"

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Internal routine for initializing the gpu backend.
 * \author Ole Schuett
 ******************************************************************************/
void dbm_multiply_gpu_start(const int max_batch_size, const int nshards,
                            dbm_shard_t *shards_c_host,
                            dbm_multiply_gpu_context_t *ctx) {
  // Select GPU device.
  offload_activate_chosen_device();

  ctx->nshards = nshards;
  ctx->max_batch_size = max_batch_size;
  ctx->shards_c_host = shards_c_host;

  // Up to |DBM_MULTIPLY_HYBRID| threads compute on the host while the GPU is
  // busy, and a negative value reports the share of the host.
  const char *const hybrid_env = getenv("DBM_MULTIPLY_HYBRID");
  ctx->hybrid = (NULL == hybrid_env ? 0 : atoi(hybrid_env));
  static bool hybrid_reported = false; // once per process
  if (0 > ctx->hybrid && !hybrid_reported &&
      !dbm_multiply_cpu_generated(0, 0, 0, 1.0, 0)) {
    fprintf(stderr, "INFO DBM: no generated host kernels, hence no hybrid\n");
    hybrid_reported = true;
  }
  ctx->hybrid_active = 0;
  ctx->flops[0] = ctx->flops[1] = 0;
  offloadStreamCreate(&ctx->main_stream);
  offloadEventCreate(&ctx->upload_event);

  // Kernels read host batches directly if requested and possible, which saves
  // an upload per batch (DBM_MULTIPLY_UNIFIED, default: unified memory build).
  const char *const unified_env = getenv("DBM_MULTIPLY_UNIFIED");
#if defined(__OFFLOAD_UNIFIED_MEMORY)
  const bool unified = (NULL == unified_env || 0 != atoi(unified_env));
#else
  const bool unified = (NULL != unified_env && 0 != atoi(unified_env));
#endif
  ctx->unified = (unified && offloadHostMemoryDeviceAccessible());
  static bool unified_reported = false; // once per process
  if (unified && !ctx->unified && NULL != unified_env && !unified_reported) {
    fprintf(stderr, "INFO DBM: host memory is not device-accessible, hence "
                    "batches are uploaded\n");
    unified_reported = true;
  }
  ctx->nthreads = (ctx->unified ? omp_get_max_threads() : 0);
  ctx->batches_host =
      (0 < ctx->nthreads ? calloc(ctx->nthreads, sizeof(dbm_batch_gpu_t))
                         : NULL);
  assert(NULL != ctx->batches_host || 0 == ctx->nthreads);

  // Allocate device storage for batches (unless kernels read host batches).
  const size_t size = nshards * max_batch_size * sizeof(dbm_task_t);
  ctx->batches_dev =
      (ctx->unified ? NULL : offload_mempool_device_malloc(size));

  // Allocate and upload shards of result matrix C.
  ctx->shards_c_dev = malloc(nshards * sizeof(dbm_shard_gpu_t));
  assert(ctx->shards_c_dev != NULL || nshards == 0);
  for (int i = 0; i < nshards; i++) {
    const dbm_shard_t *const shard_c_host = &shards_c_host[i];
    dbm_shard_gpu_t *const shard_g = &ctx->shards_c_dev[i];
    shard_g->data_size = shard_c_host->data_size;
    offloadStreamCreate(&shard_g->stream);
    offloadEventCreate(&shard_g->event);
    offloadEventCreate(&shard_g->done);
    offloadEventCreate(&shard_g->download);
    shard_g->downloading = false;
    shard_g->host_data = NULL;
    shard_g->host_size = shard_g->host_allocated = 0;
    // only allocate data_size on device rather than data_allocated
    shard_g->data_allocated = shard_c_host->data_size;
    shard_g->data =
        offload_mempool_device_malloc(shard_g->data_allocated * sizeof(double));
    offloadMemcpyAsyncHtoD(shard_g->data, shard_c_host->data,
                           shard_g->data_size * sizeof(double),
                           shard_g->stream);
  }
}

/*******************************************************************************
 * \brief Private routine for uploading a single pack onto the device.
 * \author Ole Schuett
 ******************************************************************************/
static void upload_pack(const dbm_pack_t *pack_host, dbm_pack_gpu_t *pack_dev,
                        const offloadStream_t stream) {
  // Reallocate only when the device buffer is too small; data_allocated tracks
  // the device capacity independently from the host-side data_size that changes
  // each tick (previous code compared against the wrong field).
  const size_t size = pack_host->data_size * sizeof(double);
  if (pack_dev->data_allocated < pack_host->data_size) {
    offload_mempool_device_free(pack_dev->data);
    pack_dev->data = offload_mempool_device_malloc(size);
    pack_dev->data_allocated = pack_host->data_size;
  }
  offloadMemcpyAsyncHtoD(pack_dev->data, pack_host->data, size, stream);
}

/*******************************************************************************
 * \brief Internal routine for uploading newly arrived packs onto the device.
 * \author Ole Schuett and Hans Pabst
 ******************************************************************************/
bool dbm_multiply_gpu_upload_packs(const dbm_pack_t *pack_a,
                                   const dbm_pack_t *pack_b,
                                   dbm_multiply_gpu_context_t *ctx) {
  // Assume GPU device was activated earlier.
  // Wait for all c-streams to complete before overwriting old packs.
  for (int i = 0; i < ctx->nshards; i++) {
    offloadEventRecord(ctx->upload_event, ctx->shards_c_dev[i].stream);
    offloadStreamWaitEvent(ctx->main_stream, ctx->upload_event);
  }
  // Record event to check if all c-streams already completed.
  offloadEventRecord(ctx->upload_event, ctx->main_stream);

  bool uploaded = false;
  /*if (offloadEventQuery(ctx->upload_event))*/
  {
    upload_pack(pack_a, &ctx->pack_a_dev, ctx->main_stream);
    upload_pack(pack_b, &ctx->pack_b_dev, ctx->main_stream);

    // Have all c-streams wait until new packs are uploaded.
    offloadEventRecord(ctx->upload_event, ctx->main_stream);
    for (int i = 0; i < ctx->nshards; i++) {
      offloadStreamWaitEvent(ctx->shards_c_dev[i].stream, ctx->upload_event);
    }
    uploaded = true;
  }

  return uploaded;
}

/*******************************************************************************
 * \brief Internal routine for obtaining the calling thread's host batch.
 * \author Hans Pabst
 ******************************************************************************/
dbm_task_t *dbm_multiply_gpu_batch_acquire(dbm_multiply_gpu_context_t *ctx) {
  const size_t size = ctx->max_batch_size * sizeof(dbm_task_t);
  dbm_task_t *result = NULL;
  if (ctx->unified) { // a pair per thread, kept until the backend stops
    const int tid = omp_get_thread_num();
    assert(tid < ctx->nthreads);
    dbm_batch_gpu_t *const pair = &ctx->batches_host[tid];
    if (NULL == pair->batch[0]) {
      for (int i = 0; i < 2; ++i) {
        pair->batch[i] = offload_mempool_host_malloc(size);
        offloadEventCreate(&pair->done[i]);
        pair->pending[i] = false;
      }
    }
    if (pair->pending[0]) { // kernel of a previous pack may still read it
      offloadEventSynchronize(pair->done[0]);
      pair->pending[0] = false;
    }
    result = pair->batch[0];
  } else {
    result = offload_mempool_host_malloc(size);
  }
  assert(NULL != result);
  return result;
}

/*******************************************************************************
 * \brief Internal routine for returning the calling thread's host batch once
 *        no kernel reads it anymore.
 * \author Hans Pabst
 ******************************************************************************/
void dbm_multiply_gpu_batch_release(dbm_task_t *batch,
                                    dbm_multiply_gpu_context_t *ctx) {
  if (ctx->unified) { // the pair stays for the next pack
    dbm_batch_gpu_t *const pair = &ctx->batches_host[omp_get_thread_num()];
    assert(batch == pair->batch[0] || batch == pair->batch[1]);
    (void)batch; // mark used
  } else {
    offload_mempool_host_free(batch);
  }
}

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
                                           dbm_multiply_gpu_context_t *ctx) {
  // Assume GPU device was activated earlier.
  dbm_shard_gpu_t *const shard_g = &ctx->shards_c_dev[kshard];
  dbm_task_t *const batch_dev =
      (ctx->unified ? batch : &ctx->batches_dev[kshard * ctx->max_batch_size]);
  double *old_data_dev = NULL;

  dbm_task_t *result = batch;
  bool grown = false;

  if (0 < ntasks && !ctx->unified) {
    assert(NULL != shard_c && NULL != shard_g);

    // Upload new batch.
    const size_t size = ntasks * sizeof(dbm_task_t);
    offloadMemcpyAsyncHtoD(batch_dev, batch, size, shard_g->stream);
  }

  // Blocks promised by batches computed on the host grow the shard at finish.
  if (0 < ntasks || finish) {
    // Reallocate shard_g->data if necessary.
    if (shard_c->data_promised > shard_g->data_allocated) {
      shard_g->data_allocated = DBM_ALLOCATION_FACTOR * shard_c->data_promised;
      assert(shard_c->data_promised <= shard_g->data_allocated);
      old_data_dev = shard_g->data;
      shard_g->data = offload_mempool_device_malloc(shard_g->data_allocated *
                                                    sizeof(double));
      // Omit to wait for copy before freeing old buffer.
      offloadMemcpyAsyncDtoD(shard_g->data, old_data_dev,
                             shard_g->data_size * sizeof(double),
                             shard_g->stream);
      grown = true;
    }
    if ((0 < ntasks && !ctx->unified) || grown) {
      offloadEventRecord(shard_g->event, shard_g->stream);
    }

    // Zero the headroom once per allocation rather than new blocks per batch:
    // a memset is mostly launch latency, and kernels write promised blocks.
    if (grown) {
      const size_t headroom = shard_g->data_allocated - shard_g->data_size;
      offloadMemsetAsync(&shard_g->data[shard_g->data_size], 0,
                         headroom * sizeof(double), shard_g->stream);
    }
    if (shard_c->data_promised > shard_g->data_size) {
      shard_g->data_size = shard_c->data_promised; // within zeroed headroom
    }
  }

  if (0 < ntasks) {
    OFFLOAD_CHECK(offloadGetLastError());
    assert(0 != shard_g->data_size);

    // Launch kernel.
    dbm_multiply_gpu_launch_kernel(shard_g->stream, alpha, ntasks, shape, batch,
                                   batch_dev, ctx->pack_a_dev.data,
                                   ctx->pack_b_dev.data, shard_g->data);
    OFFLOAD_CHECK(offloadGetLastError());
    offloadEventRecord(shard_g->done, shard_g->stream);
    if (0 > ctx->hybrid) {
#pragma omp atomic
      ctx->flops[0] += shape->flops;
    }
    if (ctx->unified) { // fill the other batch while the kernel reads this one
      dbm_batch_gpu_t *const pair = &ctx->batches_host[omp_get_thread_num()];
      const int i = (batch == pair->batch[0] ? 0 : 1), j = 1 - i;
      assert(batch == pair->batch[i]);
      offloadEventRecord(pair->done[i], shard_g->stream);
      pair->pending[i] = true;
      if (pair->pending[j]) {
        offloadEventSynchronize(pair->done[j]);
        pair->pending[j] = false;
      }
      result = pair->batch[j];
    }
  }

  if (finish) { // Start downloading the current shard of matrix_c.
    // Growing the host buffer frees the old one, into which the previous
    // download may still write: batches computed on the host never wait.
    if (shard_g->downloading &&
        shard_c->data_allocated < shard_c->data_promised) {
      offloadEventSynchronize(shard_g->download);
      shard_g->downloading = false;
    }
    // Grow host buffer if necessary.
    dbm_shard_allocate_promised_blocks(shard_c);
    // Download results from device.
    assert(shard_c->data_size == shard_g->data_size);
    offloadMemcpyAsyncDtoH(shard_c->data, shard_g->data,
                           shard_g->data_size * sizeof(double),
                           shard_g->stream);
    offloadEventRecord(shard_g->download, shard_g->stream);
    shard_g->downloading = true;
  }

  if ((0 < ntasks && !ctx->unified) || grown) {
    // Wait for:
    // - Batch to be uploaded (before refilling it).
    // - Safely freeing device buffer (if resized).
    offloadEventSynchronize(shard_g->event);

    if (NULL != old_data_dev) {
      offload_mempool_device_free(old_data_dev);
    }
  }

  return result;
}

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
    const int cpu_options, dbm_multiply_gpu_context_t *ctx) {
  dbm_shard_gpu_t *const shard_g = &ctx->shards_c_dev[kshard];
  bool result = false;

  // Only generated kernels let the host rival the GPU.
  if (0 != ctx->hybrid && 0 < ntasks && !offloadEventQuery(shard_g->done) &&
      dbm_multiply_cpu_generated(shape->max_m, shape->max_n, shape->max_k,
                                 alpha, cpu_options)) {
    int active;
#pragma omp atomic capture
    active = ctx->hybrid_active++;
    result = (active < abs(ctx->hybrid));
    if (result) {
      // The contributions start from zero and are added to C at the end.
      if (shard_g->host_size < shard_c->data_promised) {
        if (shard_g->host_allocated < shard_c->data_promised) {
          shard_g->host_allocated =
              DBM_ALLOCATION_FACTOR * shard_c->data_promised;
          shard_g->host_data = realloc(
              shard_g->host_data, shard_g->host_allocated * sizeof(double));
          assert(NULL != shard_g->host_data);
        }
        memset(&shard_g->host_data[shard_g->host_size], 0,
               (shard_c->data_promised - shard_g->host_size) * sizeof(double));
        shard_g->host_size = shard_c->data_promised;
      }
      dbm_multiply_cpu_process_tasks(ntasks, batch, alpha, pack_a, pack_b,
                                     shard_g->host_data, cpu_options);
      if (0 > ctx->hybrid) {
#pragma omp atomic
        ctx->flops[1] += shape->flops;
      }
    }
#pragma omp atomic
    --ctx->hybrid_active;
    if (result && finish) { // download as if the GPU processed the batch
      dbm_multiply_gpu_process_batch(0, NULL, shape, alpha, shard_c, kshard,
                                     true, ctx);
    }
  }

  return result;
}

/*******************************************************************************
 * \brief Internal routine for shutting down the gpu backend.
 * \author Ole Schuett
 ******************************************************************************/
void dbm_multiply_gpu_stop(dbm_multiply_gpu_context_t *ctx) {
  // Assume GPU device was activated earlier.
  // Wait for completion, then free gpu ressources.
#pragma omp parallel for DBM_OMP_SCHEDULE
  for (int i = 0; i < ctx->nshards; i++) {
    dbm_shard_gpu_t *const shard_g = &ctx->shards_c_dev[i];
    offloadStreamSynchronize(shard_g->stream);
    if (NULL != shard_g->host_data) { // add contributions computed on the host
      double *const data = ctx->shards_c_host[i].data;
      assert(shard_g->host_size <= ctx->shards_c_host[i].data_size);
      for (int j = 0; j < shard_g->host_size; j++) {
        data[j] += shard_g->host_data[j];
      }
      free(shard_g->host_data);
    }
    offloadStreamDestroy(shard_g->stream);
    offloadEventDestroy(shard_g->event);
    offloadEventDestroy(shard_g->done);
    offloadEventDestroy(shard_g->download);
    offload_mempool_device_free(shard_g->data);
  }
  free(ctx->shards_c_dev);

  // All streams are synchronized, hence no kernel reads host batches anymore.
  for (int i = 0; i < ctx->nthreads; i++) {
    dbm_batch_gpu_t *const pair = &ctx->batches_host[i];
    if (NULL != pair->batch[0]) {
      for (int j = 0; j < 2; ++j) {
        offloadEventDestroy(pair->done[j]);
        offload_mempool_host_free(pair->batch[j]);
      }
    }
  }
  free(ctx->batches_host);

  const int64_t flops = ctx->flops[0] + ctx->flops[1];
  if (0 > ctx->hybrid && 0 < flops) {
    fprintf(stderr, "INFO DBM: %.0f%% of %.1f GFLOP computed on the host\n",
            100.0 * ctx->flops[1] / flops, 1E-9 * flops);
  }

  offload_mempool_device_free(ctx->pack_a_dev.data);
  offload_mempool_device_free(ctx->pack_b_dev.data);
  offload_mempool_device_free(ctx->batches_dev);
  offloadStreamDestroy(ctx->main_stream);
  offloadEventDestroy(ctx->upload_event);
}

#endif // defined(__OFFLOAD) && !defined(__NO_OFFLOAD_DBM)

// EOF
