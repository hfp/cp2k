/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../offload/offload_runtime.h"
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_DBM)

#include "../offload/offload_library.h"
#include "../offload/offload_mempool.h"
#include "dbm_hyperparams.h"
#include "dbm_multiply_gpu.h"
#include "dbm_multiply_gpu_kernel.h"

#include <assert.h>
#include <stdio.h>

#define DBM_MULTIPLY_GPU_STAGES 2

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
  offloadStreamCreate(&ctx->main_stream);
  offloadEventCreate(&ctx->upload_event);

  const size_t batch_bytes = (size_t)max_batch_size * sizeof(dbm_task_t);
  const size_t batches_bytes = (size_t)nshards * batch_bytes;
  ctx->batches_dev = offload_mempool_device_malloc(batches_bytes);

  if (0 < nshards) {
    ctx->batches_host =
        offload_mempool_host_malloc(DBM_MULTIPLY_GPU_STAGES * batches_bytes);
    ctx->batch_events = malloc((size_t)nshards * DBM_MULTIPLY_GPU_STAGES *
                               sizeof(offloadEvent_t));
    ctx->batch_slot_ready = malloc((size_t)nshards * DBM_MULTIPLY_GPU_STAGES *
                                   sizeof(unsigned char));
    ctx->batch_slot_index = malloc((size_t)nshards * sizeof(int));
    assert(NULL != ctx->batches_host || 0 == batches_bytes);
    assert(NULL != ctx->batch_events || 0 == batches_bytes);
    assert(NULL != ctx->batch_slot_ready || 0 == batches_bytes);
    assert(NULL != ctx->batch_slot_index || 0 == batches_bytes);
    for (int i = 0; i < nshards; ++i) {
      ctx->batch_slot_index[i] = 0;
      for (int s = 0; s < DBM_MULTIPLY_GPU_STAGES; ++s) {
        const size_t idx = (size_t)i * DBM_MULTIPLY_GPU_STAGES + s;
        offloadEventCreate(&ctx->batch_events[idx]);
        ctx->batch_slot_ready[idx] = 1;
      }
    }
  } else {
    ctx->batches_host = NULL;
    ctx->batch_events = NULL;
    ctx->batch_slot_ready = NULL;
    ctx->batch_slot_index = NULL;
  }

  // Allocate and upload shards of result matrix C.
  ctx->shards_c_dev = malloc(nshards * sizeof(dbm_shard_gpu_t));
  assert(ctx->shards_c_dev != NULL || nshards == 0);
  for (int i = 0; i < nshards; i++) {
    const dbm_shard_t *const shard_c_host = &shards_c_host[i];
    dbm_shard_gpu_t *const shard_g = &ctx->shards_c_dev[i];
    shard_g->data_size = shard_c_host->data_size;
    offloadStreamCreate(&shard_g->stream);
    offloadEventCreate(&shard_g->event);
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
static void upload_pack(const dbm_pack_t *pack_host, dbm_pack_t *pack_dev,
                        const offloadStream_t stream) {
  const size_t size = pack_host->data_size * sizeof(double);
  if (pack_dev->data_size < pack_host->data_size) {
    offload_mempool_device_free(pack_dev->data);
    pack_dev->data = offload_mempool_device_malloc(size);
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
 * \brief Internal routine for executing the tasks in given batch on the GPU.
 * \author Ole Schuett
 ******************************************************************************/
void dbm_multiply_gpu_process_batch(const int ntasks, const dbm_task_t *tasks,
                                    const double alpha, dbm_shard_t *shard_c,
                                    const int kshard, const bool finish,
                                    dbm_multiply_gpu_context_t *ctx) {
  // Assume GPU device was activated earlier.
  dbm_shard_gpu_t *const shard_g = &ctx->shards_c_dev[kshard];
  double *old_data_dev = NULL;

  if (0 < ntasks) {
    assert(NULL != shard_c && NULL != shard_g);
    assert(ntasks <= ctx->max_batch_size);
    assert(NULL != ctx->batches_host);

    // Prepare double-buffered upload.
    dbm_task_t *const batch = &ctx->batches_dev[kshard * ctx->max_batch_size];
    const size_t batch_size = (size_t)ntasks * sizeof(dbm_task_t);
    const int slot_count = DBM_MULTIPLY_GPU_STAGES;
    const size_t base = (size_t)kshard * slot_count;
    const int slot = ctx->batch_slot_index[kshard];
    const size_t slot_index = base + (size_t)slot;
    dbm_task_t *const batch_host =
        ctx->batches_host + slot_index * ctx->max_batch_size;
    unsigned char *const slot_ready = &ctx->batch_slot_ready[slot_index];
    if (!*slot_ready) {
      offloadEventSynchronize(ctx->batch_events[slot_index]);
      *slot_ready = 1;
    }
    memcpy(batch_host, tasks, batch_size);
    *slot_ready = 0;

    // Upload new batch.
    offloadMemcpyAsyncHtoD(batch, batch_host, batch_size, shard_g->stream);
    offloadEventRecord(ctx->batch_events[slot_index], shard_g->stream);
    ctx->batch_slot_index[kshard] = (slot + 1) % slot_count;

    // Reallocate shard's data if necessary.
    if (shard_g->data_allocated < shard_c->data_promised) {
      shard_g->data_allocated = DBM_ALLOCATION_FACTOR * shard_c->data_promised;
      assert(shard_c->data_promised <= shard_g->data_allocated);
      old_data_dev = shard_g->data;
      shard_g->data = offload_mempool_device_malloc(shard_g->data_allocated *
                                                    sizeof(double));
      // Omit to wait for copy before freeing old buffer.
      offloadMemcpyAsyncDtoD(shard_g->data, old_data_dev,
                             shard_g->data_size * sizeof(double),
                             shard_g->stream);
    }
    offloadEventRecord(shard_g->event, shard_g->stream);

    // Zero new blocks if necessary.
    if (shard_g->data_size < shard_c->data_promised) {
      const int tail = shard_c->data_promised - shard_g->data_size;
      offloadMemsetAsync(&shard_g->data[shard_g->data_size], 0,
                         tail * sizeof(double), shard_g->stream);
      shard_g->data_size = shard_c->data_promised;
    }

    OFFLOAD_CHECK(offloadGetLastError());
    assert(0 != shard_g->data_size);

    // Launch kernel.
    dbm_multiply_gpu_launch_kernel(shard_g->stream, alpha, ntasks, tasks, batch,
                                   ctx->pack_a_dev.data, ctx->pack_b_dev.data,
                                   shard_g->data);
    OFFLOAD_CHECK(offloadGetLastError());
  }

  if (finish) { // Start downloading the current shard of matrix_c.
    // Grow host buffer if necessary.
    dbm_shard_allocate_promised_blocks(shard_c);
    // Download results from device.
    assert(shard_c->data_size == shard_g->data_size);
    offloadMemcpyAsyncDtoH(shard_c->data, shard_g->data,
                           shard_g->data_size * sizeof(double),
                           shard_g->stream);

    if (NULL == old_data_dev) {
      offloadEventSynchronize(shard_g->event);
    }
  }

  if (NULL != old_data_dev) {
    offloadEventSynchronize(shard_g->event);
    offload_mempool_device_free(old_data_dev);
  }
}

/*******************************************************************************
 * \brief Internal routine for shutting down the gpu backend.
 * \author Ole Schuett
 ******************************************************************************/
void dbm_multiply_gpu_stop(dbm_multiply_gpu_context_t *ctx) {
  // Assume GPU device was activated earlier.
  // Wait for completion, then free gpu ressources.
#pragma omp parallel for
  for (int i = 0; i < ctx->nshards; i++) {
    dbm_shard_gpu_t *const shard_g = &ctx->shards_c_dev[i];
    offloadStreamSynchronize(shard_g->stream);
    offloadStreamDestroy(shard_g->stream);
    offloadEventDestroy(shard_g->event);
    offload_mempool_device_free(shard_g->data);
  }
  free(ctx->shards_c_dev);

  if (NULL != ctx->batch_events) {
    const size_t total_slots = (size_t)ctx->nshards * DBM_MULTIPLY_GPU_STAGES;
    for (size_t i = 0; i < total_slots; ++i) {
      offloadEventDestroy(ctx->batch_events[i]);
    }
    free(ctx->batch_events);
  }
  free(ctx->batch_slot_ready);
  free(ctx->batch_slot_index);
  if (NULL != ctx->batches_host) {
    offload_mempool_host_free(ctx->batches_host);
  }

  offload_mempool_device_free(ctx->pack_a_dev.data);
  offload_mempool_device_free(ctx->pack_b_dev.data);
  offload_mempool_device_free(ctx->batches_dev);
  offloadStreamDestroy(ctx->main_stream);
  offloadEventDestroy(ctx->upload_event);
}

#endif // defined(__OFFLOAD) && !defined(__NO_OFFLOAD_DBM)

// EOF
