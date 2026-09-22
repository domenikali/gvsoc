#ifdef __pulp_cluster__

#include <carfield_lib/dma.h>

#include "pulp.h"

dma_transfer_id_t dma_transfer_create() { return plp_dma_counter_alloc(); }

void dma_transfer_free(dma_transfer_id_t transfer) {
  plp_dma_counter_free(transfer);
}

void dma_transfer_wait(dma_transfer_id_t transfer) {
  plp_dma_wait(transfer);
  // free in plp_dma_wait
}

void dma_transfer_hwc_to_chw(dma_transfer_cfg_t conf) {
  int start_fm_channel = 0, end_fm_channel = conf.length_1d_copy;
  void *loc = conf.loc + conf.number_of_1d_copies * conf.number_of_2d_copies *
                             start_fm_channel;
  void *ext = conf.ext + start_fm_channel;
  const int size_2d = conf.number_of_1d_copies * conf.number_of_2d_copies;

  for (int i = start_fm_channel; i < end_fm_channel; i++) {
    unsigned int dma_cmd = plp_dma_getCmd(conf.dir, size_2d, 1, 1, 1, 1);
    plp_dma_cmd_push_2d(dma_cmd, loc, ext, conf.stride_1d, 1);
    ext += 1; // next channel // TODO check if this is correct for other types different than uint8_t
    loc += conf.number_of_1d_copies * conf.number_of_2d_copies;
  }
}

void dma_transfer_1d_async(dma_transfer_cfg_t conf) {
  unsigned int dma_cmd =
      plp_dma_getCmd(conf.dir, conf.length_1d_copy, 0, 1, 1, 1);
  plp_dma_cmd_push(dma_cmd, conf.loc, conf.ext);
}

void dma_transfer_2d_async(dma_transfer_cfg_t conf) {
  const int size_2d = conf.number_of_1d_copies * conf.length_1d_copy;
  unsigned int dma_cmd = plp_dma_getCmd(conf.dir, size_2d, 1, 1, 1, 1);
  plp_dma_cmd_push_2d(dma_cmd, conf.loc, conf.ext, conf.stride_1d,
                      conf.length_1d_copy);
}

void dma_transfer_3d_async(dma_transfer_cfg_t conf) {
  const int size_2d = conf.number_of_1d_copies * conf.length_1d_copy;

  for (int i = 0; i < conf.number_of_2d_copies; i++) {
    dma_transfer_2d_async(conf);
    conf.loc += size_2d;
    conf.ext += conf.stride_2d;
  }
}

void dma_transfer_async(dma_transfer_cfg_t conf) {
  if (conf.hwc_to_chw == 1) {
    dma_transfer_hwc_to_chw(conf);
  } else if (conf.number_of_2d_copies == 1 && conf.number_of_1d_copies == 1) {
    dma_transfer_1d_async(conf);
  } else if (conf.number_of_2d_copies == 1) {
    dma_transfer_2d_async(conf);
  } else {
    dma_transfer_3d_async(conf);
  }
}

static uint32_t dma_mutex;

void dma_mutex_init() {
  dma_mutex = eu_mutex_addr(0);
}

void dma_mutex_lock() {
  eu_mutex_lock(dma_mutex);
}

void dma_mutex_unlock() {
  eu_mutex_unlock(dma_mutex);
}

#else

#include "dma.h"

#include <stddef.h>
#include <string.h>

/*
 * The standalone mailbox dma tests
 */
static void dma_copy(dma_transfer_cfg_t conf)
{
  uint8_t *ext = (uint8_t *)(uintptr_t)conf.ext;
  uint8_t *loc = (uint8_t *)(uintptr_t)conf.loc;
  size_t length = (size_t)conf.length_1d_copy;
  int copies = conf.number_of_1d_copies > 0 ? conf.number_of_1d_copies : 1;
  int stride = conf.stride_1d > 0 ? conf.stride_1d : (int)length;

  for (int copy = 0; copy < copies; copy++) {
    uint8_t *src = conf.dir == DMA_DIR_L2_TO_L1 ? ext : loc;
    uint8_t *dst = conf.dir == DMA_DIR_L2_TO_L1 ? loc : ext;
    memcpy(dst + copy * stride, src + copy * stride, length);
  }
}

void dma_transfer_1d_async(dma_transfer_cfg_t conf) { dma_copy(conf); }

void dma_transfer_2d_async(dma_transfer_cfg_t conf)
{
  int rows = conf.number_of_2d_copies > 0 ? conf.number_of_2d_copies : 1;
  int stride = conf.stride_2d > 0 ? conf.stride_2d : conf.length_1d_copy;

  for (int row = 0; row < rows; row++) {
    dma_transfer_cfg_t row_conf = conf;
    row_conf.ext += (uint32_t)(row * stride);
    row_conf.loc += (uint32_t)(row * stride);
    row_conf.number_of_2d_copies = 1;
    dma_copy(row_conf);
  }
}

void dma_transfer_3d_async(dma_transfer_cfg_t conf)
{
  int planes = conf.number_of_2d_copies > 0 ? conf.number_of_2d_copies : 1;
  int stride = conf.stride_2d > 0 ? conf.stride_2d : conf.length_1d_copy;

  for (int plane = 0; plane < planes; plane++) {
    dma_transfer_cfg_t plane_conf = conf;
    plane_conf.ext += (uint32_t)(plane * stride);
    plane_conf.loc += (uint32_t)(plane * stride);
    plane_conf.number_of_2d_copies = 1;
    dma_transfer_2d_async(plane_conf);
  }
}

void dma_transfer_async(dma_transfer_cfg_t conf)
{
  if (conf.hwc_to_chw) {
    dma_transfer_1d_async(conf);
  } else if (conf.number_of_2d_copies > 1) {
    dma_transfer_3d_async(conf);
  } else if (conf.number_of_1d_copies > 1) {
    dma_transfer_2d_async(conf);
  } else {
    dma_transfer_1d_async(conf);
  }
}

void dma_transfer_hwc_to_chw(dma_transfer_cfg_t conf)
{
  dma_transfer_1d_async(conf);
}

dma_transfer_id_t dma_transfer_create(void) { return 0; }
void dma_transfer_free(dma_transfer_id_t transfer) { (void)transfer; }
void dma_transfer_wait(dma_transfer_id_t transfer) { (void)transfer; }
void dma_mutex_init(void) {}
void dma_mutex_lock(void) {}
void dma_mutex_unlock(void) {}

#endif // __pulp_cluster__