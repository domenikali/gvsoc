#include "pmsis.h"
#include <stdint.h>
#include <stdio.h>
#include <memory.h>
#include <bsp/bsp.h>
#include <stdlib.h>
#include <math.h>

#include "light_redmule.h"


#define NB_ITER 1

#define ERROR_THRES 0x09

#define M_SIZE 32
#define N_SIZE 32
#define K_SIZE 32

PI_L1 __attribute__((aligned(512))) uint16_t w [N_SIZE * K_SIZE];
PI_L1 __attribute__((aligned(512))) uint16_t x [M_SIZE * N_SIZE];
PI_L1 __attribute__((aligned(512))) uint16_t y [M_SIZE * K_SIZE];
PI_L1 __attribute__((aligned(512))) uint16_t z [M_SIZE * K_SIZE];
PI_L1 __attribute__((aligned(512))) uint16_t c [M_SIZE * K_SIZE];

void fma_uint16(uint16_t * z, uint16_t * y, uint16_t * x, uint16_t * w, uint16_t m_size, uint16_t n_size, uint16_t k_size){
    for (int i = 0; i < m_size; ++i)
    {
        for (int j = 0; j < k_size; ++j)
        {
            z[i * k_size + j] = y[i * k_size + j];
            for (int k = 0; k < n_size; ++k)
            {
                z[i * k_size + j] += x[i * n_size + k] * w[k * k_size + j];
            }
        }
    }
}


void init_matmul_data(){

  //Y matrix
  for (int i = 0; i < M_SIZE; ++i)
  {
    for (int j = 0; j < K_SIZE; ++j)
    {
      uint16_t row_id = j/16;
      uint16_t col_id = i/16;
      uint16_t mtx_id = col_id * ((K_SIZE + 15)/16) + row_id + 1;
      y[i*K_SIZE + j] = mtx_id;
    }
  }

  //X matrix
  for (int i = 0; i < M_SIZE; ++i)
  {
    for (int j = 0; j < N_SIZE; ++j)
    {
      uint16_t row_id = j/16;
      uint16_t col_id = i/16;
      uint16_t mtx_id = col_id * ((N_SIZE + 15)/16) + row_id + 1;
      x[i*N_SIZE + j] = mtx_id;
    }
  }

  //W matrix
  for (int i = 0; i < N_SIZE; ++i)
  {
    for (int j = 0; j < K_SIZE; ++j)
    {
      uint16_t row_id = j/16;
      uint16_t col_id = i/16;
      uint16_t mtx_id = col_id * ((K_SIZE + 15)/16) + row_id + 1;
      w[i*K_SIZE + j] = mtx_id;
    }
  }

  //C matrix
  for (int i = 0; i < M_SIZE; ++i)
  {
    for (int j = 0; j < K_SIZE; ++j)
    {
      c[i*K_SIZE + j] = 0;
    }
  }
}

void display_matrix(uint16_t* ptr, uint16_t row, uint16_t col){
  for (int i = 0; i < col; ++i)
  {
    for (int j = 0; j < row; ++j)
    {
      printf("%3d", ptr[i*row + j]);
    }
    printf("\n");
  }
}

static int glob_errors;

int run_test() {

  pi_perf_conf(1 << PI_PERF_CYCLES);
  pi_perf_reset();

  //Initialize RedMule with MNK and addresses
  init_matmul_data();
  light_redmule_set_M(M_SIZE);
  light_redmule_set_N(N_SIZE);
  light_redmule_set_K(K_SIZE);
  light_redmule_set_X((uint32_t)x);
  light_redmule_set_W((uint32_t)w);
  light_redmule_set_Y((uint32_t)y);
  light_redmule_set_Z((uint32_t)z);

  printf("Start Test Program\n");
  printf("MatMul Address:\n");
  printf("    X addr: 0x%x\n", (uint32_t)x);
  printf("    W addr: 0x%x\n", (uint32_t)w);
  printf("    Y addr: 0x%x\n", (uint32_t)y);
  printf("    Z addr: 0x%x\n", (uint32_t)z);

  fma_uint16(c,y,x,w,M_SIZE,N_SIZE,K_SIZE);

  printf("--- X matrix: \n");
  display_matrix(x,N_SIZE,M_SIZE);

  printf("--- Z matrix: \n");
  display_matrix(z,K_SIZE,M_SIZE);

  pi_perf_start();
  light_redmule_trigger_async();
  light_redmule_trigger_wait();
  pi_perf_stop();

  printf("--- Z matrix: \n");
  display_matrix(z,K_SIZE,M_SIZE);

  printf("--- C matrix: \n");
  display_matrix(c,K_SIZE,M_SIZE);
  printf("%d cycles\n", pi_perf_read(PI_PERF_CYCLES));

  return 0;
}

static struct pi_cluster_task task[1];
static struct pi_task events[1];

static void pe_entry(void *arg) {
  if(pi_core_id() == 0) {
    glob_errors = run_test();
  }
  pi_cl_team_barrier();
}

static void cluster_entry(void *arg) {
  pi_cl_team_fork(0, pe_entry, 0);
}

static int launch_cluster_task() {
  struct pi_device cluster_dev;
  struct pi_cluster_conf conf;
  struct pi_cluster_task task;

  pi_cluster_conf_init(&conf);
  conf.id = 0;
  glob_errors = 0;

  pi_open_from_conf(&cluster_dev, &conf);
  pi_cluster_open(&cluster_dev);

  pi_cluster_task(&task, cluster_entry, NULL);
  pi_cluster_send_task_to_cl(&cluster_dev, &task);
  pi_cluster_close(&cluster_dev);

  return glob_errors;
}

int test_entry() {
  printf("Starting test\n");
  int errors = launch_cluster_task();
  if (errors)
    printf("Test failure\n");
  else
    printf("Test success\n");
  return errors;
}

void test_kickoff(void *arg) {
  int ret = test_entry();
  pmsis_exit(ret);
}

int main() {
  return pmsis_kickoff((void *)test_kickoff);
}
