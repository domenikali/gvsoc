#include "pmsis.h"
#include <stdint.h>
#include <stdio.h>
#include <memory.h>
#include <bsp/bsp.h>
#include <stdlib.h>
#include <math.h>


#define NB_ITER 1

#define ERROR_THRES 0x09

#define M_SIZE 16
#define N_SIZE 16
#define K_SIZE 16

PI_L1 __attribute__((aligned(32))) uint16_t w [N_SIZE * K_SIZE];
PI_L1 __attribute__((aligned(32))) uint16_t x [M_SIZE * N_SIZE];
PI_L1 __attribute__((aligned(32))) uint16_t y [M_SIZE * K_SIZE];
PI_L1 __attribute__((aligned(32))) uint16_t z [M_SIZE * K_SIZE];



// void check_errors() {
//   for (int i = 0; i < M_SIZE; i++) {
//     for (int j = 0; j < K_SIZE; j++) {
//       if (abs(z_gold[i * K_SIZE + j] - z[i * K_SIZE + j]) >= ERROR_THRES) {
//         printf("DATA MISMATCH: (%d,%d),\tdiff: 0x%x\texpected: 0x%x\twas: 0x%x\n", i, j, abs(z_gold[i * K_SIZE + j] - z[i * K_SIZE + j]), z_gold[i * K_SIZE + j], z[i * K_SIZE + j]);
//       }
//     }
//   }
// }

static int glob_errors;

int run_test() {

  pi_perf_conf(1 << PI_PERF_CYCLES);
  pi_perf_reset();

  printf("Start Test Program\n"); 
  pi_perf_start();
  pi_perf_stop();
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
