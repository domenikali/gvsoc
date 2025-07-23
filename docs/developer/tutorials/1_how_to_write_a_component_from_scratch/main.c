#include <stdio.h>
#include <stdint.h>
#include "alloc.h"

//pcm adresses
#define PCM_ADDR            0x20001000
#define AIMC_CMD_OFFSET     0x00000000
#define AIMC_VECTOR_OFFSET  0x00000004
#define PCM_SIZE_OFFSET     0x00000200

//aimc cmd
#define AIMC_SETTINGS       0x10000000
#define AIMC_COMPUTE        0x00000000
#define SECTOR_MASK         0x00FFFF00


typedef struct
{
  uint32_t *pcm_addr;
  uint32_t *aimc_cmd_addr;
  uint32_t *aimc_vector_addr;
  uint32_t *pcm_matrix_addr;
}pcm_t;

pcm_t* init_pcm() {
  pcm_t *pcm = (pcm_t *)pi_malloc(sizeof(pcm_t));
  pcm->pcm_addr =(uint32_t*) PCM_ADDR;
  pcm->aimc_cmd_addr = (uint32_t*)(PCM_ADDR + AIMC_CMD_OFFSET);
  pcm->aimc_vector_addr = (uint32_t*)(PCM_ADDR + AIMC_CMD_OFFSET+ AIMC_VECTOR_OFFSET);
  pcm->pcm_matrix_addr = (uint32_t*)(PCM_ADDR + PCM_SIZE_OFFSET+AIMC_CMD_OFFSET+AIMC_VECTOR_OFFSET);
  return pcm;
}
 

static inline uint32_t pcm_read_32(uint32_t *pcm, uint32_t offset) {
  return *(volatile uint32_t *)(pcm + offset);
}
  
static inline void pcm_write_32(uint32_t *pcm, uint32_t offset, uint32_t value) {
  *(volatile uint32_t *)(pcm + offset) = value;
}

static inline void pcm_write_64(uint32_t *pcm, uint64_t offset, uint64_t value) {
  *(volatile uint64_t *)(pcm + offset) = value;
}

static inline uint64_t pcm_read_64(uint32_t *pcm, uint64_t offset) {
  return *(volatile uint64_t *)(pcm + offset);
}

void load_Xi_vect(){

}

int main(){   
  pcm_t *pcm=init_pcm();
  uint32_t *ad =(uint32_t*)(PCM_ADDR);
  if (pcm == NULL) {
    printf("Failed to initialize PCM\n");
    return -1;
  }

  //write and read value to the pcm Xi vector
  pcm_write_32(pcm->aimc_vector_addr, 0x00000000, 0x0101010101010101); 
  // uint64_t value = pcm_read_64(pcm->aimc_vector_addr, 0x000000000);
  // printf("Read value: 0x%xl\n", value);

  // //write and read to the pcm matrix
  // pcm_write_64(pcm->pcm_matrix_addr, 0x00000000, 0x1010101010101010); 
  // uint64_t val = pcm_read_64(pcm->pcm_matrix_addr, 0x00000000);
  // printf("Read value: 0x%lx\n", val);
  /*
     3                   2                   1                   0
   1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |  cmd  |sub cmd| arr 1 | arr 2 | arr 3 | arr 4 |     layers    | 
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  */
  
  //set sectors       
  uint32_t sectors = 0b00010000101010101010101000000000;  
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,sectors);
  //start computation
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,AIMC_COMPUTE);




  return 0;
}
