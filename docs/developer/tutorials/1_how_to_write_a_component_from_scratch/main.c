#include <stdio.h>
#include <stdint.h>
#include "alloc.h"

//pcm adresses
#define PCM_ADDR            0x20001000
#define AIMC_CMD_OFFSET     0x00000000
#define AIMC_VECTOR_OFFSET  0x00000001
#define PCM_SIZE_OFFSET     0x00000080

//aimc cmd
#define AIMC_SETTINGS       0x10000000
#define AIMC_COMPUTE        0x00000000
#define SECTOR_MASK         0x00FFFF00

//usefull macros for std module
#define PCM_MATRIX_SIZE 0x200000
#define XI_VECTOR_SIZE 0x200

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
  *((volatile uint64_t *)pcm + offset) =value;
}

static inline uint64_t pcm_read_64(uint32_t *pcm, uint64_t offset) {
  return *(volatile uint64_t *)(pcm + offset);
}

void load_Xi_vect(){
  uint32_t value = 0x01010101;
  for(size_t i=0;i<XI_VECTOR_SIZE/4;i++){
    pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i, value);
  }
}

void load_pcm_matrix(){
  uint32_t value = 0x01010101;
  for(size_t i=0;i<PCM_MATRIX_SIZE/4;i++){
    pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET + PCM_SIZE_OFFSET), i, value);
  }
}

uint8_t * get_Yi(){
  uint8_t *Yi = (uint8_t *)pi_malloc(XI_VECTOR_SIZE);
  for(size_t i=0;i<XI_VECTOR_SIZE/4;i++){
    uint32_t value = pcm_read_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i);
    Yi[i*4] = (value & 0xFF000000) >> 24;
    Yi[i*4+1] = (value & 0x00FF0000) >> 16;
    Yi[i*4+2] = (value & 0x0000FF00) >> 8;
    Yi[i*4+3] = (value & 0x000000FF);
  }
  return Yi;
}

int main(){   
  pcm_t *pcm=init_pcm();
  uint32_t *ad =(uint32_t*)(PCM_ADDR);
  if (pcm == NULL) {
    printf("Failed to initialize PCM\n");
    return -1;
  }

  
  load_Xi_vect();
  load_pcm_matrix();
  

  /*set sectors   
  
     3                   2                   1                   0
   1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |  cmd  |sub cmd| arr 1 | arr 2 | arr 3 | arr 4 |     layers    | 
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  */    
  uint32_t sectors = 0b00010000101010101010101000000000;  
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,sectors);
  //start computation
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,AIMC_COMPUTE);
 
  uint8_t *res =get_Yi();

  for(size_t i=0;i<XI_VECTOR_SIZE;i++){
    printf("Yi[%ld]: %d\n", i, res[i]);
  }

  return 0;
}
