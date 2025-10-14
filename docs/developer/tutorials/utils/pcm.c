#include "pcm.h"

pcm_t * init_pcm() {
    pcm_t *pcm = (pcm_t *)pi_malloc(sizeof(pcm_t));
    pcm->pcm_addr =(uint32_t*) PCM_ADDR;
    pcm->aimc_cmd_addr = (uint32_t*)(PCM_ADDR + AIMC_CMD_OFFSET);
    pcm->aimc_vector_addr = (uint32_t*)(PCM_ADDR + AIMC_CMD_OFFSET+ AIMC_VECTOR_OFFSET);
    pcm->pcm_matrix_addr = (uint32_t*)(PCM_ADDR + PCM_SIZE_OFFSET+AIMC_CMD_OFFSET+AIMC_VECTOR_OFFSET);
    return pcm;
}

void pcm_write_word( int layer, int sector, int line, int cell, uint32_t value){
    long long start_offset=(((sector*8+layer)*128+line)*512+cell)/4;
    pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET + PCM_SIZE_OFFSET), start_offset, value);
}

void pcm_fill_line( int layer, int sector, int line, uint32_t value){
    for(size_t i=0;i<512;i+=4){
      pcm_write_word(layer,sector,line,i,value);
    }
}

void pcm_fill_sector( int layer, int sector, uint32_t value){
    for(int i=0;i<128;i++){
      pcm_fill_line(layer,sector,i,value);
    }
}

void load_vector(uint32_t value){
    for(size_t i=0;i<XI_VECTOR_SIZE/4;i++){
      pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i, value);
    }
}

void load_vector_partial(uint32_t value){
    for(size_t i=0;i<XI_VECTOR_SIZE/4;i++){
      if(i%2==0)
        pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i, value);
      else
        pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i, 0x00000000);
    }
}
void pcm_set_precision( uint8_t precision){
    if(precision>7){
      printf("Precision too high, setting to max 7 bits\n");
      precision = 7;
    }
    if(precision==0)
      printf("Edge case not handled\n");
  
    uint32_t cmd = AIMC_SETTINGS | SUBCMD_PRECISION | precision;
    pcm_write_32(PCM_CMD_ADDR,0x00000000,cmd);
}

uint8_t * pcm_get_Yi(){
    uint8_t *Yi = (uint8_t *)pi_malloc(XI_VECTOR_SIZE*sizeof(uint8_t));
    for(size_t i=0;i<XI_VECTOR_SIZE/4;i++){
        uint32_t value = pcm_read_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i);
        Yi[i*4] = (value & 0x000000FF);
        Yi[i*4+1] = (value & 0x0000FF00) >> 8;
        Yi[i*4+2] = (value & 0x00FF0000) >> 16;
        Yi[i*4+3] = (value & 0xFF000000) >> 24;
    }
    return Yi;
}