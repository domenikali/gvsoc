#include <stdio.h>
#include <stdint.h>

#define PCM_ADDR 0x20001000

typedef uint32_t pcm_t;

static inline uint32_t pcm_read_reg(pcm_t *pcm, uint32_t offset) {
    return *(volatile uint32_t *)(pcm + offset);
}
  
static inline void pcm_write_reg(pcm_t *pcm, uint32_t offset, uint32_t value) {
    *(volatile uint32_t *)(pcm + offset) = value;
}

int main()
{   
    pcm_t *pcm = (pcm_t *)PCM_ADDR;
    
    pcm_write_reg(pcm, 0x00000003, 0x00000000); 
    uint32_t value = pcm_read_reg(pcm, 0x00000000);
    printf("Read value: 0x%08X\n", value);
    return 0;
}
