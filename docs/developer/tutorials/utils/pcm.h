#ifndef PCM_H
#define PCM_H
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "alloc.h"

#ifndef PCM_BASE_ADDR
#define PCM_BASE_ADDR 0
#endif

#define PCM_DEBUG 0

//pcm adresses
#define PCM_ADDR            (0x20001000+PCM_BASE_ADDR)
#define AIMC_CMD_OFFSET     (0x00000000+PCM_BASE_ADDR)
#define AIMC_VECTOR_OFFSET  (0x00000001+PCM_BASE_ADDR)
#define PCM_SIZE_OFFSET     (0x00000080+PCM_BASE_ADDR)

//pcm address macros
#define PCM_CMD_ADDR       ((uint32_t*)(PCM_ADDR + AIMC_CMD_OFFSET))
#define PCM_VECTOR_ADDR    ((uint32_t*)(PCM_ADDR + AIMC_CMD_OFFSET+ AIMC_VECTOR_OFFSET))
#define PCM_MATRIX_ADDR    ((uint32_t*)(PCM_ADDR + PCM_SIZE_OFFSET+AIMC_CMD_OFFSET+AIMC_VECTOR_OFFSET))

//aimc cmd
#define AIMC_SETTINGS       0x10000000
#define AIMC_COMPUTE        0x00000000
#define SECTOR_MASK         0x00FFFF00

//aimc sub cmd
#define SUBCMD_MASK         0x0F000000
#define SUBCMD_SET_SECTORS  0x00000000
#define SUBCMD_TWO_STEP_U   0x01000000
#define SUBCMD_TWO_STEP_UDW 0x02000000
#define SUBCMD_TWO_STEP_S   0x03000000
#define SUBCMD_TWO_STEP_SDW 0x04000000
#define SUBCMD_SINGLE_STEP  0x05000000
#define SUBCMD_FAST_SS      0x06000000
#define SUBCMD_PRECISION    0x07000000
#define SUBCMD_BL           0x08000000
//usefull macros for std module
#define PCM_MATRIX_SIZE 0x200000
#define XI_VECTOR_SIZE 0x200
#define CELL_SIZE 4


typedef struct{
    uint32_t *pcm_addr;
    uint32_t *aimc_cmd_addr;
    uint32_t *aimc_vector_addr;
    uint32_t *pcm_matrix_addr;
}pcm_t;
  
/**
 * @brief This method initialize a pcm_t object with pointers to the necessary positions within the module
 */
pcm_t* init_pcm();
   
/**
 * @brief 32 bit read method
 * This method take a pointer to the pcm member and read at a 4 bytes value at the specifyed offset
 * @param pcm pointer to the memory locations
 * @param offset from the pointer
 * @return 4 byte unsigned integer with the informations read
*/
inline uint32_t pcm_read_32(uint32_t *pcm, uint32_t offset) {
    return *(volatile uint32_t *)(pcm + offset);
}
  
/**
 * @brief 32 bit write method
 * This method take a pointer to the pcm member and writes the value at a specifyed offset
 * @param pcm pointer to the memory locations
 * @param offset from the pointer
 * @param value 4 byte unsigned integer to write
 */
inline void pcm_write_32(uint32_t *pcm, uint32_t offset, uint32_t value) {
    *(volatile uint32_t *)(pcm + offset) = value;
}

  /**
 * @brief PCM cell write method
 * Write 4 bytes to a specific alligned locations of the PCM matrix
 * @param layer layer number (0-7)
 * @param sector sector number (0-4)
 * @param line line number (0-127)
 * @param cell cell number (0-128, 4 increment)
 * @param value 4 byte unsigned integer to write
 */
void pcm_write_word( int layer, int sector, int line, int cell, uint32_t value);
  
/**
 * @brief PCM line fill method
 * Fill a line of the PCM matrix with the same value provided by calling the pcm_write_word method
 * @param layer layer number (0-7)
 * @param sector sector number (0-4)
 * @param line line number (0-127)
 * @param value 4 byte unsigned integer to write
 */
void pcm_fill_line( int layer, int sector, int line, uint32_t value);

/**
 * @brief PCM sector fill method
 * Fill a sector of the PCM matrix with the same value provided by calling the pcm_fill_line method
 * @param layer layer number (0-7)
 * @param sector sector number (0-4)
 * @param value 4 byte unsigned integer to write
 */
void pcm_fill_sector( int layer, int sector, uint32_t value);

/**
 * @brief PCM input vector load method
 * Fill the input vector with the same value provided by calling the pcm_write_32 method
 * @param value 4 byte unsigned integer to write
 */
void load_vector(uint32_t value);
  
/**
 * @brief PCM input vector load method one every two
 * Fill the input vector with the same value provided by calling the pcm_write_32 method
 * @param value 4 byte unsigned integer to write
 */
void load_vector_partial(uint32_t value);

/**
 * @brief PCM set precision method
 * This method set the input precision of the aimc computation by writing the appropriate cmd to the aimc_cmd register
 * @param precision number of bits used for the input precision (1-7)
 */
void pcm_set_precision( uint8_t precision);

/**
 * @brief PCM set mode method
 * This method set the aimc computation mode by writing the appropriate cmd to the aimc_cmd register
 * @param mode aimc computation mode
 */
inline void pcm_set_mode(uint32_t mode){
    pcm_write_32(PCM_CMD_ADDR,0x00000000,AIMC_SETTINGS|mode);
}

/**
 * @brief PCM get output vector method
 * This method read the output vector from the aimc_vector register and return it as an array of 8 bit unsigned integers
 * @return pointer to the output vector
 * @note the output vector is 512 bytes long by default and the caller is responsible for the memory deallocation
 */
uint8_t * pcm_get_Yi();

/**
 * @brief PCM compute method
 * This method start the aimc computation by writing the AIMC_COMPUTE cmd to the aimc_cmd register
 * @note this is simply a wrapper around the pcm_write_32 method
 */
inline void pcm_compute(void){
    pcm_write_32(PCM_CMD_ADDR,0x00000000,AIMC_COMPUTE);
}

#endif


