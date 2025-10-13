#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "alloc.h"

#ifndef PCM_BASE_ADDR
#define PCM_BASE_ADDR 0
#endif

#define PCM_DEBUG 0

//pcm adresses
#define PCM_ADDR            0x20001000+PCM_BASE_ADDR
#define AIMC_CMD_OFFSET     0x00000000+PCM_BASE_ADDR
#define AIMC_VECTOR_OFFSET  0x00000001+PCM_BASE_ADDR
#define PCM_SIZE_OFFSET     0x00000080+PCM_BASE_ADDR

//pcm address macros
#define PCM_CMD_ADDR       PCM_ADDR + AIMC_CMD_OFFSET
#define PCM_VECTOR_ADDR    PCM_ADDR + AIMC_CMD_OFFSET+ AIMC_VECTOR_OFFSET
#define PCM_MATRIX_ADDR    PCM_ADDR + PCM_SIZE_OFFSET+AIMC_CMD_OFFSET+AIMC_VECTOR_OFFSET

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

#define SUCCESS "\e[1;92m SUCCESS \e[0m\n"
#define FAIL    "\e[1;91m FAIL \e[0m\n"

typedef struct{
  uint32_t code;
  const char* literal_name;
} compute_type;

static const compute_type names[] = {
  {SUBCMD_TWO_STEP_U,   "Two steps unsiged" },
  {SUBCMD_TWO_STEP_UDW, "Two steps unsiged double weight"},
  {SUBCMD_TWO_STEP_S,   "Two steps siged" },
  {SUBCMD_TWO_STEP_SDW, "Two steps siged double weight"},
  {SUBCMD_SINGLE_STEP,  "Singe step differential mode"},
  {SUBCMD_FAST_SS,      "Fast single step differential mode"},
};

static const char *compute_names(uint32_t mode){   
  mode = mode & SUBCMD_MASK;
  for (size_t i = 0; i < 6; ++i) {
    if (names[i].code == mode)
      return names[i].literal_name;
  }
}


uint32_t reverse_bytes(uint32_t value){
  return (value & 0x000000FFU) << 24 | (value & 0x0000FF00U) << 8 |
          (value & 0x00FF0000U) >> 8 | (value & 0xFF000000U) >> 24;
}

typedef struct{
  uint32_t *pcm_addr;
  uint32_t *aimc_cmd_addr;
  uint32_t *aimc_vector_addr;
  uint32_t *pcm_matrix_addr;
}pcm_t;

/**
 * @brief This method initialize a pcm_t object with pointers to the necessary positions within the module
 */
pcm_t* init_pcm() {
  pcm_t *pcm = (pcm_t *)pi_malloc(sizeof(pcm_t));
  pcm->pcm_addr =(uint32_t*) PCM_ADDR;
  pcm->aimc_cmd_addr = (uint32_t*)(PCM_ADDR + AIMC_CMD_OFFSET);
  pcm->aimc_vector_addr = (uint32_t*)(PCM_ADDR + AIMC_CMD_OFFSET+ AIMC_VECTOR_OFFSET);
  pcm->pcm_matrix_addr = (uint32_t*)(PCM_ADDR + PCM_SIZE_OFFSET+AIMC_CMD_OFFSET+AIMC_VECTOR_OFFSET);
  return pcm;
}
 
/**
 * @brief 32 bit read method
 * This method take a pointer to the pcm member and read at a 4 bytes value at the specifyed offset
 * @param pcm pointer to the memory locations
 * @param offset from the pointer
 * @return 4 byte unsigned integer with the informations read
 */
static inline uint32_t pcm_read_32(uint32_t *pcm, uint32_t offset) {
  return *(volatile uint32_t *)(pcm + offset);
}

/**
 * @brief 32 bit write method
 * This method take a pointer to the pcm member and writes the value at a specifyed offset
 * @param pcm pointer to the memory locations
 * @param offset from the pointer
 * @param value 4 byte unsigned integer to write
 */
static inline void pcm_write_32(uint32_t *pcm, uint32_t offset, uint32_t value) {
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
void pcm_write_word( int layer, int sector, int line, int cell, uint32_t value){
  long long start_offset=(((sector*8+layer)*128+line)*512+cell)/4;
  pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET + PCM_SIZE_OFFSET), start_offset, value);
}

/**
 * @brief PCM line fill method
 * Fill a line of the PCM matrix with the same value provided by calling the pcm_write_word method
 * @param layer layer number (0-7)
 * @param sector sector number (0-4)
 * @param line line number (0-127)
 * @param value 4 byte unsigned integer to write
 */
void pcm_fill_line( int layer, int sector, int line, uint32_t value){
  for(size_t i=0;i<512;i+=4){
    pcm_write_word(layer,sector,line,i,value);
  }
}

/**
 * @brief PCM sector fill method
 * Fill a sector of the PCM matrix with the same value provided by calling the pcm_fill_line method
 * @param layer layer number (0-7)
 * @param sector sector number (0-4)
 * @param value 4 byte unsigned integer to write
 */
void pcm_fill_sector( int layer, int sector, uint32_t value){
  for(int i=0;i<128;i++){
    pcm_fill_line(layer,sector,i,value);
  }
}

/**
 * @brief PCM input vector load method
 * Fill the input vector with the same value provided by calling the pcm_write_32 method
 * @param value 4 byte unsigned integer to write
 */
void load_vector(uint32_t value){
  for(size_t i=0;i<XI_VECTOR_SIZE/4;i++){
    pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i, value);
  }
}

/**
 * @brief PCM input vector load method one every two
 * Fill the input vector with the same value provided by calling the pcm_write_32 method
 * @param value 4 byte unsigned integer to write
 */
void load_vector_partial(uint32_t value){
  for(size_t i=0;i<XI_VECTOR_SIZE/4;i++){
    if(i%2==0)
      pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i, value);
    else
      pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i, 0x00000000);
  }
}

/**
 * @brief PCM set precision method
 * This method set the input precision of the aimc computation by writing the appropriate cmd to the aimc_cmd register
 * @param precision number of bits used for the input precision (1-7)
 */
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

/**
 * @brief PCM set mode method
 * This method set the aimc computation mode by writing the appropriate cmd to the aimc_cmd register
 * @param mode aimc computation mode
 */
void pcm_set_mode(uint32_t mode){
  pcm_write_32(PCM_CMD_ADDR,0x00000000,AIMC_SETTINGS|mode);
}

/**
 * @brief PCM get output vector method
 * This method read the output vector from the aimc_vector register and return it as an array of 8 bit unsigned integers
 * @return pointer to the output vector
 * @note the output vector is 512 bytes long by default and the caller is responsible for the memory deallocation
 */
uint8_t * pcm_get_Yi(){
  uint8_t *Yi = (uint8_t *)pi_malloc(XI_VECTOR_SIZE);
  for(size_t i=0;i<XI_VECTOR_SIZE/4;i++){
    uint32_t value = pcm_read_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i);
    Yi[i*4] = (value & 0x000000FF);
    Yi[i*4+1] = (value & 0x0000FF00) >> 8;
    Yi[i*4+2] = (value & 0x00FF0000) >> 16;
    Yi[i*4+3] = (value & 0xFF000000) >> 24;
  }
  return Yi;
}

/**
 * @brief PCM compute method
 * This method start the aimc computation by writing the AIMC_COMPUTE cmd to the aimc_cmd register
 * @note this is simply a wrapper around the pcm_write_32 method
 */
void pcm_compute(void){
  pcm_write_32(PCM_CMD_ADDR,0x00000000,AIMC_COMPUTE);
}

/**
 * @brief Simple compute test
 * This test fill the PCM crossbar with one value and perform an aimc computation specified by the mode attribute with an input vector filled by the vector_val attribute
 * The expected result is a vector with expected_val value in every position of the resulting vector
 * @param pcm pointer to the memory locations
 * @param vector_val value to fill the input vector
 * @param cell_val value to fill the PCM crossbar
 * @param mode aimc computation mode (using double weight mode is useless in this test since just the first sector has values different from 0) 
 * @param expected_val expected value in the output vector
 * @param load_v pointer to the function used to load the input vector (full or partial)
 * @note This test does not cover all the edge cases, it is just a simple test to verify that the basic functionalities are working
 */
void test_simple_compute(pcm_t *pcm,uint32_t vector_val,uint32_t cell_val,uint32_t mode,uint8_t expected_val,void (*load_v)(uint32_t)){
  for(int i=0;i<4;++i){
    pcm_fill_sector(0,i,cell_val);
  }
  (*load_v)(vector_val);

  uint32_t sectors = 0b00010000101010101010101000000001;  
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,sectors);

  pcm_set_mode(mode);
  pcm_compute(); 

  uint8_t flag=0;
  uint8_t *res =pcm_get_Yi();


  for(int i=0;i<512;++i){

    if(res[i]!=expected_val){
      flag=1;
      printf("Yi[%d]=%d\n",i,res[i]);
    }
    else if (PCM_DEBUG){
      printf("Yi[%d]=%d\n",i,res[i]);
    }
  }
  printf("Mode %s: %s",compute_names(mode), flag == 0 ? SUCCESS : FAIL);
}

void random_test(pcm_t *pcm,int8_t*vec_val,uint32_t mode,uint8_t expected_val){

  for(int i=0;i<4;++i){
    pcm_fill_sector(0,i,0x01010101);
  }

  for(int i=0;i<512;i+=4){
    pcm_write_32(((uint32_t*)PCM_ADDR + AIMC_VECTOR_OFFSET), i/4, (vec_val[i] & 0x000000FF) | ((vec_val[i+1] & 0x000000FF) << 8) | ((vec_val[i+2] & 0x000000FF) << 16) | ((vec_val[i+3] & 0x000000FF) << 24));
  }

  uint32_t sectors = 0b00010000101010101010101000000001;  
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,sectors);

  pcm_set_mode(mode);
  pcm_compute(); 

  uint8_t flag=0;
  uint8_t *res =pcm_get_Yi();


  for(int i=0;i<512;++i){

    if(res[i]!=expected_val){
      flag=1;
      printf("Yi[%d]=%d\n",i,res[i]);
    }
    else if (PCM_DEBUG){
      printf("Yi[%d]=%d\n",i,res[i]);
    }
  }
  printf("Mode %s: %s",compute_names(mode), flag == 0 ? SUCCESS : FAIL);
}

void pcm_test(pcm_t *pcm){

  
  pcm_fill_sector(0,0,0x01010101);
  pcm_fill_sector(0,1,0x01010101);
  
  pcm_fill_sector(0,2,0x01010101);
  pcm_fill_sector(0,3,0x01010101);
  // for(int i=0;i<512;++i){
  //   for(int j=0;j<512;j+=4){
  //     pcm_write_word(0,0,i,j,0x01010101);
  //   }
  // }

  //load_pcm_matrix_1();
  load_vector(0x00000001);


  /*set sectors   
  
     3                   2                   1                   0
   1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  |  cmd  |sub cmd| arr 1 | arr 2 | arr 3 | arr 4 |    precision  | 
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  */    
  uint32_t sectors = 0b00010000101010101010101000000000;  
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,sectors);
  //start computation
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,AIMC_COMPUTE);
  
  uint8_t flag=0;
  uint8_t *res =pcm_get_Yi();
  for(int i=0;i<512;++i){
    if(res[i]!=0x0){
      printf("Yi[%d]=%d\n",i,res[i]);
      // printf(SUCCESS);
    }
    else flag++;
  }
  if(flag)
    printf(FAIL);
  else  
    printf(SUCCESS);  

}



void pcm_sector_test(pcm_t*pcm){
  printf("Sector test\n");
  uint32_t  sectors = 0b00010000100010001000100011111111;  
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,sectors);
            sectors = 0b00010000000000000000100011111111;  
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,sectors);
            sectors = 0b00010000000000001000000011111111;  
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,sectors);
            sectors = 0b00010000000010000000000011111111;  
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,sectors);
            sectors = 0b00010000100000000000000011111111;  
  pcm_write_32(pcm->aimc_cmd_addr,0x00000000,sectors);

  printf("Sector test done\n");
}

//FIXME: this tests heavily relies on the default behavior of memory set to 0, this is fairly dangerous and wrong but for now it works and it's much faster than initializing the whole memory eveytime
void run_all_tests(pcm_t *pcm){
  uint16_t layers =0x0202;
  printf("Running all tests...\n");
  //this tests covers the basic clipping functionalities of an aimc computation of positive integers
  printf("Positive cplipping tests:\n");
  test_simple_compute(pcm,0x00000001,0x01010101,SUBCMD_TWO_STEP_S,127,load_vector);  
  test_simple_compute(pcm,0x00000001,0x01010101,SUBCMD_TWO_STEP_U,127,load_vector);
  test_simple_compute(pcm,0x00000001,0x01010101,SUBCMD_TWO_STEP_SDW|layers,127,load_vector);  
  test_simple_compute(pcm,0x00000001,0x01010101,SUBCMD_TWO_STEP_UDW|layers,127,load_vector);

  //this tests covers the basic clipping functionalities of an aimc computation of negative integers
  printf("Negative clipping tests:\n");
  test_simple_compute(pcm,0x0000000F6,0x01010101,SUBCMD_TWO_STEP_S,128,load_vector);//the actual result would be -1280, however the clipping moves it to -128 which is rappresented as 128 for 8 bit unsigned integers
  test_simple_compute(pcm,0x0000000F6,0x01010101,SUBCMD_TWO_STEP_U,127,load_vector);
  test_simple_compute(pcm,0x0000000F6,0x01010101,SUBCMD_TWO_STEP_SDW|layers,128,load_vector); 
  test_simple_compute(pcm,0x0000000F6,0x01010101,SUBCMD_TWO_STEP_UDW|layers,127,load_vector);

  printf("Positive tests:\n");
  test_simple_compute(pcm,0x00000001,0x01010101,SUBCMD_TWO_STEP_S,64,load_vector_partial);  
  test_simple_compute(pcm,0x00000001,0x01010101,SUBCMD_TWO_STEP_U,64,load_vector_partial);
  test_simple_compute(pcm,0x00000001,0x01010101,SUBCMD_TWO_STEP_SDW|layers,64,load_vector_partial);  
  test_simple_compute(pcm,0x00000001,0x01010101,SUBCMD_TWO_STEP_UDW|layers,64,load_vector_partial);

  printf("Negative tests:\n");
  test_simple_compute(pcm,0x0000000FF,0x01010101,SUBCMD_TWO_STEP_S,192,load_vector_partial);//-64 = 192 for 8 bit unsigned integers
  test_simple_compute(pcm,0x0000000FF,0x01010101,SUBCMD_TWO_STEP_U,64,load_vector_partial);
  test_simple_compute(pcm,0x0000000FF,0x01010101,SUBCMD_TWO_STEP_SDW|layers,192,load_vector_partial); 
  test_simple_compute(pcm,0x0000000FF,0x01010101,SUBCMD_TWO_STEP_UDW|layers,64,load_vector_partial);

  //the idea of this test is to fill the crossbar with 1s and the input vector with a random sequence of positive and negative numbers that will sum up to something in between -128 and 127. If the matrix only contains 1s the mvm can be easily calculated as the sum of the input vector
  printf("Negative randomised tests:\n");
  int8_t v1[512]={111, -21, 50, 119, 4, 3, -4, 96, 115, -84, 127, -54, 75, -53, 72, 19, -18, -23, 75, -15, -87, -97, -115, -50, 56, -28, -15, 21, -35, -61, 127, -22, 51, 35, 25, -62, 39, -96, 127, 118, -1, -73, -75, -68, 120, -65, -46, -39, 43, -47, -20, 26, 115, 47, 115, 43, -29, 103, 61, 62, -2, 69, 16, 57, 56, 66, -72, -55, -83, -80, 115, -42, 58, -104, -48, 111, -67, 26, 31, -116, -21, -107, -93, 19, 109, -80, -113, -107, -51, 19, 123, -67, -116, 118, -101, 26, 127, -69, -51, -118, 71, 101, 57, 127, -15, 28, 126, -105, -6, 104, 12, -35, 73, -35, -56, -54, -57, -48, 49, -115, -59, -63, 86, 86, 27, -22, -24, -22, -31, 106, -101, 84, 109, 117, -41, 33, -19, 127, 53, 4, 93, -87, 57, -106, -46, 25, 15, -7, -3, 102, -97, -109, -62, 0, -67, -20, -43, -4, 41, -18, -64, -53, 58, 100, -65, 58, -7, 19, -106, 28, -115, -74, 18, 61, 10, -61, -41, 93, 57, 55, 95, 34, 18, -59, -76, 39, -34, 127, -103, 90, -124, 21, 126, 103, 67, 23, 76, 46, 63, -72, 33, -28, 82, -71, -42, -45, 77, 115, -87, -106, 108, 84, -86, 91, -17, 76, -105, 24, 14, 14, -68, 80, 29, 4, -59, 85, 127, -75, 127, 127, -50, -15, -56, -61, 127, 25, 1, 117, -36, 17, 108, 9, 82, 44, 74, 111, 30, 52, -106, -115, -113, 28, -54, 89, -61, -59, -66, 127, -48, 12, 43, -31, -34, 43, -97, 97, 38, -101, -109, 69, 50, 127, -50, -21, -18, 127, 63, 61, 121, -59, -64, -19, 104, 15, 48, 72, -28, 22, -62, -85, -20, -21, 15, -55, 53, -51, 44, -52, 7, -121, -24, 53, -69, -22, 124, -112, -50, 49, -91, 51, -110, -28, 65, -47, -106, 107, -20, -33, -123, -28, 32, 55, -112, -82, -89, -39, -49, -66, 39, -117, 65, 106, -63, 127, 66, -104, 45, -60, 38, 6, -31, -21, 71, -104, 76, 63, -44, 50, -58, -36, 96, -110, -95, 56, 73, 123, -22, -42, -93, 30, -43, -5, -104, 56, -87, -50, 123, -90, -73, -82, -25, 21, -46, 42, -69, -123, -68, -104, -15, -122, -6, -2, -60, -34, 85, -38, 122, 106, 118, -123, -103, -22, -112, -98, -66, -74, -99, 127, 4, 3, -110, -124, -86, -53, -71, -100, -77, -52, 93, 127, 48, 127, -108, -56, 4, -39, -94, 122, 67, 43, -42, 101, -50, 124, -16, -37, -98, 2, -41, 119, -57, 42, 51, 53, -11, -25, -15, 69, 6, 44, -10, -51, 50, -111, 83, -108, -78, -108, 17, -79, -53, 10, 122, 89, -93, -12, 115, 127, 41, -84, 51, -94, 79, -59, -50, 89, -104, -37, 21, 25, -94, -38, 101, 78, -60, -5, 118, -104, -64, -48, -120, -36, -79, -33, -43, 19, 22, 95, 110, -66, 83, -37, -25, -60, 122, 15, 1, -112, 85, 69, 104, -3, 73, 4, -77, 32, 116, 40, -82, -62, 60, 96};
  random_test(pcm,v1,SUBCMD_TWO_STEP_S,138);//-118 = 138 for 8 bit unsigned integers
  random_test(pcm,v1,SUBCMD_TWO_STEP_U,118);
  random_test(pcm,v1,SUBCMD_TWO_STEP_SDW|layers,138);
  random_test(pcm,v1,SUBCMD_TWO_STEP_UDW|layers,118);

  printf("Positive randomised tests:\n");
  int8_t v2[512]={-124, 2, -80, -107, -118, 125, 127, 87, 82, -73, -25, -83, 71, 18, -110, 27, 9, 87, 10, 58, -65, -107, 46, 72, 77, 25, -10, -83, -32, -120, 74, 7, 6, -123, -40, 117, -73, -24, -88, -37, -95, 121, 65, 113, 118, 56, -71, 127, 109, 21, -62, -86, 116, -78, 108, -90, 13, 127, 69, 50, -43, -65, -90, -116, 0, -15, -126, -51, 118, 21, 9, 84, 27, -44, -83, 84, 40, 120, -35, 111, 99, 9, -110, -121, 56, -53, 92, 127, 27, -101, -57, -100, 70, -8, -74, -21, 20, -32, 86, 111, -78, 32, 8, 94, -112, -90, 80, 36, -93, 116, -42, 57, -33, 56, 67, 23, -13, 22, -36, -58, -77, -38, 99, -36, -4, 35, -21, 55, -117, -61, -97, -114, 69, -72, 99, 127, -52, 12, -113, -48, 7, -83, -92, -103, 15, -71, -94, -52, -70, -14, 120, 90, -37, 63, -94, 12, 12, -98, -123, -36, -46, -89, -36, -1, -64, -70, 89, 0, -52, 46, 60, -5, 4, -35, 124, -41, 60, -66, -50, 3, 5, -20, -104, -40, 53, 7, 74, 0, -19, 3, -114, 127, -15, 122, 36, 4, -6, 111, -104, 67, -78, -103, -24, -18, 63, -23, 19, -9, -83, 99, -80, 104, 46, -83, 38, 29, -57, -94, 9, -11, 34, -87, 41, 60, -108, -46, -122, 0, 58, -85, 75, -68, -81, -9, -42, 76, -59, 113, -106, 70, 67, -89, 69, 49, 97, 60, -44, -31, -44, -102, -120, 4, -55, 29, 111, 68, 95, 70, 112, 88, -25, -73, -41, 82, -94, -55, 65, -35, 47, 4, -67, 10, 56, 88, -104, -63, -20, -97, -61, 3, 61, -56, -117, 113, 68, -107, -46, -72, 101, -72, 79, -104, -11, -4, 114, -72, 17, -45, -96, -54, -80, 67, -25, -4, -80, -36, -81, 34, -110, -77, -51, -114, -33, -70, -69, -22, 23, -78, 92, 127, -4, 124, -89, -85, 50, 14, 115, 89, 22, -86, -92, 96, 72, -13, -90, 120, 120, 126, 95, -122, 91, 75, -20, 54, 4, 49, 116, 37, -24, 56, 96, -106, -101, 34, -5, 88, 28, 10, 53, -101, 115, 114, 65, 41, 38, 106, 61, -118, 49, 81, -36, -17, -74, 54, -99, 127, 4, 110, -98, -74, -124, -121, 50, 24, -36, -81, -75, -22, -15, -47, -97, 124, -107, -63, -16, -7, 39, -85, 40, 112, -74, 73, -51, 28, -86, -30, -97, -2, -102, 110, 108, 94, 55, 111, -56, -13, 0, -36, -116, 103, -42, -17, -58, 34, 68, 100, -88, -15, -70, 54, 118, 118, -110, -28, -110, 84, -82, -49, 105, 85, 66, 127, 24, -21, -11, 94, -89, 79, 87, 34, 84, -100, 31, 18, 14, 6, 7, 38, -15, -70, 87, 105, 96, 15, -124, -49, 126, 112, 98, -11, -13, -96, 14, 6, -90, 20, 7, 11, 15, 93, 26, 8, -10, -75, 69, -75, 24, -62, 16, 33, -41, 58, 47, -89, 69, 12, -91, 20, 10, 72, 119, -3, 30, -75, -24, 124, -79, 114, 66, 11, -27, 51};
  random_test(pcm,v2,SUBCMD_TWO_STEP_S,83);
  random_test(pcm,v2,SUBCMD_TWO_STEP_U,83);
  random_test(pcm,v2,SUBCMD_TWO_STEP_SDW|layers,83);
  random_test(pcm,v2,SUBCMD_TWO_STEP_UDW|layers,83);


  printf("All tests done\n");
}

int main(){   
  pcm_t *pcm=init_pcm();
  if (pcm == NULL) {
    printf("Failed to initialize PCM\n");
    return -1;
  }

  //pcm_sector_test(pcm);

  //pcm_set_precision(pcm, 1);
  //pcm_set_mode(pcm, SUBCMD_FAST_SS);

  //load_pcm_matrix();


  //simple test
  run_all_tests(pcm);
  // uint16_t layers =0x0202;

  // test_simple_compute(pcm,0x00000001,0x01010101,SUBCMD_TWO_STEP_SDW|layers,127,load_vector);  
  //test_simple_compute(pcm,0x0000000FF,0x01010101,SUBCMD_TWO_STEP_U,64,load_vector_partial);


  return 0;
}
