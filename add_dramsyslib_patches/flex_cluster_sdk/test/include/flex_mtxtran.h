#ifndef _FLEX_MTXTRAN_H_
#define _FLEX_MTXTRAN_H_

#include "flex_cluster_arch.h"

void flex_mtxtran_set_M(uint32_t size){
    volatile uint32_t * mtxtran_reg  = ARCH_MTXTRAN_REG_BASE;
    *mtxtran_reg            = size;
}

void flex_mtxtran_set_N(uint32_t size){
    volatile uint32_t * mtxtran_reg  = ARCH_MTXTRAN_REG_BASE + 4;
    *mtxtran_reg            = size;
}

void flex_mtxtran_set_X(uint32_t addr){
    volatile uint32_t * mtxtran_reg  = ARCH_MTXTRAN_REG_BASE + 8;
    *mtxtran_reg            = addr;
}

void flex_mtxtran_set_Y(uint32_t addr){
    volatile uint32_t * mtxtran_reg  = ARCH_MTXTRAN_REG_BASE + 12;
    *mtxtran_reg            = addr;
}

uint32_t flex_mtxtran_trigger_block(){
    volatile uint32_t * mtxtran_reg  = ARCH_MTXTRAN_REG_BASE;
    return (*mtxtran_reg);
}

#endif