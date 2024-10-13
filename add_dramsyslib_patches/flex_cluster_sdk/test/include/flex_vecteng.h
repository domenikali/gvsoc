#ifndef _FLEX_VectEng_H_
#define _FLEX_VectEng_H_

#include "flex_cluster_arch.h"

void flex_vecteng_set_M(uint32_t size){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE;
    *vecteng_reg            = size;
}

void flex_vecteng_set_N(uint32_t size){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 4;
    *vecteng_reg            = size;
}

void flex_vecteng_set_X(uint32_t addr){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 8;
    *vecteng_reg            = addr;
}

void flex_vecteng_set_Y(uint32_t addr){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 12;
    *vecteng_reg            = addr;
}

uint32_t flex_vecteng_trigger_Max(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_Sum(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_SumExpSub(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 4;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_ExpMaxDiv(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 8;
    return (*vecteng_reg);
}

#endif