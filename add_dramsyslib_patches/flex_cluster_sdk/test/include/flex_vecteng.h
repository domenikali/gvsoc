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

uint32_t flex_vecteng_trigger_RowMax(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_RowSum(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 4;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_ExpSubMtx(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 8;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_RowDivV(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 12;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_RowDotpV(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 16;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_MtxAdd(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 20;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_VDotpV(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 24;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_VAdd(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 28;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_VMax(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 32;
    return (*vecteng_reg);
}

uint32_t flex_vecteng_trigger_ExpSubV(){
    volatile uint32_t * vecteng_reg  = ARCH_VECTENG_REG_BASE + 36;
    return (*vecteng_reg);
}

#endif