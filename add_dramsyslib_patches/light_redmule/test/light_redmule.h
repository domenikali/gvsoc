#ifndef _LIGHT_REDMULE_H_
#define _LIGHT_REDMULE_H_

#define ARCH_REDMULE_REG_BASE 0x10201000

void light_redmule_set_M(uint32_t size){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE;
    *redmule_reg            = size;
}

void light_redmule_set_N(uint32_t size){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE + 4;
    *redmule_reg            = size;
}

void light_redmule_set_K(uint32_t size){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE + 8;
    *redmule_reg            = size;
}

void light_redmule_set_X(uint32_t addr){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE + 12;
    *redmule_reg            = addr;
}

void light_redmule_set_Y(uint32_t addr){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE + 16;
    *redmule_reg            = addr;
}

void light_redmule_set_Z(uint32_t addr){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE + 20;
    *redmule_reg            = addr;
}

void light_redmule_set_W(uint32_t addr){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE + 24;
    *redmule_reg            = addr;
}

uint32_t light_redmule_trigger_block(){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE + 32;
    *redmule_reg            = 32;
    return (*redmule_reg);
}

uint32_t light_redmule_trigger_async(){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE + 36;
    return (*redmule_reg);
}

uint32_t light_redmule_trigger_wait(){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE + 40;
    return (*redmule_reg);
}


#endif