#include <stdio.h>
#include <stdint.h>

#include "dma.h"

#define MAILBOX_BASE 0x20000000
#define PLIC_BASE    0x0C000000


#define INT_SND_SET (MAILBOX_BASE + 0x04)
#define INT_SND_CLR (MAILBOX_BASE + 0x08)
#define INT_RCV_EN  (MAILBOX_BASE + 0x4C)
#define LETTER_0    (MAILBOX_BASE + 0x80)

#define PLIC_PRIORITY_1  (PLIC_BASE + 0x000004) 
#define PLIC_ENABLE_H1_M (PLIC_BASE + 0x002100)
#define PLIC_THRESH_H1_M (PLIC_BASE + 0x202000)
#define PLIC_CLAIM_H1_M  (PLIC_BASE + 0x202004)

#define MBOX_1_LETTER0 (LETTER_0 + 0x100)
#define MBOX_1_SND_CLR (INT_SND_CLR + 0x100)
#define MBOX_1_RCV_EN  (INT_RCV_EN + 0x100)
#define MBOX_1_SND_SET (INT_SND_SET + 0x100)


volatile uint32_t* mbox_snd_set = (volatile uint32_t*)INT_SND_SET;
volatile uint32_t* mbox_snd_clr = (volatile uint32_t*)INT_SND_CLR;
volatile uint32_t* mbox_rcv_en  = (volatile uint32_t*)INT_RCV_EN;
volatile uint32_t* mbox_letter0 = (volatile uint32_t*)LETTER_0;

#define MSG 0x1EEDC0FF
#define CLUSTER_REQUEST 0xCAFE0007
#define CLUSTER_RESPONSE (LETTER_0)

static int dma_compatibility_test(void) {
    static uint32_t source = 0xD00DFEED;
    static uint32_t destination;
    dma_transfer_cfg_t transfer = {
        .ext = (uint32_t)(uintptr_t)&source,
        .loc = (uint32_t)(uintptr_t)&destination,
        .length_1d_copy = sizeof(source),
        .number_of_1d_copies = 1,
        .dir = DMA_DIR_L1_TO_L2,
    };

    dma_transfer_1d_async(transfer);
    return destination == source;
}

static inline uint64_t get_hartid() {
    uint64_t hartid;
    __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
    return hartid;
}

__attribute__((interrupt, aligned(4)))
void trap_handler(void) {
    uint64_t mcause;
    __asm__ volatile("csrr %0, mcause" : "=r"(mcause));

    if ((mcause & (1ULL << 63)) && ((mcause & ~(1ULL << 63)) == 11)) {
        
        volatile uint32_t* claim_reg = (volatile uint32_t*)PLIC_CLAIM_H1_M;
        uint32_t irq_source = *claim_reg;
        
        if (irq_source == 1) { // Mailbox IRQ Source ID = 1
            printf("[Core 1] PLIC triggered by Mailbox!\n");
            
            volatile uint32_t* mbox_letter = (volatile uint32_t*)MBOX_1_LETTER0;
            printf("[Core 1] Message Received: 0x%08X\n", *mbox_letter);
            
            volatile uint32_t* mbox_snd_clr = (volatile uint32_t*)MBOX_1_SND_CLR;
            *mbox_snd_clr = 1;
        }

        *claim_reg = irq_source;
    }
}

__attribute__((interrupt, aligned(4)))
void mbox_trap_handler(void) {
    uint64_t mcause;
    __asm__ volatile ("csrr %0, mcause" : "=r" (mcause));

    if ((mcause & (1ULL << 63)) && ((mcause & ~(1ULL << 63)) == 3)) {
        printf("\n>>> CPU CAUGHT THE MAILBOX INTERRUPT! <<<\n");
        
        uint32_t msg = *mbox_letter0;
        printf(">>> Mailbox Message Received: 0x%08X <<<\n\n", msg);
        
        *mbox_snd_clr = 1; 
        printf(">>> Mailbox interrupt cleared. <<<\n");
    }
}

void test_0(){

    printf("1. Setting up CPU interrupt registers...\n");

    __asm__ volatile ("csrw mtvec, %0" : : "r" (mbox_trap_handler));

    __asm__ volatile ("csrs mie, %0" : : "r" (1 << 3));

    __asm__ volatile ("csrs mstatus, %0" : : "r" (1 << 3));

    printf("2. Configuring Mailbox 0...\n");
    *mbox_rcv_en = 1;

    printf("3. Writing \"%08x\" as message to LETTER_0...\n",MSG);
    *mbox_letter0 = MSG;

    printf("4. Triggering the Mailbox IRQ (writing to SND_SET)...\n");
    *mbox_snd_set = 1; 

    //delay exit
    for(int i=0; i<1000000; i++); 
    return;
}

int main() {
    uint64_t hartid = get_hartid();
    

    //reciver core
    if (hartid == 1) {
        printf("[Core 1] Booted. Configuring PLIC...\n");

        __asm__ volatile("csrw mtvec, %0" : : "r"(trap_handler));

        *(volatile uint32_t*)PLIC_PRIORITY_1 = 1;
        *(volatile uint32_t*)PLIC_ENABLE_H1_M = (1 << 1);
        *(volatile uint32_t*)PLIC_THRESH_H1_M = 0;
        *(volatile uint32_t*)MBOX_1_RCV_EN = 1;

        // Enable MEI
        __asm__ volatile("csrs mie, %0" : : "r"(1 << 11));

        // Enable global interrupts
        __asm__ volatile("csrs mstatus, %0" : : "r"(1 << 3));

        printf("[Core 1] Sleeping, waiting for Mailbox via PLIC...\n");
        while(1) {
            __asm__ volatile("wfi"); // Wait For Interrupt
        }

    } else if (hartid == 0) {//sender core
        printf("[Core 0] Booted. Preparing to send message...\n");

        printf("[Core 0] Running local DMA compatibility check...\n");
        printf("[Core 0] DMA compatibility check: %s\n",
               dma_compatibility_test() ? "PASS" : "FAIL");

        for(volatile int i=0; i<10000; i++); 

        printf("[Core 0] Writing message to Mailbox 1...\n");
        volatile uint32_t* mbox_letter = (volatile uint32_t*)MBOX_1_LETTER0;
        *mbox_letter = MSG;

        printf("[Core 0] Triggering Mailbox 1 Hardware IRQ...\n");
        volatile uint32_t* mbox_snd_set = (volatile uint32_t*)MBOX_1_SND_SET;
        *mbox_snd_set = 1;

        printf("[Core 0] Sending square request to Snitch cluster...\n");
        *(volatile uint32_t*)MBOX_1_LETTER0 = CLUSTER_REQUEST;
        *mbox_snd_set = 1;

        volatile uint32_t* cluster_response = (volatile uint32_t*)CLUSTER_RESPONSE;
        for (volatile int i = 0; i < 1000000 && *cluster_response == 0; i++);
        printf("[Core 0] Snitch cluster response: %s\n", *cluster_response == 49 ? "PASS (7 * 7 = 49)" : "FAIL");
        
        printf("[Core 0] Finished.\n");
        while(1);
    }

    return 0;
}