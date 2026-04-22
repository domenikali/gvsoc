#include <stdio.h>
#include <stdint.h>

#define MAILBOX_0_BASE 0x20000000

#define INT_SND_SET (MAILBOX_0_BASE + 0x04)
#define INT_SND_CLR (MAILBOX_0_BASE + 0x08)
#define INT_RCV_EN  (MAILBOX_0_BASE + 0x4C)
#define LETTER_0    (MAILBOX_0_BASE + 0x80)

volatile uint32_t* mbox_snd_set = (volatile uint32_t*)INT_SND_SET;
volatile uint32_t* mbox_snd_clr = (volatile uint32_t*)INT_SND_CLR;
volatile uint32_t* mbox_rcv_en  = (volatile uint32_t*)INT_RCV_EN;
volatile uint32_t* mbox_letter0 = (volatile uint32_t*)LETTER_0;

#define MSG 0x1EEDC0FF

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

int main() {
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
    return 0;
}