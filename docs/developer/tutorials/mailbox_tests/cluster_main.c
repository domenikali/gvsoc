#include <stdint.h>

#define MAILBOX_BASE 0x20000000u
#define INT_SND_SET (MAILBOX_BASE + 0x04u)
#define LETTER_0 (MAILBOX_BASE + 0x80u)

#define REQUEST_MAILBOX (LETTER_0 + 0x100u)
#define RESPONSE_MAILBOX LETTER_0
#define RESPONSE_SEND (INT_SND_SET)

#define REQUEST_MAGIC 0xCAFE0000u

static inline uint32_t get_hartid(void)
{
  uint32_t hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  return hartid;
}

void cluster_main(void)
{
  if (get_hartid() != 2)
    for (;;) __asm__ volatile("wfi");

  for (;;) {
    volatile uint32_t *request = (volatile uint32_t *)REQUEST_MAILBOX;
    uint32_t value = *request;
    if ((value & 0xffff0000u) == REQUEST_MAGIC) {
      uint32_t operand = value & 0xffu;
      volatile uint32_t *response = (volatile uint32_t *)RESPONSE_MAILBOX;
      *response = operand * operand;
      *request = 0;
      *(volatile uint32_t *)RESPONSE_SEND = 1;
    }
  }
}
