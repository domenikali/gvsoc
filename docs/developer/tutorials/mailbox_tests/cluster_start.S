.section .text.init
.global _start
_start:
  li sp, 0x5001fff0
  call cluster_main
1:
  wfi
  j 1b
