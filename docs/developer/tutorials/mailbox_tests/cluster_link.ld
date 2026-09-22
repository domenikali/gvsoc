OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY
{
  MEM (rwx) : ORIGIN = 0x000f0000, LENGTH = 0x00010000
}

SECTIONS
{
  .text : { *(.text.init) *(.text) *(.text.*) } > MEM
  .rodata : { *(.rodata*) } > MEM
  .data : { *(.data*) } > MEM
  .bss : { *(.bss*) *(COMMON) } > MEM
}
