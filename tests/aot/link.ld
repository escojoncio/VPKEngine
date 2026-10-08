/* Test image: code first, then read-only data (jump tables), contiguous at CODE_BASE. */
ENTRY(_start)
SECTIONS {
    . = 0x400000;
    .text : { *(.text*) }
    . = ALIGN(16);
    .rodata : { *(.rodata*) }
    /DISCARD/ : { *(.note*) *(.eh_frame*) *(.comment) }
}
