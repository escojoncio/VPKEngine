/* Four LOAD segments at file offsets 0, 2, 4 and 6 MiB (links with -z max-page-size=0x200000): the SELF is under 1 KB, the ELF inside has 6 MiB. */
PHDRS { a PT_LOAD FILEHDR PHDRS FLAGS(4); b PT_LOAD FLAGS(5); c PT_LOAD FLAGS(4); d PT_LOAD FLAGS(6); }
SECTIONS {
  . = 0x400000 + SIZEOF_HEADERS;
  .t1 : { *(.t1) } :a
  . = 0x600000;
  .text : { *(.text) } :b
  . = 0x800000;
  .rodata : { *(.rodata) } :c
  . = 0xa00000;
  .data : { *(.data) } :d
  /DISCARD/ : { *(.note*) *(.comment) }
}
