#!/usr/bin/env python3
"""Wraps an x86-64 ELF in the PS4 SELF container layout vpaot's loader understands (a plaintext
SELF: segment table, then the ELF header and program headers, segment data at the SELF offsets).
Used by the tests to exercise the SELF path without any game file.  usage: self_wrap.py in.elf out.self
"""
import struct
import sys


def main(src, dst):
    elf = open(src, "rb").read()
    phoff = struct.unpack_from("<Q", elf, 32)[0]
    phnum = struct.unpack_from("<H", elf, 56)[0]
    ph = [struct.unpack_from("<IIQQQQQQ", elf, phoff + i * 56) for i in range(phnum)]
    header_end = phoff + phnum * 56
    # One SELF segment per program header with file data, "blocked" (flag 0x800), index in bits 20..31.
    segs = [(i, p) for i, p in enumerate(ph) if p[5]]
    count = len(segs)
    base = 32 + count * 32
    out = bytearray()
    out += b"O\x15=\x1d" + bytes(20) + struct.pack("<H", count) + bytes(6)
    data_off = base + len(elf[:header_end])
    data_off = (data_off + 15) & ~15
    table = bytearray()
    blobs = bytearray()
    for i, p in segs:
        flags = 0x800 | (i << 20)
        off = data_off + len(blobs)
        table += struct.pack("<QQQQ", flags, off, p[5], p[5])
        blobs += elf[p[2]:p[2] + p[5]]
        while len(blobs) % 16:
            blobs += b"\0"
    out += table
    out += elf[:header_end]
    while len(out) < data_off:
        out += b"\0"
    out += blobs
    open(dst, "wb").write(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
