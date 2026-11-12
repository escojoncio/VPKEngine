# Valid SELF containers for tests/aot/self_valid/run.sh (and the exact edge of the size limit shared with runtime/vp_loader.c).
# usage: gen.py OUTDIR (writes <case>.self, prints the case names); each must pass the SELF unwrapper ("bad SELF" is its error).
import struct, sys
d = sys.argv[1]
def elf_hdr(phoff, phnum):
    h = bytearray(64)
    h[0:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", h, 18, 62)
    struct.pack_into("<Q", h, 32, phoff)
    struct.pack_into("<H", h, 56, phnum)
    return bytes(h)
def ph(offset, filesz, align, vaddr=0x400000):
    return struct.pack("<IIQQQQQQ", 1, 5, offset, vaddr, vaddr, filesz, filesz, align)
def many(n, align, far):  # n backed one-byte headers + a far one
    # n one-byte segments just after the headers (not over the ELF header), the data (a nop) at the end of the SELF
    hdr_end = 64 + (n + 1) * 56
    p = b"".join(ph(hdr_end + i, 1, align, 0x400000 + i) for i in range(n)) + ph(far, 16, 0x1000, 0x500000)
    data = 32 + 32 * n + hdr_end
    segs = [(0x800 | (i << 20), data, 1, 1) for i in range(n)]
    out = b"O\x15=\x1d" + bytes(20) + struct.pack("<H", len(segs)) + bytes(6)
    for s in segs: out += struct.pack("<QQQQ", *s)
    return out + elf_hdr(64, n + 1) + p + b"\x90" + bytes(15)
def edge(n):  # the farthest end the limit allows: N + (n + 1) * 2 MiB (n backed headers of 2 MiB alignment, n <= 32)
    total = 32 + 32 * n + 64 + (n + 1) * 56 + 16
    return many(n, 1 << 21, total + ((n + 1) << 21) - 16)
cases = {"edge_ok": edge(32), "edge16_ok": edge(16)}
for k, v in cases.items(): open(f"{d}/{k}.self", "wb").write(v)
print(" ".join(cases))
