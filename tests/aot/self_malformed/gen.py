# Malformed PS4 SELF containers for tests/aot/self_malformed/run.sh: vpaot must refuse each with an error message.
# usage: gen.py OUTDIR (writes <case>.self, prints the case names)
import struct, sys
d = sys.argv[1]
def elf_hdr(phoff, phnum):
    h = bytearray(64)
    h[0:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", h, 18, 62)
    struct.pack_into("<Q", h, 32, phoff)
    struct.pack_into("<H", h, 56, phnum)
    return bytes(h)
def self_file(count, phoff, phnum, phdrs=b"", segs=(), tail=b""):
    out = b"O\x15=\x1d" + bytes(20) + struct.pack("<H", count) + bytes(6)
    for s in segs: out += struct.pack("<QQQQ", *s)
    out += elf_hdr(phoff, phnum) + phdrs + tail
    return out
def ph(offset, filesz, vaddr=0, memsz=None):
    return struct.pack("<IIQQQQQQ", 1, 5, offset, vaddr, vaddr, filesz, filesz if memsz is None else memsz, 0x1000)
cases = {
 "huge_phoff": self_file(0, 0xFFFFFFFFFFFF0000, 1),
 "huge_phnum": self_file(0, 64, 65535, ph(0, 0)),
 "phnum_zero_data": self_file(1, 64, 0, segs=[(0x800, 200, 8, 8)]),
 "huge_filesz": self_file(0, 64, 1, ph(0, 1 << 40)),
 "big_filesz": self_file(0, 64, 1, ph(0, 1 << 30)),  # below the 2 GiB cap but far more than the file holds: refused before reserving it
 "big_offset": self_file(0, 64, 1, ph((1 << 31) - 16, 16), tail=bytes(16)),  # a few bytes of data far into the image: refused before reserving 2 GiB
 "offset_past_align": self_file(0, 64, 1, ph(3 << 21, 0x14), tail=bytes(0x14)),  # 6 MiB with 4 KiB alignment: padding does not justify it (a 2 MiB-aligned segment at 2 MiB is in ../self_valid)
 "offset_overflow": self_file(0, 64, 1, ph(0xFFFFFFFFFFFFFFF0, 0x20)),
 "seg_off_overflow": self_file(1, 64, 1, ph(0, 16), segs=[(0x800, 0xFFFFFFFFFFFFFFF8, 16, 16)], tail=bytes(16)),
 "seg_index": self_file(1, 64, 1, ph(0, 16), segs=[(0x800 | (5 << 20), 0, 16, 16)], tail=bytes(16)),
 "truncated_128": (self_file(1, 64, 3, ph(0, 16) * 3, segs=[(0x800, 0, 16, 16)]))[:128],
}

# Many program headers must not add up to a justification for a far segment: padding counts only for headers a SELF segment backs,
# and at most 64 MiB in all.
def many(n, align, far, backed, flags=0x800):
    p = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, 1, 1, align) * n + ph(far, 16)
    segs = [(flags | (i << 20), 0, 1, 1) for i in range(min(n, 4095))] if backed else []
    return self_file(len(segs), 64, n + 1, p, segs=segs, tail=bytes(16))
cases["many_phdrs_unbacked"] = many(1024, 1 << 21, (1 << 31) - 16, False)
cases["many_phdrs_backed"] = many(1024, 1 << 21, (1 << 31) - 16, True)
cases["many_phdrs_400m"] = many(200, 1 << 21, 400 << 20, False)
cases["many_phdrs_align0"] = many(60000, 0, 234 << 20, False)
cases["many_phdrs_backed_4095"] = many(4095, 1 << 21, 100 << 20, True)  # 4095 x 2 MiB is capped at 64 MiB: 100 MiB is still refused
# Exact edge of the limit: 32 backed headers of 2 MiB alignment give 64 MiB of padding, so a segment may end at N + 66 MiB (N = the
# file size); one byte more is refused. (The accepted twin, "edge_ok", is made by ../self_valid/gen.py.)
def edge(extra, n=32):
    total = 32 + 32 * n + 64 + (n + 1) * 56 + 16
    return many(n, 1 << 21, total + ((n + 1) << 21) - 16 + extra, True)
cases["edge_plus1"] = edge(1)
cases["edge16_plus1"] = edge(1, 16)  # fewer than 32 backed headers (below the 64 MiB cap): N + 17 * 2 MiB is the edge
cases["noflag_8x2m"] = many(8, 1 << 21, 10 << 20, True, flags=0)  # SELF segment entries without the 0x800 (blocked) flag back nothing
cases["unbacked_8x2m"] = many(8, 1 << 21, 10 << 20, False)  # 8 headers of 2 MiB alignment without a SELF segment count for nothing: 10 MiB is refused
cases["unbacked_32x2m_edge"] = many(32, 1 << 21, (66 << 20) - 16, False)  # the same edge as edge_ok, but with unbacked headers: refused
for k, v in cases.items(): open(f"{d}/{k}.self", "wb").write(v)
print(" ".join(cases))
