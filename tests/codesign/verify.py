#!/usr/bin/env python3
"""Independent check of a signature written by runtime/vp_codesign.c: parses the Mach-O and the
SuperBlob, recomputes every page hash and the special slots, checks the CodeDirectory fields and
the cdhash attributes, and writes the CMS and the CodeDirectory out so that `openssl cms -verify`
can check the signature itself.
    verify.py FILE OUTDIR TEAM IDENTIFIER"""
import hashlib, plistlib, struct, sys

path, outdir, team, ident = sys.argv[1:5]
data = open(path, "rb").read()
magic, cpu, sub, ftype, ncmds, sizeofcmds = struct.unpack_from("<IiiIII", data, 0)
assert magic == 0xFEEDFACF and cpu == 0x0100000C, "not arm64 Mach-O"
off = 32
cs = linkedit = text = None
for _ in range(ncmds):
    cmd, size = struct.unpack_from("<II", data, off)
    if cmd == 0x1D:
        cs = struct.unpack_from("<II", data, off + 8)
    if cmd == 0x19:
        name = data[off + 8:off + 24].rstrip(b"\0")
        vmaddr, vmsize, fileoff, filesize = struct.unpack_from("<QQQQ", data, off + 24)
        if name == b"__LINKEDIT": linkedit = (vmsize, fileoff, filesize)
        if name == b"__TEXT": text = (fileoff, filesize)
    off += size
dataoff, datasize = cs
assert dataoff + datasize == len(data), "the signature must end the file"
assert linkedit[1] + linkedit[2] == len(data), "__LINKEDIT must end at the end of the file"
assert linkedit[0] >= linkedit[2] and linkedit[0] % 0x4000 == 0, "__LINKEDIT vmsize"
sb = data[dataoff:]
m, length, count = struct.unpack_from(">III", sb, 0)
assert m == 0xFADE0CC0 and length == datasize
slots = dict(struct.unpack_from(">II", sb, 12 + 8 * i) for i in range(count))
assert set(slots) == {0, 2, 0x10000}, slots
def blob(at):
    mg, ln = struct.unpack_from(">II", sb, at)
    return mg, sb[at:at + ln]
mg, cd = blob(slots[0]); assert mg == 0xFADE0C02
mg, req = blob(slots[2]); assert mg == 0xFADE0C01 and req == struct.pack(">III", 0xFADE0C01, 12, 0)
mg, cmsblob = blob(slots[0x10000]); assert mg == 0xFADE0B01
(cmagic, clen, version, flags, hashoff, identoff, nspecial, ncode, codelimit, hsize, htype, plat, pshift,
 _s2, _scatter, teamoff, _s3, _cl64, esbase, eslimit, esflags) = struct.unpack_from(">IIIIIIIIIBBBBIIIIQQQQ", cd, 0)
assert version == 0x20400 and flags == 0 and hsize == 32 and htype == 2 and pshift == 12
assert codelimit == dataoff and ncode == (codelimit + 4095) // 4096
assert cd[identoff:cd.index(b"\0", identoff)].decode() == ident
assert cd[teamoff:cd.index(b"\0", teamoff)].decode() == team
assert (esbase, eslimit, esflags) == (text[0], text[1], 0)
for i in range(ncode):
    page = data[i * 4096:min((i + 1) * 4096, codelimit)]
    assert cd[hashoff + 32 * i:hashoff + 32 * (i + 1)] == hashlib.sha256(page).digest(), f"page {i}"
assert cd[hashoff - 64:hashoff - 32] == hashlib.sha256(req).digest(), "requirements slot"
assert cd[hashoff - 32:hashoff] == bytes(32), "Info.plist slot"
# The DER inside the CMS blob (zero padded after it).
der = cmsblob[8:]
n = der[1]
dlen = (int.from_bytes(der[2:2 + (n & 0x7F)], "big") + 2 + (n & 0x7F)) if n & 0x80 else n + 2
assert all(b == 0 for b in der[dlen:]), "padding after the CMS"
open(f"{outdir}/cms.der", "wb").write(der[:dlen])
open(f"{outdir}/cd.bin", "wb").write(cd)
cdhash = hashlib.sha256(cd).digest()
# The cdhashes property list must be inside the signed attributes.
start = der.find(b"<?xml")
end = der.find(b"</plist>\n", start) + len(b"</plist>\n")
pl = plistlib.loads(der[start:end])
assert pl["cdhashes"] == [cdhash[:20]], "cdhashes attribute"
assert der.find(cdhash) > 0, "full cdhash attribute"
print(f"{path}: {ncode} pages, cdhash {cdhash[:20].hex()}, team {team}: structure OK")
