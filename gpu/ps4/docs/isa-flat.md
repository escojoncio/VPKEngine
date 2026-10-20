# FLAT memory instructions

All 46 FLAT opcodes lift (`engine/gcn/src/lift_flat.inc`). A FLAT instruction addresses memory with a raw
64-bit pointer in a VGPR pair. On the console that pointer is a guest GPU virtual address, meaningless to the
host GPU, so the access goes through a guest-memory map.

## Design

1. Lifter: `liftFlat` asks for one extra resource, a constant buffer with `DescriptorSource::GuestMap`
   (`engine/gcn/include/vpgcn/lift.h`, layout in `kGuestMapEntries`): `u32 count`, pad, then `count` entries
   `{baseLo, baseHi, sizeLo, sizeHi, devLo, devHi, 0, 0}` sorted by guest base, always bound at
   `kGuestMapBytes` (16 + 32 * 256).
2. Shader: an unrolled 8-step binary search finds the last region whose base is <= the pointer, checks that
   `[va, va + bytes)` lies inside it and forms `dev + (va - base)` as the u32x2 address of VPIR's `device`
   memory target (`kTargetDevice`: Metal `gpuAddress` pointers / Vulkan buffer device address). Outside every
   region loads return 0 and stores/atomics are dropped (the hardware would fault). The backend does not
   bounds-check device accesses, so this check is the only one.
3. Renderer (`src/renderer.cpp`, `Impl::guestMap`): builds the table when a stage binds a `GuestMap`:
   every guest buffer the renderer already aliases (`guestBuffers`: V#s, indirect data, index buffers...) plus the
   regions that 64-bit pairs in the stage's user data point at (`region` hook then `guestBuffer`), with
   `Device::bufferAddress` as the device address. The table goes to the upload ring; each region is also added to
   the draw's accesses as a write on slot `0xffff` (no shader slot) so hazards and the busy ranges cover them.
   If a buffer has no device address (`bufferAddress == 0`: Metal without the residency set, Vulkan without
   `bufferDeviceAddress`) the draw is skipped with a reason. The null backend has fake ones.

Limit: a pointer reached only through guest memory (a pointer stored in a buffer) reads zeros until some other
binding maps its region. A scan of user-data-pointed memory could widen this later.

## Semantics

* loads: ubyte/sbyte/ushort/sshort/dword/x2/x3/x4; sub-dword loads read the aligned dword and extract.
* stores: dword..x4 are plain stores; byte/short are `and`/`or` atomics on the aligned dword.
* 32-bit atomics: swap add sub smin umin smax umax and or xor map to VPIR atomics; inc/dec/fmin/fmax/fcmpswap are
  compare-exchange loops; cmpswap takes `{swap, compare}`. `glc` returns the old value.
* 64-bit atomics: add/sub (carry through the low dword), and/or/xor/swap are atomic per dword; min/max/inc/dec/
  cmpswap/fmin/fmax/fcmpswap are a non-atomic read-modify-write. Each is noted by the lifter.
* No aperture split (flat_scratch / LDS / private): every pointer is global memory. EXEC is honoured (stores and
  atomics are guarded, loads are not). `tfe` writes 0 to the extra register.

## Tests

`gcn.flat.lowering` (`engine/gcn/tests/diff/flat_test.cpp`: 32 hand-encoded programs on the VPIR interpreter with a
guest map: widths, sub-dword, every atomic, carry/borrow, outside the map, crossing the region end),
`vpps4.flat_access` (renderer on the null backend), golden lift `lift/cs_flat.*`.
