# VPEngine — architecture and plan

## Why

Every PS4 port to the Vision Pro so far (AstroVisionPro, the planned Bloodborne port) runs the
game's x86-64 code through FEXCore's JIT. On visionOS that costs: a debugger (StikDebug) attached
at every start to get executable memory, a code arena of 0.5–1 GB of dirty pages that count
against the 8 GB process limit, stalls while blocks compile, and FEX's Darwin patches to maintain.
The industry's answer for a fixed program is ahead-of-time translation: Rosetta 2 translates a
whole binary once, N64Recomp and XenonRecomp turn console executables into C++ that a normal
compiler optimises. A PS4 game cannot generate code at run time, so all of its code is static and
can be translated before the app is built.

## Layers

```
 game profile (per game)      camera constants, shader swaps, patches, runtime (lean or generic)
 ─────────────────────────────────────────────────────────────────────────────────────────────
 renderer                     shadPS4 video core + KosmicKrisp, tile-GPU passes, MetalFX,
                              output modes: window / stereo window / immersive
 PS4 runtime                  bbport-style lean HLE per game, or shadPS4's generic HLE
 translated code              C from vpaot, compiled into the app (this repository)
 platform (visionOS)          VPS4 folder, controllers, tracking, logs, language, diagnostics
```

A new game is a profile plus a translation run, not a new port.

## vpaot: x86-64 → C

**Input.** A PS4 SELF (decrypted dump) or any x86-64 ELF; `--raw` blobs for tests. The loader
flattens the segments into one image at the program's own addresses (`image.cpp`).

**Discovery.** Functions are found from the entry point, every relocation whose value points
into executable memory (vtables, function pointers), every `.eh_frame` FDE start, and the direct
`call` targets met while decoding. Inside a function, blocks are split at branch targets; jump
tables are read from the `lea T(%rip) / movslq (T,i,4) / add / jmp *` (PIE) and `jmp *T(,i,8)`
shapes, bounded by the preceding `cmp $N`.

**Output.** One C function per guest function: `static void fn_<addr>(VpCpu* cpu, uint32_t
entry)`; blocks are labels, branches `goto`, direct calls C calls, `ret` pops the return address
into `cpu->rip` and returns; the caller checks it came back to the expected address (an unwind to
somewhere else returns up the C stack until a dispatcher can continue). Indirect calls and
unresolved jumps go through `vp_dispatch`, a sorted table of every translated function.

**Semantics.** `runtime/vp_cpu.h`: registers, exact flags for every ALU operation (CF/PF/ZF/SF/OF;
AF kept where defined), partial-register writes (32-bit zero-extends, 8/16-bit keep the rest,
AH–BH), MUL/DIV in 128-bit, SSE scalar and packed arithmetic in C floats/doubles, conversions
with x86's out-of-range "integer indefinite", COMISS flags. The host compiler inlines all of it
and drops the flag computations nothing reads; whole functions are optimised, which a block JIT
cannot do.

**Memory.** Guest addresses are host addresses: the console's memory is mapped flat where the
game expects it (below 1 TiB, as bbport and shadPS4 do). `VP_TSO=1` turns aligned loads and
stores into acquire/release accesses (LRCPC on Apple silicon) for games that need x86 ordering.

**Verification.** `tests/aot/`: each case runs natively on x86-64 (an assembly trampoline loads
and saves the whole state) and as translated C, from identical registers, stack and scratch
memory; registers, flags and a hash of the memory must match. Cases include compiled C at -O2
(sorting, hashing, 128-bit arithmetic, float/double math, structs, jump tables). The arm64 job
replays the translated C against the recorded x86 states.

### Done

- Loader: ELF64, PS4 SELF wrapper, PT_GNU_EH_FRAME starts, DT_RELA / PS4 dynlib relocations.
- Discovery with jump tables; per-function entry switch for extra entries (landing pads).
- Integer ISA (ALU, shifts/rotates, mul/div, bit ops, setcc/cmovcc, string ops, cpuid/rdtsc),
  SSE/SSE2 moves and arithmetic, conversions, compares, shuffles, a few SSE2 integer ops.
- Differential suite (9 cases) and CI (x86 differential + arm64 replay, on demand only).
- Coverage on real compiler output: 99.6 % of `/bin/ls`'s instructions; the rest is x87.

### Next, in order

1. **Run a whole program**: a static test program (printf-free) translated and run end to end;
   then the PS4 import mechanism: an import stub's address maps to a native function in
   `vp_dispatch` (the lean runtime of bbport provides them).
2. **Landing pads**: parse the LSDA of each FDE so that C++ exception handlers enter mid-function
   through the entry switch; the guest unwinder (PS4 libc, linked into the image as bbport does)
   restores registers and jumps to them.
3. **Coverage for a game**: run `vpaot --elf eboot.bin` on the real executable (on the owner's PC;
   nothing of it goes to the repository) and read `unsupported_by_mnemonic`: expected gaps are
   SSE3/SSSE3/SSE4.1 integer ops (pshufb, pmulld, blend, round), movbe, and AVX if the game uses
   it (the PS4 Jaguar has AVX; most games were built for SSE4.2). Add them from the report.
4. **Performance**: lazy flags (keep the last ALU result and compute flags only at the reading
   instruction); keep `cpu->r[]` in locals across a block; `VP_TSO` only where a game needs it;
   measure against FEX on the same hardware.
5. **The app**: the translated C compiled into the game's Xcode target; the lean runtime ported
   from Linux to Darwin (memory via a Mach memory object, threads, files, audio, pad); the
   renderer from AstroVisionPro (KosmicKrisp, MetalFX, GPU_PASSES).
6. **Output modes**: window (RealityView plane), real stereo window (two cameras per frame with
   the game clock frozen for the second eye), immersive (AstroVisionPro's compositor path).

### Known limits

- x87 (fld/fstp…) is not translated: PS4 games use SSE; `long double` paths fault with a report.
- Self-modifying or runtime-generated code cannot exist on a PS4 and is not supported.
- A `ret` to an address that is not the caller's return address (longjmp, exceptions) relies on
  the dispatcher knowing the target as a function entry or a landing pad.
