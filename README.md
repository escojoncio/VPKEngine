# VPKEngine

A native translation engine for Apple Vision Pro: it takes code and graphics written for other
platforms and runs them on visionOS without JIT and without MoltenVK. It is one shared layer that
several ports use:

- **Graphics translation to Metal, written only for Vision Pro** (`gpu/`): a pass graph with hazard
  tracking and tile-memory planning (`gpu/core`), a Metal backend (`gpu/metal`, `gpu/metal2`), shader
  front ends for SPIR-V and AMD GCN into one IR with an MSL emitter (`gpu/shader`), and **VPVulkan**,
  VPKEngine's own Vulkan driver (`gpu/vulkan`, see [docs/VULKAN.md](docs/VULKAN.md)). Foveated
  rendering and VPKTAA (temporal antialiasing applied inside the frame, with the exact jitter the
  translator applied) are part of the engine.
  - **VisionQuestPro** runs Quest games (Vulkan) through VPVulkan.
  - **AVP** runs Half-Life: Alyx (Vulkan) through VPVulkan.
  - **AstroVisionPro** draws PS4 games through the GCN path and the PS4 renderer (`gpu/ps4`); its
    adapter lives in the AstroVisionPro repository.
- **Ahead-of-time CPU translation** (`tools/vpaot`): an x86-64 executable becomes C, which clang
  compiles into the app as ordinary arm64 code. No JIT at run time, so no debugger attachment, no
  executable memory, and the translated code is clean, signed pages that do not count against the
  process's memory limit. `tools/vpconvert` does the same on the headset. See
  [docs/PLAN.md](docs/PLAN.md).
- **`runtime/`** — the x86-64 machine state and instruction semantics the translated C calls
  (header-only, exact flags, SSE), and a small host that enters translated code.
- **`platform/visionos/`** — the Swift layer the game apps share: the VPS4 folder, controllers
  (DualSense, DualShock 4, PlayStation VR2 Sense with tracking), conversion on the headset, signing,
  logs, language.
- **`tests/`** — differential tests: every CPU case runs natively on an x86-64 host and as
  translated C, from the same state, and the results must match bit for bit (replayed on ARM in
  CI); GPU and Vulkan driver tests under `gpu/*/tests`.

Status: the translator covers what the PS4's CPU (AMD Jaguar) runs: integer and control flow,
SSE through SSE4.2 (including the string compares and CRC32), SSE4a, AVX at 128 and 256 bits,
F16C, AES-NI, PCLMULQDQ, BMI1, a bit-exact x87 FPU (80-bit, via Berkeley SoftFloat), and locked
operations, which stay atomic even on misaligned addresses. It also handles jump tables, C++
exceptions through the game's own unwinder, and context switches (fibers, coroutines, `longjmp`).

- **Verification:** every case is compared bit for bit with the hardware, on Intel locally and on
  AMD in CI, and replayed on ARM.
- **Coverage on real binaries:**
  - python3 (1.5M instructions) and libstdc++ (1M) translate at 100.00 %.
  - libstdc++ and libc, translated whole, compile with no error.
- **Speed:** translated code runs at 1.05–1.13× the time of native x86 (clang -O2).
- **Embedding:** an app links the runtime (`runtime/`) and the translated modules; the adapter
  between an emulator and this engine belongs to that app's own repository. What a whole game still
  needs is listed in [docs/PLAN.md](docs/PLAN.md) and `bitacora.md`.

No game data is included or needed to build or test.

## Build and test

```sh
git clone --recursive <this repository> && cd VPKEngine
cmake -S tools/vpaot -B build/vpaot -G Ninja && ninja -C build/vpaot
python3 tests/aot/run.py            # needs gcc/clang and binutils; native comparison on x86-64
./build/vpaot/vpaot --elf /bin/ls   # coverage report of any x86-64 ELF (or a PS4 SELF)
```

## License and credits

PolyForm Noncommercial 1.0.0 (see `LICENSE`).

**VPKEngine was conceived entirely by 256k and programmed entirely by Claude.**

### Credit when you use it
If you use VPKEngine or any part of it in your own project, port or fork, please credit it visibly
with the name **VPKEngine**, its author **256k** and a link to this repository. The PolyForm Noncommercial 1.0.0 terms
also require keeping the existing copyright and licence notices in anything you redistribute.

Third-party code it uses (licences and details in [THIRD_PARTY.md](THIRD_PARTY.md)):
- [Zydis](https://github.com/zyantific/zydis) (MIT): x86-64 instruction decoding in the translator.
- [Berkeley SoftFloat 3e](http://www.jhauser.us/arithmetic/SoftFloat.html) (BSD-3-Clause,
  `runtime/softfloat/COPYING.txt`): exact 80-bit x87 arithmetic.
- [Khronos Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) (Apache-2.0 OR MIT): the
  Vulkan API declarations VPVulkan implements.
- [LLVM, clang and lld](https://github.com/llvm/llvm-project) (Apache-2.0 with LLVM exceptions):
  compiler and linker inside `vpconvert`; `llvm-mc` as a test oracle only.

Design references studied, no code taken from them: the N64Recomp and XenonRecomp static
recompilers and bbport. Everything else was written from public specifications (AMD GCN ISA, Vulkan,
SPIR-V, Metal, Mach-O, PKCS#12).
