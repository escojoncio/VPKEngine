# VPKEngine

VPKEngine is a native, JIT-free translation engine for Apple Vision Pro / visionOS. It translates
GPU APIs (Vulkan, OpenGL ES, AMD GCN) to Metal and translates x86-64 executables ahead of time into
C that clang compiles as ordinary arm64 code. Everything is own code written from public
specifications, apart from the third-party components listed in [THIRD_PARTY.md](THIRD_PARTY.md): no MoltenVK, no SPIRV-Cross, no Mesa.

## Capabilities

### Graphics (`gpu/`)

- **VPVulkan**, an own Vulkan driver over Metal (`gpu/vulkan`, see [docs/VULKAN.md](docs/VULKAN.md)):
  reports Vulkan 1.3 by default (a 1.1 Adreno-compatible identity is opt-in), with dynamic
  state, buffer device address, imageless framebuffers, sync2, timeline semaphores, subgroup size
  control and more; exports the standard ICD entry points.
- **OpenGL ES 3.1 / EGL 1.5** (`gpu/gles`, library `vpgles`) with its own GLSL ES 3.10/3.20
  compiler that emits SPIR-V and runs through VPVulkan (compute, SSBOs, images and atomics,
  framebuffer fetch).
- **Shader pipeline** (`gpu/shader`): SPIR-V and AMD GCN (GCN 2 decoder, disassembler and lifter)
  are lowered into VPIR, VPKEngine's own structured SSA IR, which has an optimizer and an MSL
  emitter. The GCN decoder is verified against `llvm-mc` (used only as a test oracle, not shipped), and the lifter with differential tests.
- **PS4 GCN renderer layer** (`gpu/ps4`, library `vpps4`): GCN draw state to engine calls, guest
  shaders through the GCN front end. Work in progress (geometry/tessellation stages are missing).
- **Pass graph core** (`gpu/core`, platform-neutral C++): merges compatible passes into one encoder,
  places barriers only where a real hazard exists, plans tile memory (memoryless, tile-local
  attachments), keeps a pipeline cache and writes per-pass diagnostics. It is unit-tested on Linux.
- **Metal backend** (`gpu/metal`, `gpu/metal2`): tile/memoryless MSAA, multiview through vertex
  amplification (opt-in, experimental), asynchronous pipeline creation, shader-recipe and `MTLBinaryArchive` caches with
  background prewarm, command-buffer budgeting and hang splitting, device-loss handling, memory
  trimming. The Metal code (`.mm` / `.metal`) is built by macOS CI, and several of its features are
  not yet validated on hardware.
- **Visual quality**: native VPKTAA temporal antialiasing (Halton jitter, YCoCg variance clipping,
  disocclusion, UI mask), RCAS sharpening, and own foveated rendering with per-eye rasterization
  rate maps (experimental: off by default and not yet validated on a headset).
- **Tooling**: frame capture (VPCAP) and replay of the Vulkan call stream, image diff tools,
  Vulkan CTS runner scripts.

### CPU (`tools/`, `runtime/`)

- **`vpaot`**: ahead-of-time x86-64 to C translator. One C function per function in the binary,
  lazy flags, jump tables, C++ exception landing pads, context switches (fibers, coroutines,
  `longjmp`). Covers integer code, SSE through SSE4.2, SSE4a, AVX at 128 and 256 bits, F16C,
  AES-NI, PCLMULQDQ, BMI1, atomic locked operations and TSO memory ordering.
- **x87** with bit-exact 80-bit arithmetic through Berkeley SoftFloat.
- **`vpconvert`**: the same translation as a resumable on-device converter with an in-process compiler and
  linker (not yet validated on hardware).
- **Runtime** (`runtime/`): x86-64 machine state and instruction semantics (header-only), ELF and
  PE32+ loading, on-demand translation, and code signing (CodeDirectory, CMS, PKCS#12) so the
  translated code is ordinary signed pages.
- **Verification**: every CPU case runs natively on an x86-64 host and as translated C from the
  same state, and the results must match bit for bit (replayed on ARM in CI). On a sort/hash/float
  benchmark, translated code ran at 1.05 to 1.13 times the time of native x86 (clang -O2).

## Layout

| Directory | Contents |
|---|---|
| `gpu/core` | Pass graph, resources, hazards, pipeline cache, diagnostics |
| `gpu/metal`, `gpu/metal2` | Metal backends |
| `gpu/shader` | VPIR, SPIR-V and GCN front ends, MSL emitter |
| `gpu/vulkan` | VPVulkan driver, capture and replay tools |
| `gpu/gles` | OpenGL ES / EGL implementation and GLSL compiler |
| `gpu/ps4` | PS4 GCN renderer layer |
| `gpu/visionos` | visionOS presentation shaders (VPKTAA) |
| `tools/vpaot`, `tools/vpconvert` | x86-64 to C translators (host and on-device) |
| `runtime/` | Machine state, instruction semantics, loader, code signing, SoftFloat |
| `platform/visionos` | visionOS support layer (Swift) |
| `tests/` | Differential CPU tests; GPU tests live under `gpu/*/tests` |
| `docs/`, `notas/` | Design documents and technical notes |

## Build and test

```sh
git clone --recursive https://github.com/escojoncio/VPKEngine && cd vpkengine
cmake -S tools/vpaot -B build/vpaot -G Ninja && ninja -C build/vpaot
python3 tests/aot/run.py            # needs gcc/clang and binutils; native comparison on x86-64
./build/vpaot/vpaot --elf /bin/ls   # coverage report of any x86-64 ELF
```

No game data is included or needed to build or test.

## Authorship

**VPKEngine was conceived and directed entirely by 256k — architecture, design and every technical decision. The code was written by Claude under his direction.**

256k on X: [@256k](https://x.com/256k) / [@256kilustracion](https://x.com/256kilustracion).

## License

VPKEngine is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). It is free for
noncommercial use only.

- The `Required Notice` line in `LICENSE` (VPKEngine, 256k, link to this repository) must be kept
  in every copy, fork or derivative. This is binding.
- Visible credit in your documentation or UI (the name **VPKEngine**, its author **256k** and a link
  to this repository) is requested.
- Any commercial use requires a separate written license from 256k. To request one, open an issue
  on this repository.

### Third-party code

These components keep their own licenses (details in [THIRD_PARTY.md](THIRD_PARTY.md)):

- [Zydis](https://github.com/zyantific/zydis) (MIT): x86-64 instruction decoding.
- [Berkeley SoftFloat 3e](http://www.jhauser.us/arithmetic/SoftFloat.html) (BSD-3-Clause): exact
  80-bit x87 arithmetic.
- [Khronos Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) (Apache-2.0 OR MIT): the
  Vulkan API declarations.
- [LLVM, clang and lld](https://github.com/llvm/llvm-project) (Apache-2.0 with LLVM exceptions):
  compiler and linker inside `vpconvert`.
