# VPEngine GPU — a native graphics translator for Apple Vision Pro

Goal: replace the Vulkan path (shadPS4 renderer_vulkan → KosmicKrisp/MoltenVK → Metal) with a graphics
layer written for visionOS only: Metal 4, Compositor Services, no Vulkan, no code for other platforms.
Everything under `gpu/` is original VPEngine code written from public specifications (AMD's GCN ISA
manuals, Apple's Metal documentation). Emulators and translation layers (shadPS4, Mesa, SPIRV-Cross,
MoltenVK, DXVK…) are not read or copied for it; LLVM's `llvm-mc` is used only as a black-box test
oracle, the same way the CPU tests use the real processor.

## Why (data: AstroVisionPro bitácora, «Informe de rendimiento», ASTRO BOT, M2)
- ~30 ms GPU per frame, GPU 94–100 %. Floor 16–18 ms that does not scale with pixels or vertices = fixed
  cost of Vulkan-on-Metal: ~49–55 render encoders per frame for ~42–46 passes, ALL→ALL barriers, copies
  through compute, RectList/QuadList through compute tessellation, robust-access checks everywhere.
- 816 ≈ 1440 per eye: pixels are not the limit today. Order: remove the fixed cost, then foveation, then
  temporal upscaling. MoltenVK 77 ms vs KK 30 ms: the layer alone is worth 2.5×.

## Architecture

```
 guest GPU (PS4: PM4 + GCN)          other clients later (Windows/D3D, …)
          │                                     │
 gpu/shader/gcn  ──► VPIR ◄── (other frontends) │
                      │                         │
               gpu/shader/msl                   │
                      ▼                         ▼
                 gpu/core  (pass graph, resources, hazards, pipeline cache, diagnostics)
                      │
                 gpu/metal (Metal 4 backend)  ──  gpu/visionos (Compositor Services, foveation, MetalFX)
```

The core and VPIR know nothing about the PS4. A client is a thin adapter (`integrations/<client>/`).

### VPIR — VPEngine shader IR (`gpu/shader/ir`)
- Typed SSA values inside **structured** control flow (`if` / `loop` / `break` / `continue` / `return`
  tree, no goto, no phi: values that cross regions go through function-local variables). Structured IR
  maps 1:1 to MSL and keeps the emitter trivial; structurization is the frontend's job.
- Types: `bool`, `i32`, `u32`, `f16`, `f32`, and vectors of 2–4 of them.
- Ops: arithmetic, compare, select, conversions/bitcasts, bitfield, math (fma, min/max/clamp, floor,
  fract, rcp, rsq, sqrt, exp2, log2, sin, cos…), byte-address buffer load/store/atomics, image
  sample/gather/fetch/store/size (lod, bias, grad, offset, compare), interpolation, derivatives,
  barriers, discard, subgroup ops (ballot, broadcast, read-first, any/all: GCN's wave ops map to
  Metal SIMD-group functions).
- Every instruction carries `precise`; the emitter only keeps strict math where it is set.
- Resources are slots (kind, index); the backend maps slots to Metal argument buffers.
- Stages: vertex, fragment, compute, object, mesh (geometry shaders become object+mesh).
- Canonical text form with parser and printer: every test is a readable `.vpir` file.

### GCN frontend (`gpu/shader/gcn`)
- Decoder + disassembler for GCN 2 (Sea Islands, the PS4's Liverpool), verified bit for bit against
  `llvm-mc -mcpu=bonaire` on every encoding family.
- Lifting to VPIR: SGPR/VGPR/EXEC/VCC/SCC as variables, uniform branches as structured `if`/`loop`,
  divergent control as EXEC-masked regions.

### MSL emitter (`gpu/shader/msl`)
VPIR → Metal Shading Language text. Fast math by default, `precise::` where marked.

### gpu/core (`gpu/core`, platform-neutral C++)
- Resources: buffers in shared storage (unified memory: guest memory mapped directly, no staging),
  textures (format table; detile/swizzle only when needed), samplers, residency sets.
- Pass graph: the client declares passes with attachments and load/store actions; the core merges
  consecutive passes with the same attachments into one encoder and batches copies before/after.
- Hazards: tracked per resource (last writer + stage) → a barrier only where a real hazard exists.
- Pipelines: key → compiled state; disk cache in VPS4; background compilation with priorities.
- Diagnostics in the log from day one: per-pass GPU time, encoders per frame with the reason each was
  cut, pipeline compile times, binding validation. (No Mac, no debugger: the log is the debugger.)
- The graph/hazard/merge logic is pure C++ and is unit-tested on Linux; only `gpu/metal` needs Apple.

### gpu/visionos
Compositor Services presenter: eye images rendered straight into drawable textures; foveation = the
drawable's rasterization rate map on the scene passes; MetalFX temporal upscaler (depth + motion
vectors from depth and head pose delta; per-title switch for ghosting).

### Client: shadPS4 (`integrations/shadps4/metal`)
Thin adapter: Liverpool PM4 → gpu/core calls, guest shaders → gpu/shader/gcn. RectList/QuadList: the
4th vertex in the vertex shader (v1+v2−v0), no tessellation; fans/restart rewritten on the GPU.

## Verification without the headset
- Linux CI: VPIR parser/printer round trips, GCN decoder vs `llvm-mc`, MSL emitter golden files,
  pass-graph/hazard unit tests.
- macOS CI runner: every emitted MSL compiles with `xcrun metal`; small headless renders compared with
  golden images.
- Corpus: ASTRO BOT's shaders (dumped from the logs) through GCN → VPIR → MSL → `xcrun metal`.

## Steps (each testable; device rounds start at 5)
1. VPIR + text form + MSL emitter (Linux tests, macOS compile).
2. GCN decoder/disassembler vs llvm-mc.
3. gpu/core pass graph + hazards (Linux), Metal backend + headless render test (macOS runner).
4. GCN → VPIR lifting; ASTRO BOT shader corpus compiles.
5. shadPS4 adapter behind `ENABLE_VPENGINE_GPU`, KK build untouched; device: menus/logos.
6. ASTRO BOT world: correctness, then encoders/barriers vs KK, then ms.
7. Foveation; 8. MetalFX temporal; 9. thermal fps cap option.

Success criterion per step: log numbers (GPU ms/frame, encoders/frame) against KK's 30 ms / ~50.
