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
                 gpu/metal (Metal 4 backend)  ──  gpu/visionos (Compositor Services, foveation, VPKTAA)
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
- Lifting to VPIR (`vpgcn/lift.h`): one VPIR invocation per lane; SGPR/VGPR/M0/SCC as variables,
  EXEC/VCC/mask pairs as per-lane bools, VGPR writes EXEC-predicated once EXEC is written. Acyclic
  CFGs become nested `if`s (immediate post-dominators); other CFGs a dispatch loop. Descriptors are
  traced to user data or a constant offset from a user-data pointer and become resource slots with
  a `Binding` the client resolves per draw.

### MSL emitter (`gpu/shader/msl`)
VPIR → Metal Shading Language text. Fast math by default, `precise::` where marked.
- Optimizer (`vpir::optimize`, run by `emitMsl` on a copy): constant folding, exact identities, CSE,
  dead pure values; `precise` math is never reassociated. The module is verified before and again
  after it, so a pass bug is an error instead of bad MSL.
- `framebuffer_fetch n` reads colour attachment n at the fragment's own pixel; it becomes an entry
  parameter `[[color(n)]]`. `vpir::toFramebufferFetch` rewrites provable own-pixel G-buffer reads
  (fetch at `position.xy`, nearest sample at `position.xy / size`) into it and reports the rest.
- `EmitResult::shader.colorOutputs`: bit n = writes `[[color(n)]]`; clients copy it to
  `MslShader::colorOutputs` so unwritten attachments keep their contents.

### Argument-table contract (MSL emitter ↔ Metal backend)
One table per stage at `[[buffer(argumentBufferIndex)]]`; one 8-byte entry per declared resource at
its listed offset: buffers = GPU address + range begin, textures/samplers = `MTLResourceID`. The
backend fills each declared (kind, slot) from the draw's bindings; a missing one skips the draw
(`gpu.err`). Pixel-local bindings are never in the table: they are `[[color(n)]]` inputs, and a
draw's fetches must equal what its pipeline was specialised for. Needs Metal 3 + argument-buffer
tier 2. Resources are resident through a residency set; without one (simulator) each encoder gets
`useResource` for the resources its tables reach.

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
- Tile-memory G-buffer (Apple TBDR): a pass that reads the open encoder's attachments only
  pixel-locally (`Access::pixelLocal`) merges into it. An attachment released inside that encoder
  (store DontCare/Resolve or `TextureDesc::transient`) and not `persistent` becomes tile-local:
  memoryless, never stored; non-persistent render targets get memory only on first real use.
  Merges are capped by the tile budget (`CompileOptions::tileBytesPerPixel`, default 64 B per
  pixel and sample).

### gpu/visionos
Compositor Services presenter: eye images rendered straight into drawable textures, composed on the
render's own command queue; real depth to the compositor; EDR/P3. Temporal antialiasing = VPKTAA (own,
from PTVisionPro: one pass per view replacing the resolve; jitter injected into the translated vertex
position, camera motion from depth and head pose delta, reactive mask from blended draws), not MetalFX.
Foveation: to be measured (rate map on geometry passes only vs. quad views); see bitácora.

### Client: shadPS4 (`integrations/shadps4/metal`)
Thin adapter: Liverpool PM4 → gpu/core calls, guest shaders → gpu/shader/gcn. RectList/QuadList: the
4th vertex in the vertex shader (v1+v2−v0), no tessellation; fans/restart rewritten on the GPU.

## Verification without the headset
- Linux CI: VPIR parser/printer round trips, GCN decoder vs `llvm-mc`, MSL emitter golden files,
  pass-graph/hazard unit tests.
- `visionos-device` job: everything built for the headset SDK (xros, warnings are errors); every
  golden MSL compiles with `xcrun metal` and links into one `.metallib`.
- `visionos-simulator` job: built for the simulator SDK; all tests, including headless Metal renders
  checked pixel by pixel, run inside an Apple Vision Pro simulator. Groups the simulator GPU lacks
  (tile memory, GPU frame times) print `SKIP` and exit 77 (CTest: skipped); they need the device.
- Corpus: ASTRO BOT's shaders (dumped from the logs) through GCN → VPIR → MSL → `xcrun metal`.

## Steps (each testable; device rounds start at 5)
1. VPIR + text form + MSL emitter (Linux tests, visionos-device compile).
2. GCN decoder/disassembler vs llvm-mc.
3. gpu/core pass graph + hazards (Linux), Metal backend + headless render tests (visionos-simulator).
4. GCN → VPIR lifting; ASTRO BOT shader corpus compiles.
5. shadPS4 adapter behind `ENABLE_VPENGINE_GPU`, KK build untouched; device: menus/logos.
6. ASTRO BOT world: correctness, then encoders/barriers vs KK, then ms.
7. Foveation; 8. VPKTAA temporal antialiasing; 9. thermal fps cap option.

Success criterion per step: log numbers (GPU ms/frame, encoders/frame) against KK's 30 ms / ~50.
