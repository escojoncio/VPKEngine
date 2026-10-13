# VPKEngine GPU client for shadPS4 (AstroVisionPro) — design

Goal: shadPS4 draws through VPKEngine (gpu/core + gpu/metal + the GCN lifter) instead of its Vulkan
renderer on KosmicKrisp/MoltenVK, on Apple Vision Pro, selectable in the app, with the Vulkan path
untouched until VPKEngine matches it.

## What native Metal backends of other emulators converge on (the bar to meet)

- **Guest-memory-keyed caches.** Textures, render targets and buffers are found by guest address +
  descriptor (format, size, tiling); a write by the CPU (page protection) or by the GPU (render
  target, storage write) invalidates the overlapping entries. Render targets stay in GPU memory and
  are only written back when the guest reads that memory.
- **Unified memory used in place.** Buffers alias guest memory (no staging copies); only formats the
  GPU cannot read natively are converted, once per upload.
- **Pipelines from a hashed state key**, compiled in the background, kept on disk (here: VPS4), with a
  fallback (skip the draw, never stall the frame) while a pipeline compiles.
- **Few render passes.** Consecutive draws with the same targets share one encoder; barriers only on
  real hazards (gpu/core already does both: frame compiler, pass merging, tile-memory G-buffer).
- **Capture and replay.** A frame's command stream + touched memory saved on the device and replayed
  off-device is how these backends are debugged without the hardware at hand.

Common bottlenecks to design against: detiling cost and its stalls; the same memory used as colour,
depth and texture (aliasing); CPU waits on GPU writes (labels, fences, readbacks: ASTRO BOT's frame
labels, KITCHEN's raw layer copies); shader compile hitches; per-draw descriptor churn.

## Options considered

1. **A VPKEngine renderer behind shadPS4's Liverpool** (chosen). Liverpool keeps parsing PM4 and
   registers; a new rasterizer turns each draw/dispatch/copy into gpu/core calls. Own texture/buffer
   caches built on gpu/core. Most work, but it removes the whole Vulkan-on-Metal layer, which is where
   ASTRO BOT's fixed cost lives (~16–18 ms floor, ~50 encoders/frame).
2. Keep shadPS4's caches and replace only shaders/pipelines: its caches are Vulkan objects, so this
   keeps the translation layer and its cost. Rejected.
3. Improve the Vulkan driver (KosmicKrisp): decided against (no more KK work).

## Seams in shadPS4 (AstroVisionPro tree)

- `Vulkan::Rasterizer` is referenced by `AmdGpu::Liverpool`, `Core::MemoryManager` and
  `VideoCore::PageManager`. Step 1 extracts an abstract `VideoCore::Rasterizer` with the calls they
  make (Draw, DrawIndirect, DispatchDirect/Indirect, FillBuffer, CopyBuffer, ReadDataFromGds,
  Invalidate/Read/MapMemory, CpSync, Flush, Finish, OnSubmit, markers); the Vulkan one implements it
  unchanged.
- The VR presenter (`core/vr/openxr_host_visionos.mm`) receives eye images as MTLTextures exported
  from Vulkan images; the VPKEngine path hands it MTLTextures directly.
- Shaders: `vpgcn::lift` with the draw's `Shader::Info`/`RuntimeInfo` (already wired in the
  diagnostic shader check, `src/platform/visionos/vpkengine_shader_check.mm`).

## The VPKEngine renderer (src/video_core/renderer_vpkengine/ in the shadPS4 tree)

Per draw: registers (colour/depth targets, blend, depth/stencil, raster, viewport/scissor, primitive,
index buffer) → `vpgpu::RenderPassDesc` + `vpgpu::Draw`; each shader stage → lifted module (cached by
program hash + the runtime info it depends on) → `vpgpu::PipelineKey`; each `vpgcn::Binding` → the
descriptor read at its source (user data / pointer path / program image) → a buffer, texture or
sampler from the caches. Compute dispatches the same way. Copies and fills go to gpu/core copies.
Labels/EOP events the guest waits on are written at the end of the command buffer that produced them.

Texture cache: key = (address, T# shape and format); linear textures alias guest memory; tiled ones
are detiled into a private texture on first use and on CPU writes (page-protected). Render targets
are created on first bind and written back to guest memory only when the CPU or a raw copy reads them.

## Milestones (each one a build the user can test; the log says what happened)

1. Interface extraction (no behaviour change). Setting «Motor gráfico: KosmicKrisp / VPKEngine
   (experimental)», VPKEngine path logs what it would draw.
   Status (AstroVisionPro 1514525): `VideoCore::RasterizerInterface` done; in «Modo diagnóstico» every
   shader is lifted with the draw's real descriptors and compiled by the headset's Metal, and a draw
   census (`VPKENGINE_DRAWS`) gives the share of draws/dispatches whose shaders VPKEngine runs and the
   shaders blocking the most. The setting and the renderer itself are next.
2. Clears and presentation through VPKEngine (menus' backgrounds, the eye images reach the headset).
3. Draws with lifted shaders, untextured; then linear textures; then tiled textures.
4. Depth, blending, MSAA resolve; compute dispatches.
5. ASTRO BOT world: correctness, then encoders/frame and ms/frame against KosmicKrisp (30 ms, ~50).
6. VPKTAA (gpu/visionos/shaders/vpktaa.metal) and foveation in the presenter.

## Debugging without a Mac: frame capture/replay

A diagnostic shortcut writes one frame (PM4 + registers + every memory range it reads + the shader
programs) to VPS4 on the headset. Decided: captures are game data and never go to any repository or
CI storage; the user hands a capture over directly and it is replayed and inspected in the dev
sandbox (gpu/core on Linux with the VPIR interpreter; Metal only on the headset itself).
