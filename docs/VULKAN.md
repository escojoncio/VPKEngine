# VPVulkan — VPEngine's own Vulkan driver (`gpu/vulkan`)

A Vulkan 1.1 driver for Apple Vision Pro built on `gpu/core` (pass graph, hazards, pipeline cache) and
`gpu/metal`. Goal: Quest titles (Vulkan, Unity/Unreal) running through Klepton load `libVPVulkan.dylib`
where they loaded MoltenVK today. Written from the Vulkan specification; uses only the Khronos headers
(`third_party/vulkan-headers`). No MoltenVK, Mesa or SPIRV-Cross code.

## Loading
- The dylib exports every implemented `vk*` under its own name plus `vkGetInstanceProcAddr`,
  `vkGetDeviceProcAddr`, `vk_icdGetInstanceProcAddr`, `vk_icdNegotiateLoaderICDInterfaceVersion`.
  Internals are hidden (`-fvisibility=hidden`).
- Klepton (`runtime/gfx/kl_vulkan.c`): `KL_MVK_DYLIB=<path>/libVPVulkan.dylib` makes it load this
  driver unchanged. kl_vulkan.c resolves most entry points through the proc-addr functions and a few
  vkCmd* (`vkCmdCopyImage`, `vkCmdBlitImage`, `vkCmdBeginRendering`, …) with `dlsym`; unknown names
  return null and log `vk.unimpl <name>` once.
- Dispatchable handles start with the ICD loader word, so a Vulkan loader can sit in front later.

## Model
- **One physical device**, reported as Adreno 740 (vendor 0x5143) with Quest-3-like limits so engines
  pick their mobile Vulkan path. One queue family (graphics+compute+transfer), one queue.
- **Memory**: one DEVICE_LOCAL heap; types 0 device, 1 host-visible coherent, 2 +cached, 3 lazily
  allocated (transient attachments only). Every `VkDeviceMemory` is page-aligned (16 KiB) host memory;
  a bound `VkBuffer` is a core buffer aliasing `memory + offset` (`BufferDesc::guestMemory`, Metal
  `newBufferWithBytesNoCopy`): no staging, no flush. Images own Metal textures (private, tiled);
  their `VkDeviceMemory` only accounts for them (image/buffer aliasing of one allocation: not yet).
- **Command buffers** are deferred op lists (`src/commands.h`). `vkQueueSubmit` translates all its
  command buffers into **one** core `Frame`, so the frame compiler merges passes across command buffers
  and places barriers only for real hazards. `vkCmdPipelineBarrier` is counted, never executed.
- **Render passes**: subpass 0 → one core pass. `STORE_OP_DONT_CARE/NONE` → `StoreAction::DontCare`
  (the core keeps it on tile); `TRANSIENT_ATTACHMENT` images are `transient`, all others `persistent`.
  Multiview view mask → `layerCount = popcount(mask)` (backend: vertex amplification / layered RT).
- **Sync**: single in-order queue, so semaphores need no GPU work. Fences are signalled at submit and
  `vkWaitForFences` waits for the device to go idle (correct, coarse; per-submission completion is
  pending).
- **VK_EXT_metal_objects**: `vkExportMetalObjectsEXT` returns the image's MTLTexture / the MTLDevice
  through `vpvk::Platform` (needs the accessor in gpu/metal, see pending).

## Log lines (`vk.` prefix, stderr → app log)
`vk.instance`, `vk.device` (enabled multiview / fdm / metal_objects), `vk.submit` (first 3, every 600th,
and any with dropped or skipped work: passes, copies, clears, Vulkan barriers dropped, draws/dispatches
skipped), `vk.err what=…`, `vk.unimpl <name>`.

## Shaders: what the SPIR-V reader (`gpu/shader/spirv`) must provide
Phase 3 needs, per `VkShaderModule` + entry point + specialisation constants:
```
// Parse once at vkCreateShaderModule; lower per pipeline (specialisation, multiview).
struct SpirvModule;                       // opaque, owns the parsed words
bool spirv::parse(const uint32_t* words, size_t count, SpirvModule&, std::string& error);
struct SpirvLowerOptions {
  std::string entry;                      // OpEntryPoint name
  vpir::Stage stage;
  std::vector<std::pair<uint32_t, uint64_t>> specConstants;  // SpecId -> value bits
  uint32_t viewCount = 1;                 // multiview: ViewIndex → amplification id
};
// Resource interface: every (set, binding[, array element]) used, its kind, and the VPIR slot it
// became, so the driver can map descriptor sets to argument-table slots; push constants as one
// constant-buffer slot; vertex inputs (location → format) for the vertex fetch the driver emits.
struct SpirvBinding { uint32_t set, binding, arraySize; vpgpu::BindKind kind; uint16_t slot; };
struct SpirvInterface {
  std::vector<SpirvBinding> bindings;
  int pushConstantSlot = -1;             // ConstantBuffer slot, -1 = none
  std::vector<std::pair<uint32_t, uint32_t>> vertexInputs;  // location -> component count/type
  uint8_t colorOutputs = 0;              // fragment: written locations
  std::array<uint32_t, 3> localSize{};   // compute
};
bool spirv::lower(const SpirvModule&, const SpirvLowerOptions&, vpir::Module& out,
                  SpirvInterface& iface, std::string& error);
```
Then `vpmsl::emitMsl` → `metal::MslShader` through a `ShaderProvider` the driver owns.
Needed SPIR-V features for Unity mobile: Shader, Matrix, ImageQuery, SampledBuffer, MultiView
(`ViewIndex`), InputAttachment (→ framebuffer fetch), `FragCoord`, `FrontFacing`, combined image
samplers, uniform/storage buffers, push constants, spec constants.

## Phases
1. Instance, physical device, device, queue, memory, buffers, images, views, samplers, metal_objects — **done (Linux tests)**.
2. Command pools/buffers as deferred lists, render passes, framebuffers, copies/clears/uploads → core
   frame — **done for subpass 0**; pending: multi-subpass (input attachments → pixel-local),
   `vkCmdBeginRendering`, image copies/blits/resolves, word fills, secondary command buffers.
3. Descriptor set layouts/pools/sets, update templates, pipeline layouts, push constants, graphics and
   compute pipelines (SPIR-V reader), dynamic state, draws/dispatches → core `Draw`/`Dispatch`.
4. Sync: per-submission completion for fences (backend completion handler), events, queries/timestamps.
5. Multiview → vertex amplification; fragment density map → VPEngine foveation (rasterization rate maps).

## Foveado y VPKTAA (configuración del anfitrión)
- El foveado del juego (Quest) queda **siempre apagado**: `VK_EXT_fragment_density_map`/`_2` no se anuncian y
  `fragmentDensityMap*` = `VK_FALSE`. Si el juego los habilita igualmente, `vkCreateDevice` los quita de la lista
  (log `vk.fov ignored=device_ext`) y un `VkRenderPassFragmentDensityMapCreateInfoEXT` se ignora
  (`vk.fov ignored=render_pass_density_map`, `RenderPassObj::densityMap` queda en -1; el adjunto queda sin uso).
- API C `include/vpvk/config.h` (`src/config.cpp`), llamada por la app (VisionQuestPro: ajustes dentro de la app):
  `vpvk_config_set_foveation` (off/M2/M5/personalizado: zona nítida, periferia, altura), `vpvk_config_set_vpktaa`
  (activado, porcentaje, nitidez), `vpvk_config_set_render_scale`, `vpvk_config_set_asw_disabled`,
  `vpvk_config_set_quest3_profile`, `vpvk_config_get`, `vpvk_config_generation`. Valores acotados, log `vk.fov`/`vk.cfg`.
- Dónde se aplicará (pendiente, depende de fases 2b/3 y del backend Metal):
  1. **Rate map**: en cada render pass multiview de ojo (subpasada con `viewMask` 0b11 cuyo color acaba en la imagen
     del swapchain XR), el driver construye/reutiliza un `MTLRasterizationRateMap` (2 capas) a partir de
     `vpvk_config_get().foveation` cuando cambia `vpvk_config_generation()`, igual que PTVisionPro
     (presets M2/M5, zona nítida/periferia/altura) o el rate map del drawable de CompositorServices si se entrega
     directamente; se pasa en `MTLRenderPassDescriptor.rasterizationRateMap`. Hace falta: campo en el pase de gpu/core,
     soporte en gpu/metal, y que los shaders que leen `FragCoord`/muestrean el color del ojo usen coordenadas
     físicas↔lógicas (`map_screen_to_physical_coordinates`) — o reescalar con un pase de des-foveado antes del compositor.
  2. **VPKTAA**: con `vpktaa.enabled`, el driver crea el adjunto de color del ojo con 1 muestra (se quita el MSAA:
     `samples` → 1 y las resoluciones se convierten en copias) y, al liberar la imagen del swapchain XR
     (`xrReleaseSwapchainImage` → Klepton → hook del driver), encola un pase final VPKTAA (shader `vpktaa` portado de
     PTVisionPro: historial + vectores de movimiento si el juego los da con SpaceWarp) antes de entregar la textura al
     compositor. Hace falta: pipelines/MSL propios del driver (fase 3), textura de historial por ojo y el punto de
     enganche "fin de frame" que exponga Klepton.
