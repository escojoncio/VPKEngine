# VPVulkan — VPKEngine's own Vulkan driver (`gpu/vulkan`)

A Vulkan 1.3 driver for Apple Vision Pro built on `gpu/core` (pass graph, hazards, pipeline cache) and
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
  pick their mobile Vulkan path. One queue family (graphics+compute+transfer) with 4 VkQueues (Quest: queueCount 4, timestampValidBits 48): each
  keeps its own submission order and all run on the one Metal command queue in the order their batches become ready.
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
5. Multiview → vertex amplification; fragment density map → VPKEngine foveation (rasterization rate maps).

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
  1. **Rate map**: implementado, tras interruptor apagado por defecto: ver «Foveado propio» más abajo.
  2. **VPKTAA**: con `vpktaa.enabled`, el driver crea el adjunto de color del ojo con 1 muestra (se quita el MSAA:
     `samples` → 1 y las resoluciones se convierten en copias) y, al liberar la imagen del swapchain XR
     (`xrReleaseSwapchainImage` → Klepton → hook del driver), encola un pase final VPKTAA (shader `vpktaa` portado de
     PTVisionPro: historial + vectores de movimiento si el juego los da con SpaceWarp) antes de entregar la textura al
     compositor. Hace falta: pipelines/MSL propios del driver (fase 3), textura de historial por ojo y el punto de
     enganche "fin de frame" que exponga Klepton.
     **Estado actual:** el pase VPKTAA vive en el compositor de Klepton (`KleptonTAA.swift`, VisionQuestPro): un
     compute por ojo y frame del juego, historial reproyectado por rotación de cabeza (sin profundidad), clipping de
     varianza YCoCg y nitidez tipo RCAS; vale para cualquier driver. Sin jitter inyectado todavía y el MSAA del juego
     se mantiene (forzar `samples=1` queda pendiente).
  3. **Anisotrópico x16** (`vpvk_config_set_anisotropy`, por defecto 16): `vkCreateSampler` sube `maxAnisotropy` a 16
     en samplers linear/linear con mips (`maxLod > minLod`), sin comparación ni coordenadas no normalizadas.

## Foveado propio (rate map por ojo; interruptor apagado por defecto, sin probar en dispositivo)
Código: `gpu/vulkan/src/foveation.{h,cpp}` (matemática pura, testeada en Linux), `config.cpp` (interruptor, `FoveationState`),
`commands.cpp` (marca de pases), `gpu/metal2/src/metal_backend.mm` (mapa, sombras, plegado/desplegado; **Obj-C++ sin compilar ni probar**).
- **Interruptor**: `vpvk_config_set_foveation_apply(1)` (o `VPVK_FOVEATION=1`; `=0` lo anula) además de un modo distinto de apagado
  en `vpvk_config_set_foveation` y un dispositivo con `supportsRasterizationRateMapWithLayerCount:2` (`vpvk_foveation_active()`).
  Por defecto 0: nada cambia (ni MSL, ni recetas, ni pases). Hay que activarlo **antes** de crear los pipelines.
  `vpvk_config_set_foveation_eyes(const float t[8])` (tangentes l r t b de cada ojo) centra la zona nítida en el centro óptico de
  cada ojo (`x = l/(l+r)`, `y = t/(t+b)`); sin ellas, el centro geométrico. Log `vk.fov plan ... kept= est_saving=`.
- **Qué pases**: los pases de ojo que identifica el registro del VPKTAA nativo (`taaEyePass`: necesita jitter), estéreo de 2 vistas
  (`layerCount` 2), 1x, con color 0 y profundidad. `RenderPassDesc::rateMap` (`vpgpu::RateMap`: tamaño lógico, calidades por zona y
  eje, una serie por capa); su clave forma parte de `sameTargets`/`compatible`: no se funden pases de distinto mapa.
- **Perfil**: radial redondeado y separable por eje (cuadrado redondeado): calidad 1 hasta `sharp*(1-fade)`, caída smootherstep (`fade` 0.15, como el
  `pad.y` del inset de PTVisionPro) hasta la periferia en `sharp`, periferia constante más allá. Presets = PTVisionPro: M2 zona nítida
  60 grados (`sharp 0.481` = tan(30)/1.2) / periferia 0.10 / centro 10 grados bajo el eje (`height -0.0735`); M5 70 grados (`0.583`) /
  0.15 / igual. Personalizado = ajustes de la app; `height` desplaza el centro en vertical (en los presets se suma al suyo).
- **Regla fija** (bitácora): los vertex stages multiview de 2 vistas escriben `[[viewport_array_index]] = vista`
  (`SpirvLowerOptions::viewportIndex`, builtin IR `ViewportIndex`) y `metal2` fija un viewport y un scissor por vista. La capa del
  mapa la elige ese índice, no la capa del attachment.
- **Espacio físico/lógico**: el pase se pinta en attachments físicos más pequeños (sombras `Private`, 2DArray, tamaño máximo de
  `physicalSizeForLayer`) con `rp.rasterizationRateMap`; viewports y scissors siguen en espacio lógico. Un encoder previo **pliega**
  (lógico → físico) lo que el pase carga y otro posterior **despliega** (físico → lógico) lo que guarda: los attachments del juego
  conservan su tamaño, así que el post del juego, el VPKTAA nativo y el compositor ven lo de siempre (fold/expand con los datos del
  mapa en el buffer 0). `FragCoord`: los fragment stages multiview con `Position` reciben `constant rasterization_rate_map_data&
  vp_rrm [[buffer(28)]]` y reconstruyen la posición de pantalla con `map_physical_to_screen_coordinates(pos.xy, capa)`
  (`Module::rateMapPosition`, `SpirvLowerOptions::rateMapPosition`); fuera de un pase foveado se enlaza un mapa identidad.
- **VPKTAA nativo**: trabaja en espacio lógico (ve el ojo ya desplegado): historial, máscara reactiva R8 y nitidez no cambian.
  Coste: la periferia tiene menos resolución real y se despliega con filtro bilineal (color) o lectura directa (profundidad).
- **UI/2D en el buffer del ojo** (clase UI del clasificador del VPKTAA; requiere la máscara UI activa): desde el primer draw de UI el
  pase continúa **sin** mapa, cargando todo, y el texto no pierde resolución. `VPVK_FOVEATION_UI=0` lo desactiva.
- **Se rechaza (el pase va sin foveado; log `gpu.fov decline reason=`, una vez por motivo)**: clears de área o quads de clear (sin
  índice de vista), pipelines sin índice de vista o con `FragCoord` sin mapa, MSAA, attachments de otro tamaño / mip / capa distinta de
  0 / formato entero o con vista de formato, profundidad distinta de D32F o D16 o con stencil, memoryless, sin `ShaderRead` si se carga,
  con resolve, volcados (`dumping`) y mapa no soportado.
- **Logs**: `vk.fov plan` (lógico, perfil, ahorro estimado), `vk.fov frame` (pases foveados, cortes de UI), `gpu.fov map` (lógico vs
  físico, % de fragmentos), `gpu.fov enc` (pliegues y despliegues), `gpu.fov decline`.
- **Límites**: solo 1x y 2 vistas; plegar y desplegar cuesta ancho de banda (sin medir: puede comerse parte del ahorro); los pipelines
  creados con el interruptor apagado no valen (MSL distinto): activarlo invalida su receta/prewarm; el orden entre encoders usa los
  mismos `MTLFence` que el resto (`frontier`), sin verificar en visor.
