# Bitácora BB — Bloodborne en Apple Vision Pro

Análisis y decisiones previas al código. El port se hará sobre VPEngine (este repo) y la capa de
AstroVisionPro; el repo del juego se creará cuando Astro sea jugable con el motor AOT.

## Fuentes revisadas

- `MaSieS4Fun/ARM_bloodborne_pc`, rama `arm64-fex`: fork de bbport (deadinside28). El código
  x86-64 del juego corre en el JIT de FEXCore; runtime y GPU (Vulkan) nativos aarch64. Solo
  CUSA03173 v1.09. Páginas de 16 KB (como Apple). Direcciones del juego fijas entre 64 GiB y
  ~1 TiB (`src/runtime_memory.c`: `USER_MIN 0x1000000000`, `USER_MAX 0xfc00000000`). Memoria
  con `memfd`/`dma-buf` (solo Linux). Ficheros con APIs solo Linux: `probe.c`, `runtime_cpu.c`,
  `runtime_kernel.c`, `runtime_memory.c`, `runtime_services.c`. Preparación del eboot: scripts
  Python (`scripts/prepare.py`: SELF → imagen plana + relocaciones + imports). Sin medidas en
  ningún dispositivo ARM. SteamDeckHQ (2026-10-06) vio artefactos gráficos.
- `Supermedo/bloodborne_pc`, rama `windows`: otro fork del mismo bbport, un solo commit sin
  historial. Añade capa Win32 de memoria (`VirtualAlloc2`/`MapViewOfFile3`), online, DLSS, trucos,
  launcher en 13 idiomas; 123 diferencias en `gpu/` (casi todas Windows/DLSS/red). En Windows el
  x86 del juego corre nativo: esa ventaja no existe en el visor. Útil como **oráculo de pruebas**
  (x86 real = verdad exacta) y por el online/trucos.
- bbport localiza la cámara del juego: reconoce el bloque de constantes de escena de 864 bytes
  (`gpu/.../vk_camera_motion.cpp`, `OnConstants`) y lee vista 3x4, vista inversa 3x4 y proyección
  (x scale, y scale, z scale, z offset). Lo usa para vectores de movimiento. **Es la base del 3D.**
- Parches útiles en `patches/Bloodborne.xml` (v1.09): 60 FPS sin deltatime (línea ~458), cámara
  sin autorrotación (~1205), distancia de cámara (~1213), cámara libre de Lance McDonald (~2865).

## Comparativa de código (quién tiene mejor qué)

| Capa | bbport (ARM/Windows) | AstroVisionPro |
|---|---|---|
| Runtime PS4 | ~6k líneas, solo lo que usa Bloodborne, con tests. Mejor. | HLE genérico de shadPS4 |
| Renderer | misma base + pipeline 2 hilos, memoización, vectores de movimiento, HUD aparte. Mejor para BB. | KosmicKrisp, `GpuTimer`, pasada sin adjuntos, MetalFX. Imprescindible en visionOS. |
| CPU en ARM | FEX en Linux, sin probar en dispositivo | FEX parcheado para Darwin (x18, 16 KB, SIGBUS/ESR, arena TXM). Lo único que corre en el visor. |
| Plataforma visionOS | no existe | toda (VPS4, mandos, diagnósticos) |

Los problemas de FEX en Astro son de integración (señales Darwin, arena RWX bajo TXM, 0,5–1 GB
de RAM sucia de JIT, StikDebug en cada arranque). El traductor AOT de VPEngine los elimina.

## Decisiones

- **Versión del juego**: CUSA03173 **v1.09** (confirmada). Los parches de `patches/Bloodborne.xml`
  y las direcciones que localizó bbport (incluido el bloque de cámara) son de esa versión.
- **Base del port (revisada 2026-10-10)**: opción A — BB sobre AstroVisionPro (shadPS4) con el motor
  AOT de VPEngine (`integrations/shadps4/aot_guest_engine.cpp`), sin FEX. De bbport solo se portan al
  renderer de shadPS4: la detección del bloque de constantes de escena de 864 bytes (cámara → 3D real)
  y los vectores de movimiento (→ MetalFX). Opción B (runtime y renderer de bbport + un embebido
  nuevo del motor + `memfd` → objeto Mach) solo si A no cabe en memoria o rendimiento. La decisión
  anterior (fork ARM + FEX) queda descartada: ver "Cambio de plan".
- **CPU**: sin JIT, con el mismo flujo que Astro: conversión del eboot y de los `.prx` en el propio
  visor (`tools/vpconvert`) o en el PC con `vpaot`, paquete firmado (`req4k`) cargado con `dlopen`,
  guardado en VPS4. Los ficheros del juego no van al repo.
- **Parches del juego**: vía `VpConvertCallbacks.patch_image` (el mismo mecanismo que
  `KnownTitle::OnGameLoaded` de Astro): 60 FPS sin deltatime (~458), cámara sin autorrotación
  (~1205), distancia de cámara (~1213), cámara libre (~2865). Se aplican a la imagen antes de
  traducir; cambiar un parche en la app obliga a reconvertir solo los módulos afectados.
- **3D**: real, no reproyección. "Secuencial sincronizado": frame A ojo izquierdo (vista
  desplazada −IPD/2), frame B ojo derecho (+IPD/2) **sin avanzar el reloj del juego** (el reloj lo
  da nuestro runtime). Cámaras paralelas + convergencia por desplazamiento de imagen; historial
  MetalFX por ojo; desactivar motion blur y AA del juego (opciones del menú de bbport). Ventana:
  `RealityView` con plano y material ShaderGraph *Camera Index Switch* (textura por ojo).
  Sombras/compute no dependen del ojo: dibujar una vez (segundo ojo ≈ 70 % del coste).
  Multiview / vertex amplification para que la CPU no duplique draws (~120k draws/s si no).
- **Resolución de salida**: 1080p (ventana de 1,6 m a 2 m ≈ 44°, ~1500 px útiles). Parches de
  >1080p de bbport (+4 GB) prohibidos en el visor.
- **Memoria (riesgo nº1)**: límite 8191 MB; BB reserva pool de 5056 MB (`sceKernelGetDirectMemorySize`);
  Astro muere por jetsam al cargar mundo. Modelo "en sitio" de bbport depende de que KosmicKrisp
  importe memoria del host (sin verificar). El AOT recupera 0,5–1 GB del JIT.
- **FSR 4/4.1.1** imposibles en Metal (int8 + extensión Valve) → MetalFX temporal con los vectores
  de movimiento de bbport. MoltenVK descartado (2,5× más caro por draw que KosmicKrisp en Astro).

## Estimaciones de rendimiento en M2 (±30 %; confirmar con `GPU_PASSES` en la primera build)

PS4 1,84 TFLOPS / 176 GB/s; M2 ~3,6 TFLOPS / **100 GB/s**: el doble de cálculo, 57 % del ancho
de banda. Emulado sin optimizar ≈ 0,4–0,6× PS4.

| Modo | Sin optimizar | Objetivo optimizado |
|---|---|---|
| Mono 720p → 1080p | 30 | 50–60 |
| 3D real 720p/ojo | 17–20 | 30 bloqueados |
| 3D real 576p/ojo | 22,5 | 40–45 |

Palancas: arreglos KK ya medidos en Astro (pasada vacía 16384², ~0,25 ms fijos por pasada pequeña,
barreras) −15–25 %; pasadas en memoria de GPU (G-buffer sin store/load, fusión) −10–20 %; ~20
shaders más caros a mano + FP16 −10–20 %; sombras una vez; multiview. CPU no limita (FEX 50–70 %
de nativo, núcleo P ≈ 3 Jaguar; AOT 1,2–1,5× FEX). M5: ~2–2,5× GPU → 30 fps estéreo a 720p/ojo.

## Cambio de plan (2026-10-10)

Estado de VPEngine con Astro Bot (CUSA12392) en el visor, según `bitacora.md`:
- Hecho y probado en el visor: conversión en el visor (vpconvert + clang/lld), firma del paquete
  aceptada (`req4k`), `dlopen` sin JIT, eboot + `libc_prx` + `libscefios2_prx` enganchados por
  huella; el juego corre traducido (audio PHASE, `js::GpuDevice`, colas Gnm, VideoOut, Fios).
- Pendiente: auditoría del ISA de Jaguar (diferencial 42/59 en local, goldens sin escribir), nueva
  IPA, y que Astro llegue a jugarse con AOT. `libscenptoolkit2_prx` sin enganchar.

Consecuencias para BB:
- La fase 4 del plan anterior (AOT, "la parte más incierta", 5–8 días) ya está casi resuelta por
  Astro y se hereda; no hace falta pasar por FEX en ningún momento.
- El motor AOT está integrado en shadPS4, no en bbport (cuyo runtime es solo Linux) → opción A.
  Además, BB sin ajustes por juego en el motor es la prueba del objetivo "cualquier juego de PS4".
- **No se empieza BB en paralelo**: tropezaría con los mismos huecos de ISA/motor que Astro. Cada
  fallo cerrado con Astro lo es también para BB.

## Plazos (tras Astro jugable con AOT; estimación con el ritmo de trabajo de Claude)

0. Requisito: auditoría ISA cerrada + goldens en CI + IPA nueva + Astro jugable sin FEX.
1. BB 1.09 en ventana plana sobre AstroVisionPro + VPEngine: 1 sesión (convertir, cerrar entradas
   perdidas y huecos del HLE). Medir memoria en el primer arranque.
2. 3D real (bloque de cámara de bbport portado al renderer de shadPS4): ~1 día.
3. Optimización con medidas: 2–3 días.

## Pendiente para el repo del juego

- Medir memoria antes que nada (`PACE`, `MEMORY`, `GPU_PASSES` de Astro): pool de 5056 MB de BB
  frente al límite de 8191 MB; cuánto recupera el AOT frente a FEX. Si no cabe → evaluar opción B.
- Portar de bbport al renderer de shadPS4: `vk_camera_motion.cpp` (`OnConstants`, bloque de 864 B:
  vista 3x4, vista inversa 3x4, proyección) y los vectores de movimiento.
- Parches 1.09 como opciones de la app aplicadas con `patch_image`; reconversión automática al
  cambiarlos (pendiente también en Astro).
- Ajustes en la app (no en .txt): profundidad 3D (0–100 %), convergencia, "salir de la ventana",
  presets M2/M5, resolución por ojo, MetalFX, efectos del juego.
- Solo si se va a la opción B: mover el runtime de bbport de `memfd` al objeto Mach de
  `address_space.cpp` (Astro) y escribir su embebido del motor AOT.
