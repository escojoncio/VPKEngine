# Bitácora BB — Bloodborne en Apple Vision Pro

Análisis y decisiones previas al código. El port se hará sobre VPEngine (este repo); el repo del
juego se creará cuando el motor arranque un programa entero.

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

- **Base del port**: el fork ARM (POSIX, FEX integrado, historial) sobre la capa visionOS de
  AstroVisionPro; cuando salga el parche "GnmDriver → Vulkan sin emular el procesador de
  comandos" de bbport, portar ese renderer (más ligero para Apple).
- **CPU**: sin JIT. El eboot se traduce a C con `tools/vpaot` en el PC del usuario (los ficheros
  del juego no salen de su casa; nada con copyright va al repo), el C se compila dentro del IPA;
  SideStore lo refirma cada 7 días sin problema.
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

## Plazos acordados (sesiones diarias, builds de ~15 min)

1. BB en ventana plana (con FEX de momento): 1–3 días.
2. 3D real: ~1 día más.
3. Optimización con medidas: 2–3 días.
4. AOT: estadísticas sobre el eboot 1 día; corriendo el juego 5–8 días (parte más incierta);
   sustituye a FEX en BB y en Astro a la vez.

## Pendiente para el repo del juego

- Mover el runtime de `memfd` al objeto Mach de `address_space.cpp` (Astro).
- Conversión del eboot dentro de la app la primera vez → VPS4 (o en el PC con vpaot).
- Ajustes en la app (no en .txt): profundidad 3D (0–100 %), convergencia, "salir de la ventana",
  presets M2/M5, resolución por ojo, MetalFX, efectos del juego.
- Medir memoria antes que nada (`PACE`, `MEMORY`, `GPU_PASSES` de Astro).
