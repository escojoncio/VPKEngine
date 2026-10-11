# Bitácora — VPEngine

Motor para juegos de PS4 en Apple Vision Pro sin JIT: traductor AOT x86-64 → C (`tools/vpaot`), runtime
(`runtime/`), conversión en el visor (`tools/vpconvert`: vpaot + clang + lld en proceso), firma/carga de paquetes
(`runtime/vp_codesign.c`, `vp_pkcs12.c`, `platform/visionos/Sources/VPGamePack.swift`, `VPConversion.swift`), motor
para shadPS4 (`integrations/shadps4/aot_guest_engine.cpp`, sustituye a FEX en la build `visionos-vpengine` de
AstroVisionPro). Plan: `docs/PLAN.md`. Bloodborne: `Bitacora BB.md`. Historial detallado: `git log -p bitacora.md`.

## Estado (2026-10-11)

| Parte | Estado |
|---|---|
| Traductor | ISA completa de Jaguar (2174 formas idénticas al hardware Intel y AMD), x87 80 bits exacto, AVX-256/F16C, atómicos LOCK, excepciones C++, fibers/corrutinas |
| Conversión en el visor | Funciona (~17–27 min el juego entero; reanudable; reutiliza objetos por hash de C) |
| Firma y carga | Aceptada por visionOS con la variante `req4k` |
| Astro en el visor | Eboot, libc y fios2 enganchados; el juego corre traducido hasta código no descubierto estáticamente |
| Traducción bajo demanda | Escrita y probada en Linux (e8a1fc7); **sin build ni prueba en el visor** |
| CPU (traducción AOT) | Aparcada: el límite de Astro es la GPU (~30 ms/imagen, ver bitácora de AstroVisionPro) |
| **GPU nativa Metal (foco)** | `gpu/`: core, VPIR, emisor MSL, backend Metal, GCN escalar. 71/71 tests Linux; Apple sin compilar aún (CI visionOS en curso). Ver sección GPU |

## Hecho y verificado (referencia breve)

- **Traductor** (`translate.cpp`): una función C por función invitada, bloques = etiquetas, flags perezosos, registros
  en locales (regcache), `--pic` (base en `VP_MOD.base`), `--split`, tablas de salto PIE/absolutas, landing pads de
  `.eh_frame`, puntos de reanudación por cada `call`, límites de función por `.eh_frame`, filtro de datos-como-código
  (`impossible_in_user_code`, `--keep-data`), `--roots`, `--patch`, `--fragment/--known`.
- **Carga**: ELF, SELF de PS4, PE32+ (aparcado). `VpModule` con huella FNV-1a sobre código (sin sitios de
  relocación) → enganche por huella en cualquier base; registro por referencia (`--registry`), paquetes (`--pack`).
- **Runtime** (`vp_host.c`): `vp_dispatch` con caché por hilo y generación, `vp_run` anidado, `vp_call_guest`,
  rangos de salida, hooks del embebedor (`VpEmbedderHooks`), registro de entradas perdidas (`VPENGINE_MISSING_LOG`).
- **x87**: SoftFloat 3e renombrado `vp_sf_*` en `vp_x87.c`. **TSO**: `VP_TSO` por defecto 1 fuera de x86
  (LDAPR/STLR, 128 bits y no alineados ordenados, `dmb ish` tras LOCK, `rep movs/stos` planos entre barreras).
- **Firma propia**: CodeDirectory SHA-256, CMS, requisitos de la app copiados; PKCS#12 propio (OpenSSL 3 / iloader).
- **vpconvert**: auto-prueba de clang, recuperación de fallos de LLVM, `llvm::allocate_buffer` propio, calor
  (`max_jobs`), piezas por contenido (`<mod>_u<off>.c`), `compile_key` (cabeceras SDK + flags), `vp_convert_is_current`
  (ignora la línea `roots`), `patch_image` del embebedor (parches de KnownTitle de shadPS4).
- **Tests** (`tests/aot/run.py`, CI `aot-tests.yml` x86 + ARM): diferencial 105 casos (incluidos 57 de la auditoría del
  ISA y `jaguar_sse4a_00` con golden AMD), programa, roots, ondemand, loader, módulos, hilos, atómicos, TSO (message
  passing en ARM + comprobación LDAPR/STLR), excepciones, libstdc++, pack, codesign, data_in_text, shadps4 (motor con
  veneers, callbacks, hilos, señales, fibers, bajo demanda).

## Traducción bajo demanda (e8a1fc7, sin probar en el visor)

- Motivo: el análisis estático nunca encuentra todo (punteros calculados en ejecución); en Astro cada carga nueva
  destapaba otra entrada (`eboot+0xd2df64`, `eboot+0xd86ae2` en `SoundUpdate_ATQT`) → crash, reconvertir, repetir.
- `aot_guest_engine.cpp` `TranslateOnDemand`: fallo de despacho dentro de un módulo enganchado → handler de la app
  (`runtime/vp_ondemand.h`, fuera del SDK) → `vp_convert_fragment` → `work/ondemand/<mod>_ondemand_<hex>.dylib` →
  `VPGamePack.loadFragment` (firma, dlopen) → engancha por huella → sigue. Fallidos no se reintentan.
- Swift: `missing.txt` por juego en `VPS4/VPEngine/<juego>/`; `enableOnDemand(game:)` carga los fragmentos previos;
  `onDemandActivity` para avisar en la app. La parte de AstroVisionPro (llamar a `enableOnDemand` en `startGame`,
  `vp_ondemand.h` en el bridging header, aviso en el visor) **no está en su repo**.
- Límite de diseño: cada entrada nueva para el hilo 1–3 s. Inaceptable si es frecuente.

## GPU (`gpu/`, diseño en docs/GPU.md; todo código propio, sin shadPS4/Mesa/SPIRV-Cross/MoltenVK)

Hecho (487b201, 71/71 tests en Linux con -Werror):
- `gpu/core`: grafo de pases → encoders (frame tipo Astro 93 → 15), barreras solo por riesgo real, motivo de cada corte,
  caché de pipelines con prioridades y disco, diagnóstico `gpu.frame/gpu.cut/gpu.enc/gpu.pso/gpu.err`.
  G-buffer en tesela: lecturas `pixelLocal` fusionan pases; solo `transient` o store DontCare/Resolve va a memoria de
  tesela (`persistent` veta); validación de tamaño/muestras/slots; draws con fetch imposible se saltan con `gpu.err`;
  presupuesto `CompileOptions::tileBytesPerPixel` (64; Apple8 admite 128). Fuzz 4000 frames sin violaciones.
- `gpu/shader/ir` (VPIR): IR estructurada, texto canónico, verificador, `optimize` (plegado + CSE), `toFramebufferFetch`.
- `gpu/shader/msl`: VPIR → MSL sin comprobaciones de límites, constantes directas, `[[color(n)]]`; 11 goldens.
- `gpu/metal`: backend Metal 3 (bindless + residency set; puntos `M4:` para Metal 4), memoryless, MSAA resolve,
  stencil, tablas de argumentos por etapa según `MslShader::arguments`. **Nunca compilado en Apple**.
- `gpu/shader/gcn`: SOP2/SOP1/SOPC/SOPP/SOPK/SMRD verificados (corpus 38 406 entradas). llvm-mc 18 no desensambla
  gfx7: el corpus usa el ensamblador como oráculo (`tests/gen_corpus.py`).
- CI `gpu.yml`: linux + visionos-device (SDK xros, metallib de goldens) + visionos-simulator (ctest vía
  `gpu/metal/ci/simctl-run.sh`). Primer run: 38176237184.

Pendiente (orden): 1) resultado del CI visionOS y arreglos; SKIP en simulador (sin memoryless/fetch, residency set
puede ser nil, timings 0). 2) GCN vectorial: VOP2, VOP1, VOPC, VOP3, EXP, MUBUF, MTBUF, MIMG, DS, VINTRP, FLAT.
3) Lifter GCN → VPIR. 4) `emitMsl` re-verificar tras optimize; emisor rellena `colorOutputs`. 5) Adaptador shadPS4
(`integrations/shadps4/metal`, PM4 → gpu/core) + presentador visionOS. 6) Build de prueba con FEX (CPU) + GPU propia;
opción de TAA de PTVisionPro. 7) Foveado, MetalFX.

## Siguiente CPU (si se retoma)

1. **Intérprete de reserva** para entradas no traducidas (sin pausa: ejecuta al momento, compila en segundo plano y
   sustituye). Generarlo a partir de la semántica del traductor (formas verificadas por la auditoría).
2. Caché compartida de `.prx` de Sony entre juegos (misma huella).
3. Analizar el origen de las entradas perdidas (el log de bajo demanda da la cima de la pila) para mejorar el
   descubrimiento estático.
4. Pausa real de todos los hilos invitados durante una traducción bajo demanda.
5. Medir FPS/CPU frente a FEX con el mismo juego; si no mejora, mantener aparcado.

## Notas técnicas

- `ret` deja `cpu->rip` = dirección de retorno; el llamador hace `if (cpu->rip != next) return;`. `vp_run` termina en
  una dirección de salida (`vp_is_exit`).
- Cambiar `runtime/vp_cpu.h|vp_emit.h|vp_regs.h|vp_pack.h` cambia `compile_key` → el visor recompila **todo** (~25
  min). `vp_host.c` y `vp_ondemand.h` no van al SDK. Cambiar el traductor solo retraduce (objetos reutilizados si el
  C es igual).
- AstroVisionPro `visionos-vpengine` exige que la release `vpconvert-visionos` corresponda a VPEngine en
  `tools/vpaot|vpconvert|sdk`, `runtime`, `third_party`: tras tocarlos, dispatch de `vpconvert-visionos` antes de la IPA.
- Los runners x86 de GitHub son Intel o AMD al azar; flags indefinidos y algunos campos (MXCSR_MASK, XSTATE_BV[0],
  OF de `rcl` con cuenta > 1, NOP reservados) difieren entre fabricantes: los tests no los miran.
- Excepciones x87 sin máscara no se modelan; flags de excepción de MXCSR no se modelan; `rcpps/rsqrtps` exactos.
- Enlaces de descarga: releases `visionos-vpengine` (AstroVisionPro) y `vpaot-windows` (este repo).
