# Bitácora — VPEngine

Motor para juegos de PS4 en Apple Vision Pro sin JIT: traductor AOT x86-64 → C (`tools/vpaot`), runtime
(`runtime/`), conversión en el visor (`tools/vpconvert`: vpaot + clang + lld en proceso), firma/carga de paquetes
(`runtime/vp_codesign.c`, `vp_pkcs12.c`, `platform/visionos/Sources/VPGamePack.swift`, `VPConversion.swift`), motor
para shadPS4 (`integrations/shadps4/aot_guest_engine.cpp`, sustituye a FEX en la build `visionos-vpengine` de
AstroVisionPro). Plan: `docs/PLAN.md`. Bloodborne: `Bitacora BB.md`. Historial detallado: `git log -p bitacora.md`.

## Estado (2026-10-12)

| Parte | Estado |
|---|---|
| Traductor | ISA completa de Jaguar (2174 formas idénticas al hardware Intel y AMD), x87 80 bits exacto, AVX-256/F16C, atómicos LOCK, excepciones C++, fibers/corrutinas |
| Conversión en el visor | Funciona (~17–27 min el juego entero; reanudable; reutiliza objetos por hash de C) |
| Firma y carga | Aceptada por visionOS con la variante `req4k` |
| Astro en el visor | Eboot, libc y fios2 enganchados; el juego corre traducido hasta código no descubierto estáticamente |
| Traducción bajo demanda | Escrita y probada en Linux (e8a1fc7); **sin build ni prueba en el visor** |
| CPU (traducción AOT) | Aparcada: el límite de Astro es la GPU (~30 ms/imagen, ver bitácora de AstroVisionPro) |
| **GPU nativa Metal (foco)** | `gpu/`: core, VPIR, emisor MSL, backend Metal, decodificador GCN completo. 99/99 tests Linux (gcc y clang, -Werror); CI visionOS (dispositivo + simulador) en verde desde cc1dfa5. Ver sección GPU |

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
- `gpu/shader/gcn`: decodificador + desensamblador de **todas** las codificaciones de gfx7: SOP2/SOP1/SOPC/SOPP/SOPK/
  SMRD, VOP2/VOP1/VOPC/VOP3, VINTRP, DS, MUBUF, MTBUF, MIMG, FLAT, EXP. Corpus 139 565 entradas (0 fallos) +
  `fieldChecks()` en `gcn_test.cpp` para lo que el oráculo no juzga. llvm-mc 18 no desensambla gfx7: el corpus se
  genera ensamblando (`tests/gen_corpus.py`); lo que llvm rechaza pero el hardware ejecuta (lds, r128, ttmp como
  descriptor, tfe en FLAT/atómicos de imagen, attr > 32, GWS sin gds) queda "?" (no juzgado).
  Semántica para el lifter: `Instruction::src[4]` en orden impreso; `dst` = lo que vuelve (cargas, atómicos con glc;
  cmpswap devuelve la mitad; carga con `lds` no escribe VGPR); MIMG datos = popcount(dmask) (gather4: 4) + (tfe|lwe);
  la cuenta de VGPR de dirección de MIMG es la de 2D sin array (la real depende del descriptor: la recalcula el
  lifter); EXP: fuentes apagadas = None (compr: src[0] = vsrc0, src[1] = vsrc1); MUBUF addr64 con offen/idxen se
  rechaza. GWS: el operando va en el campo addr (como lo codifica llvm-mc; el manual dice data0; sin uso en gráficos).
- CI `gpu.yml`: linux + visionos-device (SDK xros, metallib de goldens) + visionos-simulator (ctest vía
  `gpu/metal/ci/simctl-run.sh`). Primer run: 38176237184.

- **Lifter GCN → VPIR** (`gpu/shader/gcn/src/lift.cpp`, `include/vpgcn/lift.h`, lib `vpgcn_lift`): modelo un hilo por
  carril; SGPR/VGPR/M0/SCC variables, EXEC/VCC/pares de máscara bools por carril (`findMasks`), escrituras VGPR con
  `select(exec)` si el programa escribe EXEC; bucles naturales (`findLoops`: bloques h..l contiguos, una vuelta atrás,
  una salida E; anidables) → `while (true)` con `continue`/`break`; ramas → `if` anidados por post-dominador inmediato
  sin aristas de vuelta (`walk`, pasada en seco con límite de duplicación 4×); si no encaja → bucle de despacho. Descriptores: `traceOrigins`
  (user SGPR o `s_load` a offset constante de un puntero en user data) → `Binding` (slot, origen, dwords; 0 = la fuente
  es una dirección: memoria ahí). User SGPRs: constant buffer `user_data`. Fetch (`s_swappc_b64`) en línea. PS: VGPRs
  por `SPI_PS_INPUT_ADDR`, `v_interp_p2` = `load_input` (modo por el par baricéntrico), cara frontal 1.0/0. Exports:
  mrt → color N, mrtz → depth, pos0 → Position, paramN → location N, `compr` según `LiftInput::compressedExport`; PS
  descarta con `vm`/`done` y EXEC apagado. Aproximaciones → `LiftResult::notes` (readlane, mbcnt fuera de cómputo,
  swizzle, div_scale, _cl, cubos con gradientes...). Revisión adversarial (Sonnet) aplicada: ds_cmpst (DATA = comparando),
  ffbl(0) = -1, literal SMRD en dwords, bfe definido con off+w > 32, arrays de cubos (slice*8+face), *_clamp, pk_* con
  saturación, ldexp en dos pasos, rangos.
  Pruebas: `tests/lift/*.s` (fuente) → `assemble.py` (llvm-mc a objeto + llvm-objcopy: `-show-encoding` NO resuelve
  saltos) → `.lift`; `vpgcn_lift_test` exige traducción + verificación correctas y `.txt` (bindings/notas) idéntico; el `.metal`
  no se compara (su texto depende del orden de evaluación de argumentos de gcc/clang) pero el CI lo compila con Metal.
  Comprobar también con clang en local (`CXX=clang++`): el CI de Apple da error en avisos que gcc no da.
  `compile_all.sh` también compila `tests/lift/*.metal` con Metal (CI visionos-device).
- AstroVisionPro e6cf5b2: «Modo diagnóstico» vuelca el GCN de cada shader a `VPS4/VPEngine/shaders/dumps`
  (`<etapa>_<hash>.bin`, `vs_<hash>.fetch.bin`); pendiente que el usuario los suba.

- Corpus real (volcado de AstroVisionPro e6cf5b2, ASTRO BOT + KITCHEN, 174 programas: 36 cs, 75 fs, 66 vs; NUNCA al
  repo, son datos del juego): `vpgcn_lift_batch DIR [OUT]` (`tools/lift_batch.cpp`) traduce todos e histograma de
  errores/notas; infiere user SGPRs (leídos antes de escribirse, ≤16), PS = centro + posición. **174/174 a MSL válido**,
  0 errores de verificador, todos estructurados. `fs_0x...bd149c40` usa LDS en píxel como spill de registros
  (dirección = `mbcnt`·4 + offset, cada carril lee solo lo suyo) → memoria privada por invocación (abajo).
  Lo que hizo falta (patrones del compilador de Sony): base/destino SMRD solapados (`s_load_dwordx4 s[16:19], s[16:17]`),
  punteros de dos niveles (`DescriptorSource::path`), `vcc_lo/hi` y pares de máscara también como datos (doble forma:
  u32 + bool, sincronizadas), `v_cmp_*_u64` (máscaras por bit del carril / pares por dwords), `v_mov_b32` VOP3 con
  abs/neg (decoder `11/a`, corpus "?"), `v_movrel*` (cadena de select por M0), tablas de constantes tras el código
  (`s_getpc_b64` + `s_add_u32` → `DescriptorSource::Code`, offset en bytes desde el inicio del programa),
  `v_mad_u64_u32`, `ds_swizzle` sin LDS, V# construido desde una dirección cargada (`s_or_b32 hi, stride`:
  `Origin::LoadHi`, binding con dwords 0 = memoria en la dirección guardada en la fuente; `rawBuffer`).

- VPIR `quad_shuffle v, lane` / `quad_any b` (no se fusionan en optimize; MSL `quad_shuffle`/`quad_any`; test
  `msl/tests/quad_ops.vpir`). Lifter: `ds_swizzle_b32` QUAD_PERM y BITMASK dentro del quad → `quad_shuffle` con
  `quadLane` (PS: `x&1 + 2(y&1)` de la posición; CS: índice local & 3); `s_wqm_b64` de máscaras → `quad_any`.
  Limitación: dentro de ramas divergentes el vecino puede estar inactivo (en GCN se lee igual).
- **VPKTAA en MSL propio** (`gpu/visionos/shaders/vpktaa.metal`): `vpktaa_vertex` (triángulo completo) +
  `vpktaa_fragment`; texturas 0 color (render), 1 profundidad (`depth2d`), 2 historia RGBA16F (a = distancia), 3 reactivo
  opcional, 4 movimiento opcional; `VpktaaParams` (clipToPrevClip = clip actual sin jitter → clip anterior, jitter en uv
  de render, tamaños, `viewZ` (dist = 1/(x·depth+y)), nitidez, flags HISTORY_VALID/PERIPHERY/REVERSE_Z/REACTIVE/MOTION);
  salidas color(0) y historia(1). Movimiento de cámara desde profundidad (VPEngine no tiene vectores de objetos);
  distancia esperada = prev.w × distancia actual (la reproyección parte de NDC). Compilado por `compile_all.sh` (CI).
  Mismas constantes que PT build 50 (sin probar allí): afinar con su log.

- **Test diferencial** (`gcn.lift.differential`, `tests/diff/`): intérprete GCN de referencia independiente
  (`tests/ref/gcn_ref`, onda de 64 carriles, `-ffp-contract=off`, selftest `gcn.ref.selftest`) frente a intérprete
  VPIR (`tests/diff/vpir_eval`, IEEE exacto, min/max/clamp/saturate/fract como Metal: cualquier NaN, también
  señalizador, devuelve el otro operando; neg/abs de float por bit de signo) sobre el módulo traducido. NaN de
  resultado canónico (0x7fc00000) en ambos lados (salida float de la referencia; resultados float y `select` float
  del intérprete): la carga útil no está fijada ni entre GPUs ni entre compiladores host (orden de operandos), y
  reinterpretada como entero daba falsos fallos. `programs.txt` = 1000 programas aleatorios (`gen_programs.py 1000 99`,
  llvm-mc bonaire): **1000/1000**; semillas extra (3000 s31337, 4000 s4242): 0 fallos, con gcc y con clang.
  Depurar un programa: `VPGCN_DIFF_PRINT=i` (VPIR + exportaciones por carril),
  `VPGCN_DIFF_TRACE_LANE=l` (traza de valores a stderr).
  Arreglado con él: min/max legacy (select, NaN → b), med3 con NaN → min3, `bfe_i32` ancho (shift aritmético),
  ldexp (exponente acotado ±400, 4 pasos de ≤2^127, potencia desde bits `(e+127)<<23`), pkrtz (`halfRtz`), f32→i32
  saturado/NaN → 0, `cvt_u32` ≥ 2^32, pknorm NaN → 0, frexp con denormales, EXEC como operando `c` (decoder).
  Referencia: unorm/snorm16 con producto f32 (como la ALU). Generador: `s_mulk`/`s_movk` no tocan SCC.
  **Con control de flujo** (`gcn.lift.differential_cf`, `programs_cf.txt` = `gen_programs.py 400 7 cf`): if/else
  con el idioma saveexec de LLVM (anidados hasta 3, condición anidada en `s[36:37]`), bucles uniformes por SCC y
  bucles por carril que salen por EXEC (`s_andn2_b64 exec` + `s_cbranch_execnz`); el driver de `gcn_diff` ejecuta los
  saltos a nivel de onda. **400/400** (y 1000/1000 s7, 2000/2000 s2024; gcc y clang). Cuerpos
  divergentes solo con VALU que escribe VGPRs (compares/carries fuera: en GCN escriben 0 en carriles inactivos).
  Cada programa exporta también `pos0` (no se compara) para ser un VS completo: CI (job visionos-device, paso «The MSL
  of every differential program…») compila con un `vpgcn_diff` del host macOS `--emit-msl` las 1400 MSL y las pasa por
  `xcrun -sdk xros metal` (`diff/compile_msl.sh`): el emisor con combinaciones aleatorias y control de flujo.

- VPIR `image_atomic_{add,sub,min,max,and,or,xor,exchange,compare_exchange}` (storage read_write i32/u32, no cubo;
  MSL 3.1 `tex.atomic_fetch_*(coord[, capa], v).x`, CAS con bucle sobre `atomic_compare_exchange_weak` con vec4
  esperado; sin uso → sentencia). Lifter `image_atomic_*` (cmpswap: data[0] nuevo, data[1] comparando; glc devuelve
  el anterior); textura con atómicos declarada u32 (i32 si hay smin/smax) desde `resource()`. El host tendrá que crear
  la textura con `MTLTextureUsageShaderAtomic` y formato R32Uint/Sint. Goldens `cs_image_atomic`, `msl/tests/image_atomics`.
- VPIR `image_query_lod tex, smp, coord -> vec2<f32>` (fragment; MSL `calculate_clamped_lod` /
  `calculate_unclamped_lod` sin la capa de arrays; no se fusiona en optimize). Lifter `image_get_lod` (data[0]
  clamped, data[1] unclamped por dmask; fuera de píxel → 0 con nota). Goldens `ps_get_lod`, `msl/tests/query_lod`,
  `reject/query_lod_vertex`.
- VPIR `private` (`kTargetPrivate`, `private_bytes N`): memoria por invocación en cualquier etapa, sin atómicos
  (verificador; test `reject/private_atomic`); MSL `uint4 vp_private[..] = {}` en `thread`. Lifter: LDS fuera de
  compute → `private` solo con el idioma del spill (cada base DS = `v_mbcnt` de máscara completa, desplazado o con
  constante, último escritor en orden de programa; si no → error), tamaño = mayor offset DS + 16 (≤ 4096, `privateLds`),
  direcciones acotadas con `min` (sin UB), b8/b16 por RMW, atómicos DS → error. Golden `ps_lds_spill`. Stores `buffer_store_byte/short` y `ds_write_b8/b16` en
  compute: `atomic_and(~máscara)` + `atomic_or(campo)` sobre el dword (carriles vecinos no se pisan; recurso
  read_write). Golden `cs_bytes`. Stores tipados de todos los formatos salvo 11/10-bit float (`storeFormat`, inverso de
  `loadFormat`: unorm/snorm con RNE y NaN → 0, scaled saturado, f16; elementos < dword por `storeSubDword`; el formato
  se resuelve antes de declarar el recurso); golden `cs_typed_store` con bytes comprobados en `vpir_eval`. Loads de
  10_11_11 / 11_11_10 (antes leídos como 0): floats sin signo de exponente 5 (`smallFloat`, normales, denormales,
  inf/NaN); golden `cs_small_float`. Orden de campos (`packedLayout`): los nombres de AMD van del bit más alto, x en
  el bit 0 (2_10_10_10 = A2B10G10R10: x 10 bits; 10_10_10_2: x 2 bits; 10_11_11 = B10G11R11: x 11 bits). Antes
  10_10_10_2 y 2_10_10_10 estaban al revés (normales de vértice mal). Stores tipados parciales (menos componentes
  que el formato) escriben solo esos bits (`storeBits`: and+or atómicos; buffer read_write); golden
  `cs_partial_store`. unorm/sscaled de 32 bits acotados al mayor float convertible. Límites conocidos: short con `addr&3==3` (cruza dword) pierde el byte alto; dos
  carriles al mismo byte dan OR en vez de último-gana (haría falta bucle CAS). `vpir_eval` evalúa shared/private (una invocación).

Pendiente (orden): 1) Lifter: validar en el visor (Metal real) — AstroVisionPro aa35479 lo hace en «Modo diagnóstico»;
falta: FLAT, f64, atómicos de 64 bits, atómicos LDS fuera de compute, image_atomic inc/dec/rsub/f*. 2) Adaptador
shadPS4 (`integrations/shadps4/metal`, PM4 → gpu/core) + presentador visionOS. 3) Build de prueba con FEX (CPU) + GPU
propia; opción VPKTAA (ver «Aportes de PTVisionPro»). 4) Foveado (ver lecciones abajo). MetalFX temporal: descartado,
lo sustituye VPKTAA. Hecho: CI visionOS en verde (cc1dfa5), `emitMsl` re-verifica tras optimize y rellena
`colorOutputs`, decodificador GCN completo.

## Aportes de PTVisionPro (revisión de d830b7b; nada portado aún)

**VPKTAA** = el TAA propio del visor. Fuente: PTVisionPro `patches/0001-visionos-menus.patch` → `shaders/vpktaa.frag`
(303 líneas) y `RecordVpkTaa` en `src/engine/render/upscale/scene_upscale.cpp`. Build 50 de PT: lanzada, **sin probar
en el visor** (último log en `ci-logs` = run 49). Antes de portarlo: leer el log de la build 50 de PT (coste `vpktaa`,
objetivo ≤ 0,7 ms por vista nítida; antes TAA viejo 0,8–1,0 + copia 1,5–1,8 ms) y las constantes que se afinen allí.
Sustituye a MetalFX temporal (en PT separaba la estereoscopia y costaba 3–4 fps). FXAA queda solo para comparar.

Algoritmo (una pasada por vista, en lugar del resolve; salidas MRT: color, profundidad, velocidad, historia siguiente):
- Actual: 3x3 de muestras con jitter → gaussiana (Blackman-Harris, `exp(-2.29·d²)`, d en píxeles de salida) + afilado
  tipo CAS contra la media del 3x3 (×1,2·nitidez, ×1,6 en movimiento, recortado al min/max; sin afilado en periferia).
- Todo en espacio comprimido reversible `c/(1+max(c))` + YCoCg; `Sane()` quita NaN/Inf (si no, quedan para siempre).
- Movimiento del más cercano del 3x3 (reverse-Z): objeto desde su vector o, si no hay, cámara calculada desde la
  profundidad con la view-proj anterior.
- Historia: Catmull-Rom de 5 taps; caja de varianza 1,25σ (plano) → 0,85σ (borde de profundidad), recortada al min/max,
  clip hacia el centro (no clamp).
- Peso de historia 0,93 quieto → 0,80 a 8 px/frame, tope 0,97; antiparpadeo por acuerdo de luma (−0,12);
  disoclusión: la historia guarda la distancia (ViewZ) en `.a` y se rechaza si guardaba algo más cercano
  (tolerancia max(5 %, 1,5× rango de profundidad del 3x3)); reactivo (partículas/transparentes) ×(1−0,75·r).
- Resolución interna 100/90/85/80 %: fases de jitter max(8, 8/r²), mip bias −0,5 + log2(1/r), confianza por distancia a la
  muestra más cercana. Historia RGBA16F en HDR plano (no comprimido: bandea cerca del blanco), una por vista, doble buffer.
- Ajustes de PT: `aa` (VPKTAA/FXAA), `taa_res`, `taa_sharpness` (def. 50), `taa_periphery`. Log `vpktaa: on RxR -> OxO view N`,
  `vpktaa: view N history starts again (motivo)`.

Lo que PT tiene gratis (motor nativo) y VPEngine debe resolver para un juego de PS4 cualquiera:
1. Jitter: sumar `jitter·w` al xy de la posición exportada (EXP pos0) en el VS traducido, por vista; mip bias en los
   samplers traducidos (VPEngine controla los descriptores).
2. Profundidad: identificar el depth target de la escena y si es Z invertida (varía por juego).
3. Movimiento: sin vectores de objetos genéricos → solo cámara: delta de pose de cabeza (la conoce el host) + proyección
   (tangentes de Compositor Services) + profundidad. Objetos móviles: los cubre la caja de color (algo más de estela).
4. Reactivo: gpu/core sabe qué draws mezclan alfa tras el opaco → marcar su cobertura (salida o stencil extra).
5. Punto de inserción: tras la escena y antes del post/HUD del juego (heurística: primer pase a pantalla completa que lee
   el color de escena). Alternativa simple pero peor: al final, en LDR, con el HUD dentro de la historia.
6. Lee vecinos (3x3, Catmull-Rom): no fusionable en memoria de tesela; pase propio con destino en memoria.

Lecciones del host de PT para el presentador `gpu/visionos`:
- visionOS mata la app tras ~2 s sin fotogramas: mantener fotogramas durante cargas.
- Composición en la misma `MTLCommandQueue` que el render (orden GPU sin esperas de CPU).
- Profundidad real al compositor (`[[depth(any)]]`) para la reproyección del sistema; HDR: EDR + P3, drawable `rgba16Float`.
- La app no elige los Hz: 45 fps con `cp_layer_renderer_set_minimum_frame_repeat_count`; medir y loguear los Hz reales.
- Ritmo: `query_next_frame` → `predict_timing` → esperar `optimal_input_time` → drawable + device anchor → `encode_present`.

Descartado en PT (no repetir a ciegas):
- Rate map (VRR) aplicado a todas las pasadas: sin ganancia y cuadrados al girar. En VPEngine solo tendría sentido en
  pases de geometría con el post consciente del mapa: medir antes de construir encima.
- MetalFX espacial sobre el ojo entero (6,7 ms de composición); resolución dinámica (imagen inaceptable); bajar la
  resolución base; luces por teselas y sonda TBDR sin ganancia.
- Foveado que sí funcionó en PT: quad views (inset nítido + vista ancha a baja resolución, compuestos con círculo y
  fundido 15 %). En un juego de PS4 exige repetir el command buffer del juego con otra proyección (vértices ×2): evaluar
  coste antes de elegirlo frente al rate map propio.
- Datos de P.T. en M2: centro 100 % → GPU 32–40 ms; domina el coste por píxel (sombras, luces), no los draws.

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
