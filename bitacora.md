# Bitácora — VPEngine

Motor común para juegos de PS4 en Apple Vision Pro: traductor AOT x86-64 → C (`tools/vpaot`),
runtime de semántica (`runtime/`), capa visionOS compartida (`platform/visionos/`), tests
diferenciales (`tests/aot/`). Plan de arquitectura en `docs/PLAN.md`; Bloodborne en `Bitacora BB.md`.

## Estado (sin build de CI aún; todo verificado en local en x86-64 Linux)

Añadido en la misma sesión tras el primer bloque (ver abajo "Primer bloque"):
- VEX-128 (`emit_vex`: V* → mnemónico legacy con dst = src1 op src2; fuentes copiadas a `vsrc1`/
  `vsrc2`/`vmask` ANTES de escribir el destino, porque el destino puede ser también máscara:
  `vblendvpd xmm0, xmm15, xmm14, xmm0`); ymm → `unsupported("ymm")`.
- SSE3/SSSE3/SSE4.1: blendv*/blendps/pd, pinsr*/pextr*, pshufb, pmulld/pmuludq/pmullw/pmulhw/
  pmulhuw/pmaddwd, round ss/sd/ps, pmin/pmax (sd/ud/sw/ub), pcmpgt b/w/d, pcmpeq w/q, padd/psub b/w,
  pavg, min/max ps/pd, sqrtpd, rcp/rsqrt ss/ps, pabs, psign, cvtps2dq, punpck bw/wd/dq, movs(h/l)dup,
  movddup, haddps, phaddd, cvtdq2pd, cvtps2pd, cvttpd2dq, cvtpd2ps, pmovzx*/pmovsx*, psll/psrl/psra
  w/d/q (imm y xmm), ptest, palignr, shufpd, pshuflw/hw, insertps, extractps, dpps, cmpps/pd,
  pack*, psadbw, movmskpd, movbe. Atómicos: `lock xadd/cmpxchg/add/sub/and/or/xor` → `__atomic_*`,
  `xchg` con memoria siempre atómico, `cmpxchg16b`; shld/shrd; BMI1/2 (andn, blsr/blsi/blsmsk,
  sarx/shlx/shrx, rorx, bzhi, mulx, pdep/pext); rdseed/rdrand (determinista).
- Descubrimiento: `lea X(%rip)` con X en código → función (punteros a función en PIE).
- **Bug corregido**: `call *mem(rsp…)` leía el destino DESPUÉS del push (x86 lo lee antes).
- Imports nativos: `--native ADDR` (no se traduce; `call` directo → `vp_call_native`, `jmp` → tail
  call); `vp_register_native`/`vp_call_native` en `vp_host.c`; `vp_dispatch` también los resuelve.
  Un nativo termina con `cpu->rip = vp_pop64(cpu)`. Caso `native_import.s` (`# native: ext`).
- `--split N`: N funciones por fichero `<out>_NNN.c` + `<out>_decl.h` + `<out>_files.txt`
  (libstdc++: 906k instrucciones, 17 unidades, gcc -O2 ×8 en 2m19s; en un solo fichero >10 min).
- Bench `tests/aot/bench/run_bench.sh` (qsort 1M ×3 + hash + float): ver "Registros en locales"
  para las cifras actuales. (El experimento `--locals`, copia del struct entero, se eliminó:
  superado por regcache.)
- Landing pads (excepciones C++): `image.cpp` `parse_eh_frame` recorre .eh_frame desde el
  `eh_frame_ptr` del hdr (CIE: augmentation z/L/R/P; FDE: start, range, LSDA), y de cada LSDA la
  tabla de call sites → `Image::landing_pads`. `discover()` las asigna a la función que las
  contiene (`extra_entries`, explorando desde ellas si no estaban alcanzadas); el C emite
  `switch (entry) { case <offset>: goto L_...; }` y una tabla `vp_extra_entries[]` (guest, fn,
  offset) que `vp_dispatch` consulta (libstdc++: 2601 pads; unidades con switch compilan).
  Pendiente: un test con excepciones reales (necesita un unwinder dentro de la imagen).
- SELF: `tests/aot/self_wrap.py` envuelve un ELF como SELF de PS4; `run.py` comprueba que la
  traducción del SELF es idéntica a la del ELF (`ok self_loader`).
- Cobertura: `/bin/ls` 99,62 %; **libstdc++ 99,96 %** (906k instrucciones; resto x87 + `in`).
- Revisión adversarial (subagente) de VEX + eh_frame: 2 bugs reales corregidos — (1) VEX shift
  por inmediato (`vpslld $imm, src, dst` y familia, `vpslldq/vpsrldq`) no copiaba src1→dst y leía
  el inmediato de un operando registro; (2) `vmovss/vmovsd` reg,reg, `vmovlhps/vmovhlps`,
  `insertps` leían src2 tras escribir dst (fallaba cuando dst == src2). Caso `vex_review.s` los
  cubre. Parser eh_frame: comprobaciones de rango (`need`), uleb/sleb con `shift < 64`, tabla de
  call sites acotada (cs_enc 0xff / tamaño absurdo → se descarta esa LSDA).
- Flags perezosos (ON por defecto, `--no-lazy-flags`): por bloque, pasada hacia atrás con
  `insn.cpu_flags` de Zydis (modified|set_0|set_1|undefined vs tested); al final del bloque todo
  vivo. El C emite `#define VP_FLAG_MASK 0x..` alrededor de cada instrucción y los helpers
  `vp_flags_*` son macros sobre `vp_flags_*_m(..., mask)` (`runtime/vp_cpu.h`). Bench clang -O2:
  **1,57–1,65× nativo** (antes 1,74–1,82×).
- Tests: **14/14** contra nativo (añadidos c_avx con `-mavx`, c_fnptr, c_atomic, native_import).

### Primer bloque

Funciona 100 %:
- `vpaot` compila (CMake + Ninja, Zydis 5 como submódulo `third_party/zydis`, recursivo: zycore).
- Carga ELF64 x86-64 y SELF de PS4 (`image.cpp`: desenvuelve segmentos SELF, PT_GNU_EH_FRAME,
  relocaciones DT_RELA y las de PS4 en `PT_SCE_DYNLIBDATA` tags 0x61000029/2d y 0x6100002f/31).
- Descubrimiento: entrada + punteros a código + FDEs + `call` directos; tablas de salto PIE
  (`lea T(%rip)` … `movslq (T,i,4)` … `jmp *`) y absolutas (`jmp *T(,i,8)`), acotadas por `cmp $N`;
  el historial de instrucciones se hereda al bloque de caída (el `lea` queda antes del `ja`).
- Emisión C: una función por función guest `fn_<addr>(VpCpu*, uint32_t entry)`, bloques = labels,
  Jcc = `if (vp_cc) goto`, `call` directo = llamada C + comprobación de `cpu->rip` al volver,
  indirectos = `vp_dispatch`, tablas de salto = `switch(t)`.
- ISA: mov/movzx/movsx/lea/xchg/cmov/setcc/push/pop/leave/cdqe…cwd; add/sub/adc/sbb/cmp/and/or/
  xor/test/inc/dec/neg/not; shl/shr/sar/rol/ror (cuenta 0 en 32 bits sigue cero-extendiendo);
  imul 1/2/3 op, mul, div/idiv (128 bits, #DE → `vp_divide_error`); bt/bts/btr/btc (no mem,reg),
  bsf/bsr/tzcnt/lzcnt/popcnt/bswap; clc/stc/cmc/cld/std; jmp/call/ret(imm)/jcxz/loop*; cpuid,
  rdtsc, ldmxcsr/stmxcsr, fences; movs/stos (rep); SSE: movaps/ups/dqa/dqu/nt*, movss/sd, movd/q,
  movl/hps/pd, movlhps/hlps, add/sub/mul/div ss/sd/ps/pd, min/max ss/sd, sqrt ss/sd/ps, xor/and/
  or/andn ps/pd + pxor/pand/por/pandn, (u)comiss/sd, cvtsi2ss/sd, cvt(t)ss/sd2si, cvtss2sd,
  cvtsd2ss, cvtdq2ps, cvttps2dq, shufps, pshufd, unpck l/h ps/pd, punpck l/h qdq, paddd/psubd/
  paddq/psubq, pcmpeqd/b, pmovmskb, movmskps, cmpss/sd, pslldq/psrldq. Resto → `vp_unsupported`
  (registrado en `stats.json` por mnemónico).
- Runtime `runtime/vp_cpu.h`: estado, memoria plana (`VP_TSO=1` → acquire/release alineados),
  flags exactos, mul/div 128, conversiones con "integer indefinite", `vp_cc`. `vp_host.c`:
  `vp_dispatch` (tabla ordenada), `vp_run` (setjmp), cpuid fijo, rdtsc contador, mxcsr→fesetround.
- Tests `tests/aot/run.py`: 9 casos, **9/9 contra ejecución nativa x86** (trampolín
  `native_x86.c` con offsets verificados por `_Static_assert`; `harness.c` compara registros,
  flags (no AF) y hash de datos+pila). Casos: alu_basic, shifts_mul, memory_stack, branches,
  sse_scalar, switch_table, c_sort, c_float, c_mixed (C a -O2 vía `tests/aot/mkcase.sh`, flags
  `-mpopcnt -mlzcnt -mbmi -msse4.2`). Goldens en `tests/aot/golden/`. Enlace con `tests/aot/link.ld`
  (.text + .rodata contiguos en 0x400000).
- Cobertura en `/bin/ls`: 71.852 instrucciones, 99,61 % soportadas (resto x87 y `hlt`), 19 tablas
  resueltas, el C generado (7 MB) compila con gcc -O1 sin errores.
- `platform/visionos/`: `VPPlatform.swift` (hooks: documents, coreLogPath, controllerSink,
  aimSink, logSubsystem, detectedHeadset/hardwareModel), `VPS4Folder.swift`, `Language.swift`,
  `LogFiles.swift`, `PlayStationController.swift` (→ `VPControllerState`; l2/r2 analógicos, l3),
  `SenseTracking.swift` (→ `aimSink`), `vp_platform.h` (`vp_controller`). Copiados de PTVisionPro
  y desacoplados; **sin compilar aún** (no hay Swift en el entorno): workflow `platform-check.yml`
  (`swiftc -typecheck` con SDK xros 26) pendiente de ejecutar.
- CI `aot-tests.yml`: job x86 (diferencial, sube goldens) + job `ubuntu-24.04-arm` (compila el C
  traducido con binutils cruzados `x86_64-linux-gnu-*` y compara con goldens). Solo dispatch o
  `[build]`. Sin ejecutar todavía.

## App de demostración para el visor (`apps/Demo`, workflow `demo-visionos.yml`, dispatch)

Job `translate` (ubuntu): `prog.c`+`start.s` → static PIE enlazado en `0x200000000` (por encima
de los 4 GB de page-zero de una app arm64) → `vpaot --elf` → artifact `prog.c` + `prog.elf`. Job
`app` (macos-26): descarga a `apps/Demo/Generated/`, `xcodegen`, `xcodebuild` sin firma, IPA en
la release `demo-visionos` (bundle `com.kdt.livecontainer`). La app (SwiftUI, ventana) carga
`prog.elf` del bundle, mapea los segmentos en sus direcciones (`demo.c`, un solo `mmap` para toda
la imagen: las páginas del visor son de 16 K y los segmentos van a 4 K), ejecuta `vp_run(entry)`
y compara con `prog.c` compilado nativo para arm64; muestra resultado y tiempos. La lógica de
`demo.c` está probada en Linux con el mismo ELF (OK). **Sin ejecutar en CI ni en el visor.**
Enlace directo cuando exista: https://github.com/escojoncio/VPEngine/releases/download/demo-visionos/VPEngineDemo.ipa
- Jump tables: base del `lea` izada fuera del bucle → `lea_by_reg` por función (registro →
  último `lea T(%rip)`); `movslq (reg,i,4)` en el historial identifica el registro.
- Imports: `image.cpp` lee símbolos (PS4: tags 0x61000039/3f + 0x61000035/37 del blob dynlib;
  ELF: DT_SYMTAB/DT_STRTAB/DT_STRSZ) y de las relocaciones 1/6/7 contra símbolos no definidos saca
  `Image::imports` (nombre, slot, función/objeto); DT_JMPREL también se escanea. El C emite
  `vp_imports[]` (`VpImport {name, slot, function}`). `/bin/ls`: 118 imports.
- `syscall` → `vp_syscall(cpu)` (hook weak en `vp_host.c`, por defecto `unsupported`), `rdtscp`,
  `xgetbv` (XCR0 = 7), `vzeroupper/vzeroall`.
- `vp_host.c`: estado de `vp_run` `_Thread_local` (un juego corre traducido en muchos hilos);
  tabla de nativos ordenada con búsqueda binaria.
- Estrés con ASan sobre libc (1,4M instr.), python3.13 (1,88M, 100 % soportado), libm, libgcc_s:
  sin fallos de memoria. Lo no soportado es AVX-512/ymm (no existe en Jaguar), x87, E/S.

## Cargador (`runtime/vp_loader.{h,c}`) — la unión traductor ↔ runtime

`vp_load_image(path, resolve, user, &img)`: lee ELF o SELF (desenvuelve el SELF como `image.cpp`),
mapea todos los PT_LOAD/RELRO en un solo `vp_map_fixed` (granularidad 64 K) en sus direcciones de
enlace, copia los segmentos, y aplica relocaciones: RELATIVE (= addend, carga en dirección de
enlace), y 1/6/7: si el slot está en `vp_imports[]` → dirección stub `VP_IMPORT_STUB_BASE +
i*16` escrita en el slot y `vp_register_native(stub, resolve(nombre))` (cuenta resueltos /
ausentes); si es un símbolo definido → valor del símbolo (+addend) desde DT_SYMTAB (ELF) o el blob
dynlib (PS4). `image.cpp` añade como `code_pointers` los símbolos definidos de esas relocaciones
(PLT de funciones propias). Test `tests/aot/loader/`: `prog2.c` con 2 imports (`vp_ext_double`,
`vp_ext_counter`, una por puntero de datos), enlazado `ld -shared -e _start -Ttext-segment=
0x200000000` (= forma de un eboot: ET_DYN con GLOB_DAT/JUMP_SLOT), `host2.c` resuelve los imports
a nativos (que terminan con `cpu->rip = vp_pop64(cpu)`) y compara con el nativo: **OK**. En CI.
- Revisión adversarial del cargador aplicada: sin tope de tags dinámicos (`vp_dyn_tag` bajo
  demanda; un eboot tiene decenas de DT_SCE_*), **imports de datos** (`im->function == 0` →
  el resolver da la dirección del objeto, sin stub), **un stub por símbolo** (no por relocación;
  `fp == func` funciona), `R_X86_64_64` con símbolo 0 = valor absoluto, símbolos por índice en
  64 bits, comprobaciones de rango en cabeceras/segmentos/SELF (`flags & 10` = cifrado →
  rechazado), límite de imagen 2 GB, `missing[]` con los primeros nombres no resueltos.
  Resolver nuevo: `int resolve(name, function, VpNative* native, uint64_t* data, user)`.
  Pendiente (entrada malformada, no juego real): MAP_FIXED sin comprobar colisión con el host,
  `so + ss` sin comprobar desbordamiento, dynlib data en ELF plano sin acotar.
- `tests/aot/loader/threads.c`: la misma imagen en 8 hilos × 20 ejecuciones (nativos con
  `_Thread_local`), todas coinciden: ni el C traducido ni el host guardan estado compartido.
  (El `prog.c` del test de programa entero tiene un global mutable: no sirve para hilos.)

## Traducción independiente de la posición (`--pic`) — hecho

**Qué:** con `--pic` toda dirección de la imagen en el C generado es `vp_image_base + desplazamiento`
(`Emitter::A()`): RIP-relativas, direcciones de retorno empujadas, `cpu->rip != next`, destinos de
nativos, `case` de tablas de salto (`switch (t - vp_image_base)`), y los `movabs` cuyo inmediato es
un sitio de relocación (`Image::reloc_sites`, de RELATIVE/R_X86_64_64 y DIR64 de PE). Las tablas
(`vp_entries`, `vp_extra_entries`, `vp_imports`) guardan desplazamientos si `vp_tables_relative`;
`vp_host.c` resta `vp_image_base` al buscar. **Por qué:** shadPS4 (AstroVisionPro) y bbport cargan
los módulos donde decide su gestor de memoria, y las DLL de Windows piden 0x180000000, que en
iOS/visionOS es la caché compartida del sistema: la traducción no puede depender de una base fija.
Suite 15/15 también con `VPFLAGS=--pic`.

## `vp_run` anidado y `vp_call_guest` — hecho

**Qué:** cada `vp_run` guarda su propio `jmp_buf` (puntero thread-local al más interno) y restaura el
exterior al volver; `vp_call_guest(cpu, fn)` llama a código invitado desde un nativo (guarda GPR y
rip, empuja `VP_HOST_EXIT_ADDRESS`, `vp_run`, restaura, devuelve rax); `vp_run_reset()` para salidas
por `longjmp` (fin de proceso/hilo). `vp_run` entra por `vp_dispatch`, así acepta también entradas
a mitad de función y nativos. **Por qué:** el HLE de shadPS4 llama a funciones del juego (callbacks
de fios2, ngs2, fuentes, avplayer, pthread) desde dentro de una llamada del juego al HLE.

## Cargador PE32+ en `vpaot` (Windows; aparcado) — hecho, sin usar aún

**Qué:** `load_pe` (detectado por `MZ`): secciones en ImageBase+RVA, `.pdata` → inicios de función,
relocaciones DIR64 → punteros a código y `reloc_sites`, exports y callbacks TLS → raíces, tablas
de ámbito de `__C_specific_handler` → landing pads, imports `dll!nombre` con su slot de IAT.
**Por qué:** base de la capa Windows (ejecutables .exe en visionOS). Aparcado por prioridad: primero
PS4 completo para AstroVisionPro. Pendiente: cargador de ejecución PE, TEB/PEB, capa kernel32/msvcrt.

## Registro de módulos (`VpModule`) y enganche por huella — hecho

**Qué:** cada traducción define `VpModule vp_module_<NOMBRE>` (`--module NOMBRE`; por defecto `main`,
prefijo de funciones `<NOMBRE>_fn_`) con sus tablas (`entries` ordenadas, `extra`, `imports`), base
actual y de enlace, tamaño, `relative` (= `--pic`), rangos de código, sitios de relocación y la
**huella** FNV-1a 64 de los bytes de código tal como se tradujeron (relocaciones a cero). Se registra
sola (`__attribute__((constructor))`). El C generado lee la base como `VP_MOD.base`. `vp_dispatch`
busca el módulo por dirección (`vp_module_at`), luego nativos, luego `vp_dispatch_miss` (gancho débil
para el embebedor: stubs generados en ejecución, CPU de reserva). `vp_attach_module(base, size)`
encuentra la traducción de una imagen ya cargada por otro (mismo tamaño + misma huella) y la mueve
allí; una imagen distinta no se engancha. `vp_load_module(path, mod, load_at, …)` comprueba base
de enlace y huella al cargar. Imports de función pueden resolverse a código de otro módulo invitado
(`*native = NULL`, `*data = dirección`). Los símbolos exportados definidos (STT_FUNC/NOTYPE en código)
son raíces de descubrimiento (los `.prx` de PS4 exportan así sus funciones).
**Por qué:** un juego de PS4 son varios módulos (`eboot.bin` + `sce_module/*.prx`) que shadPS4 carga
donde quiere; sin registro colisionaban los símbolos globales, y sin huella no hay garantía de que la
traducción corresponda al `.prx` cargado (un juego con otra versión de `libc.prx` ejecutaría código
equivocado). Test `tests/aot/modules/` (en CI): `lib.so` y `main.so` traducidos por separado, cargados
en 0x260000000 y 0x280000000, `main` llama a `lib_mix` (invitado→invitado) que llama de vuelta a una
función de `main` por puntero y a un nativo: idéntico al nativo; enganche por huella OK; imagen
modificada rechazada OK. Suite 15/15, programa, cargador, hilos: OK.

### Revisión adversarial del registro de módulos — aplicada

1. **Huella**: cubre los rangos ejecutables saltando **todo** destino de relocación de cualquier tipo
   (`Image::loader_written`: RELATIVE, 64, GLOB_DAT, JUMP_SLOT, TLS…, DIR64 e IAT en PE). Antes solo
   saltaba RELATIVE/64 y un segmento RWX (`ld -N`) con GOT dentro nunca coincidía. Probado: el test
   de módulos ahora también corre con RWX (carga OK, imagen modificada rechazada).
2. **`attached`**: `vp_module_at`/`vp_dispatch` solo usan módulos enganchados; un `--pic` empieza sin
   enganchar (un no-PIC empieza enganchado en su base de enlace); `vp_module_set_base` engancha,
   `vp_detach_module` desengancha. Si un módulo contiene la dirección pero no la tiene en sus tablas,
   se sigue buscando en los demás (solapes: DLL con la misma base preferida, PRX recargados).
   Hosts sin cargador (`harness.c`, `bench_main.c`) enganchan el primer módulo en su base de enlace.
3. **Hilos**: escritores con `pthread_mutex`; cabeza de lista publicada con release/acquire; `base`
   relaxed, `attached` release/acquire (se baja antes de mover la base y se sube después).
4. **Cargador**: comprueba base de enlace y tamaño antes de mapear; la huella se calcula en la
   dirección nueva y solo entonces se mueve el módulo (antes un fallo dejaba la base movida).
5. `VpModule` y funciones de `--split` con visibilidad oculta; dos módulos con el mismo nombre →
   abort con mensaje (antes, en .so, uno tapaba al otro en silencio).
6. `--module` no puede empezar por dígito ni estar vacío.
7. Nº de símbolos ELF desde DT_HASH (nchain) o recorriendo DT_GNU_HASH; `to - so` solo de último
   recurso (con lld leía las tablas hash como símbolos). Resolvers IFUNC (tipo 10) son raíces.

## Motor de CPU AOT para AstroVisionPro (`integrations/shadps4/aot_guest_engine.cpp`) — hecho

**Qué:** implementa **la misma interfaz** que `src/core/fex/fex_guest_engine.h` de AstroVisionPro
(`Core::Fex::GuestEngine`: `Create`, `CreateThread`, `Run`, `CallGuest`, `Invalidate`,
`DestroyThread`, `Shutdown`, `ReturnAddress`, `ReturnRange`, `CallbackReturnRange`,
`RunControlledHarness`; y las libres `HandleGuestSignal`, `DeliverGuestOrbisSignal`,
`FlushPendingGuestOrbisSignal`, `BachataQueryGuestRipSyscall`, `BachataQueryGuestRegisters`) sobre
VPEngine. Integración = compilar este `.cpp` + `runtime/vp_host.c` + los módulos traducidos en lugar
de `fex_guest_engine.cpp` + FEX; el resto de shadPS4 no cambia.
- Módulos: cada `eboot.bin`/`.prx` traducido con `--pic --module X`; en el primer acceso a código
  sin módulo, `vp_dispatch_miss` → `TryAttach`: pide el rango ejecutable al `GuestBridge`
  (`QueryExecutableRange`) y prueba bases {inicio del rango, inicio − desplazamiento del primer
  rango de código} contra la huella de cada módulo sin enganchar.
- Veneers de shadPS4 (`mov r10,rcx; mov rax,op; syscall; ret`, escritos en ejecución): mini
  intérprete `InterpretStub` (mov reg,reg; mov r64,simm32; mov r32/movabs; syscall; ret; jmp reg;
  nop); `syscall` → `InvokeBridge`, copia exacta de `HandleSyscall` de FEX (frame con rcx = r10,
  copia de vuelta de GPR/XMM, fallo → rax = −error y `Run` devuelve el fallo).
- Retorno: páginas de `hlt` (`FunctionReturn`, `CallbackReturn`) registradas como rangos de
  salida (`vp_add_exit_range`, nuevo en `vp_host.c`; `vp_is_exit` sustituye a la comparación con
  `VP_HOST_EXIT_ADDRESS`). `Run` termina `Halted` con rip en la página; `CallGuest` imita
  `HandleCallback` (retorno en RSP−16, 7.º argumento en RSP−8) y termina `Returned`.
- Señales Orbis: cola por hilo, entrega en el siguiente límite HLE con ucontext si el árbol de
  shadPS4 tiene `core/libraries/kernel/threads/exception.h`.
**Por qué:** es la forma de sustituir FEX tocando lo mínimo: shadPS4 ya habla con su CPU a través
de esa interfaz. **Prueba** `tests/shadps4/` (en CI x86 y ARM): host que imita a shadPS4 (carga el
módulo en 0x5000000000 ≠ dirección de enlace, desengancha, imports → veneers generados en
ejecución, `GuestBridge` con 3 operaciones: cálculo, `CallGuest` anidado, hilo de guest con
`CreateThread`+`Run` y TLS propio por `fs:0x10`, `RunGuestFunction` como el de `linker.cpp`):
enganche perezoso por huella OK y resultado **idéntico al nativo** con 2 niveles de anidamiento y
4 hilos. Cabeceras de interfaz copiadas en `tests/shadps4/include` (MIT, ver README allí).
**Pendiente:** pasos de integración en el repo AstroVisionPro (CMake: opción para compilar este
fichero en vez de FEX; traducir los módulos del juego; `vp_dispatch_miss` para código no traducido
→ hoy falla con mensaje `VPENGINE: guest fault`).

### Revisión adversarial del motor (subagente) — 6 defectos confirmados, corregidos
1. **`TryAttach` solo miraba un módulo**: iteraba desde `vp_first_module()`, que es la *cola*
   de la lista (su `next` es NULL), así que con varios `.prx` ninguno más se enganchaba.
   **Arreglo:** `vp_module_list()` (cabeza de la lista, nuevo en `vp_host.c`/`vp_emit.h`).
2. **Lectura fuera de rango y huella en cada llamada HLE**: se hacía hash de bytes sin comprobar
   que estuvieran mapeados (SIGSEGV reproducido con página PROT_NONE tras los veneers), y cada
   veneer recalculaba la huella de todos los módulos sin enganchar. **Arreglo:** cada rango de
   código debe caer entero dentro de un rango ejecutable de shadPS4 (`InsideExecutable`); los
   módulos no `--pic` solo en su base de enlace; caché negativa por `range.Begin`
   (`NoTranslation`, se vacía en `Invalidate`).
3. **La señal Orbis machacaba el estado del HLE**: solo se guardaban GPR+rip; xmm0–7 (argumentos
   float de la llamada HLE) y xmm0 (resultado float), flags y MXCSR quedaban como los dejara el
   handler. **Arreglo:** copia completa de `VpCpu` y `vp_apply_mxcsr` al volver. Además el
   mcontext ahora lleva **todos** los GPR y rflags (nuestro estado es exacto, a diferencia de un
   JIT a mitad de bloque). La cola pasa a atómicos sin bloqueo (escrita desde un handler de señal
   del mismo hilo: `Handler` publicado el último y tomado con `exchange`).
4. **Un HLE que fallaba tras un callback dejaba RSP corrupto**: `CallGuest` escribía el estado
   compartido del hilo y en el camino de fallo nadie lo restauraba → el `ret` del veneer saltaba
   a basura. **Arreglo:** `CallGuest` guarda el `VpCpu` exterior y lo repone siempre (como
   `HandleCallback` de FEX); DF=0 al entrar al callback (ABI).
5. **Rangos de salida agotados tras 4 motores**: `vp_add_exit_range` descartaba en silencio a
   partir de 8 y `Create` registraba antes del CAS. **Arreglo:** las dos páginas `hlt` se crean
   una vez por proceso y las comparten todos los motores; `vp_add_exit_range` devuelve error y
   deduplica.
6. **`Invalidate` no hacía nada**: ahora desengancha los módulos que solapan el rango (vuelven a
   engancharse solo si la huella sigue cuadrando). shadPS4 no lo llama hoy fuera del arnés de FEX.
Otros: `jmp [rip+disp32]` en el intérprete de stubs; `cpu.rip` = sitio del `syscall` durante el
HLE (para `BachataQueryGuestRipSyscall`); MXCSR aplicado al crear hilo; `Shutdown` con hilos
vivos no libera el estado (evita uso tras liberar). Descartado tras leer el código real de
AstroVisionPro: `validate_range`/`publish_host_range` los rellena el propio `HleGuestBridge`.
**Test ampliado** (`tests/shadps4`): segundo módulo traducido (`lib.c`, enlazado *antes* que el
juego y cargado en 0x6000000000), HLE con argumentos/resultado `double` durante el cual llega una
señal cuyo handler hace cálculo float, HLE que falla con EIO tras un callback (el invitado vuelve
por su pila con rax=−EIO), y 7 motores creados/destruidos seguidos. El motor anterior (con solo
el arreglo 1) da **MISMATCH** en este test; el actual, idéntico al nativo. CI: los pasos con
herramientas cruzadas x86 (loader, módulos, motor) estaban por error en el job x86; movidos al job ARM.

## AVX de 256 bits y F16C (la ISA de Jaguar) — hecho

**Qué:** `VpCpu.ymmh[16]` (bits 128–255 de ymm0–15). Reglas de x86: SSE heredado **conserva** la
mitad alta; todo VEX.128 que escribe un registro vectorial la **pone a cero** (`emit_vex` añade
`ymmh[d] = 0` si el operando 0 tiene acción de escritura según Zydis; `vcomiss`/`vptest` no).
- **Ops por carriles** (`lane_split`): las 256 de AVX1 que son dos operaciones de 128 independientes
  (mov*/aritmética/min/max/sqrt/rcp/rsqrt/lógicas/shufps/unpck/blend/blendv/cmp/cvtdq2ps/cvt(t)ps2dq/
  haddps/movddup/movs[hl]dup/dpps/roundps) se emiten **dos veces** con el mismo código del emisor:
  la segunda con `hi = true` (registros → `cpu->ymmh[i]`, memoria → `ea + 16`), cada mitad en su
  bloque `{}`. Inmediatos con bits por mitad: `vshufpd`/`vblendpd` (>>2), `vblendps` (>>4) en la copia
  de operandos de la mitad alta. Jaguar no tiene AVX2 (enteros de 256) ni FMA: no hacen falta.
- **Ops con su propio código** (`emit_avx`): `vzeroupper`/`vzeroall`, `vbroadcastss/sd/f128`,
  `vinsertf128`, `vextractf128`, `vperm2f128` (bits de cero 3 y 7), `vpermilps/pd` (imm y variable),
  `vmaskmovps/pd` (carga: elementos sin máscara a cero y sin leer; almacenamiento: no se tocan),
  `vtestps/pd` y `vptest` 256 (ZF/CF), `vmovmskps/pd` 256, conversiones que cambian de anchura
  (`vcvtps2pd`/`vcvtdq2pd` ymm←xmm, `vcvtpd2ps`/`vcvt(t)pd2dq` xmm←ymm), **F16C**
  `vcvtph2ps`/`vcvtps2ph` (redondeo del inmediato o de MXCSR si bit 2).
- `vp_f16_to_f32`/`vp_f32_to_f16` en `vp_cpu.h`: redondeo entero propio (no depende del modo del
  host). **Verificado exhaustivo** contra el hardware (F16C del host x86): las 65 536 mitades y
  80 M floats aleatorios × 4 modos de redondeo, 0 diferencias.
- Lo no cubierto a 256 sale como `unsupported` `ymm-<mnemónico>` en el informe (antes: `ymm` genérico).
**Por qué:** el SDK de PS4 compila para `btver2` (Jaguar), que tiene AVX y F16C; código de juego
con bucles vectorizados de 256 y conversiones a half (vértices, animación) es normal. Sin esto,
cualquier eboot con `ymm` fallaba en esas funciones.
**Prueba:** caso `avx256.s` (≈100 instrucciones: todas las anteriores, mitad alta conservada por
`addps` y borrada por `vaddps xmm`, `vzeroupper` seguido de SSE y 256) y caso `c_jaguar`
(C compilado por gcc con `-march=btver2`: vectoriza a 256, `vperm2f128`, `vinsertf128`, F16C). El
arnés diferencial ahora inicializa, compara y guarda **ymm completos** (nativo: `vinsertf128`/
`vextractf128` en `native_x86.c`); goldens regenerados. 17/17 normal, `--pic`, `--no-regcache`,
`--no-lazy-flags`. Comprobado que el test detecta errores (romper el desplazamiento de `vshufpd` → FAIL).

### Revisión adversarial del cambio AVX (subagente) — corregido
Operandos, inmediatos por mitad, aliasing en la pasada alta y puesta a cero VEX.128: verificados
correctos por experimento. Defectos (antiguos en helpers compartidos, ahora alcanzables a 256):
1. **Predicados de `vcmp*` 8–31** se reducían a `pred & 7` (p. ej. `_CMP_GT_OQ`=30, `NEQ_OQ`=12
   daban otra máscara con NaN). `vp_fcmp_pred` ahora decodifica bit 3 (resultado si no ordenado);
   bit 4 (si QNaN señaliza) no cambia el resultado.
2. **`roundps/pd/ss/sd` imm 0** usaba `rint` (sigue MXCSR); imm 0 es par-más-cercano fijo →
   `__builtin_roundeven`. `nearbyint` solo con imm bit 2.
3. **`dpps`** sumaba en secuencia; el hardware suma en árbol `(p0+p1)+(p2+p3)` con productos
   enmascarados = +0 ({1e8,1,−1e8,1}: nativo 0, antes 1).
4. **`vcvtps2ph` con MXCSR.DAZ**: un denormal de entrada cuenta como cero (FTZ no aplica, como en hardware).
5. **CPUID** decía Intel sin AVX mientras XGETBV decía AVX activo: ahora identidad **Jaguar**
   (AuthenticAMD, 0x730F01) con solo lo que el traductor ejecuta: SSE3/SSSE3/SSE4.1/4.2, CX16,
   MOVBE, POPCNT, XSAVE/OSXSAVE, AVX, F16C, BMI1 (hoja 7), ABM; hoja 0xD con estado AVX.
6. **`-frounding-math`** al compilar el C generado (tests, app de demo, test shadPS4): sin él gcc
   inline `rint` con el truco del número mágico (mal con redondeo dirigido: `cvtps2dq(−2.5)` con
   RC=arriba daba −3) y clang puede mover operaciones FP alrededor de `fesetround`. Coste medido en
   el bench: 1,08× → 1,09× (ruido). **Obligatorio también en la integración de AstroVisionPro.**
Cobertura añadida: `haddpd`, `hsubps/pd`, `addsubps/pd`, `roundpd` (SSE, VEX.128 y 256).
Casos de la revisión conservados como regresión: `avx_review_{cmp,dp,rc,f16daz}.s` (21/21, también `--pic`).
Aceptado y documentado: `rcpps/rsqrtps` dan el valor exacto, no la aproximación del CPU (no está
definida bit a bit y difiere entre Intel y AMD); un acceso de 256 que falla en su mitad alta deja
la baja hecha (solo importa si un handler reintenta la instrucción). `GuestExecutionRequest` de
shadPS4 no lleva ymm: un hilo nuevo empieza con las mitades altas a cero (correcto por ABI).

## Kit de integración en AstroVisionPro + ciclo de entradas perdidas — hecho

**Qué:**
- `integrations/shadps4/vpengine.cmake` + `astrovisionpro.patch` (verificado con `git apply
  --check` contra AstroVisionPro actual): opción `ENABLE_VPENGINE_GUEST_CPU` (OFF por defecto; la
  build FEX no cambia). Con ella: fuera `fex_guest_engine.cpp` y libs FEXCore; dentro
  `aot_guest_engine.cpp`, `vp_host.c` y el C traducido de `VPENGINE_GAME_DIR`, con
  `-O2 -frounding-math`; se mantiene `SHADPS4_ENABLE_FEX_GUEST_CPU` para que el resto de shadPS4 siga
  por sus caminos FEX (veneers, bridge, fibers, señales). Comprobado con un proyecto CMake mínimo.
- **Registro de módulos por referencia** (`vpaot --registry OUT.c MOD...` → `vp_register_game_modules()`,
  llamado por el motor en `Create`). Por qué: en visionOS shadPS4 es una **biblioteca estática** y el
  enlazador solo mete los objetos referenciados: los constructores de los módulos no bastaban. El test
  shadPS4 ahora enlaza motor + traducciones como `.a` (falla sin el registro).
- `tools/scripts/translate_game.ps1` (+ `.sh`): eboot.bin + `sce_module/*.prx|*.sprx` → `vpaot --pic
  --module --split 4000` + informe por módulo + registro. Se publica junto a `vpaot.exe` en la release.
- **Entradas perdidas**: `vp_dispatch` registra una vez cada dirección dentro de un módulo sin entrada
  traducida como `módulo+0xOFF` (`vp_set_missing_log`; por defecto `$VPENGINE_MISSING_LOG`; el motor
  usa `$HOME/Documents/vpengine_missing.txt`, visible en Archivos del visor). `vpaot --roots FICHERO`
  lo lee de vuelta (solo las líneas de su `--module`). Test `tests/aot/roots` (CI x86 y ARM): función
  alcanzada por un puntero que el análisis estático no puede ver → 1.ª traducción falla y la registra →
  2.ª con `--roots` → idéntico al nativo. Es el método de N64Recomp/XenonRecomp para cerrar cobertura.
- `vpaot` sin `--out` usaba `/dev/null` también en Windows (fallaría el primer paso del usuario):
  ahora `NUL`. Smoke test del workflow Windows sustituido por una traducción real (antes fallaba
  por el código de salida de `vpaot` sin argumentos) → **build Windows verde, release publicada**.
- CPU: el motor informa de cuántos módulos traducidos hay al crearse.
**Pendiente (decisión del usuario):** el C traducido de un juego no puede ir al repo público de
AstroVisionPro; opciones: build local en un Mac, repo privado que el workflow descarga con token, o
artefacto privado.

## Operaciones LOCK atómicas de verdad + split locks + `bt m, reg` — hecho

**Qué (bug grave encontrado):** solo `lock add/sub/and/or/xor`, `xadd`, `cmpxchg`, `cmpxchg16b` y
`xchg` eran atómicos; **`lock inc/dec/neg/not/bts/btr/btc/adc/sbb` se traducían como lectura +
escritura normales** → actualizaciones perdidas entre hilos (`lock inc/dec` = contadores de
referencias, muy habituales en juegos).
- Ahora: `lock add/sub` y `xadd` → `vp_fetch_addN`; `xchg m` → `vp_xchgN`; `cmpxchg` → `vp_casN`; el
  resto (`lock_loop`) → **bucle compare-and-swap genérico**: el emisor normal genera la operación con
  el operando de memoria redirigido a `vp_old`/`vp_new` (`lock_rmw`), y `vp_casN` cierra el bucle.
  Flags y registros se recalculan en cada reintento (idempotente).
- **Split locks** (x86 permite atómicos desalineados; ARM da fallo): los helpers de `vp_cpu.h` usan
  el atómico nativo si está alineado (lo normal, una comprobación) y si no, `vp_cas_split` en
  `vp_host.c` (un mutex de proceso + compara-y-copia).
- `bt/bts/btr/btc m, reg` (antes `unsupported`): desplazamiento de bit con signo que sale del
  operando: `ea + (sext(off) >> log2(bits)) * bytes`.
**Prueba:** caso diferencial `lock_ops.s` (todas las formas, alineadas y desalineadas, offsets de
bit negativos) y `tests/aot/atomics` (CI x86 y ARM): 8 hilos × 200 000 iteraciones de cada tipo
de operación bloqueada sobre palabras compartidas, alineadas y partidas → totales exactos.

### Revisión adversarial (atómicos + kit de integración) — corregido
1. **`lock adc/sbb` tras un CAS fallido** usaban el CF que escribió el intento fallido (cuando CF
   está vivo después): ahora `vp_cin` se lee antes del bucle. El test de contención ahora hace
   `stc; lock adc/sbb` con CF leído después: sin el arreglo da actualizaciones perdidas, con él exacto.
   También `neg`/`btc` desde valores ≠ 0 (antes el test era vacuo) y guard RAII de `lock_rmw`.
2. **Ficheros viejos de una traducción anterior** (cambio de nº de partes con `--split`) se
   compilaban y uno ganaba en el enlace en silencio: los scripts borran la salida previa de cada
   módulo y `vpengine.cmake` ya no hace GLOB: compila exactamente los módulos del registro
   (`NAME_files.txt` o `NAME.c`), con `CMAKE_CONFIGURE_DEPENDS`; `VPENGINE_GAME_DIR` sin registro → error.
3. `.sh`: ruta del log con espacios; `.prx/.PRX` sin distinguir mayúsculas. `.ps1`: `-LiteralPath`
   (carpetas tipo `CUSA03173 [Astro Bot]`: los corchetes son comodines en `-Path`), `$vpArgs` en vez
   de la automática `$args`. Ambos: error si dos módulos dan el mismo nombre C.
4. `vp_cas_split`: barrera completa antes del mutex (un locked op x86 es barrera en ambos sentidos).

## FPU x87 completa y exacta (80 bits) — hecho

**Qué:** el traductor no tenía **ninguna** instrucción x87 (solo `fnop`). Ahora: `runtime/vp_x87.c`
(incluido por `vp_host.c`: ningún build necesita ficheros ni flags nuevos) con **Berkeley SoftFloat
3e** vendorizado en `runtime/softfloat/` (BSD-3, especialización 8086 para las reglas de NaN del
x87; unity build `vp_softfloat.c`, `INLINE_LEVEL 0`, estado por hilo). Estado en `VpCpu`: `st[8]`
físicos, `fcw`, `fsw` (sin TOP), `ftop`, `ftag` (abreviado). `fcw == 0` (VpCpu a cero de un
embebedor) = estado de FNINIT 0x037F.
- Cubierto: `fld/fild/fbld` (m32/m64/m80/m16/m32/m64), `fst/fstp/fist/fistp/fisttp/fbstp`,
  `fadd/fsub/fsubr/fmul/fdiv/fdivr` en todas sus formas (reg-reg, `p`, m32/m64, enteros m16/m32),
  `fchs/fabs/fsqrt/frndint`, `fcom/fcomp/fcompp/fucom*/ficom*/ftst/fxam`, `fcomi/fucomi(p)` (EFLAGS,
  vía `VP_FC` para el regcache), `fcmovcc`, `fxch`, constantes (redondeadas según RC como el hardware),
  `fprem/fprem1` (resto exacto con enteros de 128 bits, bits del cociente en C0/C3/C1, parcial con C2
  si los exponentes distan ≥ 64), `fscale`, `fxtract`, `fnstsw ax/m16`, `fnstcw/fldcw`, `fnclex`,
  `fninit`, `fnstenv/fldenv`, `fnsave/frstor`, `fxsave(64)/fxrstor(64)`, `ffree(p)`, `fincstp/fdecstp`,
  `emms`. Transcendentes (`fsin/fcos/fsincos/fptan/fpatan/f2xm1/fyl2x/fyl2xp1`) por libm en doble:
  precisas a 53 bits, no a 64 (documentado; uso raro en PS4).
- Detalles de hardware reproducidos (encontrados uno a uno con el test aleatorio): control de
  precisión 24/53/64 (`fscale` y `frndint` lo ignoran), **C1 = redondeo hacia arriba en magnitud**
  (calculado repitiendo la operación en modo hacia-cero solo si fue inexacta; incluye desbordamiento
  a ∞, `fst m32/m64`, `fist*`, `frndint`, `fscale`, `fsqrt`), fallos de pila (underflow: IE+SF, C1=0,
  indefinido; overflow: IE+SF+C1; en `fxtract/fsincos/fptan` con overflow ambos registros quedan
  indefinidos y no se calcula nada), prioridad de excepciones (NaN o fallo de pila → sin DE;
  IE/ZE → sin DE; `fst/fist` nunca dan DE), `fcomi` no toca C0/C2/C3 ni C1, `fprem` con NaN
  conserva C0/C3, mitades reservadas de `fnstenv` a 1, `fscale` con ST(1) enorme satura (no el
  entero indefinido negativo), ES/B cuando hay excepciones sin máscara.
**Prueba:** el arnés diferencial ahora compara también el estado x87 completo (8 registros
físicos, TOP, tags, FSW, FCW) cargándolo/guardándolo en el nativo con `fxrstor/fxsave` en el mismo
formato que `vp_x87_fxsave/fxrstor`; `VP_DUMP=prefijo` vuelca la memoria de ambas ejecuciones.
Casos: `x87_basic.s` (formas, controles, excepciones, entorno) y `x87_random.s` generado por
`tests/aot/gen/x87_random.py` (3000 pasos aleatorios sobre valores especiales: NaN señalizantes y
silenciosos, ∞, ±0, denormales, extremos; con FSW guardado tras muchos pasos). **35 semillas × 3000
pasos idénticos al hardware bit a bit**; la del repo es la 1234. 24/24 normal, `--pic`, `--no-regcache`.

### Revisión adversarial del x87 — corregido
1. **`fcmovcc`** solo miraba la pila si se cumplía la condición: el hardware comprueba ST(0) y ST(i)
   siempre (vacío → IE+SF e indefinido en ST(0)). Ahora `vp_x87_fcmov(cpu, i, cond)`.
2. **`fxrstor`**: FCW sin el bit 6 forzado (0 se convertía en 0x037F) y ES/B copiados en vez de
   recalculados → como `fldenv`.
3. **`fnstenv`**: tras enmascarar todo, ES/B se borran; bytes 26–27 reservados a 0xFF.
4. **Codificaciones no soportadas** (unnormales, pseudo-NaN, pseudo-∞, exponente ≠ 0 sin bit
   entero): operandos inválidos (IE, indefinido, comparación desordenada, `fst m32/m64/int` da el
   indefinido del formato); `fxam` las clasifica como "unsupported".
5. **Transcendentes**: el argumento ya no se convierte entero a double (un 2^-10000 daba −∞, UE/ZE u
   OE espurios): exponente separado (`fyl2x` = (e + log2 m)·y con el producto en SoftFloat), `sin/tan`
   de |x| < 2^-33 = x, `cos` = 1, `f2xm1` diminuto = x·ln2, `fpatan` escalando ambos por la misma
   potencia de 2 (y/x o ±π/2 exactos en los extremos); las conversiones a double ya no filtran flags.
   Siguen siendo de 53 bits (C1 del redondeo no se conoce en ellas).
6. **Excepciones sin máscara**: no se modelan (resultado siempre el enmascarado, sin #MF); se avisa
   una vez por stderr si `fldcw/fldenv/fxrstor` desenmascaran algo. El código de PS4 va enmascarado.
7. **Símbolos**: SoftFloat se renombra entero a `vp_sf_*` (`vp_softfloat_rename.h` generado del
   objeto; guardas `#ifndef softfloat_X` → `VP_SF_INLINED_softfloat_X`, fuera los
   `#define softfloat_X softfloat_X` y los atajos CLZ/int128 de cabecera): `vp_host.o` no exporta
   ningún símbolo que no empiece por `vp_` → no puede chocar con el SoftFloat modificado de FEX.
Regresiones: `x87_env`, `x87_fcmov`, `x87_unnormal`, `x87_pseudonan`, `x87_pseudoinf` (29/29).
Semillas aleatorias 1, 2, 3, 50, 60, 65, 77, 88, 99 × 3000 pasos siguen idénticas al hardware.

## Salto no local (unwinder/longjmp) — probado

`nonlocal_jump.s`: un callee restaura la pila del marco exterior y salta a mitad de la función
exterior (lo que hace `_Unwind_Resume` o `longjmp`). Los marcos C traducidos se deshacen por la
comprobación de dirección de retorno hasta el que coincide: resultado idéntico al nativo (normal,
`--pic`, `--no-regcache`).

## Logs de CI legibles + diferencias Intel/AMD — hecho

**Qué:** el sandbox no puede leer los logs de Actions (el proxy bloquea los blobs). Ahora el job x86
guarda la salida del diferencial y el modelo de CPU del runner, y **si falla** los publica en la rama
`ci-logs-<job>` (legible por la API de contenidos). `run.py`, en un fallo, vuelve a ejecutar con
`VP_DUMP` y añade qué bytes de datos difieren (offset nativo/traducido).
**Primer hallazgo:** los runners x86 de GitHub son **AMD EPYC 7763** (la máquina local es Intel
Xeon): todo el x87 aleatorio coincide también en AMD; la única diferencia fue `MXCSR_MASK` en
`fxsave` (Intel 0xFFFF, AMD 0x2FFFF con MM). El objetivo es Jaguar (AMD) → `vp_x87_fxsave` escribe
0x2FFFF y los tests ponen ese campo a 0 tras `fxsave` (depende del fabricante).
**Por qué importa:** tener los dos fabricantes en el bucle diferencial (local Intel, CI AMD) separa
lo que es arquitectura de lo que es implementación; para el PS4 manda lo de AMD.

## Huecos de ISA de Jaguar: SSSE3/SSE4.1/SSE4.2 completos, AES, PCLMUL, CRC32, cadenas — hecho

**Qué:** auditoría de mnemónicos frente a lo que Jaguar ejecuta; faltaban y ahora están (formas SSE
y VEX.128 con puesta a cero de la mitad alta):
- SSSE3: `pmulhrsw`, `pmaddubsw`, `phaddw/phsubw/phsubd`, `phaddsw/phsubsw` (saturados),
  `psignb/psignw`. SSE4.1: `pmuldq`, `pminsb/pmaxsb/pminuw/pmaxuw`, `pblendw`, `phminposuw`, `mpsadbw`.
  SSE4.2: `pcmpgtq`, **`pcmpestri/pcmpestrm/pcmpistri/pcmpistrm`** (los 4 modos de agregación,
  formatos con/sin signo de bytes/palabras, polaridades incl. "masked negate", longitudes implícitas
  y explícitas con |rax|/|eax| saturado, índice LSB/MSB, máscara de bits o expandida; CF ZF SF OF),
  **`crc32`** (CRC-32C, todos los tamaños).
- **AES-NI** (`aesenc/aesenclast/aesdec/aesdeclast/aesimc/aeskeygenassist`) y **`pclmulqdq`**: Jaguar
  los tiene y los juegos los usan para descifrar; tablas S-box generadas, estado columna-mayor como
  Intel. Ahora también se anuncian en CPUID.
- Cadenas: `lods*`, `scas*`, `cmps*` con `rep/repe/repne` (la condición de repetición usa el resultado
  de la comparación, no el flag perezoso); el `cmpsd` de cadena ya no es `unsupported`.
- `pushfq/popfq`, `lahf/sahf`, `xlat`, `bextr` (BMI1), `movnti`, `clflush(opt)`/`prefetch` (nop).
**Prueba:** caso `isa_extra.s` (generado con semilla fija: 120 operaciones aleatorias sobre los datos
aleatorios del arnés + cadenas con terminadores para `pcmpistr*` con 17 inmediatos, flags guardados
con `pushfq`): idéntico al hardware; mutaciones (polaridad 3 de `pcmpstr`, MixColumns) → FAIL.
30/30 (normal, `--pic`, `--no-regcache`).
**Queda de Jaguar sin hacer:** SSE4a (`extrq/insertq/movntss/movntsd`, solo AMD: no verificables en
el Intel local; sí en el runner AMD de CI) y XSAVE/XRSTOR explícitos.

## SSE4a (verificado en el runner AMD) + caché de veneers HLE — hecho

**SSE4a** (`extrq/insertq` en sus dos formas, `movntss/movntsd`): instrucciones solo de AMD; el
caso `sse4a.s` lleva `# requires: sse4a` → `run.py` en un CPU sin el flag lo compara solo con su
golden (y `VP_NO_NATIVE` evita la ejecución nativa en el arnés); el golden lo escribe el runner AMD
y CI lo publica en `ci-logs-differential-x86/golden/`. Hallazgo de la primera pasada en AMD: la mitad
alta del destino (documentada como "indefinida") la **pone a cero** el hardware → igual aquí.
Otro hallazgo AMD vs Intel: tras `bextr`, AF (indefinido) sale a 1 en AMD y a 0 en Intel → el test
ya no mira flags indefinidos.
**Caché de veneers** en el motor de AstroVisionPro: la llamada HLE pasaba cada vez por
`QueryExecutableRange` (en shadPS4: un mutex global + búsqueda lineal en `HleVeneerAllocator`,
contención entre hilos) e interpretaba las 4 instrucciones. Ahora: tabla sin bloqueos (16 384
huecos, sondeo lineal ≤ 8) de direcciones de veneer ya vistas; en un acierto se comprueban los 13
bytes exactos del patrón (`mov r10,rcx; mov rax,imm32; syscall; ret`) y se ejecuta directamente;
`Invalidate` la vacía. En el host de test (cuya consulta es trivial) 58,6 → 44 ns por llamada; en
shadPS4 el ahorro es mayor (se evita el mutex). `run.sh` imprime la medida (`VP_HLE_BENCH`).

## Caché de despacho por hilo — hecho

**Qué:** `vp_dispatch` (llamadas/saltos indirectos, llamadas a imports y veneers) recorría todos
los módulos enganchados + búsqueda binaria + nativos + rangos de salida en cada llamada. Ahora una
caché por hilo de 1024 entradas (dirección → función traducida y `entry`), etiquetada con una
generación global que sube al registrar, mover o desenganchar un módulo (así un `Invalidate` nunca
deja entradas obsoletas). También recuerda los destinos que resolvió el embebedor
(`vp_dispatch_miss`: veneers HLE) para saltarse el recorrido de módulos, salvo si el propio handler
cambió los módulos. Con un juego de ~20 módulos y C++ con llamadas virtuales, esto quita el coste
lineal en módulos de cada llamada indirecta. Todas las suites siguen verdes.

### Revisión adversarial (ISA, cachés) — corregido
1. **Flags perezosos con escrituras condicionales** (bug antiguo, grave): el análisis de vida daba
   por escritos los flags de `shl/shr/sar/rol/ror/rcl/rcr/shld/shrd` por `cl` y de `rep cmps/scas`;
   con cuenta 0 / rcx = 0 la instrucción no los toca, pero el productor anterior ya no los había
   calculado → flags basura. Ahora esas instrucciones (`writes_flags_conditionally`) mantienen vivos
   los flags que podrían no escribir. (Los desplazamientos por inmediato ≠ 0 siguen optimizándose.)
2. Cadenas con prefijo de tamaño de dirección (esi/edi/ecx, con extensión a cero) y con `fs:`/`gs:`
   en el origen (también en `movs/stos`, heredado).
3. `pushfq/popfq` conservan AC e ID (`VpCpu.rflags_ac_id`): el truco de detección de CPUID funciona.
4. Caché de despacho: `vp_register_native`/`vp_add_exit_range` suben la generación; un miss cacheado
   que falla no vuelve a llamar al handler (podría ejecutar dos veces parte de un stub). Caché de
   veneers con época: un `Invalidate` concurrente no puede ser deshecho por una inserción tardía.
5. CI: los runners x86 de GitHub salen **Intel o AMD al azar**; `# requires:` admite varios flags y
   en ARM un caso sin golden se salta en vez de fallar. El golden de `sse4a` se versionará cuando
   un runner AMD lo publique.
Regresiones: `review_str2/shl/flags/a32`. 35/35, también `--no-lazy-flags`.

## Excepciones C++ reales de punta a punta — hecho (3 bugs encontrados)

**Test:** `tests/aot/exceptions/` — programa C++ (`throw` a través de varios marcos, `catch` por tipo,
`throw;`, destructores durante el desenrollado, `try` anidados, `catch (...)`) enlazado **estático**
con su propio `libsupc++` + `libgcc_eh` (como un juego trae su libc++/libunwind) y un `shim.c` mínimo
(malloc de montón fijo, `dl_iterate_phdr`/`_dl_find_object` sobre las cabeceras de programa,
stubs de stdio/pthread); se traduce **entero** (unwinder incluido, 20 348 instrucciones, 99,99 %) y
se compara con el mismo C++ compilado nativo. CI x86 y ARM (con `g++-x86-64-linux-gnu`).
**Bugs reales que destapó:**
1. **Funciones con partes antes de su entrada** (división caliente/fría de gcc/clang: `.text.unlikely`
   va antes que `.text`): los bloques se emiten por dirección y la función C empezaba ejecutando el
   bloque frío. Ahora `goto L_entrada` al principio cuando el primer bloque no es la entrada. Afecta a
   cualquier binario optimizado con PGO/`__builtin_expect`/`cold`, también los de PS4.
2. **Un registro de `.eh_frame` ilegible cortaba todo el recorrido**: la comprobación de lectura
   anticipada (16 bytes) fallaba en la última LSDA de la sección y se perdían las *landing pads* de
   todo lo que venía después. Ahora se salta solo ese registro.
3. **Landing pads fuera del rango alcanzado** (tras el último `ret` o en la parte fría): se
   descartaban; ahora, si no caen dentro de una función descubierta, son funciones propias (entrar
   en ellas con el marco que restauró el unwinder equivale a entrar a mitad de la original).
Además: `rdssp/incssp` (CET, que el unwinder de libgcc consulta) son NOPs como en un CPU sin CET; el
host del test da TLS válido (puntero a sí mismo en fs:0 y canario en fs:0x28). El registro de
entradas perdidas funcionó en la depuración (`VPENGINE: missing entry main+0x176c`).

## Cobertura sobre binarios reales grandes (informe sin ejecutar) — medido

`vpaot --elf X --stats` sobre binarios del sistema (gcc/clang x86-64 genérico, no Jaguar):
python3.12 1,49 M instrucciones **100,00 %** (solo `ud2`/`hlt`); libstdc++ 1,02 M **100,00 %**;
gdb 5,1 M 99,95 % (`ud2`, E/S privilegiada); libc 99,02 % y libcrypto 92 %: lo que falta es AVX2,
AVX-512 (`kmov*`, `vmovdqu64`, EVEX) y FMA de sus rutas para CPUs modernas, que **Jaguar no tiene**
(el PS4 nunca las ejecuta); libm: FMA/FMA4 ídem. Hueco real encontrado y cerrado: `vldmxcsr` /
`vstmxcsr` (formas VEX). MMX sigue sin implementar (Jaguar lo tiene; el SDK de PS4 no lo emite salvo
intrínsecos): aparecerá en el informe del eboot si hace falta.

## Puntos de reanudación: fibers, corrutinas y cambios de contexto — hecho

**Problema:** un cambio de contexto (sceFiberSwitch de shadPS4 — `fiber_fex.cpp` cambia RSP y los
registros preservados en el frame HLE y el `ret` del veneer sigue en la otra fibra —, corrutinas o
librerías de fibers propias del juego en asm, `longjmp` a otra pila) hace un `ret` a una dirección
de retorno cuyos marcos C traducidos **ya no existen** (son de la otra pila). Antes: los marcos se
deshacían por la comprobación de retorno hasta `vp_run`, que fallaba con "returned to an untranslated
address".
**Solución:** cada dirección de retorno de un `call` es una **entrada reanudable**: etiqueta `R_<dir>`
justo antes del `VP_IN()` que recarga los registros locales, un `case` en el `switch (entry)` de la
función y una fila en la tabla de entradas intermedias (ahora ordenada: búsqueda binaria en el
runtime). `vp_run` ya no se rinde cuando el último `ret` no llega a la salida: si la dirección es
conocida, sigue despachando desde ahí. `--no-resume-points` lo desactiva (para comparar).
**Coste:** ninguno medible en el bench (1,09×): solo añade entradas al switch y filas a la tabla.
**Pruebas:** `coroutine.s` (ping-pong entre dos pilas con su propio cambio de contexto en asm:
idéntico al nativo, normal/`--pic`/`--no-regcache`; con `--no-resume-points` falla como antes) y en
`tests/shadps4` **fibers cambiados por el bridge HLE exactamente como `fiber_fex.cpp`** (guarda y
carga rbx rsp rbp r12–r15 en el frame; el `ret` del veneer entra en la fibra): 6 idas y vueltas OK.

## Pila del host para hilos invitados — parche ampliado

Con AOT cada llamada invitada es una llamada C en la pila del host. Medido con `-fstack-usage` sobre
el programa C++ traducido: marco mediano 56 B, p90 ~1 000 B, máx 1 112 B. shadPS4 crea los hilos con
`pthread_create(..., nullptr, ...)`: 512 KiB en Apple → una recursión invitada de unos cientos de
niveles en funciones grandes desbordaría (con FEX no pasaba: el JIT no anida en el host). El parche
de AstroVisionPro ahora también toca `core/thread.cpp`: con `SHADPS4_GUEST_CPU_VPENGINE`, 16 MiB de
pila por hilo (`git apply --check` OK).

## Para el usuario (primer paso con el eboot)

Workflow `vpaot-windows.yml` (dispatch): deja `vpaot.exe` en la release `vpaot-windows` del repo.
En el PC: `vpaot.exe --elf "A:\...\CUSA03173\eboot.bin" --stats report.json` (sin `--out` solo
mide), o el juego entero con `translate_game.ps1` (misma release). `report.json` → `unsupported_by_mnemonic` dice qué instrucciones faltan. Nada del juego
sale del PC ni va al repo. Enlace directo una vez publicada:
https://github.com/escojoncio/VPEngine/releases/download/vpaot-windows/vpaot.exe

## Registros en locales (`--regcache`, ON por defecto; `--no-regcache`)

`runtime/vp_regs.h`: el C generado accede a los GPR solo por `VP_R64/32/16/8/8H(i)` y
`VP_W*(i, v)` (y `VP_PUSH`/`VP_POP`); con `VP_REGCACHE` cada función declara `g0..g15`
(`VP_DECL()`), y `VP_LOCAL` (definido por instrucción) elige local o struct. Cuerpos con helpers
que tocan `cpu->r` (mul/div, cpuid/rdtsc, llamadas `fn_*`, dispatch, nativos, fallos, `return`)
van en modo struct entre `VP_OUT()`/`VP_IN()`; el resto en locales. Sin `VP_REGCACHE` las macros
son el struct. Los flags también en local: `VpCpu vpF` por función (solo se usan sus bytes de
flags; el compilador la escalariza), `VP_FC` = `&vpF` o `cpu` según `VP_LOCAL`; todo acceso a
flags del C generado va por `VP_FC` (`vp_flags_*(VP_FC, …)`, `vp_cc(VP_FC, …)`, `VP_FC->cf`);
`VP_OUT/IN` sincronizan también los flags. **14/14 en ambos modos; programa entero OK.**
Bench clang -O2: **1,05–1,13× nativo** (gcc 1,19×); sin regcache 1,71×.
- Revisión adversarial (subagente) de regcache: lógica confirmada; test de estrés
  `regcache_mix.s` añadido (15/15). `push/pop` de 16 bits → `unsupported` (antes se emitían como 64).
- Bug corregido: `Range::contains` desbordaba con `a = 0xffff…` (`a + size <= end`); ahora
  `a < end && size <= end - a`. Lo disparó la detección de `mov $imm` como puntero a función.
- `--scan-data`: qwords alineados en datos que apuntan a código decodificable → raíces (para
  imágenes sin relocaciones, como el test de programa entero; un eboot las tiene).
- `tests/aot/program/`: programa entero freestanding (`prog.c` + `start.s`) enlazado estático en
  0x400000 → `vpaot --elf --scan-data` → `host.c` carga los PT_LOAD, ejecuta `vp_run(entry)` y
  compara `rax` con el mismo `prog.c` compilado nativo (`-Dvp_main=vp_main_native`). En CI (x86 y
  ARM, con `gcc-x86-64-linux-gnu` para el invitado).

## Siguiente sesión (por orden)

1. **Con el usuario:** pasar `translate_game.ps1` (release `vpaot-windows`) sobre su dump de Astro Bot
   y leer los `*.json` (`unsupported_by_mnemonic`, `supported_fraction`); cubrir lo que salga.
2. Decidir con él cómo entra el C traducido (privado) en la build de visionOS: Mac local, repo
   privado con token, o artefacto. Después aplicar `integrations/shadps4/astrovisionpro.patch` en
   AstroVisionPro y primera build con `ENABLE_VPENGINE_GUEST_CPU=ON`.
3. Primer arranque en el visor: leer `Documents/vpengine_missing.txt` → `-Missing` → recompilar
   (bucle hasta que no falte nada).
4. Excepciones C++: probadas de punta a punta (`tests/aot/exceptions`); en PS4 el unwinder es el
   del juego/prx (libc++abi/libunwind de Sony): mismo mecanismo.
5. Opcional: intérprete de reserva para código no traducido (hoy: fallo limpio + registro).
6. Golden de `sse4a` cuando un runner AMD lo publique (`ci-logs-differential-x86/golden/`).

## Notas técnicas

- `ret` deja `cpu->rip` = dirección de retorno; el llamador hace `if (cpu->rip != next) return;`
  para desenrollar hasta quien pueda despachar. `vp_run` termina cuando `rip == VP_HOST_EXIT_ADDRESS`
  (0x7fff0000, empujado por el host).
- Operandos de 8 bits altos (AH–BH): `vp_r8h`/`vp_w8h`; índices GPR calculados por rangos de enum
  de Zydis (`gpr_index`).
- `cmovcc` de 32 bits escribe siempre (cero-extensión aunque no mueva); fuente leída siempre.
- `imul` de 2/3 operandos: SF/ZF/PF "indefinidos" pero se calculan como el hardware real (los
  tests los comparan).
- El hash de memoria del harness pone a 0 la ranura de la dirección de retorno (difiere entre
  nativo y traducido por diseño).
