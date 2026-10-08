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

## Para el usuario (primer paso con el eboot)

Workflow `vpaot-windows.yml` (dispatch): deja `vpaot.exe` en la release `vpaot-windows` del repo.
En el PC: `vpaot.exe --elf "A:\...\CUSA03173\eboot.bin" --stats report.json` (sin `--out` solo
mide). `report.json` → `unsupported_by_mnemonic` dice qué instrucciones faltan. Nada del juego
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

0. Mandar un `[build]` para ejecutar los dos workflows por primera vez (nunca se han lanzado).
1. Lanzar los dos workflows (`[build]`) y arreglar lo que salga (Swift sin compilar; en ARM el
   runner necesita `binutils-x86-64-linux-gnu` para ensamblar los casos).
2. Programa entero: caso de test con varias funciones, recursión y punteros a función
   (indirect call por `vp_dispatch`) + un `main` que use `vp_run`; luego el mecanismo de imports
   PS4 (stub → función nativa en `vp_dispatch`).
3. Test de excepciones reales: imagen estática con libgcc_eh/libunwind enlazado (o `_Unwind_*`
   como nativos) que lance y capture; verifica la entrada por `vp_extra_entries`.
4. Pasar `vpaot --elf eboot.bin --stats` en el PC del usuario y añadir lo que falte de
   `unsupported_by_mnemonic` (esperado: pshufb, pmulld, blend*, round*, movbe, quizá AVX).
5. Rendimiento: flags perezosos; registros en locales por bloque; medir vs FEX.
6. Refinar tablas de salto: con `cmp $N` encontrado se leen N+1 entradas saltando las no-código;
   verificar que el `cmp` tomado es el del índice (ahora: el más cercano hacia atrás, ≤12 insns).

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
