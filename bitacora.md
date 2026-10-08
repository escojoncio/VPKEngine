# Bitácora — VPEngine

Motor común para juegos de PS4 en Apple Vision Pro: traductor AOT x86-64 → C (`tools/vpaot`),
runtime de semántica (`runtime/`), capa visionOS compartida (`platform/visionos/`), tests
diferenciales (`tests/aot/`). Plan de arquitectura en `docs/PLAN.md`; Bloodborne en `Bitacora BB.md`.

## Estado (sin build de CI aún; todo verificado en local en x86-64 Linux)

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

## Siguiente sesión (por orden)

1. Lanzar los dos workflows (`[build]`) y arreglar lo que salga (Swift sin compilar; en ARM el
   runner necesita `binutils-x86-64-linux-gnu` para ensamblar los casos).
2. Programa entero: caso de test con varias funciones, recursión y punteros a función
   (indirect call por `vp_dispatch`) + un `main` que use `vp_run`; luego el mecanismo de imports
   PS4 (stub → función nativa en `vp_dispatch`).
3. Landing pads desde LSDA (`image.cpp` ya tiene FDEs; falta parsear `.gcc_except_table`) →
   `Function::extra_entries` (el switch de entrada ya se emite).
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
