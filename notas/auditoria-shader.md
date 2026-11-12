# Auditoría de `gpu/shader` (lote 4, 2026-11-12)

Método de `notas/auditoria.md`: gcc 13 con ASan+UBSan (`-fno-sanitize-recover=undefined`), build `/home/claude/build/fix_shader`, sin CTS/dEQP.
Además, fuera del repo: fuzzer de mutación (SPIR-V → `parse`/`lower` en 3 etapas y varias opciones → `emitMsl`; texto VPIR →
`parse/verify/optimize/toFramebufferFetch/emitMsl`; palabras GCN, con y sin fetch → `lift` → `emitMsl`; palabras sueltas al decodificador GCN)
y bancos de escala (cadenas muertas, diamantes y ifs anidados GCN, expresiones largas). Gravedad como en `notas/auditoria.md`.
Todo lo de "Arreglado" tiene test que falla antes (comprobado contra las fuentes de HEAD) y pasa después.
Al empezar había hunks sin commitear ajenos a la auditoría (`PointCoord` en `ir.h`, `ir.cpp`, `ir/src/parse.cpp`, `msl.cpp`, `lower.cpp`, venidos
de GLES): intactos; compilan y pasan.

## Arreglado
- S1 [A, CPU] `ir/src/optimize.cpp:258` (`removeDeadValues`): recontaba los usos de toda la función una vez por eslabón de una cadena de valores
  muertos: O(n²) con tablas hash, en cada `optimize` y `toFramebufferFetch`. Medido: 4000 valores = 27 s (ASan); 20000 = 13 s de CPU (-O2).
  Arreglo: un recuento y un recorrido desde el final de cada bloque (los usuarios van tras la definición), descontando usos al borrar;
  20000 valores = 0,2 s. Test `vpir.dead_chain`.
- S2 [A, CPU] `gcn/src/lift.cpp:2982` (`postDominators`): conjuntos `std::set<int>` por bloque, punto fijo y búsqueda del inmediato en
  O(|pdom|²·log): cúbico en bloques y O(B²) de memoria. Medido (-O2): 600 bloques = 1,9 s, 2000 = 143 s, solo para levantar un shader.
  Arreglo: el grafo es acíclico → recorrido en profundidad iterativo y "meet" por orden de fin; 20000 bloques = 0,7 s. ipdom idéntico al
  viejo en los 17 `.lift` y 20000 programas mutados. Test `gcn.lift.scale` (6000 diamantes < 3 s).
- S3 [A, pila] `gcn/src/lift.cpp:2949` (`dagAcyclic`): DFS recursivo con `std::function`, un nivel por bloque de una cadena: con 512 KB
  de pila (hilo secundario en macOS) SIGSEGV a ~8000 bloques. Ahora iterativo. `lift.cpp:3053` (`walk`): recursión por nivel de if
  anidado; más de 200 niveles (`kMaxWalkDepth`) → bucle de despacho (siempre correcto). Tests `gcn.lift.scale` y `gcn.lift.scale_nested`
  (levantan en un hilo de 512 KB).
- S4 [B, pila] `gcn/src/lift.cpp:2873` (`liftFetchCall`): un fetch que contiene `s_swappc_b64` se inlinea a sí mismo sin fin (fuzzer:
  desborde de pila). Arreglo: error "s_swappc_b64 inside the fetch subroutine". Test `gcn.lift.fetch_recursion`.
- S5 [B, RAM] `msl/src/msl.cpp:584` (`planConstTables`): un slot privado de hasta 2 GiB con stores constantes pasaba a tabla
  `constant uint4 vp_kN[…]` con todas las palabras, ceros incluidos, en texto: 16 MiB de slot = 61 MB de MSL. Tope 64 KiB. Test
  `vpmsl.limits.big_private_table`.
- S6 [B, RAM] `msl/src/msl.cpp:213` (`line`): sangrado de 4 espacios por nivel sin tope: el MSL crece con el cuadrado del anidamiento
  (1000 niveles = 28 MB, 5000 = 700 MB). Tope de 32 niveles de sangrado (no cambia ningún MSL del corpus). Test `gcn.lift.scale_nested`.
- S7 [B, UB/RAM] `spirv/src/parse.cpp:56-110`: tipos y constantes sin orden ni unicidad → struct que se contiene (`naturalSize`,
  `loadMem`, `constVal`: pila agotada), compuesto con constituyente inexistente (`lower.cpp` `constVal` desreferenciaba `end()`), vector de
  2^32 componentes (`uint8_t(count)` trunca). Arreglo: tipos y constantes se declaran antes de usarse (salvo el pointee de un puntero,
  con `OpTypeForwardPointer` como marcador: test `gpu_spirv.forward_pointer`), sin repetir id; vector 1..16, matriz 1..4; función o
  etiqueta repetida = error.
- S8 [B, UB] `spirv/src/module.h:110` (`SpirvModule::w`): sin límite de `words`; una instrucción más corta de lo que su opcode pide, al final
  del módulo, leía fuera (heap-buffer-overflow en `lower.cpp` `value`). Fuera de rango = 0.
- S9 [B, UB] `lower.cpp:5584`: función de entrada sin bloques → `blocks.front()` nulo. `lower.cpp:87` (`ValVec`): `Val::v[i]` sin comprobar
  (`determinant`/`inverse` con matriz de otra anchura: heap-buffer-overflow); ahora lanza `LowerError`.
- S10 [B, CPU/RAM] `lower.cpp:406,951`: `leaves()`/`leafCount()` construían la lista de hojas (2^32 elementos para un array de 2^32,
  solo para contarla) y recorrían de forma exponencial los structs que comparten tipos; `naturalSize` igual y con desbordes de 32
  bits. Arreglo: `leafTotal`/`naturalSize` con memo, tope de 65536 hojas y 2^31-1 bytes por tipo, array de elementos vacíos = 0
  hojas; arrays de interfaz y `unrollForced` de más de 1024 elementos (`kMaxUnroll`) = error.
- S11 [B, CPU] `lower.cpp:700` (`analyse`/`use`): seguía los operandos de access chains (profundidad 32) sin recordar lo visitado:
  exponencial. Ahora un conjunto (id, bloque).
- S12 [B, pila] `lower.cpp:1189,2661,3129,3157` (`resolve`, `flow`, `unrollForced`, `unrollDynamic`): recursión sin límite con un módulo
  mal formado (chain cuya base es ella misma, bloque que lleva de vuelta a su constructo). Contador `Nest` (256 / 320 / 16 / 16).
  320 niveles de `flow` = una cadena de 300 else-if; mal formado y válido se distinguen mal, ver A4.
- S13 [B, RAM] `ir/src/builder.cpp:34`, `lower.cpp:5536`: sin tope de nodos, un array de 2^31 elementos cargado entero pedía 1,2 GiB
  de `Node` (144 B). `FunctionBuilder::setNodeLimit` (lanza `std::length_error`, `lower` lo recoge como "internal error"); 2^20 nodos.
- S14 [C, test] `gcn/tests/diff/vpir_eval.cpp:112-113`: el oráculo del test diferencial sumaba/restaba `int32_t` con desbordes (UB; con
  UBSan abortaba `gcn.lift.differential` y `_cf`). Aritmética `uint32_t`.
- Test `gpu_spirv.malformed`: 16 módulos rotos hallados por el fuzzer (`spirv/tests/fuzz/*.spv`, < 4 KB cada uno) que antes agotaban la
  pila, leían fuera, desreferenciaban nulo o colgaban.

- S15 [C] (A1) `gcn/src/lift.cpp` `buildOrder`: cada `s_swappc_b64` inlinea el fetch entero. Compartir el fetch no es seguro en el IR (sin
  llamadas: el fetch escribe SGPR/VGPR del programa y su `s_setpc` vuelve a la dirección del `s_swappc`); se limita: llamadas × instrucciones del
  fetch > `kMaxFetchInlined` (2^16) = error "the fetch subroutine (F instructions) is called C times". Test `gcn.lift.fetch_calls`.
- S16 [C] (A2) `ir/src/builder.cpp:20` `current()`: guarda el `Block*` de la región abierta en `open_` → O(1) por nodo. Test `vpir.deep_nest`
  (40000 ifs anidados, hilo de 512 KB).
- S17 [C] (A5) `lower.cpp` `memberOffset`: sumas en `uint64_t` por struct (memo) con error "lies past 2 GiB"; antes `uint32_t` sin comprobar
  (envolvía) y O(miembros²). Tests `gpu_spirv.offset_overflow` y `gpu_spirv.wide_struct`.
- S18 [C] (A6) `gcn/tools/lift_batch.cpp:88`: `hashOf` ya no hace `substr` fuera de rango. Test `gcn.lift_batch.short_names` (nombres `vs.bin`, `x.bin`, …).
- S19 [C] (A7) `gcn/src/lift.cpp` `dispatchLoop`: búsqueda binaria sobre el número de bloque (`dispatchRange`, cadena de ≤ 8) → log2(bloques)
  comparaciones por salto, no una por bloque. Test `gcn.lift.scale` (comprueba la profundidad/cantidad de comparaciones). Misma ronda:
  borrada la caché `thread_local` muerta de `usedWithDref` en `lower.cpp` (y el envoltorio sin uso; `drefGlobals_` se consulta directo).
- S20 [C] (A3) Validación estructural propia en `spirv/src/parse.cpp` `validate()` (la usan `vkCreateShaderModule` y el precompile de
  `gpu/vulkan/src/precompile.cpp`, ambos pasan por `spirv::parse`): ids bajo el `bound` y definidos una vez, destinos de salto/merge/switch =
  etiquetas de la función, llamadas a funciones con el número de argumentos, sin recursión, punto de entrada = función sin parámetros, interfaz =
  variables, punteros con pointee declarado. No es `spirv-val` (sin tipos de operandos ni capacidades). Test `gpu_spirv.structure`. Con
  `spirv-val` instalado, CMake añade `gpu_spirv.val.*` (los .spv de glslang/spirv-opt de `tests/glsl`); en esta máquina no existe: sin ejecutar.

## Comprobaciones de que no cambia la salida
- MSL de HEAD frente al actual: 28105 combinaciones (73 `.spv` × 128 variantes × 3 etapas; 7222 lowerings correctos) y 12 variantes de
  cada uno de los 17 `.lift`: hashes idénticos (`shaderId`).
- ctest `gpu/shader` con ASan+UBSan: 169 + `d24_emul` + `gcn.lift.differential(_cf)` en verde; `gpu_vulkan.stage_keys` y 87 más de
  `gpu_vulkan` en verde en Release (`replay` falla por un test ajeno de `vk_tests.cpp`, otro lote).
- Fuzz tras los arreglos: ≈300 000 iteraciones SPIR-V, ≈200 000 GCN, ≈360 000 texto VPIR, 2 M palabras al decodificador: 0 informes.

## Medido y limpio
`optimize`/CSE, `verify`, `print`, `emitMsl` y el lifter son lineales (50 000 `v_add` = 0,4 + 1,2 s; 100 000 sumas encadenadas = 0,5 s). Sin hilos
ni estado global mutable en `gpu/shader` (la caché `thread_local` de `usedWithDref` estaba muerta: borrada, ver S19). `std::getenv` en `lift.cpp:3324` (`VPGCN_DEBUG`, por cada lift) y `msl.cpp:1595` (una vez): lectura concurrente con el
`setenv` de `vkCreateDevice` ("Dudosos" del lote 2 de `notas/auditoria.md`).

## Topes (A4): consumo medido y margen
Medido con instrumentación temporal (ya retirada) sobre todos los tests de `gpu/shader` (73 `.spv` en 3 etapas, 17 `.lift`, programas del
diferencial) en `/home/claude/build/fix_shader`. No hay shaders de juegos reales en el repo ni en la máquina (el mayor `.spv` mide 5 KB, el mayor
`.lift` 7 KB; `notas/hallazgos.md` no da tamaños): el margen es sobre el corpus del repo, más bancos sintéticos legítimos. Unverified con
shaders de visor.
| Tope | Valor | Máximo medido (legítimo) | Margen | Qué lo acota |
|---|---|---|---|---|
| nodos por función (`kMaxNodes`) | 2^20 | 411 (SPIR-V, `mesh_multiview.vert`); 9829 (lifter, `vs_cbuf_dword`, 631 instr.); 160003 el test sintético `wide_struct` | 2550x / 107x / 6,5x | RAM: `sizeof(Node)` = 144 B; RSS medido 370 MB al llegar al tope, 510 MB con 1M nodos terminados |
| hojas de un compuesto (`kMaxLeaves`) | 65536 | 1 | 65536x | RAM |
| elementos desenrollados (`kMaxUnroll`) | 1024 | 4 (array de interfaz y `unrollForced`); 100 (`dyn_index_big`, descriptores, tope propio 256) | 256x | tamaño del código emitido |
| niveles de `flow` | 320 | 5 | 64x | pila |
| niveles de `walk` (lifter) | 200 | 5 | 40x | pila |
| fetch inlinado (`kMaxFetchInlined`) | 2^16 instr. | 9 | 7000x | RAM (~15 nodos por instrucción) |
| `resolve` / `unroll*` | 256 / 16 | 3 / 1 | 85x / 16x | pila |
Pila (gcc 13 x86-64, -O2, hilo de 512 KB, como el de macOS; topes sin tope con `VP_SCALE` temporal): `flow` (flow → arm → emitBlock) ≈ 0,85 KB
por nivel: cadena de else-if o ifs anidados aguanta ~600 niveles y cae a ~650; bucles anidados ~450 (2 niveles de flow cada uno); con ASan -O1 cae
a ~100-150 niveles (los tests de profundidad corren en el hilo principal). `walk` aguanta 1800 y cae a ~2800-3200 (≈ 0,17 KB por nivel).
Conclusión: ningún tope de niveles/nodos se sube: ya tienen ≥ 8x sobre lo medido en shaders legítimos y subirlos solo abre memoria (nodos: +150 MB
por cada 2^20). El límite de `flow` ya no depende de los marcos de cada arquitectura: ver S21.

## Pila (S21, cierra A8)
- `vpir::StackGuard` (`ir/include/vpir/stack.h`, `ir/src/builder.cpp`): al entrar en `spirv::lower` y `vpgcn::lift` lee el tamaño real de la pila del hilo
  (`pthread_getattr_np` en Linux, `pthread_get_stackaddr_np`/`_stacksize_np` en Apple; otra plataforma = sin comprobación, solo topes de niveles) y fija
  un suelo = fondo de la pila + max(32 KB, 1/4 de lo disponible al entrar). `vpir::stackLow()` (una comparación) se consulta en `Nest` (todas las
  recursiones de `lower.cpp`: flow, resolve, unroll → `LowerError` "nested too deep for the stack of this thread") y en `walk` del lifter (cae al bucle de despacho).
  Los topes de niveles siguen como defensa secundaria. Sin cambio para módulos válidos: 181 tests de shader (goldens MSL/`shaderId` incluidos) en verde.
- Cifras finales (Release -O2 x86-64, niveles de else-if que bajan con éxito antes de que el suelo o el tope 320 lo impidan): 512 KB = 319 (el tope), 256 KB = 230,
  128 KB = 108, 64 KB = 28; bucles anidados 159 (512 KB) y 21 (64 KB). ASan -O1: 82 niveles a 512 KB; a 64 KB ni un shader trivial (error limpio, no SIGSEGV).
  Un hilo de 512 KB llega, pues, al tope de 320 niveles; uno más pequeño recibe el error por pila en vez de caer.
- Tests: `gpu_spirv.stack_limit` (escalera de 400 else-if en hilos de 512 KB y 64 KB: error con mensaje, sin SIGSEGV, bajo ASan; 300 niveles en 64 KB: error "stack";
  300 niveles en el hilo principal: baja bien), `gcn.lift.scale_nested_64k` / `_128k` (150 ifs anidados en hilos de 64/128 KB).
- No cubierto: `emitMsl`/`verify` también recursan por la profundidad del IR (≤ 320 niveles tras `lower`, ≤ 200 tras el lifter) sin guarda propia; medido
  en Release con hilo de 512 KB sin fallo (cadena de 600 niveles con `VP_SCALE`). macOS/arm64 sin medir: la guarda mide bytes reales, no depende de ello.

## Abierto
- A8 [C] Con shaders reales del visor (UE4/Unity, log `vk.err what=shader`) repetir la tabla de arriba; ninguno estaba disponible aquí.
