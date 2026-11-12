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

## Comprobaciones de que no cambia la salida
- MSL de HEAD frente al actual: 28105 combinaciones (73 `.spv` × 128 variantes × 3 etapas; 7222 lowerings correctos) y 12 variantes de
  cada uno de los 17 `.lift`: hashes idénticos (`shaderId`).
- ctest `gpu/shader` con ASan+UBSan: 169 + `d24_emul` + `gcn.lift.differential(_cf)` en verde; `gpu_vulkan.stage_keys` y 87 más de
  `gpu_vulkan` en verde en Release (`replay` falla por un test ajeno de `vk_tests.cpp`, otro lote).
- Fuzz tras los arreglos: ≈300 000 iteraciones SPIR-V, ≈200 000 GCN, ≈360 000 texto VPIR, 2 M palabras al decodificador: 0 informes.

## Medido y limpio
`optimize`/CSE, `verify`, `print`, `emitMsl` y el lifter son lineales (50 000 `v_add` = 0,4 + 1,2 s; 100 000 sumas encadenadas = 0,5 s). Sin hilos
ni estado global mutable en `gpu/shader` (la caché `thread_local` de `lower.cpp:~838` `usedWithDref` está muerta: se construye y no se usa;
no se borra). `std::getenv` en `lift.cpp:3324` (`VPGCN_DEBUG`, por cada lift) y `msl.cpp:1595` (una vez): lectura concurrente con el
`setenv` de `vkCreateDevice` ("Dudosos" del lote 2 de `notas/auditoria.md`).

## Abierto
- A1 [C] `gcn/src/lift.cpp:249` (`buildOrder`) y `liftFetchCall`: cada `s_swappc_b64` inlinea el fetch entero (llamadas × instrucciones del
  fetch en `order` y en el VPIR); miles de llamadas con un fetch grande piden GB. Un VS real tiene una. Propuesta: `setNodeLimit` también en
  el lifter (recoger `std::length_error` en `lift()`) o error si `order` > 2^22.
- A2 [C] `ir/src/builder.cpp:23` (`current()`): recorre el camino de bloques abiertos en cada emisión, O(profundidad) por nodo (77 % del tiempo
  en 1500 ifs anidados). Con `kMaxWalkDepth` y `flow` acotados no pasa de un pico; arreglo: guardar el `Block*` en `open_`.
- A3 [C] `lower.cpp` asume SPIR-V válido: tras el fuzz no queda UB hallado, pero los operandos mal tipados que no revientan dan un
  resultado erróneo. Mitigación sin código: `spirv-val` en el precompile / CI.
- A4 [C] Los topes nuevos son arbitrarios y no verificados en juegos reales: 2^20 nodos, 65536 hojas, 1024 elementos desenrollados, 320
  niveles de `flow`, 200 de `walk`. Cualquier shader legítimo que los pase se rechaza (log `vk.err what=shader`); conviene revisar con el
  mayor shader de los logs del visor antes de la próxima build.
- A5 [C] `lower.cpp` `memberOffset`: suma `naturalSize` en `uint32_t` sin comprobar (miles de miembros de 2 GiB envuelven).
- A6 [C] `gcn/tools/lift_batch.cpp:88`: `name.substr(3, 18)` lanza `std::out_of_range` con un nombre de menos de 3 caracteres (herramienta).
- A7 [C] `gcn/src/lift.cpp` `dispatchLoop`: un `if (blk == k)` por bloque dentro del bucle → O(bloques) comparaciones por salto en la GPU; es
  diseño, solo se usa en los saltos no estructurados.
