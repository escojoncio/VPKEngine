# Auditoría 2 de gpu/vulkan (2026-11-12): capture, foveación, passdump, config, translate_pool, tools

Alcance: `gpu/vulkan/src` salvo lo que edita otro agente (pending, pipelines, commands, entry, recipes, ext) y `gpu/vulkan/tools`.
Leído entero: capture.cpp, config.cpp, foveation.*, passdump.*, translate_pool.h, texdecode.*, taa_native.*, physical_device.cpp,
precompile.cpp, vpcap.cpp, vpcap_calls.h (códec), platform_null.cpp, tools/*. `.mm` de Metal: razonados, no ejecutados.
Gravedad: A = caída, corrupción, carrera o bloqueo; B = resultado erróneo o UB que hoy funciona; C = menor.
Tests nuevos: `gpu/vulkan/tests/audit_vk2_tests.cpp` (+ `audit_vk2.cmake`, ejecutable `vpvk_audit2`), un grupo por hallazgo.
Cada grupo falló antes del arreglo (binario sin arreglos, sin sanitizers; `passdump_race` y `replay_imagedata` incluidos) y pasa
después con ASan+UBSan (los 9 grupos, build propio). TSan solo sobre los grupos que no cargan la librería (`passdump_race`,
`pool_shutdown`, `fov_nan`, `json`, `png_hostile`): 0 informes. Estado: **arreglados F1-F8**.
Regresión (ASan+UBSan, `vpvk_tests`): capture*, png, imgdiff, foveation, pass_dump, taa_native, replay, pipeline, async/sync_pipelines,
pipeline_policy, translator_modes: todos pasan.

## Hallazgos arreglados
- F1 [B] `config.cpp` `vpvk_config_set_foveation` / `_set_vpktaa` / `_set_render_scale`, `foveation.cpp` `shapeOf`, `opticalCenter`,
  `axisQuality`: `std::clamp` deja pasar NaN (un 0/0 del anfitrión, un float sin inicializar). Escenario: `sharp_zone = NaN` → las
  calidades del mapa de tasas son NaN → `std::lround(NaN)` (UB) en la clave del mapa y mapa de tasas inválido enviado a Metal;
  tangentes infinitas dan `inf/inf = NaN` en `opticalCenter`. Arreglo: `clampOr` (NaN → valor por defecto del campo, como ya hace
  `rcasClamp`) en los setters, `shapeOf`, `opticalCenter`, `axisQuality`, `smootherstep` y el desplazamiento vertical.
  Tests `fov_nan`, `config_nan`.
- F2 [A] `tools/vpimage.h` `pngDecode`: `raw(h*(row+1))` y `px.assign(w*h*4)` con w/h del IHDR sin validar. Escenario: PNG de 4 GB
  de cabecera (un dump dañado en `vpimgdiff` / `vpreplay`) → `bad_alloc` o gigabytes reservados. Arreglo: rechazar w*h > 2^26.
  Test `png_hostile` (5 tamaños, hijo con tope de memoria).
- F3 [A] `tools/vpreplay.cpp` `onImageData` (:644): el búfer de `total` bytes se reservaba con el `total` del fichero (2^62 →
  `length_error`/`terminate`) y el chunk se copiaba si `offset + size <= total` con la suma en 64 bits: `offset = 2^64-8`,
  `size = 16` pasa la comprobación y `memcpy` escribe fuera del vector (SIGSEGV / "munmap_chunk(): invalid pointer"). Además
  `expectedImageBytes` multiplicaba en 32 bits el redondeo a bloques (`w + bw - 1` envolvía). Arreglo: validar contra el formato
  de la imagen (`total == esperado`), `offset <= total && size <= total - offset`, imagen desconocida = fallo contado, y dimensiones
  > 65536 = 0 bytes esperados; un tamaño que no es el de la imagen sigue diciendo «image upload skipped» (ahora por registro, no al
  completar el subrecurso). Test `replay_imagedata` (capturas `.vpcap` fabricadas con `vpcap::Writer`).
- F4 [B] `capture.cpp` `finishCapture` (:878): el texto END iba a `char b[1024]` con `snprintf`. Escenario: una captura con varias
  llamadas sin hook (`unhooked_used_calls`) y razones de invalidez (`reason=`) pasa de 1 KiB → se pierde el final, incluido `end=`
  (`frames_done`, la razón de fin) y a veces `invalid=`. Medido: 20 llamadas sin hook = 1239 B de META+END, `end=` ausente.
  Arreglo: búfer de 1024 + longitud de las dos listas + `why`. Test `capture_endtext`.
- F5 [B] `capture.cpp` `makePath` (:852): nombre `vpcap_<segundos>_<pid>.vpcap`. Escenario: dos capturas del mismo proceso en el mismo
  segundo (p. ej. `vpvk_capture_request` desde el anfitrión dos veces seguidas con directorio) → el segundo `fopen("wb")` trunca el
  primero. Arreglo: si el fichero existe, sufijo `_2`, `_3`… (un solo fichero cuando no hay colisión, como espera `capture_env_dir`).
  Test `capture_filenames`.
- F6 [A] `passdump.cpp` `passDumpNext` (:96): `seen++`, `windows`, `windowEnd`, `nextNs` sin sincronizar sobre el `PassDump`
  global, y lo llama el hilo de submit de CADA dispositivo (cada uno con su `submitMutex`; `entry.cpp:3570-3592`). Escenario: dos
  dispositivos con `VPVK_PASS_DUMP=1` → carrera de datos (cuenta perdida, ventana abierta dos veces). `passDumpSummary` leía `seen`
  sin lock. Arreglo: mutex estático en `passDumpNext` y lectura de `seen` bajo él. Test `passdump_race` (4 hilos × 10^6 llamadas:
  antes 1,8·10^6 de 4·10^6).
- F7 [A] `translate_pool.h` `~TranslatePool`: ponía `stop_`, esperaba a los hilos y dejaba los trabajos en cola sin terminar. Escenario:
  un hilo en `wait(job, UINT32_MAX)` de un trabajo que nunca empezó (pipeline asíncrono, dispositivo destruido) se queda bloqueado para
  siempre. Arreglo: tras el `join`, cada trabajo en cola se marca terminado (sin ejecutar `fn`) y se despierta a sus esperas.
  Test `pool_shutdown` (antes: el hijo no termina en 5 s).
- F8 [B] `tools/vpjson.h` `Parser`: (a) recursión sin límite: 300000 `[` desbordan la pila (SIGSEGV); (b) `\u` truncaba a un `char`
  (`é` → byte 0xE9, no UTF-8; sin pares sustitutos) y aceptaba hexadecimales inválidos; (c) `\r \b \f` no se decodificaban;
  (d) `strtod` aceptaba `nan`, `inf` y flotantes hexadecimales. Arreglo: profundidad máx. 64 (el `index.json` anida 3), UTF-8 y
  pares, hex estricto, `\r\b\f`, número solo con `-` o dígito. Test `json`.

## Sin arreglar (informe)
- R1 [C, rendimiento] `capture.cpp` `resolveRef` (~:640-703), caso `DescSet`: copia `set->flat` bajo el `unique_lock` de
  `setLock` por cada referencia repetida de `cbRefs`; con un cb que usa el mismo set cientos de veces se copia cientos de veces.
  Arreglo: `visited` para todos los tipos, no solo algunos. Sin test (solo coste; otro agente edita esa zona).
- R2 [B, no verificado] `capture.cpp` `pipelinesHook` (:1209): `|| r < 0` sale sin registrar los pipelines que un lote con fallo
  parcial sí creó (los fallidos quedan en `VK_NULL_HANDLE`, y el bucle ya los salta). Un cb posterior que los use no los encuentra y
  la captura queda sin ellos. Arreglo: quitar `|| r < 0`. Sin test: el backend nulo no falla un pipeline suelto de un lote.
- R3 [C] `translate_pool.h`: el comentario «Jobs that never started are dropped» está duplicado (resto de otra edición). Código
  muerto/ruido; no se tocó. Tras el arreglo F7 la frase de «se descartan» sigue siendo cierta para `fn`, no para la espera.
- R4 [C] `vpreplay.cpp` `onImageData`/`upload`: `width/height/depth/mip/layer` del registro se validan contra el formato (bytes
  esperados = `total`) pero no contra el extent, los niveles y las capas de la imagen creada: un registro con mip/capa fuera de rango
  o una región mayor que el mip llega a `vkCmdCopyBufferToImage` con una región inválida (comportamiento del driver bajo prueba).
  Arreglo: comparar con `ci.extent >> mip`, `mipLevels`, `arrayLayers`. Sin test (necesita imagen real creada en el replay).

## Limpio (leído y razonado, sin hallazgo)
- `texdecode.cpp` (ASTC LDR), `taa_native.cpp` (matemática, `rcasClamp` ya trata NaN), `physical_device.cpp`, `precompile.cpp`,
  `perf.h`, `platform_null.cpp`: leídos enteros, sin hallazgo (solo lectura; no hay test propio de ASTC/TAA en este lote).
- `vpcap.cpp` (Writer/Reader): longitudes de bloque y de registro validadas contra el tamaño del fichero; `decodeImageData` comprueba
  `size <= c.size - 64`. `vpimgdiff`, `vpcap_dump`: solo lectura con comprobaciones de tamaño.

## No ejecutado
- Metal (`platform_metal.mm`, `taa_native_metal.mm`): razonado, no ejecutado. Sin CTS/dEQP. Sin pruebas con GPU real.
- El grupo `replay_imagedata` ejecuta vpreplay con una captura sin objetos (el fallo de F3 es previo a cualquier llamada Vulkan).
