# Auditoría misc (zona: tools, workflows, core/src restante, ps4 fetch/formats, shaders visionOS, tests)

Build propio: /home/claude/build/aud_misc (gcc, ASan+UBSan, make -j2). Sin CTS/dEQP. Nada commiteado.
Gravedad: A caída/corrupción/carrera; B desperdicio o fallo de CI; C menor. [F] = arreglado con test que falla antes.

## Arreglado
- M1 [B,F] `gpu/core/src/pipeline_cache.cpp:150-160` (+ `pipeline_cache.h` Entry::queuedPriority): `request()` sobre un estado ya Queued empujaba otro Job (copia de PipelineState) en cada petición, a cualquier prioridad → cola sin tope mientras los workers están ocupados, y `queuedAt` se reiniciaba (pending() daba edad errónea). Ahora solo una petición más urgente (o boost) añade un job. Test `gpu_core.audit.requeue` (audit_misc_tests.cpp).
- M2 [C,F] `pipeline_cache.cpp` `decode()`/`prefetch()`: formatos fuera del enum aceptados desde pipelines.keys/recetas dañados o de otra versión. Ahora se rechazan. Test `decoderange`.
- M3 [C,F] `diagnostics.cpp:292` `attStr`: `width >> a.mip` con mip ≥ 32 (UB). Acotado a 31. Test `dumpdeepmip` (UBSAN halt_on_error). El mismo desplazamiento está en `frame_compiler.cpp:245` `extent()` (NO es mi zona, sin tocar).
- M4 [C,F] `commands.cpp:109-111,131-133`: suma `offset + first*stride` y `x + width` envolvían (rango de hazard al inicio del buffer/origen). Saturan. Test `rangesat`.
- M5 [B,F] `tools/cts/run_lot.py:109,127`: `open(log).readlines()` / `.read()` cargaba el log entero del proceso deqp (cientos de MB) para dar 40 líneas. `tail_text()`. Test `tools/cts/tests/test_tools.py`.
- M6 [C,F] `tools/cts/diff.py:49`: caso NotRun antes y Fail ahora no salía en ningún apartado (ni regresión ni nuevo). Ahora "New cases failing". Mismo test.
- M7 [B,F] `tools/sdk/gen_runtime_tbd.sh`: sin símbolos `vp_*` (lib equivocada, nm fallando) escribía un stub vacío y exit 0. Ahora error. Test `tests/rt/audit_scripts.sh`.
- M8 [B,F] `tests/rt/ubsan_aot.sh`: vpaot o cc fallando en un caso = "SKIP" y exit 0 (el test no probaba nada si todo fallaba); además `run_local.sh` solo miraba `grep ' 0 failed'`. Ahora FAIL, exit real. Mismo test.
- M9 [B,F] `tests/aot/self_malformed/run.sh`: si `gen.py` fallaba no había casos, el bucle no corría y salía 0. Ahora falla. Mismo test.
- M10 [C] `tests/rt/pkcs12_secrets.c`: añadido control positivo (un bloque con ambos secretos liberado debe verse), si no el escaneo pasaba trivialmente si el `--wrap` no se aplicaba.
- M11 [C] `tools/vpconvert/cli.cpp`: un `STOP` dejado por la parada anterior paraba la siguiente ejecución al instante (la reanudación). Se borra al arrancar. Solo `-fsyntax-only` (necesita LLVM).
- M12 [B,F] Workflows (sin tocar `on:` ni `if:`):
  - `aot-tests.yml:12`, `metal-gpu-tests.yml:13`, `vpconvert-visionos.yml:15`: concurrencia a nivel de workflow: un push SIN [build] (todos los jobs saltados) cancelaba (aot, cancel-in-progress) o reemplazaba la ejecución pendiente (metal, vpconvert) de un [build]. El grupo lleva ahora `'build' || run_id`.
  - `demo-visionos.yml`, `vpaot-windows.yml`: publican release sin concurrencia → dos ejecuciones pisan el mismo tag. Añadida (sin cancelar).
  - `cts.yml` job focus y `gles-cts.yml` focus: `--deqp-case='${{ inputs.cases }}'` y `--shard-size "${{ inputs.shard-size }}"` en el script (inyección de shell por input) → vía `env:`.
  - `cts.yml` focus: `$(grep -c ... || echo 0)` imprimía "0\n0" sin fallos. Corregido.
  - `gles-cts.yml`: grupos por `github.sha`/`github.ref` (vpvulkan, smoke, focus) cancelaban jobs de otra ejecución del mismo commit → `run_id` como en cts.yml.
  - `aot-tests.yml:91`: el commit de logs del job x86 sin `--allow-empty` (el de arm64 sí).
  - Test: `tests/ci/check_workflows.py` (PyYAML; local, no está en ningún workflow).
- M13 [B,F] Tests de tiempo `gpu/core/tests/core_tests.cpp` `barrierscale` y `copyhoistscale` (límites absolutos 400/150 ms): ahora ratio t(4n)/t(n), interleaved, mínimo de 5, límite 10 (lineal ≈4, cuadrático ≈16). Comprobado: con el `frame_compiler.cpp` de HEAD (cuadrático) fallan (x17,6 y x20,5); con el actual pasan, también con 4 bucles ocupados en 2 núcleos.

## Segunda tanda (todo lo anotado, ahora arreglado)
- M14 [C,F] `frame_compiler.cpp:245,785,1302`, `device.cpp:548-549`: `width >> mip` con mip >= 32 (UB). Helper `mipShift()` en `resources.h`. Test `gpu_core.audit.extentmip` (UBSAN halt).
- M15 [B,F] `pipeline_cache.cpp` `save()`: escribe `pipelines.keys.tmp` y `rename` (antes `trunc` sobre el fichero real). Test `save` (cambia el inode, sin restos, falla limpio).
- M16 [B] `background.cpp` hooks de pausa con `hookMutex_`: el interbloqueo solo existe con un hook que bloquea o llama a add/remove, lo que el contrato de `background.h:55` prohíbe (el único hook real, `requestArchiveSave`, no bloquea). Descartado como defecto; test de estrés `gpu_core.audit.background` (set/add/remove desde 5 hilos, ningún hook corre tras `removePauseHook`), limpio bajo TSan.
- M17 [B,F] `gpu/ps4/src/formats.cpp:104-111` `blendFactor`: 10 (SRC_ALPHA_SATURATE) → SrcAlphaSaturate, 19/20 → ConstantAlpha/OneMinusConstantAlpha, exactos (antes 19/20 se aproximaban con Constant SIN bajar `exact`, y 10 con One). Solo SRC1_* y reservados siguen aproximados; `formats.h` lo dice. Test `vpps4.blend_factors`.
- M18 [C] `tools/cts/caselists.py`: `keep_sample` duplicada borrada.
- M19 [C,F] `tests/aot/run.py:134`: "N/M passed" contaba `self_loader` como caso.
- M20 [C] `tests/rt/loader_many.c`: directorio `mkdtemp` (TMPDIR) en vez de `/tmp/vp_loader_many.elf`; se borra al terminar.
- M21 [B] `aot-tests.yml`: apple-codesign 0.29.0 descargado a fichero, `sha256sum -c` con dbe85ced...0002 (calculado por mí del asset de la release; no hay checksum publicado con el que contrastarlo), `curl -f`.
- Solo documentado (no son defectos a corregir): `core_tests.cpp` `requireddispatch` (150 ms de compilación) y `submit_order_tests.cpp:141` (sleep 50 ms) usan temporización con falso negativo posible solo con la máquina parada; `apt-get install` sin `update` en jobs ubuntu; `.github/actions/replay-vpvk` sin hallazgos.

## Comprobado limpio
- `tools/sdk/libVPRuntime.tbd` coincide con los `vp_*` de `runtime/vp_host.c` (67 símbolos).
- scripts make_game_pack/translate_game (.sh/.ps1), compare.sh, publish-logs.sh, gles/*.sh, run_lot.py (lógica de reanudación), summarize.py, caselists: sin más defectos.
- pkcs12_secrets (wipe, threads), codesign: pasan; ubsan_aot con 6 casos reales: 0 fallos.
