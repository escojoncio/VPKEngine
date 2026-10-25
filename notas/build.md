# Cómo compilar y probar

```
cmake -S gpu -B build && cmake --build build -j2 && (cd build && ctest -j2 --output-on-failure)   # Linux: 232/232, backend nulo
ctest -R gpu_vulkan          # también gpu_core, gpu_gles, gpu_spirv, gpu_metal2, vpps4
CXX=clang++ cmake -S gpu -B build-clang ...      # comprobar también con clang: el CI de Apple falla con avisos que gcc no da
cmake -S gpu -B build-tsan -DCMAKE_CXX_FLAGS=-fsanitize=thread ...   # TSan de gpu_core / gpu_vulkan (directorio aparte, no se commitea)
python3 tests/aot/run.py     # CPU AOT (CI aot-tests.yml)
```
- Los `.mm`/`.metal` NO compilan en Linux: solo el CI (Mac). Regenerar goldens MSL con `VPMSL_UPDATE=1`.
- Tras cambiar traducción: `gpu_vulkan.stage_keys` dice si hay que subir `kLowerVersion`/`kEmitVersion` y actualizar `kCorpusDigest` (`notas/arquitectura.md`).
- Parámetros de entorno: tabla de `notas/arquitectura.md`; la app los pasa antes de cargar el driver. Logs del visor: `gpu.*`, `vk.*`, `gles.warn`, `[mem]`, `[mtl]`.
- **CI de VisionQuestPro** (`.github/workflows/visionos.yml`, push a `main`): clona VPKEngine de `main` (`fetch-depth: 1`, sparse-checkout cone `runtime` y `gpu`), construye `VPVulkan.framework` (`scripts/build-vpvulkan.sh vpkengine build`, caché por hash de `vpkengine/gpu/**`), MoltenVK 1.4.2, y pasa el Gate: `tools/ci/syntax-xros.sh` (`-fsyntax-only` visionOS) + `tools/ci/host-tests.sh vpkengine build/host-tests`, que en paralelo ejecuta `gpu_host` (`cmake -S gpu -B …/gpu -DCMAKE_BUILD_TYPE=Release`, build `-k`, `ctest`), `metal2_xros` (`cmake -S gpu -G Ninja -DCMAKE_SYSTEM_NAME=visionOS -DCMAKE_OSX_SYSROOT=xros -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 -DBUILD_TESTING=ON`, targets `vpgpu_metal2 vpgpu_metal2_tests`, que incluyen el test `manyenc`) y `klepton_tests` (ASan/UBSan). Fallo de cualquiera = sin IPA. Todo cambio debe estar en `main` de VPKEngine antes de lanzar la build de VisionQuestPro.
- Test `gpu_vulkan.async_pipelines` (CI macOS): se colgaba por el bloque nuevo de f93d027 que creaba un 2º dispositivo (`Ctx gl`, cliente interno) con el 1º vivo; pasa en Linux (232/232) pero cuelga con Metal. `persistWaitMs` ahora recibe `bool internalClient` (pipelines.cpp/vpvk.h) y el test lo comprueba sin 2º dispositivo. Sin verificar en el Mac de CI.
