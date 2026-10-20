# PROGRESS cpucost

## Medido (vpps4_tests, NullBackend, Release -O2, maquina de 2 nucleos compartida: ruido +-30 %)
- draw_bench (1 textura): 1.24-2.4 us/draw antes (min-mediana), 1.28-1.38 ahora.
- draw_bench_heavy (nuevo: ps_heavy.lift = 12 texturas + 1 sampler + 6 buffers, estado cambiante por draw: user data, T# de la textura 0 entre 16, depthFunc, culling, blend, viewport): 5.8-6.4 us/draw antes, 2.2-2.4 ahora. Instrucciones por draw (callgrind, prof): 31.6k -> 22.5k. Objetivo < 5 us cumplido con margen en x86; no medido en M2/FEX.
- Variables: VPPS4_BENCH_FRAMES (frames del bench heavy), VPPS4_BENCH_STATS=1 (contadores).

## Hecho
- renderer.cpp: Hash por palabras de 8 bytes; region() cacheada dentro de draw/dispatch (RegionScope); readDescriptor sobre array fijo; shader(): plan de sondeo por programa (ProgramInfo.probe), sin std::map/vector, descriptores leidos una vez por etapa (Snap) y reutilizados por bind() (Translated::snapIndex); texture(): cache plana por dwords crudos del T# (textureFast) con generacion (texGen); invalidate() solo invalida los programas cuyo codigo toca el rango (programGeneration solo si el rango no se puede situar); touch() con ultimo intervalo; pipeline: ready() sin etiqueta ni mutex.
- core: PipelineState::key() por palabras; tabla atomica de claves Ready (PipelineCache::ready/request/status sin mutex en el camino de aciertos; hits en atomicos); accessesOf sin copia por draw; conflicts() con indice por recurso (antes producto cartesiano); PipelineCache::readKeyList y save atomico (tmp + rename).
- Cache persistente (src/translation_cache.{h,cpp}, Renderer::setCacheRoot/waitCache/flushCache): translations.vpsc (nivel 1 bindings por programa, nivel 2 MSL + metadatos por programa+estado de descriptores), cabecera con huella (hash CMake de gcn/ir/msl/renderer + translatorFingerprint de vp4msl): otra huella = fichero reemplazado; cola torn-write reparada; hilo propio de carga/escritura; fusion por trozos en el hilo de render (128 shaders por draw) y despues prefetch de la lista de pipelines (pipelines.keys, guardada cada 30 s si hubo compilaciones). Test translation_cache.
- FetchAddressMemo (fetch.h/.cpp): memo de fetchAddress por codigo de VS verificado por palabras; test en fetch_subroutine.
- Metal (engine/metal/src/metal_backend.mm, NO compilado aqui): RenderCache por encoder (pipeline, depth, cull, winding, bias, clip, fill, stencil ref, blend color, viewport/scissor, offsets de la tabla con setVertexBufferOffset/setFragmentBufferOffset) reseteada al abrir encoder y tras encodeClear; table() con busqueda con pista.
- AstroVisionPro: vpkengine_renderer.mm (MetalOptions::cacheRoot = <CacheDir>/vpkengine/<serial>/metal, renderer->setCacheRoot(.../vpps4), memo de fetch por (direccion, generacion de mapeos, contenido)), astro_core.mm (SHADPS4_CACHE_DIR -> PathType::CacheDir), AstroSettings.swift (pasa GameFolder.caches). Sin compilar.

## Pendiente / riesgos
- Todo el .mm y el Swift sin compilar aqui: la CI de macOS debe pasar. Revision adversarial con Sonnet pendiente (no hay subagentes en esta sesion).
- programGeneration++ en cada submit se mantiene (red de seguridad frente a escrituras de CPU no notificadas); con el hash por palabras re-hashear es barato.
- Memoria: translations.vpsc se carga entera (MSL de todos los shaders vistos); tope de 512 MB (mayor = descartado).
- Metal: el archivo de binarios de pipeline ya existia en el backend (cacheRoot); ahora se activa desde Astro. Sin medir aciertos en visor.
- Posible: la fusion del cache persistente ocurre a trozos en el hilo de render; si un draw llega antes, traduce por si mismo (los duplicados no se vuelven a anexar al fichero? si: se anexa de nuevo, el fichero crece; dedupe pendiente).
