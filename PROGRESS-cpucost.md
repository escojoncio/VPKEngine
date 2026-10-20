# PROGRESS cpucost

## Hecho (CPU por draw)
- draw_bench_heavy (tests/renderer_tests.cpp, programa ps_heavy.lift: 12 texturas + 1 sampler + 6 buffers, estado cambiante por draw): 6.0 us/draw antes, ~2.8 us/draw ahora (min de varias tandas, maquina compartida; ruido +-40%). Instrucciones/draw medidas con callgrind: 31.6k -> 23.0k.
- renderer.cpp: hash de 8 bytes por paso (antes FNV byte a byte); cache de region() dentro de draw/dispatch; descriptores leidos una vez por etapa (Snap) y reutilizados en bind(); sin std::map/vector en shader(); plan de sondeo por programa; cache plana T# -> textura por dwords crudos (textureFast) con generacion; invalidate() solo invalida programas cuyo codigo toca el rango.
- core: PipelineState::key() sin FNV por byte; tabla atomica de claves Ready (request/status/ready sin mutex); accessesOf sin copia por draw.

## Pendiente
Ver abajo (se actualiza).
