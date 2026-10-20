# PROGRESS submit (rama ps4/submit)

## Hecho
- core: `Backend::waitFramesBelow/waitAll/setFrameDoneNotify` (fences parciales), `LabelQueue` (escrituras de etiquetas ordenadas por la GPU), `SubmitPolicy` (cuando se envia lo grabado), `Device::waitFrames/afterFrame/pumpLabels`, hilo de envio opcional (`startSubmitThread`, `drain`, `withBackend`), destroy ordenado tras los frames en cola.
- metal: `completedFrames()` (faltaba: la ruta Metal tomaba todo por terminado), `waitFramesBelow`, `waitAll`, notificacion al terminar un frame.
- vpps4: `Renderer::waitTarget/waitedBelow/afterGpu/labelPending/hasRecordedWork`, draws/dispatch con pipeline compilando se graban y el frame se retiene (Device hold) en vez de descartarse (salvo VP_HOLD_MS=0). `Scheduler` (scheduler.h/.cpp): un Frame por imagen, esperas parciales, etiquetas, prepareWait.
- Tests Linux: vp4gpu_core.submit, vpps4.{one_frame_per_image,partial_waits,label_ordering,finish_all,hold_pipeline}.

## Pendiente / riesgos
- El hilo de envio es opt-in (no verificado en Metal real: el backend asume un hilo para crear+enviar).

## Integracion AstroVisionPro (rama ps4/submit)
- vpkengine_renderer.mm: Engine usa `vpps4::Scheduler` (un frame por imagen; Flush ya no envia), hilo watchdog (politica de envio), `syncCpu` parcial para DMA/GDS/WRITE_DATA, `ScheduleLabelWrite`, `PrepareLabelWait`, `SyncGds`; framesInFlight 6 -> 2; el presentador envia en `present()`; hilo de envio opt-in con VPK_SUBMIT_THREAD=1.
- liverpool.cpp / pm4_cmds.h / rasterizer_interface.h: EOP, EOS, RELEASE_MEM escriben etiqueta + IRQ cuando la GPU termina el trabajo previo; WRITE_DATA ordenado; WAIT_REG_MEM llama PrepareLabelWait antes de sondear (heuristicas de reproyeccion intactas).
- Swift: GameRenderer mantiene el refresco nominal (90) aunque el sistema baje la pantalla; fps por defecto 45.
- El contador "VPKENGINE_GPU submits flip=.. dma=.." (cada 2 s) da el motivo de cada envio: medir con el.
- No compilado aqui (.mm/Swift): revisar en la CI de macOS.
