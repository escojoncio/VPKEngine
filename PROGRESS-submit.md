# PROGRESS submit (rama ps4/submit)

## Hecho
- core: `Backend::waitFramesBelow/waitAll/setFrameDoneNotify` (fences parciales), `LabelQueue` (escrituras de etiquetas ordenadas por la GPU), `SubmitPolicy` (cuando se envia lo grabado), `Device::waitFrames/afterFrame/pumpLabels`, hilo de envio opcional (`startSubmitThread`, `drain`, `withBackend`), destroy ordenado tras los frames en cola.
- metal: `completedFrames()` (faltaba: la ruta Metal tomaba todo por terminado), `waitFramesBelow`, `waitAll`, notificacion al terminar un frame.
- vpps4: `Renderer::waitTarget/waitedBelow/afterGpu/labelPending/hasRecordedWork`, draws/dispatch con pipeline compilando se graban y el frame se retiene (Device hold) en vez de descartarse (salvo VP_HOLD_MS=0). `Scheduler` (scheduler.h/.cpp): un Frame por imagen, esperas parciales, etiquetas, prepareWait.
- Tests Linux: vp4gpu_core.submit, vpps4.{one_frame_per_image,partial_waits,label_ordering,finish_all,hold_pipeline}.

## Pendiente / riesgos
- El hilo de envio es opt-in (no verificado en Metal real: el backend asume un hilo para crear+enviar).
