# PROGRESS msaa (rama ps4/msaa)

## Hecho
- gpu/ps4/include/vpps4/msaa.h, src/msaa.cpp: plan de muestras (EQAA), identidad de FMASK, modo de resolve de profundidad, color de fast clear, decision de resolve (InPass/OwnPass/Copy).
- gpu/ps4/include/vpps4/lifetime.h, src/lifetime.cpp: planificador de load/store/resolve por frame (TBDR) con historial entre frames.
- renderer.cpp: targets MS (`ms`) + companero 1x (`id`), resolve en el pase, clears pendientes (fast clear / clear de profundidad) como LoadAction::Clear del siguiente pase, destino writable bajo demanda (sin `storage` por defecto), lectura de muestras (T# 14/15) y FMASK identidad via DescriptorInfo.
- engine/gcn lift: image_load con indice de muestra en MS; lectura de FMASK = constante sin recurso.
- engine/core: Frame::mutableCommands().

## Pendiente
Ver commits siguientes.
