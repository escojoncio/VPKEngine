# PROGRESS-state (rama ps4/state)

## Hecho (VPKEngine)
- `gpu/ps4/include/vpps4/state.h`: campos nuevos en DrawState (Raster: polygon offset, clip, polygon mode,
  provoking, restart; ScissorRegs; ColorTarget force_dst_alpha_1/blend_clamp/bypass; DepthTarget write bases,
  read-only, force dirty/valid, bounds; DepthStencil opVal/bounds/colour-on-depth-fail; rop3, degamma,
  alphaToMask, indexSwap; Stage.dualSource/clipSpaceGL).
- `gpu/ps4/src/fixed_function.{h,cpp}` (nuevo): resolveDepthBias, resolveBlend, resolveStencil, combineScissor,
  viewportDepthRange, depthClamp, resolvePolygonMode, indicesNeedPrep/prepareIndices (swap + restart).
- `formats.cpp:blendFactor`: SRC_ALPHA_SATURATE, SRC1_*, CONSTANT_ALPHA exactos.
- `renderer.cpp:draw`: bias/slope/clamp, depth clamp, wireframe, stencil ref trasera + ONES/REPLACE_OP/logica,
  scissor combinado, rango z del viewport (DX / GL), alphaToCoverage, clear de un solo plano o parcial como
  ClearRects dentro del pase, alias de superficie de escritura de Z, draws de copia de profundidad
  (DepthCopy), swap/restart de indices en CPU, dual-source (Stage.dualSource), z -w..w (Stage.clipSpaceGL).
- Lifter (`engine/gcn`): LiftInput.dualSource (mrt1 -> color(0) index(1)), clipSpaceGL (z' = (z+w)/2),
  export de sample mask en MRTZ.
- NullBackend registra pipelines/draws/passes (tests).

## Tests (Linux, ctest -R vpps4)
poly_offset, blend_factors, stencil_ops, scissors, clip_viewport, index_prep, lift_exports,
state_reaches_backend, plane_clears, depth_copy, restart_index (+ los 18 anteriores).

## Pendiente / riesgos
- Sin equivalente en Metal (solo census/nota): depth bounds, ROP3, user clip planes, provoking last,
  export de stencil, color writes on depth fail/pass, multi-viewport (solo viewport 0), pull-model.
