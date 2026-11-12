// VPKTAA nativo en Metal (taa_native_metal.mm). Solo visionOS.
#pragma once

#include <cstdint>

#include "vpvk/platform.h"

namespace vpvk {

// Bytes of the history / resolved-depth textures TaaNativeMetal holds now (any thread, 0 before the first use).
uint64_t taaNativeBytes();

class TaaNativeMetal {
 public:
  explicit TaaNativeMetal(void* mtlDevice);
  ~TaaNativeMetal();
  TaaNativeMetal(const TaaNativeMetal&) = delete;
  TaaNativeMetal& operator=(const TaaNativeMetal&) = delete;
  bool encode(const Platform::TaaNativeArgs&, void* commandBuffer);
  void* resolveDepth(void* commandBuffer, void* depthMS, bool reversedZ, int mode, int key);
  // Gives back the history, resolved depth and cached views that nothing used for `idleMs` (0 = all): the TAA switched off, or a
  // memory trim. Same thread rules as encode (the caller holds the submit mutex). Returns the bytes released.
  uint64_t release(uint32_t idleMs);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace vpvk
