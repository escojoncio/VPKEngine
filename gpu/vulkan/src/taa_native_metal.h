// VPKTAA nativo en Metal (taa_native_metal.mm). Solo visionOS.
#pragma once

#include "vpvk/platform.h"

namespace vpvk {

class TaaNativeMetal {
 public:
  explicit TaaNativeMetal(void* mtlDevice);
  ~TaaNativeMetal();
  TaaNativeMetal(const TaaNativeMetal&) = delete;
  TaaNativeMetal& operator=(const TaaNativeMetal&) = delete;
  bool encode(const Platform::TaaNativeArgs&, void* commandBuffer);
  void* resolveDepth(void* commandBuffer, void* depthMS, bool reversedZ, int mode, int key);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace vpvk
