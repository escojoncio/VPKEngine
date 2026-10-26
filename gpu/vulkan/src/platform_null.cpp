// Linux/test platform: gpu/core's null backend (records calls, executes nothing).
#include <cstdlib>
#include <string>

#include "vpvk/platform.h"

namespace vpvk {
namespace {
class NullPlatform final : public Platform {
 public:
  vpgpu::Backend& backend() override { return backend_; }

 private:
  vpgpu::NullBackend backend_;
};
}  // namespace

std::unique_ptr<Platform> createPlatform() { return std::make_unique<NullPlatform>(); }

// No GPU: the counts the physical device advertises (1/2/4), like an Apple M2.
int platformMaxSampleCount() { return 4; }
std::string platformAttrSummary() { return {}; }
std::string platformDeviceName() { return "Apple GPU (VPKEngine null backend)"; }
bool platformSupportsBC() { return true; }

// No GPU: rate maps are "supported" only when a test asks for it (VPVK_NULL_RATEMAP=1), to exercise the
// foveation shader variants without a device.
bool platformRateMapSupported() {
  const char* e = std::getenv("VPVK_NULL_RATEMAP");
  return e && e[0] == '1';
}

}  // namespace vpvk
