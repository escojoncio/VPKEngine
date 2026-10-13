// Linux/test platform: gpu/core's null backend (records calls, executes nothing).
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

}  // namespace vpvk
