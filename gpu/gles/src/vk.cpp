#include "vk.h"

#include <string_view>

namespace vpgles {

bool Vk::loadInstance(PFN_vkGetInstanceProcAddr gipa, VkInstance instance) {
    vkGetInstanceProcAddr = gipa;
    if (!gipa) return false;
    bool ok = true;
#define VPGLES_LOAD(f)                                                          \
    f = reinterpret_cast<PFN_##f>(gipa(instance, #f));                          \
    if (!f && (instance || std::string_view(#f) == "vkCreateInstance")) ok = false;
    VPGLES_VK_INSTANCE(VPGLES_LOAD)
#undef VPGLES_LOAD
    return ok;
}

bool Vk::loadDevice(VkDevice device) {
    bool ok = true;
#define VPGLES_LOAD(f)                                                 \
    f = reinterpret_cast<PFN_##f>(vkGetDeviceProcAddr(device, #f));    \
    if (!f) f = reinterpret_cast<PFN_##f>(vkGetInstanceProcAddr(nullptr, #f)); \
    if (!f) ok = false;
    VPGLES_VK_DEVICE(VPGLES_LOAD)
#undef VPGLES_LOAD
    return ok;
}

}  // namespace vpgles
