// Audit test for gpu/gles/src/vk.cpp: the loader is injected (the layer never links libvulkan), so the resolution rules are
// testable with a fake vkGetInstanceProcAddr. Usage: vpgles_audit_vk_tests
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

#include "vk.h"

using namespace vpgles;

namespace {
int failures = 0;
#define CHECK(c)                                                                 \
    do {                                                                         \
        if (!(c)) {                                                              \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

std::set<std::string> gMissing;  // names the fake loader does not know
int gDeviceLookups = 0;

void dummy() {}

PFN_vkVoidFunction VKAPI_PTR fakeGetDeviceProcAddr(VkDevice, const char* name) {
    ++gDeviceLookups;
    if (gMissing.count(name)) return nullptr;
    return reinterpret_cast<PFN_vkVoidFunction>(&dummy);
}

PFN_vkVoidFunction VKAPI_PTR fakeGetInstanceProcAddr(VkInstance, const char* name) {
    if (gMissing.count(name)) return nullptr;
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(&fakeGetDeviceProcAddr);
    return reinterpret_cast<PFN_vkVoidFunction>(&dummy);
}

VkInstance fakeInstance() { return reinterpret_cast<VkInstance>(uintptr_t(0x10)); }
VkDevice fakeDevice() { return reinterpret_cast<VkDevice>(uintptr_t(0x20)); }
}  // namespace

int main() {
    {  // loadDevice before loadInstance, and after a failed one, fails instead of calling a null vkGetDeviceProcAddr
        Vk vk;
        CHECK(!vk.loadDevice(fakeDevice()));
        CHECK(!vk.loadInstance(nullptr, fakeInstance()));
        CHECK(!vk.loadDevice(fakeDevice()));
        CHECK(gDeviceLookups == 0);
    }
    {  // a loader without vkGetDeviceProcAddr: the instance load reports it and the device load does not crash
        Vk vk;
        gMissing = {"vkGetDeviceProcAddr"};
        CHECK(!vk.loadInstance(fakeGetInstanceProcAddr, fakeInstance()));
        CHECK(!vk.loadDevice(fakeDevice()));
        gMissing.clear();
    }
    {  // everything resolves; the two private hooks are optional
        Vk vk;
        gMissing = {"vpvk_cmd_pass_drop", "vpvk_recipe_stats"};
        CHECK(vk.loadInstance(fakeGetInstanceProcAddr, fakeInstance()));
        CHECK(vk.loadDevice(fakeDevice()));
        CHECK(vk.vkQueueSubmit && vk.vkCmdDraw && vk.vkCmdFillBuffer);
        CHECK(!vk.vpvk_cmd_pass_drop && !vk.vpvk_recipe_stats);
        gMissing.clear();
    }
    {  // one missing device function makes loadDevice report it
        Vk vk;
        gMissing = {"vkCmdFillBuffer"};
        CHECK(vk.loadInstance(fakeGetInstanceProcAddr, fakeInstance()));
        CHECK(!vk.loadDevice(fakeDevice()));
        gMissing.clear();
    }
    {  // with no instance only vkCreateInstance is required
        Vk vk;
        gMissing = {"vkEnumeratePhysicalDevices", "vkCreateDevice"};
        CHECK(vk.loadInstance(fakeGetInstanceProcAddr, nullptr));
        CHECK(vk.vkCreateInstance != nullptr);
        gMissing = {"vkCreateInstance"};
        Vk vk2;
        CHECK(!vk2.loadInstance(fakeGetInstanceProcAddr, nullptr));
        gMissing.clear();
    }
    std::printf("audit_vk: %s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
