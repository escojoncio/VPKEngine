// vpreplay <file.vpcap> [options]: replays a VPCAP capture (include/vpvk/vpcap.h) against any Vulkan driver loaded at run time.
//
//   --icd vpvk|system   vpvk: libVPVulkan (--lib PATH, else $VPREPLAY_VPVK, else next to this executable); system: libvulkan.so.1
//                       (the Vulkan loader: --icd-file ICD.json selects the driver, e.g. lavapipe's lvp_icd.*.json)
//   --gpu N             physical device (default 0)
//   --dump DIR          write the render targets as PNG (+ index.json) into DIR
//   --dump-at pass|draw after every render pass (default) or after every draw (the pass is ended, copied, resumed with LOAD)
//   --recapture FILE    vpvk only: capture the replay with VPVulkan's own capture (a test of the replay itself)
//   --raw               write every dump as .raw + .json (bit-exact), not as PNG
//   --strict            exit 1 when any call failed or was skipped for a missing object, a command buffer or an image upload was dropped
//   --allow-truncated   accept a capture cut short (no END block, or a last block cut off); otherwise it is an error (exit 1)
//
// What is rebuilt: the objects of STATE in creation order, then the FRAME calls in file order, on one thread. Handles are
// the stable ids of the file, mapped to the driver's own. Differences from the original run, all counted in the summary:
//  - memory types are chosen by property flags, sizes by max(captured, what the bound resources need on this driver);
//  - image usage gets TRANSFER_SRC / TRANSFER_DST when the driver allows it (dumps, uploads); images are left in GENERAL
//    after their contents are uploaded;
//  - every submit runs to completion before the next; semaphore waits / signals, fence waits, events and the host mapping
//    calls are dropped (the data comes from vpcapMemoryData / vpcapImageData); queue families collapse into one.
#define VK_NO_PROTOTYPES
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>  // _NSGetExecutablePath (no /proc on macOS)
#endif

#include <algorithm>
#include <array>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include "vpimage.h"
#include "vpjson.h"
#include "vpvk/vpcap_calls.h"

using namespace vpcap;

namespace {

template <class T>
T* mut(const T* p) {
  return const_cast<T*>(p);
}
uint64_t alignUp(uint64_t x, uint64_t a) { return a > 1 ? (x + a - 1) / a * a : x; }
template <class H>
uint64_t idOfFake(H h) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h));
}

// Block size in texels and bytes of a compressed format; false = not a compressed format (or one the driver does not hold).
bool blockOf(VkFormat f, uint32_t& bw, uint32_t& bh, uint32_t& bytes) {
  const int v = int(f);
  bw = bh = 4, bytes = 16;
  if (v >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && v <= VK_FORMAT_BC7_SRGB_BLOCK) {
    if (v <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK || v == VK_FORMAT_BC4_UNORM_BLOCK || v == VK_FORMAT_BC4_SNORM_BLOCK) bytes = 8;
    return true;
  }
  if (v >= VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK && v <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK) {
    if (v <= VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK || v == VK_FORMAT_EAC_R11_UNORM_BLOCK || v == VK_FORMAT_EAC_R11_SNORM_BLOCK) bytes = 8;
    return true;
  }
  if (v >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && v <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) {  // UNORM / SRGB pairs, in this order
    static const uint8_t dims[14][2] = {{4, 4}, {5, 4}, {5, 5}, {6, 5}, {6, 6}, {8, 5}, {8, 6}, {8, 8}, {10, 5}, {10, 6}, {10, 8}, {10, 10}, {12, 10}, {12, 12}};
    const int k = (v - VK_FORMAT_ASTC_4x4_UNORM_BLOCK) / 2;
    bw = dims[k][0], bh = dims[k][1];
    return true;
  }
  return false;
}
// Bytes one subresource of `f` takes in the file (vpcap.h ImageData: tightly packed rows of blocks, the backend's layout);
// width / height are those of the plane for a plane aspect. 0 = unknown, or a layout the file does not hold in the Vulkan one.
uint64_t expectedImageBytes(VkFormat f, uint32_t aspect, uint32_t w, uint32_t h, uint32_t d) {
  uint32_t bw = 1, bh = 1, bytes = 0;
  if (aspect & (VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT | VK_IMAGE_ASPECT_PLANE_2_BIT)) {
    const bool two = f == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM || f == VK_FORMAT_G8_B8R8_2PLANE_422_UNORM;
    const bool three = f == VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM || f == VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM || f == VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM;
    if (!two && !three) return 0;
    bytes = (aspect == VK_IMAGE_ASPECT_PLANE_1_BIT && two) ? 2 : 1;
  } else if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
    bytes = 1;
  } else if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
    if (f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_X8_D24_UNORM_PACK32) return 0;  // held as float32 on Metal: not the Vulkan layout
    bytes = vpimg::depthCopyBytes(f);
  } else if (!blockOf(f, bw, bh, bytes)) {
    bw = bh = 1;
    switch (f) {  // uncompressed formats vpimg::texelBytes does not list
      case VK_FORMAT_A1R5G5B5_UNORM_PACK16: case VK_FORMAT_B4G4R4A4_UNORM_PACK16: bytes = 2; break;
      case VK_FORMAT_A8B8G8R8_SNORM_PACK32: case VK_FORMAT_A8B8G8R8_UINT_PACK32: case VK_FORMAT_A8B8G8R8_SINT_PACK32: bytes = 4; break;
      case VK_FORMAT_R32G32B32A32_SINT: bytes = 16; break;
      default: bytes = vpimg::texelBytes(f);
    }
  }
  if (!bytes) return 0;
  return uint64_t((w + bw - 1) / bw) * ((h + bh - 1) / bh) * d * bytes;
}

struct Options {
  std::string file, icd = "vpvk", lib, icdFile, dumpDir, dumpAt = "pass", recapture;
  int gpu = 0;
  bool strict = false, raw = false, allowTruncated = false;
};

// VPVulkan's memory types (physical_device.cpp makeMem): the only producer of captures, and the file does not carry the table.
constexpr VkMemoryPropertyFlags kSrcTypes[4] = {
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT};

// ---- records of the objects the dumps need ----
struct AttDesc {
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkAttachmentStoreOp store = VK_ATTACHMENT_STORE_OP_STORE, sstore = VK_ATTACHMENT_STORE_OP_STORE;
  VkImageLayout finalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};
struct RpRec {
  uint64_t id = 0;
  bool v2 = false;
  VkRenderPassCreateInfo ci1{};
  VkRenderPassCreateInfo2 ci2{};
  std::vector<AttDesc> atts;
  VkRenderPass variant[2] = {};  // [0] stores everything, [1] also loads (the pass resumed after a cut)
};
struct FbRec {
  uint64_t id = 0;
  std::vector<VkImageView> views;
};
struct ImgRec {
  uint64_t id = 0;
  VkImageCreateInfo ci{};  // pointers are not valid
  uint32_t touchedAspects = 0;
};
struct ViewRec {
  uint64_t id = 0, imageId = 0;
  VkImage image{};
  VkImageSubresourceRange range{};
};
struct MemRec {
  VkDeviceMemory h{};
  uint64_t size = 0;
  uint8_t* map = nullptr;
  bool coherent = false;
};
struct Req {
  uint64_t size = 0, align = 1;
  uint32_t bits = 0xFFFFFFFFu;
  bool ok = false;
};
struct Need {
  uint64_t size = 0;
  uint32_t bits = 0xFFFFFFFFu;
  bool hostWrite = false;
};

// One attachment copy recorded into a command buffer (read back after the submit that runs it).
struct DumpSlot {
  VkBuffer buf{};
  VkDeviceMemory mem{};
  uint8_t* map = nullptr;
  uint64_t bytes = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  bool depth = false;
  uint32_t w = 0, h = 0, localPass = 0, draw = 0;
  std::string att, call;
  uint64_t callIndex = 0, pipeline = 0, rp = 0, fb = 0, image = 0, view = 0, ibuf = 0;
  std::vector<uint64_t> pipelines, sets, vbufs;
};
struct AttCopy {
  VkImage image{};
  uint64_t imageId = 0, viewId = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  uint32_t mip = 0, layerBase = 0, layers = 1, w = 0, h = 0;
  VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
  std::string name;
};
struct PassCtx {
  bool active = false, dynamic = false, canCut = false;
  uint32_t draws = 0, localIndex = 0;
  uint64_t rpId = 0, fbId = 0;
  RpRec* rp = nullptr;
  FbRec* fb = nullptr;
  VkRect2D area{};
  VkRenderingInfo ri{};
  std::vector<VkRenderingAttachmentInfo> col;
  VkRenderingAttachmentInfo dep{}, stn{};
  std::set<uint64_t> pipelines;
};
struct CbRec {
  uint64_t id = 0;
  std::vector<DumpSlot> dumps;
  uint32_t passes = 0;
  uint64_t pipe = 0, ibuf = 0;
  std::array<uint64_t, 8> sets{};
  std::array<uint64_t, 16> vbufs{};
  PassCtx pass;
  bool tainted = false;  // a vkCmd of it was left out (missing object, no driver function, corrupt record): it is not submitted
};

struct R;
template <uint16_t ID>
void run(R& r, const Call& c);

struct R {
  Options opt;
  // ---- driver ----
  void* lib = nullptr;
  PFN_vkGetInstanceProcAddr gipa = nullptr;
  PFN_vkGetDeviceProcAddr gdpa = nullptr;
  PFN_vkVoidFunction tbl[0x300] = {};
  PFN_vkCreateInstance fCreateInstance = nullptr;
  PFN_vkEnumerateInstanceExtensionProperties fInstExt = nullptr;
  PFN_vkEnumeratePhysicalDevices fEnumPd = nullptr;
  PFN_vkGetPhysicalDeviceProperties fPdProps = nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties fPdMem = nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyProperties fPdQf = nullptr;
  PFN_vkGetPhysicalDeviceFeatures2 fPdFeat2 = nullptr;
  PFN_vkGetPhysicalDeviceImageFormatProperties fPdImgFmt = nullptr;
  PFN_vkEnumerateDeviceExtensionProperties fDevExt = nullptr;
  PFN_vkCreateDevice fCreateDevice = nullptr;
  PFN_vkGetBufferMemoryRequirements fBufReq = nullptr;
  PFN_vkGetImageMemoryRequirements fImgReq = nullptr;
  int (*vpvkTrack)(int) = nullptr;
  int (*vpvkRequest)(uint32_t, const char*) = nullptr;

  VkInstance instance{};
  VkPhysicalDevice pd{};
  VkDevice device{};
  VkQueue queue0{};
  uint32_t family = 0, queueCount = 1;
  VkPhysicalDeviceMemoryProperties mp{};
  uint32_t apiVersion = 0;
  bool bdaEnabled = false;

  // ---- objects ----
  std::unordered_map<uint64_t, uintptr_t> h;  // id -> the driver's handle
  bool missing = false;                       // a nonzero id the call uses is unknown
  std::unordered_map<uint64_t, MemRec> mems;
  std::unordered_map<uint64_t, Need> need;
  std::unordered_map<uint64_t, Req> reqs;  // buffer / image id -> requirements on this driver
  std::unordered_map<VkRenderPass, RpRec> rps;
  std::unordered_map<VkFramebuffer, FbRec> fbs;
  std::unordered_map<VkImage, ImgRec> imgs;
  std::unordered_map<VkImageView, ViewRec> views;
  std::unordered_map<VkCommandBuffer, CbRec> cbs;
  std::vector<std::shared_ptr<void>> keep;  // decoded create infos the records above point into
  std::map<std::string, VkResult> fmtCache;
  VkCommandPool internalPool{};
  bool prescanned = false, instanceMade = false, recaptureArmed = false;

  // ---- progress / stats ----
  uint32_t frame = 0, submitInFrame = 0, curFrame = UINT32_MAX;
  uint64_t calls = 0, errors = 0, unknownIds = 0, skippedMissing = 0, unsupported = 0;
  uint64_t memWrites = 0, memBytes = 0, memLost = 0, imgUploads = 0, imgUploadFail = 0, misaligned = 0, memGrown = 0, memRemapped = 0;
  uint64_t imgUploadSkipped = 0, cbsSkipped = 0;  // uploads whose size / layout does not fit the image; command buffers not submitted
  std::map<std::string, uint64_t> skipped, failed;
  std::set<std::string> warned;

  // ---- dumps ----
  std::vector<std::string> entries;
  uint64_t dumpFiles = 0, dumpRaw = 0;
  std::map<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>, std::pair<std::vector<uint8_t>, uint64_t>> pendingImage;

  // ---- logging ----
  void note(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "vpreplay: ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
  }
  void warnOnce(const std::string& key, const std::string& text) {
    if (warned.insert(key).second) note("%s", text.c_str());
  }

  // ---- loading ----
  bool load() {
    std::string path = opt.lib;
    if (opt.icd == "system") {
      if (!opt.icdFile.empty()) {
        setenv("VK_DRIVER_FILES", opt.icdFile.c_str(), 1);
        setenv("VK_ICD_FILENAMES", opt.icdFile.c_str(), 1);
      }
      if (path.empty()) path = "libvulkan.so.1";
    } else {
      // A capture holds what the game asked of the driver. What the driver adds on its own (VPKTAA, RCAS, foveation, forced 1x /
      // MSAA / anisotropy, set by the host through the environment) is not in it: off, whatever the environment says (the
      // environment wins over every config call of the driver).
      for (const char* k : {"VPVK_TAA_NATIVE", "VPVK_RCAS", "VPVK_TAA_JITTER", "VPVK_TAA_UI_MASK", "VPVK_FOVEATION", "VPVK_FORCE_1X",
                            "VPVK_MSAA", "VPVK_FORCE_ANISO"})
        setenv(k, "0", 1);
    }
    if (opt.icd != "system" && path.empty()) {
      if (const char* e = std::getenv("VPREPLAY_VPVK")) path = e;
      else {
        char self[PATH_MAX] = {};
#ifdef __APPLE__
        uint32_t selfSize = sizeof self;
        const ssize_t n = _NSGetExecutablePath(self, &selfSize) == 0 ? ssize_t(std::strlen(self)) : -1;
        const char* name = "/libVPVulkan.dylib";
#else
        const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
        const char* name = "/libVPVulkan.so";
#endif
        path = (n > 0 ? std::filesystem::path(self).parent_path().string() : std::string(".")) + name;
      }
    }
    lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
      note("cannot load %s: %s", path.c_str(), dlerror());
      return false;
    }
    gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
    if (!gipa) {
      note("%s has no vkGetInstanceProcAddr", path.c_str());
      return false;
    }
    fCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    fInstExt = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(gipa(nullptr, "vkEnumerateInstanceExtensionProperties"));
    vpvkTrack = reinterpret_cast<int (*)(int)>(dlsym(lib, "vpvk_capture_track"));
    vpvkRequest = reinterpret_cast<int (*)(uint32_t, const char*)>(dlsym(lib, "vpvk_capture_request"));
    if (!opt.recapture.empty()) {
      if (!vpvkTrack || !vpvkRequest) {
        note("--recapture needs libVPVulkan (vpvk_capture_* not found in %s)", path.c_str());
        return false;
      }
      vpvkTrack(1);
    }
    return fCreateInstance != nullptr;
  }
  template <class F>
  F inst(const char* name) {
    return reinterpret_cast<F>(gipa(instance, name));
  }
  PFN_vkVoidFunction devFn(const char* name) {
    PFN_vkVoidFunction f = gdpa ? gdpa(device, name) : nullptr;
    if (!f) f = gipa(instance, name);
    if (!f) f = gdpa ? gdpa(device, (std::string(name) + "KHR").c_str()) : nullptr;
    return f;
  }
  void loadInstanceFns() {
    gdpa = inst<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
    fEnumPd = inst<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
    fPdProps = inst<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
    fPdMem = inst<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties");
    fPdQf = inst<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
    fPdFeat2 = inst<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2");
    if (!fPdFeat2) fPdFeat2 = inst<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2KHR");
    fPdImgFmt = inst<PFN_vkGetPhysicalDeviceImageFormatProperties>("vkGetPhysicalDeviceImageFormatProperties");
    fDevExt = inst<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
    fCreateDevice = inst<PFN_vkCreateDevice>("vkCreateDevice");
  }
  void loadDeviceFns() {
#define VPR_X(name, num) tbl[num] = devFn(#name);
    VPCAP_VKCALLS(VPR_X)
#undef VPR_X
    fBufReq = reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(devFn("vkGetBufferMemoryRequirements"));
    fImgReq = reinterpret_cast<PFN_vkGetImageMemoryRequirements>(devFn("vkGetImageMemoryRequirements"));
  }
  template <uint16_t ID>
  typename CallDef<ID>::Fn fn() const {
    return reinterpret_cast<typename CallDef<ID>::Fn>(tbl[ID]);
  }

  // ---- handles ----
  template <class H>
  H real(H x) {
    const uint64_t i = idOfFake(x);
    if (!i) return nullptr;
    auto it = h.find(i);
    if (it == h.end()) {
      missing = true;
      ++unknownIds;
      return nullptr;
    }
    return reinterpret_cast<H>(it->second);
  }
  template <class H>
  H realOpt(H x) {  // a handle that may legitimately be unknown: no complaint
    const bool m = missing;
    H r = real(x);
    if (missing && !m) --unknownIds;
    missing = m;
    return r;
  }
  template <class H>
  void reg(uint64_t id, H handle) {
    if (id && handle) h[id] = reinterpret_cast<uintptr_t>(handle);
  }
  template <class T>
  void mapOne(T& x) {
    if constexpr (kIsHandle<T>) x = real(x);
  }
  template <class... T>
  void mapTop(std::tuple<T...>& t) {
    std::apply([&](auto&... x) { (mapOne(x), ...); }, t);
  }
  template <class H>
  void realArr(const H* p, uint32_t n) {
    if (p)
      for (uint32_t i = 0; i < n; ++i) mut(p)[i] = real(p[i]);
  }

  // ---- device selection / memory ----
  uint32_t srcFlags(uint32_t type) const { return type < 4 ? kSrcTypes[type] : 0; }
  uint32_t pickType(VkMemoryPropertyFlags src, uint32_t bits, bool hostWrite, bool* satisfied) {
    VkMemoryPropertyFlags want = src & ~VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
    if (hostWrite) want |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const bool needHost = (want & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    int best = -1, bestScore = INT_MAX;
    for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
      if (!((bits >> t) & 1)) continue;
      const VkMemoryPropertyFlags f = mp.memoryTypes[t].propertyFlags;
      if ((f & ~0x1Fu) || (needHost && !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) continue;
      const int score = __builtin_popcount(f ^ want);
      if (score < bestScore) best = int(t), bestScore = score;
    }
    if (satisfied) *satisfied = best >= 0;
    if (best >= 0) return uint32_t(best);
    for (uint32_t t = 0; t < mp.memoryTypeCount; ++t)
      if ((bits >> t) & 1) return t;
    return 0;
  }

  // ---- calls into the driver that the replay itself makes ----
  uint32_t findFamily() {
    uint32_t n = 0;
    fPdQf(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> q(n);
    fPdQf(pd, &n, q.data());
    for (uint32_t i = 0; i < n; ++i)
      if ((q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) return queueCount = q[i].queueCount, i;
    for (uint32_t i = 0; i < n; ++i)
      if (q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) return queueCount = q[i].queueCount, i;
    return queueCount = n ? q[0].queueCount : 1, 0;
  }

  bool ensureInstance() {
    if (instanceMade) return true;
    // The capture started after the instance existed (META instance_in_state=0): a plain instance for the replay.
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vpreplay", app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    return makeInstance(ci);
  }
  bool makeInstance(VkInstanceCreateInfo& ci) {
    if (const VkResult res = fCreateInstance(&ci, nullptr, &instance); res != VK_SUCCESS) {
      note("vkCreateInstance failed (%d)%s", int(res), res == VK_ERROR_INCOMPATIBLE_DRIVER ? ": no Vulkan driver found (--icd-file?)" : "");
      return false;
    }
    instanceMade = true;
    loadInstanceFns();
    uint32_t n = 0;
    fEnumPd(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    fEnumPd(instance, &n, pds.data());
    if (!n || opt.gpu < 0 || uint32_t(opt.gpu) >= n) {
      note("physical device %d not found (%u available)", opt.gpu, n);
      return false;
    }
    pd = pds[size_t(opt.gpu)];
    VkPhysicalDeviceProperties props{};
    fPdProps(pd, &props);
    apiVersion = props.apiVersion;
    fPdMem(pd, &mp);
    family = findFamily();
    note("driver: %s (api %u.%u.%u), queue family %u, %u memory types", props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
         VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion), family, mp.memoryTypeCount);
    return true;
  }

  // The usage a dump / upload needs on every image, when the driver allows it for the format.
  void augmentImage(VkImageCreateInfo& ci) {
    if (ci.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) return;
    const VkImageUsageFlags extra[2] = {VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    for (const VkImageUsageFlags e : extra) {
      const VkImageUsageFlags u = ci.usage | e;
      char key[96];
      std::snprintf(key, sizeof key, "%d/%d/%d/%x/%x", int(ci.format), int(ci.imageType), int(ci.tiling), u, ci.flags);
      auto it = fmtCache.find(key);
      if (it == fmtCache.end()) {
        VkImageFormatProperties p{};
        it = fmtCache.emplace(key, fPdImgFmt ? fPdImgFmt(pd, ci.format, ci.imageType, ci.tiling, u, ci.flags, &p) : VK_ERROR_FORMAT_NOT_SUPPORTED).first;
      }
      if (it->second == VK_SUCCESS) {
        ci.usage = u;
        return;
      }
    }
  }
  static void exclusive(VkSharingMode& mode, uint32_t& n, const uint32_t*& p) {
    if (mode == VK_SHARING_MODE_CONCURRENT) mode = VK_SHARING_MODE_EXCLUSIVE, n = 0, p = nullptr;  // one family now
  }
  void patchBuffer(VkBufferCreateInfo& ci) { exclusive(ci.sharingMode, ci.queueFamilyIndexCount, ci.pQueueFamilyIndices); }
  void patchImage(VkImageCreateInfo& ci) {
    exclusive(ci.sharingMode, ci.queueFamilyIndexCount, ci.pQueueFamilyIndices);
    augmentImage(ci);
  }

  // ---- requirements of every resource of the file, so a memory is allocated once with a size that fits this driver ----
  void prescan() {
    if (prescanned) return;
    prescanned = true;
    Reader rd;
    if (!rd.open(opt.file)) return;
    Call c;
    auto addBind = [&](uint64_t res, uint64_t mem, uint64_t off) {
      auto q = reqs.find(res);
      if (q == reqs.end() || !q->second.ok) return;
      Need& n = need[mem];
      n.size = std::max(n.size, alignUp(off, q->second.align) + q->second.size);
      n.bits &= q->second.bits;
    };
    while (rd.next(c)) {
      if (c.id == id::vpcapMemoryData) {
        MemoryData m;
        if (decodeMemoryData(c, m)) need[m.memory].hostWrite = true;
      } else if (c.id == id::vkCreateBuffer) {
        auto d = decode<id::vkCreateBuffer>(c);
        if (!d->ok || !std::get<1>(d->args)) continue;
        VkBufferCreateInfo ci = *std::get<1>(d->args);
        patchBuffer(ci);
        VkBuffer b{};
        Req q;
        if (fn<id::vkCreateBuffer>()(device, &ci, nullptr, &b) == VK_SUCCESS) {
          VkMemoryRequirements mr{};
          fBufReq(device, b, &mr);
          q = {mr.size, mr.alignment, mr.memoryTypeBits, true};
          fn<id::vkDestroyBuffer>()(device, b, nullptr);
        }
        reqs[idOfFake(*std::get<3>(d->args))] = q;
      } else if (c.id == id::vkCreateImage) {
        auto d = decode<id::vkCreateImage>(c);
        if (!d->ok || !std::get<1>(d->args)) continue;
        VkImageCreateInfo ci = *std::get<1>(d->args);
        patchImage(ci);
        VkImage im{};
        Req q;
        if (fn<id::vkCreateImage>()(device, &ci, nullptr, &im) == VK_SUCCESS) {
          VkMemoryRequirements mr{};
          fImgReq(device, im, &mr);
          q = {mr.size, mr.alignment, mr.memoryTypeBits, true};
          fn<id::vkDestroyImage>()(device, im, nullptr);
        }
        reqs[idOfFake(*std::get<3>(d->args))] = q;
      } else if (c.id == id::vkBindBufferMemory || c.id == id::vkBindImageMemory) {
        uint64_t v[3];  // payload: device id, then resource id, memory id, offset (VPCAP_ALL)
        if (c.size >= 32) {
          std::memcpy(v, c.data + 8, 24);
          addBind(v[0], v[1], v[2]);
        }
      }
    }
  }

  // ---- upload of vpcapMemoryData / vpcapImageData ----
  void onMemoryData(const Call& c) {
    MemoryData m;
    if (!decodeMemoryData(c, m)) return void(++errors);
    auto it = mems.find(m.memory);
    if (it == mems.end() || !it->second.map || m.offset >= it->second.size) {
      ++memLost;
      return;
    }
    const uint64_t n = std::min<uint64_t>(m.size, it->second.size - m.offset);
    std::memcpy(it->second.map + m.offset, m.data, n);
    if (!it->second.coherent) {
      VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      r.memory = it->second.h, r.offset = 0, r.size = VK_WHOLE_SIZE;
      fn<id::vkFlushMappedMemoryRanges>()(device, 1, &r);
    }
    ++memWrites, memBytes += n;
  }

  template <class F>
  bool immediate(F&& record) {
    if (!queue0) return false;
    auto create = fn<id::vkCreateCommandPool>();
    if (!internalPool) {
      VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      pi.queueFamilyIndex = family;
      if (create(device, &pi, nullptr, &internalPool) != VK_SUCCESS) return false;
    }
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = internalPool, ai.commandBufferCount = 1, ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VkCommandBuffer cb{};
    if (fn<id::vkAllocateCommandBuffers>()(device, &ai, &cb) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    fn<id::vkBeginCommandBuffer>()(cb, &bi);
    record(cb);
    fn<id::vkEndCommandBuffer>()(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &cb;
    const bool ok = fn<id::vkQueueSubmit>()(queue0, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS;
    fn<id::vkQueueWaitIdle>()(queue0);
    fn<id::vkFreeCommandBuffers>()(device, internalPool, 1, &cb);
    return ok;
  }
  // A host-visible buffer (TRANSFER_SRC | TRANSFER_DST) with its memory mapped.
  bool makeStaging(uint64_t bytes, VkBuffer& buf, VkDeviceMemory& mem, uint8_t*& map) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = std::max<uint64_t>(bytes, 4), bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (fn<id::vkCreateBuffer>()(device, &bi, nullptr, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr{};
    fBufReq(device, buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = pickType(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                  mr.memoryTypeBits, true, nullptr);
    if (fn<id::vkAllocateMemory>()(device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
    void* p = nullptr;
    if (fn<id::vkBindBufferMemory>()(device, buf, mem, 0) != VK_SUCCESS ||
        fn<id::vkMapMemory>()(device, mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS)
      return false;
    map = static_cast<uint8_t*>(p);
    std::memset(map, 0, size_t(bi.size));  // a dump of a driver that copies nothing is all zero, not stale memory
    return true;
  }
  void barrier(VkCommandBuffer cb, VkImage im, VkImageAspectFlags aspect, uint32_t mip, uint32_t mips, uint32_t layer, uint32_t layers,
               VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src, b.dstAccessMask = dst, b.oldLayout = from, b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im, b.subresourceRange = {aspect, mip, mips, layer, layers};
    fn<id::vkCmdPipelineBarrier>()(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }

  void onImageData(const Call& c) {
    ImageData d;
    if (!decodeImageData(c, d)) return void(++errors);
    auto key = std::make_tuple(d.image, d.mip, d.layer, d.aspect);
    auto& pend = pendingImage[key];
    if (d.offset == 0 && d.size == d.total) {
      upload(d, d.data, d.total);
      pendingImage.erase(key);
      return;
    }
    if (pend.first.size() != d.total) pend.first.assign(d.total, 0), pend.second = 0;
    if (d.offset + d.size <= d.total) std::memcpy(pend.first.data() + d.offset, d.data, d.size);
    pend.second += d.size;
    if (pend.second >= d.total) {
      upload(d, pend.first.data(), d.total);
      pendingImage.erase(key);
    }
  }
  void upload(const ImageData& d, const uint8_t* bytes, uint64_t total) {
    auto hi = h.find(d.image);
    if (hi == h.end()) return void(++imgUploadFail);
    const VkImage im = reinterpret_cast<VkImage>(hi->second);
    ImgRec& rec = imgs[im];
    // The staging buffer holds `total` bytes and the copy reads width x height x depth of this format from it: they must agree,
    // and the file's layout must be one this can write to the Vulkan format (or convert).
    const uint64_t want = expectedImageBytes(rec.ci.format, d.aspect, d.width, d.height, d.depth);
    if (d.store > kStoreSwappedRB4 || (d.store == kStoreSwappedRB4 && rec.ci.format != VK_FORMAT_B4G4R4A4_UNORM_PACK16) || !want ||
        want != total) {
      ++imgUploadSkipped;
      char why[160];
      std::snprintf(why, sizeof why, "image upload skipped: %llu bytes for format %d aspect 0x%x %ux%ux%u (expected %llu, store %u); more are counted",
                    (unsigned long long)total, int(rec.ci.format), d.aspect, d.width, d.height, d.depth, (unsigned long long)want, d.store);
      return warnOnce("imgsize", why);
    }
    VkBuffer buf{};
    VkDeviceMemory mem{};
    uint8_t* map = nullptr;
    bool ok = makeStaging(total, buf, mem, map);
    if (ok) {
      std::memcpy(map, bytes, total);
      if (d.store == kStoreSwappedRB4)  // ABGR4 held for B4G4R4A4: the R and B nibbles trade places
        for (uint64_t i = 0; i + 1 < total; i += 2) {
          uint16_t v;
          std::memcpy(&v, map + i, 2);
          v = uint16_t((v & 0x0F0F) | ((v & 0xF000) >> 8) | ((v & 0x00F0) << 8));
          std::memcpy(map + i, &v, 2);
        }
      const VkImageAspectFlags asp = d.aspect;
      const bool first = !(rec.touchedAspects & asp);
      rec.touchedAspects |= asp;
      ok = immediate([&](VkCommandBuffer cb) {
        if (first)  // the whole image to GENERAL once: later uploads must keep what earlier ones wrote
          barrier(cb, im, asp, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
                  VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        barrier(cb, im, asp, d.mip, 1, d.layer, 1, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy r{};
        r.imageSubresource = {asp, d.mip, d.layer, 1};
        r.imageExtent = {d.width, d.height, d.depth};
        fn<id::vkCmdCopyBufferToImage>()(cb, buf, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        barrier(cb, im, asp, d.mip, 1, d.layer, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
      });
    }
    ok ? void(++imgUploads) : void(++imgUploadFail);
    if (buf) fn<id::vkDestroyBuffer>()(device, buf, nullptr);
    if (mem) fn<id::vkFreeMemory>()(device, mem, nullptr);
  }

  // The capture of the replay starts at the end of an empty submit, right before the first FRAME call; nothing else runs
  // between the request and that submit, so the FRAME calls land in the new file whole.
  void armRecapture() {
    recaptureArmed = true;
    if (!queue0) return;
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = family;
    VkCommandPool pool{};
    fn<id::vkCreateCommandPool>()(device, &pi, nullptr, &pool);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool, ai.commandBufferCount = 1, ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VkCommandBuffer cb{};
    fn<id::vkAllocateCommandBuffers>()(device, &ai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    fn<id::vkBeginCommandBuffer>()(cb, &bi);
    fn<id::vkEndCommandBuffer>()(cb);
    vpvkRequest(0x7fffffffu, opt.recapture.c_str());
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &cb;
    fn<id::vkQueueSubmit>()(queue0, 1, &si, VK_NULL_HANDLE);
  }

  // ---- dumps ----
  bool dumpOn() const { return !opt.dumpDir.empty(); }
  bool atDraw() const { return opt.dumpAt == "draw"; }

  CbRec* cbOfId(uint64_t id) {
    auto it = h.find(id);
    if (it == h.end()) return nullptr;
    auto c = cbs.find(reinterpret_cast<VkCommandBuffer>(it->second));
    return c == cbs.end() ? nullptr : &c->second;
  }
  // A vkCmd of this command buffer is not sent: what it records is no longer what the game recorded, so it is not submitted.
  void taint(uint64_t cbId) {
    if (CbRec* cb = cbOfId(cbId)) cb->tainted = true;
  }

  VkRenderPass variantOf(RpRec& rp, bool resume) {
    VkRenderPass& v = rp.variant[resume];
    if (v) return v;
    auto edit = [&](auto& a) {
      if (a.finalLayout == VK_IMAGE_LAYOUT_UNDEFINED)  // not valid: the layout the pass leaves a target in (as recordDumps assumes)
        a.finalLayout = vpimg::isDepthFormat(a.format) ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      a.storeOp = VK_ATTACHMENT_STORE_OP_STORE, a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
      if (resume) a.loadOp = a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD, a.initialLayout = a.finalLayout;
    };
    if (!rp.v2) {
      std::vector<VkAttachmentDescription> at(rp.ci1.pAttachments, rp.ci1.pAttachments + rp.ci1.attachmentCount);
      for (auto& a : at) edit(a);
      VkRenderPassCreateInfo ci = rp.ci1;
      ci.pAttachments = at.data();
      if (fn<id::vkCreateRenderPass>()(device, &ci, nullptr, &v) != VK_SUCCESS) v = VK_NULL_HANDLE;
    } else {
      std::vector<VkAttachmentDescription2> at(rp.ci2.pAttachments, rp.ci2.pAttachments + rp.ci2.attachmentCount);
      for (auto& a : at) edit(a);
      VkRenderPassCreateInfo2 ci = rp.ci2;
      ci.pAttachments = at.data();
      if (fn<id::vkCreateRenderPass2>()(device, &ci, nullptr, &v) != VK_SUCCESS) v = VK_NULL_HANDLE;
    }
    return v;
  }

  void addAtt(std::vector<AttCopy>& out, VkImageView view, VkImageLayout layout, const std::string& name) {
    auto vi = views.find(view);
    if (vi == views.end()) return;
    const ViewRec& v = vi->second;
    auto ii = imgs.find(v.image);
    if (ii == imgs.end()) return;
    const VkImageCreateInfo& ci = ii->second.ci;
    if (ci.samples != VK_SAMPLE_COUNT_1_BIT) return;  // a multisampled image cannot be copied out
    if (!(ci.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) return;  // transient, or a format the driver does not copy
    AttCopy a;
    a.image = v.image, a.imageId = v.imageId, a.viewId = v.id, a.format = ci.format, a.layout = layout, a.name = name;
    a.mip = v.range.baseMipLevel, a.layerBase = v.range.baseArrayLayer;
    a.layers = std::min<uint32_t>(v.range.layerCount == VK_REMAINING_ARRAY_LAYERS ? ci.arrayLayers - a.layerBase : v.range.layerCount, 8);
    a.w = std::max(1u, ci.extent.width >> a.mip), a.h = std::max(1u, ci.extent.height >> a.mip);
    if (vpimg::isDepthFormat(ci.format)) {
      a.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
      a.name = "d";
    }
    out.push_back(a);
  }
  std::vector<AttCopy> gatherAttachments(PassCtx& p, bool forcedStore) {
    std::vector<AttCopy> out;
    if (!p.dynamic) {
      if (!p.rp || !p.fb) return out;
      for (size_t k = 0; k < p.fb->views.size() && k < p.rp->atts.size(); ++k) {
        const AttDesc& d = p.rp->atts[k];
        if (!forcedStore && d.store == VK_ATTACHMENT_STORE_OP_DONT_CARE && d.sstore == VK_ATTACHMENT_STORE_OP_DONT_CARE) continue;
        addAtt(out, p.fb->views[k], d.finalLayout, "c" + std::to_string(k));
      }
    } else {
      for (size_t k = 0; k < p.col.size(); ++k) {
        const VkRenderingAttachmentInfo& a = p.col[k];
        const bool res = a.resolveMode != VK_RESOLVE_MODE_NONE && a.resolveImageView;
        if (!res && !forcedStore && a.storeOp == VK_ATTACHMENT_STORE_OP_DONT_CARE) continue;
        addAtt(out, res ? a.resolveImageView : a.imageView, res ? a.resolveImageLayout : a.imageLayout, "c" + std::to_string(k));
      }
      if (p.dep.imageView) {
        const bool res = p.dep.resolveMode != VK_RESOLVE_MODE_NONE && p.dep.resolveImageView;
        if (res || forcedStore || p.dep.storeOp != VK_ATTACHMENT_STORE_OP_DONT_CARE)
          addAtt(out, res ? p.dep.resolveImageView : p.dep.imageView, res ? p.dep.resolveImageLayout : p.dep.imageLayout, "d");
      }
    }
    return out;
  }

  // Records, after the pass (or the cut), the copy of every attachment to a staging buffer of its own.
  void recordDumps(VkCommandBuffer cb, CbRec& rec, bool forcedStore, uint32_t draw, const Call& c) {
    PassCtx& p = rec.pass;
    for (const AttCopy& a : gatherAttachments(p, forcedStore)) {
      const bool depth = a.aspect == VK_IMAGE_ASPECT_DEPTH_BIT;
      const uint32_t bpt = depth ? vpimg::depthCopyBytes(a.format) : vpimg::texelBytes(a.format);
      if (!bpt) {
        warnOnce("fmt" + std::to_string(int(a.format)), "attachment format " + std::to_string(int(a.format)) + " has no known texel size: not dumped");
        continue;
      }
      for (uint32_t l = 0; l < a.layers; ++l) {
        DumpSlot s;
        s.bytes = uint64_t(a.w) * a.h * bpt;
        if (!makeStaging(s.bytes, s.buf, s.mem, s.map)) {
          warnOnce("staging", "staging buffer allocation failed: dumps are incomplete");
          continue;
        }
        s.format = a.format, s.depth = depth, s.w = a.w, s.h = a.h, s.localPass = p.localIndex, s.draw = draw;
        s.att = a.name + (a.layers > 1 ? "_l" + std::to_string(l) : std::string());
        s.call = callName(c.id), s.callIndex = c.index, s.pipeline = rec.pipe;
        s.pipelines.assign(p.pipelines.begin(), p.pipelines.end());
        s.rp = p.rpId, s.fb = p.fbId, s.image = a.imageId, s.view = a.viewId, s.ibuf = rec.ibuf;
        s.sets.assign(rec.sets.begin(), rec.sets.end());
        s.vbufs.assign(rec.vbufs.begin(), rec.vbufs.end());
        const uint32_t layer = a.layerBase + l;
        // A finalLayout of UNDEFINED (not valid) is taken as the layout the pass leaves a target in.
        const VkImageLayout layout = a.layout != VK_IMAGE_LAYOUT_UNDEFINED ? a.layout
                                     : depth                                 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                                                             : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier(cb, a.image, a.aspect, a.mip, 1, layer, 1, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy r{};
        r.imageSubresource = {a.aspect, a.mip, layer, 1};
        r.imageExtent = {a.w, a.h, 1};
        fn<id::vkCmdCopyImageToBuffer>()(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.buf, 1, &r);
        barrier(cb, a.image, a.aspect, a.mip, 1, layer, 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout, VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        rec.dumps.push_back(std::move(s));
      }
    }
  }

  void endPass(VkCommandBuffer cb, CbRec& rec, const Call& c) {
    PassCtx& p = rec.pass;
    if (!p.active) return;
    if (dumpOn() && (!atDraw() || !p.canCut)) recordDumps(cb, rec, false, p.draws, c);
    p.active = false;
  }
  // After a draw in draw mode: end the pass, copy, resume it with LOAD.
  void cutPass(VkCommandBuffer cb, CbRec& rec, const Call& c) {
    PassCtx& p = rec.pass;
    if (p.dynamic) {
      fn<id::vkCmdEndRendering>()(cb);
      recordDumps(cb, rec, true, p.draws - 1, c);
      std::vector<VkRenderingAttachmentInfo> col = p.col;
      for (auto& a : col) a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      VkRenderingAttachmentInfo dep = p.dep, stn = p.stn;
      dep.loadOp = stn.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      VkRenderingInfo ri = p.ri;
      ri.pColorAttachments = col.data(), ri.pDepthAttachment = p.dep.imageView ? &dep : nullptr;
      ri.pStencilAttachment = p.stn.imageView ? &stn : nullptr;
      fn<id::vkCmdBeginRendering>()(cb, &ri);
    } else {
      fn<id::vkCmdEndRenderPass>()(cb);
      recordDumps(cb, rec, true, p.draws - 1, c);
      VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      bi.renderPass = variantOf(*p.rp, true);
      bi.framebuffer = reinterpret_cast<VkFramebuffer>(h[p.fbId]);
      bi.renderArea = p.area;
      fn<id::vkCmdBeginRenderPass>()(cb, &bi, VK_SUBPASS_CONTENTS_INLINE);
    }
  }

  static const char* fmtName(VkFormat f) {
    switch (f) {
      case VK_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
      case VK_FORMAT_R8G8B8A8_SRGB: return "R8G8B8A8_SRGB";
      case VK_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
      case VK_FORMAT_B8G8R8A8_SRGB: return "B8G8R8A8_SRGB";
      case VK_FORMAT_R16G16B16A16_SFLOAT: return "R16G16B16A16_SFLOAT";
      case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return "B10G11R11_UFLOAT_PACK32";
      case VK_FORMAT_D32_SFLOAT: return "D32_SFLOAT";
      case VK_FORMAT_D16_UNORM: return "D16_UNORM";
      case VK_FORMAT_D24_UNORM_S8_UINT: return "D24_UNORM_S8_UINT";
      case VK_FORMAT_D32_SFLOAT_S8_UINT: return "D32_SFLOAT_S8_UINT";
      default: return "other";
    }
  }
  static std::string arr(const std::vector<uint64_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
  }

  // After a submit has completed: the staging buffers of the command buffers it ran become files.
  void collectDumps(const std::vector<VkCommandBuffer>& order, uint32_t frameIdx, uint32_t submitIdx) {
    if (!dumpOn()) return;
    uint32_t passBase = 0;
    for (VkCommandBuffer cb : order) {
      auto it = cbs.find(cb);
      if (it == cbs.end()) continue;
      CbRec& rec = it->second;
      for (const DumpSlot& s : rec.dumps) {
        char base[128];
        std::snprintf(base, sizeof base, "f%u_s%u_p%u_d%u_%s", frameIdx, submitIdx, passBase + s.localPass, s.draw, s.att.c_str());
        vpimg::Image img;
        const bool png = !opt.raw && vpimg::convert(s.format, s.depth, s.map, s.w, s.h, img);
        std::string file = std::string(base) + (png ? ".png" : ".raw");
        const std::string path = opt.dumpDir + "/" + file;
        if (png) {
          const auto bytes = vpimg::pngEncode(img);
          std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
          ++dumpFiles;
        } else {
          std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(s.map), std::streamsize(s.bytes));
          std::ofstream(path + ".json") << "{\"format\":" << int(s.format) << ",\"width\":" << s.w << ",\"height\":" << s.h
                                        << ",\"bytes_per_texel\":" << (s.bytes / std::max<uint64_t>(1, uint64_t(s.w) * s.h)) << "}\n";
          ++dumpRaw;
        }
        std::ostringstream e;
        e << "{\"file\":\"" << vpjson::escape(file) << "\",\"kind\":\"" << (png ? "png" : "raw") << "\",\"frame\":" << frameIdx
          << ",\"submit\":" << submitIdx << ",\"pass\":" << passBase + s.localPass << ",\"draw\":" << s.draw << ",\"att\":\"" << s.att
          << "\",\"format\":" << int(s.format) << ",\"format_name\":\"" << fmtName(s.format) << "\",\"width\":" << s.w << ",\"height\":" << s.h
          << ",\"call\":\"" << s.call << "\",\"call_index\":" << s.callIndex << ",\"pipeline\":" << s.pipeline
          << ",\"pipelines_in_pass\":" << arr(s.pipelines) << ",\"render_pass\":" << s.rp << ",\"framebuffer\":" << s.fb
          << ",\"image\":" << s.image << ",\"view\":" << s.view << ",\"descriptor_sets\":" << arr(s.sets)
          << ",\"vertex_buffers\":" << arr(s.vbufs) << ",\"index_buffer\":" << s.ibuf << ",\"command_buffer\":" << rec.id << "}";
        entries.push_back(e.str());
      }
      passBase += rec.passes;
    }
  }
  void writeIndex() {
    if (!dumpOn()) return;
    std::ofstream f(opt.dumpDir + "/index.json");
    f << "{\"version\":1,\"capture\":\"" << vpjson::escape(opt.file) << "\",\"icd\":\"" << opt.icd << "\",\"dump_at\":\"" << opt.dumpAt
      << "\",\"dumps\":[\n";
    for (size_t i = 0; i < entries.size(); ++i) f << " " << entries[i] << (i + 1 < entries.size() ? ",\n" : "\n");
    f << "]}\n";
  }

  // ---- calls ----
  void corrupt(const Call& c) {
    ++errors;
    if (c.id >= 0x0200 && c.size >= 8) {  // a vkCmd: its first argument is the command buffer
      uint64_t cbId = 0;
      std::memcpy(&cbId, c.data, 8);
      taint(cbId);
    }
    warnOnce(std::string("corrupt") + callName(c.id), std::string("payload of ") + callName(c.id) + " is shorter than its codec expects");
  }
  void skip(const Call& c) { ++skipped[callName(c.id)]; }
  void fail(const Call& c, VkResult r) {
    ++errors;
    ++failed[std::string(callName(c.id)) + "=" + std::to_string(int(r))];
  }

  void handle(const Call& c);

  // ---- per-call hooks ----
  // Before the handles are replaced (arrays still hold ids): what the dumps report as bound.
  template <uint16_t ID, class A>
  void track(const Call&, A& a) {
    if constexpr (ID == id::vkBeginCommandBuffer) {
      if (CbRec* cb = cbOfId(idOfFake(std::get<0>(a)))) {
        cb->dumps.clear(), cb->passes = 0, cb->pipe = cb->ibuf = 0, cb->sets.fill(0), cb->vbufs.fill(0), cb->pass = PassCtx{}, cb->tainted = false;
      }
    } else if constexpr (ID == id::vkCmdBindPipeline) {
      if (CbRec* cb = cbOfId(idOfFake(std::get<0>(a))); cb && std::get<1>(a) == VK_PIPELINE_BIND_POINT_GRAPHICS) cb->pipe = idOfFake(std::get<2>(a));
    } else if constexpr (ID == id::vkCmdBindDescriptorSets) {
      if (CbRec* cb = cbOfId(idOfFake(std::get<0>(a))); cb && std::get<1>(a) == VK_PIPELINE_BIND_POINT_GRAPHICS && std::get<5>(a))
        for (uint32_t i = 0; i < std::get<4>(a); ++i)
          if (std::get<3>(a) + i < cb->sets.size()) cb->sets[std::get<3>(a) + i] = idOfFake(std::get<5>(a)[i]);
    } else if constexpr (ID == id::vkCmdBindVertexBuffers || ID == id::vkCmdBindVertexBuffers2) {
      if (CbRec* cb = cbOfId(idOfFake(std::get<0>(a))); cb && std::get<3>(a))
        for (uint32_t i = 0; i < std::get<2>(a); ++i)
          if (std::get<1>(a) + i < cb->vbufs.size()) cb->vbufs[std::get<1>(a) + i] = idOfFake(std::get<3>(a)[i]);
    } else if constexpr (ID == id::vkCmdBindIndexBuffer) {
      if (CbRec* cb = cbOfId(idOfFake(std::get<0>(a)))) cb->ibuf = idOfFake(std::get<1>(a));
    } else if constexpr (ID == id::vkCmdNextSubpass) {
      if (CbRec* cb = cbOfId(idOfFake(std::get<0>(a)))) cb->pass.canCut = false;
    } else if constexpr (ID == id::vkCmdDraw || ID == id::vkCmdDrawIndexed || ID == id::vkCmdDrawIndirect ||
                         ID == id::vkCmdDrawIndexedIndirect || ID == id::vkCmdDrawIndirectCount || ID == id::vkCmdDrawIndexedIndirectCount) {
      if (CbRec* cb = cbOfId(idOfFake(std::get<0>(a))); cb && cb->pass.active) {
        ++cb->pass.draws;
        cb->pass.pipelines.insert(cb->pipe);
      }
    }
  }

  void fixBarrierQf(uint32_t& s, uint32_t& d) { s = d = VK_QUEUE_FAMILY_IGNORED; }
  void fixDep(const VkDependencyInfo* d) {
    if (!d) return;
    for (uint32_t i = 0; i < d->bufferMemoryBarrierCount; ++i) {
      VkBufferMemoryBarrier2& b = mut(d->pBufferMemoryBarriers)[i];
      b.buffer = real(b.buffer);
      fixBarrierQf(b.srcQueueFamilyIndex, b.dstQueueFamilyIndex);
    }
    for (uint32_t i = 0; i < d->imageMemoryBarrierCount; ++i) {
      VkImageMemoryBarrier2& b = mut(d->pImageMemoryBarriers)[i];
      b.image = real(b.image);
      fixBarrierQf(b.srcQueueFamilyIndex, b.dstQueueFamilyIndex);
    }
  }
  void fixBufBarriers(const VkBufferMemoryBarrier* p, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) mut(p)[i].buffer = real(p[i].buffer), fixBarrierQf(mut(p)[i].srcQueueFamilyIndex, mut(p)[i].dstQueueFamilyIndex);
  }
  void fixImgBarriers(const VkImageMemoryBarrier* p, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) mut(p)[i].image = real(p[i].image), fixBarrierQf(mut(p)[i].srcQueueFamilyIndex, mut(p)[i].dstQueueFamilyIndex);
  }
  void fixRenderingAtt(const VkRenderingAttachmentInfo* a) {
    if (!a) return;
    mut(a)->imageView = real(a->imageView);
    mut(a)->resolveImageView = real(a->resolveImageView);
  }

  // The handles inside the structs of a call (arrays and structs are in the decoder's arena: edited in place).
  template <uint16_t ID, class A>
  void patch(A& a) {
    if constexpr (ID == id::vkCreateBuffer) {
      if (auto* ci = std::get<1>(a)) patchBuffer(*mut(ci));
    } else if constexpr (ID == id::vkCreateImage) {
      if (auto* ci = std::get<1>(a)) patchImage(*mut(ci));
    } else if constexpr (ID == id::vkCreateBufferView) {
      if (auto* ci = std::get<1>(a)) mut(ci)->buffer = real(ci->buffer);
    } else if constexpr (ID == id::vkCreateImageView) {
      if (auto* ci = std::get<1>(a)) mut(ci)->image = real(ci->image);
    } else if constexpr (ID == id::vkCreateDescriptorSetLayout) {
      if (auto* ci = std::get<1>(a))
        for (uint32_t i = 0; i < ci->bindingCount; ++i) {
          const VkDescriptorSetLayoutBinding& b = ci->pBindings[i];
          if (b.pImmutableSamplers) realArr(b.pImmutableSamplers, b.descriptorCount);
        }
    } else if constexpr (ID == id::vkCreatePipelineLayout) {
      if (auto* ci = std::get<1>(a)) realArr(ci->pSetLayouts, ci->setLayoutCount);
    } else if constexpr (ID == id::vkAllocateDescriptorSets) {
      if (auto* ai = std::get<1>(a)) mut(ai)->descriptorPool = real(ai->descriptorPool), realArr(ai->pSetLayouts, ai->descriptorSetCount);
    } else if constexpr (ID == id::vkFreeDescriptorSets) {
      realArr(std::get<3>(a), std::get<2>(a));
    } else if constexpr (ID == id::vkCreateFramebuffer) {
      if (auto* ci = std::get<1>(a)) {
        mut(ci)->renderPass = real(ci->renderPass);
        if (!(ci->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT)) realArr(ci->pAttachments, ci->attachmentCount);
      }
    } else if constexpr (ID == id::vkCreateGraphicsPipelines) {
      const auto* infos = std::get<3>(a);
      for (uint32_t i = 0; infos && i < std::get<2>(a); ++i) {
        VkGraphicsPipelineCreateInfo& g = mut(infos)[i];
        g.layout = real(g.layout), g.renderPass = real(g.renderPass), g.basePipelineHandle = real(g.basePipelineHandle);
        for (uint32_t s = 0; s < g.stageCount; ++s) mut(g.pStages)[s].module = real(g.pStages[s].module);
      }
    } else if constexpr (ID == id::vkCreateComputePipelines) {
      const auto* infos = std::get<3>(a);
      for (uint32_t i = 0; infos && i < std::get<2>(a); ++i) {
        VkComputePipelineCreateInfo& g = mut(infos)[i];
        g.layout = real(g.layout), g.basePipelineHandle = real(g.basePipelineHandle), g.stage.module = real(g.stage.module);
      }
    } else if constexpr (ID == id::vkCreateCommandPool) {
      if (auto* ci = std::get<1>(a)) mut(ci)->queueFamilyIndex = family;
    } else if constexpr (ID == id::vkAllocateCommandBuffers) {
      if (auto* ai = std::get<1>(a)) mut(ai)->commandPool = real(ai->commandPool);
    } else if constexpr (ID == id::vkFreeCommandBuffers) {
      realArr(std::get<3>(a), std::get<2>(a));
    } else if constexpr (ID == id::vkResetFences) {
      realArr(std::get<2>(a), std::get<1>(a));
    } else if constexpr (ID == id::vkBeginCommandBuffer) {
      if (auto* bi = std::get<1>(a); bi && bi->pInheritanceInfo) {
        mut(bi->pInheritanceInfo)->renderPass = real(bi->pInheritanceInfo->renderPass);
        mut(bi->pInheritanceInfo)->framebuffer = real(bi->pInheritanceInfo->framebuffer);
      }
    } else if constexpr (ID == id::vkCmdBeginRenderPass || ID == id::vkCmdBeginRenderPass2) {
      if (auto* bi = std::get<1>(a)) mut(bi)->renderPass = real(bi->renderPass), mut(bi)->framebuffer = real(bi->framebuffer);
    } else if constexpr (ID == id::vkCmdBeginRendering) {
      if (auto* ri = std::get<1>(a)) {
        for (uint32_t i = 0; i < ri->colorAttachmentCount; ++i) fixRenderingAtt(ri->pColorAttachments + i);
        fixRenderingAtt(ri->pDepthAttachment);
        fixRenderingAtt(ri->pStencilAttachment);
      }
    } else if constexpr (ID == id::vkCmdBindDescriptorSets) {
      realArr(std::get<5>(a), std::get<4>(a));
    } else if constexpr (ID == id::vkCmdBindVertexBuffers || ID == id::vkCmdBindVertexBuffers2) {
      realArr(std::get<3>(a), std::get<2>(a));
    } else if constexpr (ID == id::vkCmdExecuteCommands) {
      realArr(std::get<2>(a), std::get<1>(a));
    } else if constexpr (ID == id::vkCmdPipelineBarrier) {
      fixBufBarriers(std::get<7>(a), std::get<6>(a));
      fixImgBarriers(std::get<9>(a), std::get<8>(a));
    } else if constexpr (ID == id::vkCmdPipelineBarrier2) {
      fixDep(std::get<1>(a));
    } else if constexpr (ID == id::vkCmdCopyBuffer2) {
      if (auto* i = std::get<1>(a)) mut(i)->srcBuffer = real(i->srcBuffer), mut(i)->dstBuffer = real(i->dstBuffer);
    } else if constexpr (ID == id::vkCmdCopyImage2) {
      if (auto* i = std::get<1>(a)) mut(i)->srcImage = real(i->srcImage), mut(i)->dstImage = real(i->dstImage);
    } else if constexpr (ID == id::vkCmdCopyBufferToImage2) {
      if (auto* i = std::get<1>(a)) mut(i)->srcBuffer = real(i->srcBuffer), mut(i)->dstImage = real(i->dstImage);
    } else if constexpr (ID == id::vkCmdCopyImageToBuffer2) {
      if (auto* i = std::get<1>(a)) mut(i)->srcImage = real(i->srcImage), mut(i)->dstBuffer = real(i->dstBuffer);
    } else if constexpr (ID == id::vkCmdBlitImage2) {
      if (auto* i = std::get<1>(a)) mut(i)->srcImage = real(i->srcImage), mut(i)->dstImage = real(i->dstImage);
    } else if constexpr (ID == id::vkCmdResolveImage2) {
      if (auto* i = std::get<1>(a)) mut(i)->srcImage = real(i->srcImage), mut(i)->dstImage = real(i->dstImage);
    }
  }

  // After the handles are the driver's, before the call: the passes start here (and draw mode swaps the render pass).
  template <uint16_t ID, class A>
  void pre(const Call&, A& a) {
    if constexpr (ID == id::vkCmdBeginRenderPass || ID == id::vkCmdBeginRenderPass2) {
      auto* bi = std::get<1>(a);
      CbRec* cb = cbOfReal(std::get<0>(a));
      if (!cb || !bi) return;
      PassCtx& p = cb->pass;
      p = PassCtx{};
      p.active = true, p.localIndex = cb->passes++, p.area = bi->renderArea;
      auto rp = rps.find(bi->renderPass);
      auto fb = fbs.find(bi->framebuffer);
      if (rp == rps.end() || fb == fbs.end()) return;
      p.rp = &rp->second, p.fb = &fb->second, p.rpId = rp->second.id, p.fbId = fb->second.id;
      uint32_t subpasses = rp->second.v2 ? rp->second.ci2.subpassCount : rp->second.ci1.subpassCount;
      VkSubpassContents contents = VK_SUBPASS_CONTENTS_INLINE;
      if constexpr (ID == id::vkCmdBeginRenderPass) contents = std::get<2>(a);
      else if (std::get<2>(a)) contents = std::get<2>(a)->contents;
      p.canCut = dumpOn() && atDraw() && subpasses == 1 && contents == VK_SUBPASS_CONTENTS_INLINE;
      if (p.canCut) {
        const VkRenderPass v = variantOf(rp->second, false);
        if (v) mut(bi)->renderPass = v;
        else p.canCut = false;
      }
    } else if constexpr (ID == id::vkCmdExecuteCommands) {
      if (CbRec* cb = cbOfReal(std::get<0>(a)))
        for (uint32_t i = 0; i < std::get<1>(a); ++i)
          if (CbRec* sec = cbOfReal(std::get<2>(a)[i]); sec && sec->tainted) cb->tainted = true;
    } else if constexpr (ID == id::vkCmdBeginRendering) {
      auto* ri = std::get<1>(a);
      CbRec* cb = cbOfReal(std::get<0>(a));
      if (!cb || !ri) return;
      PassCtx& p = cb->pass;
      p = PassCtx{};
      p.active = p.dynamic = true, p.localIndex = cb->passes++;
      p.canCut = dumpOn() && atDraw() && !(ri->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT | VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT));
      if (p.canCut) {  // every attachment is stored: the cut copies them
        for (uint32_t i = 0; i < ri->colorAttachmentCount; ++i) mut(ri->pColorAttachments)[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (ri->pDepthAttachment) mut(ri->pDepthAttachment)->storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (ri->pStencilAttachment) mut(ri->pStencilAttachment)->storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      }
      p.ri = *ri;
      p.col.assign(ri->pColorAttachments, ri->pColorAttachments + ri->colorAttachmentCount);
      if (ri->pDepthAttachment) p.dep = *ri->pDepthAttachment;
      if (ri->pStencilAttachment) p.stn = *ri->pStencilAttachment;
      p.ri.pColorAttachments = nullptr, p.ri.pDepthAttachment = p.ri.pStencilAttachment = nullptr;
    }
  }

  CbRec* cbOfReal(VkCommandBuffer cb) {
    auto it = cbs.find(cb);
    return it == cbs.end() ? nullptr : &it->second;
  }
  // A command buffer with a command left out is not submitted (counted).
  bool dropped(VkCommandBuffer cb) {
    const CbRec* rec = cbOfReal(cb);
    if (!rec || !rec->tainted) return false;
    ++cbsSkipped;
    warnOnce("cbskip", "a command buffer with a command that could not be replayed is not submitted (more are counted)");
    return true;
  }

  template <uint16_t ID, class A>
  void after(const Call& c, A& a) {
    if constexpr (ID == id::vkCmdEndRenderPass || ID == id::vkCmdEndRenderPass2 || ID == id::vkCmdEndRendering) {
      if (CbRec* cb = cbOfReal(std::get<0>(a))) endPass(std::get<0>(a), *cb, c);
    } else if constexpr (ID == id::vkCmdDraw || ID == id::vkCmdDrawIndexed || ID == id::vkCmdDrawIndirect || ID == id::vkCmdDrawIndexedIndirect ||
                         ID == id::vkCmdDrawIndirectCount || ID == id::vkCmdDrawIndexedIndirectCount) {
      CbRec* cb = cbOfReal(std::get<0>(a));
      if (cb && dumpOn() && atDraw() && cb->pass.active && cb->pass.canCut) cutPass(std::get<0>(a), *cb, c);
    }
  }

  // Records the objects the dumps and the draw-mode cuts look at, once the driver made them.
  template <uint16_t ID, class A>
  void made(const Call&, A& a, const std::vector<uint64_t>& ids) {
    if constexpr (ID == id::vkCreateImage) {
      ImgRec r;
      r.id = ids[0], r.ci = *std::get<1>(a);
      imgs[*std::get<3>(a)] = r;
    } else if constexpr (ID == id::vkCreateImageView) {
      ViewRec v;
      v.id = ids[0], v.image = std::get<1>(a)->image, v.range = std::get<1>(a)->subresourceRange;
      if (auto it = imgs.find(v.image); it != imgs.end()) v.imageId = it->second.id;
      views[*std::get<3>(a)] = v;
    } else if constexpr (ID == id::vkCreateFramebuffer) {
      FbRec f;
      f.id = ids[0];
      if (!(std::get<1>(a)->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT))
        f.views.assign(std::get<1>(a)->pAttachments, std::get<1>(a)->pAttachments + std::get<1>(a)->attachmentCount);
      fbs[*std::get<3>(a)] = f;
    } else if constexpr (ID == id::vkCreateRenderPass) {
      RpRec r;
      r.id = ids[0], r.ci1 = *std::get<1>(a);
      for (uint32_t i = 0; i < r.ci1.attachmentCount; ++i) {
        const VkAttachmentDescription& d = r.ci1.pAttachments[i];
        r.atts.push_back({d.format, d.storeOp, d.stencilStoreOp, d.finalLayout});
      }
      rps[*std::get<3>(a)] = r;
    } else if constexpr (ID == id::vkCreateRenderPass2) {
      RpRec r;
      r.id = ids[0], r.v2 = true, r.ci2 = *std::get<1>(a);
      for (uint32_t i = 0; i < r.ci2.attachmentCount; ++i) {
        const VkAttachmentDescription2& d = r.ci2.pAttachments[i];
        r.atts.push_back({d.format, d.storeOp, d.stencilStoreOp, d.finalLayout});
      }
      rps[*std::get<3>(a)] = r;
    } else if constexpr (ID == id::vkAllocateCommandBuffers) {
      for (size_t i = 0; i < ids.size(); ++i) {
        CbRec& r = cbs[std::get<2>(a)[i]];
        r = CbRec{};
        r.id = ids[i];
      }
    }
  }
  template <uint16_t ID>
  static constexpr bool kKeeps = ID == id::vkCreateRenderPass || ID == id::vkCreateRenderPass2;
  template <uint16_t ID>
  static constexpr bool kDestroys = ID == id::vkDestroyBuffer || ID == id::vkDestroyBufferView || ID == id::vkDestroyImage ||
                                    ID == id::vkDestroyImageView || ID == id::vkDestroySampler || ID == id::vkDestroyDescriptorSetLayout ||
                                    ID == id::vkDestroyPipelineLayout || ID == id::vkDestroyDescriptorPool || ID == id::vkDestroyRenderPass ||
                                    ID == id::vkDestroyFramebuffer || ID == id::vkDestroyShaderModule || ID == id::vkDestroyPipeline ||
                                    ID == id::vkDestroyCommandPool || ID == id::vkDestroyFence || ID == id::vkDestroySemaphore ||
                                    ID == id::vkDestroyEvent || ID == id::vkDestroyQueryPool;
  template <uint16_t ID>
  static constexpr bool kSkips = ID == id::vkMapMemory || ID == id::vkUnmapMemory || ID == id::vkFlushMappedMemoryRanges ||
                                 ID == id::vkWaitForFences || ID == id::vkCmdSetEvent || ID == id::vkCmdSetEvent2 || ID == id::vkCmdResetEvent ||
                                 ID == id::vkCmdResetEvent2 || ID == id::vkDestroyInstance || ID == id::vkDestroyDevice;

  // The last argument is a non-const pointer to a handle: the call makes objects.
  template <class A>
  using LastArg = std::tuple_element_t<std::tuple_size_v<A> - 1, A>;
  template <class A>
  static constexpr bool kHasOut = std::is_pointer_v<LastArg<A>> && !std::is_const_v<std::remove_pointer_t<LastArg<A>>> &&
                                  kIsHandle<std::remove_pointer_t<LastArg<A>>>;

  template <uint16_t ID, class A>
  uint32_t outCount(A& a) {
    if constexpr (ID == id::vkAllocateDescriptorSets) return std::get<1>(a) ? std::get<1>(a)->descriptorSetCount : 0;
    else if constexpr (ID == id::vkAllocateCommandBuffers) return std::get<1>(a) ? std::get<1>(a)->commandBufferCount : 0;
    else if constexpr (ID == id::vkCreateGraphicsPipelines || ID == id::vkCreateComputePipelines) return std::get<2>(a);
    else return 1;
  }

  template <uint16_t ID, class D>
  void execute(const Call& c, D& d) {
    auto& a = d.args;
    auto f = fn<ID>();
    if (!f) {
      ++unsupported;
      warnOnce(std::string("nofn") + callName(c.id), std::string("driver has no ") + callName(c.id));
      if constexpr (ID >= 0x0200) {
        if (CbRec* cb = cbOfReal(std::get<0>(d.args))) cb->tainted = true;
      }
      return;
    }
    using Args = std::remove_reference_t<decltype(a)>;
    std::vector<uint64_t> ids;
    if constexpr (kHasOut<Args>) {
      auto* out = std::get<std::tuple_size_v<Args> - 1>(a);
      if (out)
        for (uint32_t i = 0, n = outCount<ID>(a); i < n; ++i) ids.push_back(idOfFake(out[i]));
    }
    using Ret = decltype(std::apply(f, a));
    if constexpr (std::is_void_v<Ret>) {
      std::apply(f, a);
    } else {
      const auto res = std::apply(f, a);
      if (res < 0) return fail(c, res);
      if constexpr (kHasOut<Args>) {
        auto* out = std::get<std::tuple_size_v<Args> - 1>(a);
        for (size_t i = 0; i < ids.size(); ++i) reg(ids[i], out[i]);
        if (!ids.empty()) made<ID>(c, a, ids);
      }
    }
    after<ID>(c, a);
  }
};

template <uint16_t ID>
void run(R& r, const Call& c) {
  if constexpr (R::kSkips<ID>) {
    r.skip(c);
  } else {
    auto d = decode<ID>(c);
    if (!d->ok) return r.corrupt(c);
    auto& a = d->args;
    r.missing = false;
    uint64_t destroyed = 0, cbId = 0;
    if constexpr (R::kDestroys<ID>) destroyed = idOfFake(std::get<1>(a));
    if constexpr (ID >= 0x0200) cbId = idOfFake(std::get<0>(a));
    r.track<ID>(c, a);
    r.mapTop(a);
    r.patch<ID>(a);
    if (r.missing) {
      if constexpr (ID >= 0x0200) r.taint(cbId);
      ++r.skippedMissing;
      r.warnOnce(std::string("miss") + callName(c.id), std::string(callName(c.id)) + " uses an object that is not in the file: skipped (more are counted)");
      return;
    }
    r.pre<ID>(c, a);
    r.execute<ID>(c, *d);
    if (destroyed) r.h.erase(destroyed);
    if constexpr (R::kKeeps<ID>) r.keep.emplace_back(std::move(d));
  }
}

// ---- the calls that are not a plain pass-through ----
template <>
void run<id::vkCreateInstance>(R& r, const Call& c) {
  auto d = decode<id::vkCreateInstance>(c);
  if (!d->ok) return r.corrupt(c);
  VkInstanceCreateInfo ci = *std::get<0>(d->args);
  uint32_t n = 0;
  std::vector<VkExtensionProperties> have;
  if (r.fInstExt) {
    r.fInstExt(nullptr, &n, nullptr);
    have.resize(n);
    r.fInstExt(nullptr, &n, have.data());
  }
  std::vector<const char*> exts;
  for (uint32_t i = 0; i < ci.enabledExtensionCount; ++i) {
    const bool ok = std::any_of(have.begin(), have.end(), [&](const VkExtensionProperties& e) { return !std::strcmp(e.extensionName, ci.ppEnabledExtensionNames[i]); });
    if (ok) exts.push_back(ci.ppEnabledExtensionNames[i]);
    else r.warnOnce(std::string("ie") + ci.ppEnabledExtensionNames[i], std::string("instance extension dropped: ") + ci.ppEnabledExtensionNames[i]);
  }
  ci.enabledLayerCount = 0, ci.ppEnabledLayerNames = nullptr;
  ci.enabledExtensionCount = uint32_t(exts.size()), ci.ppEnabledExtensionNames = exts.data();
  if (r.instanceMade) return;
  if (r.makeInstance(ci)) r.reg(idOfFake(*std::get<2>(d->args)), r.instance);
  else std::exit(1);
}

template <>
void run<id::vkCreateDevice>(R& r, const Call& c) {
  auto d = decode<id::vkCreateDevice>(c);
  if (!d->ok) return r.corrupt(c);
  if (!r.ensureInstance()) std::exit(1);
  VkDeviceCreateInfo ci = *std::get<1>(d->args);
  // One queue family: every captured queue becomes a queue of it.
  uint32_t wanted = 0;
  for (uint32_t i = 0; i < ci.queueCreateInfoCount; ++i) wanted += ci.pQueueCreateInfos[i].queueCount;
  const uint32_t count = std::max(1u, std::min(wanted, r.queueCount));
  std::vector<float> prio(count, 1.0f);
  VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qi.queueFamilyIndex = r.family, qi.queueCount = count, qi.pQueuePriorities = prio.data();
  ci.queueCreateInfoCount = 1, ci.pQueueCreateInfos = &qi;
  ci.enabledLayerCount = 0, ci.ppEnabledLayerNames = nullptr;
  uint32_t n = 0;
  r.fDevExt(r.pd, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> have(n);
  r.fDevExt(r.pd, nullptr, &n, have.data());
  std::vector<const char*> exts;
  for (uint32_t i = 0; i < ci.enabledExtensionCount; ++i) {
    const char* e = ci.ppEnabledExtensionNames[i];
    if (std::any_of(have.begin(), have.end(), [&](const VkExtensionProperties& x) { return !std::strcmp(x.extensionName, e); })) exts.push_back(e);
    else r.warnOnce(std::string("de") + e, std::string("device extension dropped: ") + e);
  }
  ci.enabledExtensionCount = uint32_t(exts.size()), ci.ppEnabledExtensionNames = exts.data();
  // Features the driver lacks are switched off (a feature struct is a run of VkBool32 after its header).
  VkPhysicalDeviceVulkan11Features s11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features s12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features s13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceFeatures2 s2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  if (r.apiVersion >= VK_API_VERSION_1_2) s2.pNext = &s11, s11.pNext = &s12;
  if (r.apiVersion >= VK_API_VERSION_1_3) s12.pNext = &s13;
  if (r.fPdFeat2) r.fPdFeat2(r.pd, &s2);
  auto andBools = [](void* dst, const void* sup, size_t off, size_t bytes) {
    auto* x = static_cast<VkBool32*>(static_cast<void*>(static_cast<char*>(dst) + off));
    const auto* y = static_cast<const VkBool32*>(static_cast<const void*>(static_cast<const char*>(sup) + off));
    for (size_t i = 0; i < bytes / 4; ++i) x[i] = x[i] && y[i];
  };
  constexpr size_t kHdr = 16;
  if (ci.pEnabledFeatures) {
    auto* f = mut(ci.pEnabledFeatures);
    andBools(f, &s2.features, 0, sizeof *f);
  }
  for (auto* p = static_cast<VkBaseOutStructure*>(const_cast<void*>(ci.pNext)); p; p = p->pNext) {
    switch (p->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2:
        andBools(&reinterpret_cast<VkPhysicalDeviceFeatures2*>(p)->features, &s2.features, 0, sizeof(VkPhysicalDeviceFeatures));
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
        andBools(p, &s11, kHdr, sizeof s11 - kHdr);
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
        andBools(p, &s12, kHdr, sizeof s12 - kHdr);
        r.bdaEnabled = reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(p)->bufferDeviceAddress != 0;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
        andBools(p, &s13, kHdr, sizeof s13 - kHdr);
        break;
      default: break;
    }
  }
  const VkResult res = r.fCreateDevice(r.pd, &ci, nullptr, &r.device);
  if (res != VK_SUCCESS) {
    r.note("vkCreateDevice failed (%d)", int(res));
    std::exit(1);
  }
  r.reg(idOfFake(*std::get<3>(d->args)), r.device);
  r.loadDeviceFns();
  r.fn<id::vkGetDeviceQueue>()(r.device, r.family, 0, &r.queue0);
}

template <>
void run<id::vkGetDeviceQueue>(R& r, const Call& c) {
  auto d = decode<id::vkGetDeviceQueue>(c);
  if (!d->ok) return r.corrupt(c);
  VkQueue q{};
  r.fn<id::vkGetDeviceQueue>()(r.device, r.family, std::get<2>(d->args) % r.queueCount, &q);
  r.reg(idOfFake(*std::get<3>(d->args)), q);
}

template <>
void run<id::vkAllocateMemory>(R& r, const Call& c) {
  auto d = decode<id::vkAllocateMemory>(c);
  if (!d->ok) return r.corrupt(c);
  r.prescan();
  VkMemoryAllocateInfo* ai = mut(std::get<1>(d->args));
  const uint64_t memId = idOfFake(*std::get<3>(d->args));
  VkBaseOutStructure* prev = reinterpret_cast<VkBaseOutStructure*>(ai);
  for (auto* p = prev->pNext; p; p = p->pNext) {
    if (p->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO) {
      auto* dd = reinterpret_cast<VkMemoryDedicatedAllocateInfo*>(p);
      dd->image = r.realOpt(dd->image), dd->buffer = r.realOpt(dd->buffer);
    } else if (p->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO && !r.bdaEnabled) {
      prev->pNext = p->pNext;  // device addresses need a feature this device may not have
      continue;
    }
    prev = p;
  }
  const Need& n = r.need[memId];
  const uint32_t srcType = ai->memoryTypeIndex;
  const uint64_t captured = ai->allocationSize;
  ai->allocationSize = std::max(captured, n.size);
  if (ai->allocationSize > captured) ++r.memGrown;
  bool sat = true;
  ai->memoryTypeIndex = r.pickType(r.srcFlags(srcType), n.bits, n.hostWrite, &sat);
  if (!sat) r.warnOnce("memhost", "no host-visible memory type fits a memory that receives data: its contents cannot be written");
  if (ai->memoryTypeIndex != srcType) ++r.memRemapped;
  VkDeviceMemory mem{};
  const VkResult res = r.fn<id::vkAllocateMemory>()(r.device, ai, nullptr, &mem);
  if (res < 0) {
    r.note("memory %llu: allocating %llu bytes (captured %llu, type %u -> %u) failed (%d)", (unsigned long long)memId,
           (unsigned long long)ai->allocationSize, (unsigned long long)captured, srcType, ai->memoryTypeIndex, int(res));
    return r.fail(c, res);
  }
  MemRec m;
  m.h = mem, m.size = ai->allocationSize;
  const VkMemoryPropertyFlags f = r.mp.memoryTypes[ai->memoryTypeIndex].propertyFlags;
  m.coherent = (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
  if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    void* p = nullptr;
    if (r.fn<id::vkMapMemory>()(r.device, mem, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS) {
      m.map = static_cast<uint8_t*>(p);
      std::memset(m.map, 0, m.size);  // a new memory reads as zero in the capture
    }
  }
  r.mems[memId] = m;
  r.reg(memId, mem);
}

template <>
void run<id::vkFreeMemory>(R& r, const Call& c) {
  auto d = decode<id::vkFreeMemory>(c);
  if (!d->ok) return r.corrupt(c);
  const uint64_t id = idOfFake(std::get<1>(d->args));
  auto it = r.mems.find(id);
  if (it == r.mems.end()) return;
  r.fn<id::vkFreeMemory>()(r.device, it->second.h, nullptr);  // implicitly unmaps
  r.mems.erase(it);
  r.h.erase(id);
}

template <uint16_t ID>
void bindMemory(R& r, const Call& c) {
  auto d = decode<ID>(c);
  if (!d->ok) return r.corrupt(c);
  r.missing = false;
  const uint64_t resId = idOfFake(std::get<1>(d->args));
  auto& a = d->args;
  r.mapTop(a);
  if (r.missing) return void(++r.skippedMissing);
  auto q = r.reqs.find(resId);
  if (q != r.reqs.end() && q->second.ok && std::get<3>(a) % q->second.align) {
    ++r.misaligned;
    r.warnOnce("misalign", "a bind offset is not aligned for this driver: aligned up (contents written at the captured offset may not match)");
    std::get<3>(a) = alignUp(std::get<3>(a), q->second.align);
  }
  const VkResult res = std::apply(r.fn<ID>(), a);
  if (res < 0) r.fail(c, res);
}
template <>
void run<id::vkBindBufferMemory>(R& r, const Call& c) {
  bindMemory<id::vkBindBufferMemory>(r, c);
}
template <>
void run<id::vkBindImageMemory>(R& r, const Call& c) {
  bindMemory<id::vkBindImageMemory>(r, c);
}

template <>
void run<id::vkUpdateDescriptorSets>(R& r, const Call& c) {
  auto d = decode<id::vkUpdateDescriptorSets>(c);
  if (!d->ok) return r.corrupt(c);
  auto& [dev, wn, writes, cn, copies] = d->args;
  (void)dev;
  std::vector<VkWriteDescriptorSet> good;
  for (uint32_t i = 0; i < wn; ++i) {
    VkWriteDescriptorSet w = writes[i];
    r.missing = false;
    w.dstSet = r.real(w.dstSet);
    if (w.pImageInfo)
      for (uint32_t k = 0; k < w.descriptorCount; ++k) {
        auto* ii = mut(w.pImageInfo) + k;
        ii->sampler = r.real(ii->sampler), ii->imageView = r.real(ii->imageView);
      }
    if (w.pBufferInfo)
      for (uint32_t k = 0; k < w.descriptorCount; ++k) mut(w.pBufferInfo)[k].buffer = r.real(w.pBufferInfo[k].buffer);
    if (w.pTexelBufferView) r.realArr(w.pTexelBufferView, w.descriptorCount);
    if (!r.missing && w.dstSet) good.push_back(w);
    else ++r.skippedMissing;
  }
  std::vector<VkCopyDescriptorSet> goodCopies;
  for (uint32_t i = 0; i < cn; ++i) {
    VkCopyDescriptorSet x = copies[i];
    r.missing = false;
    x.srcSet = r.real(x.srcSet), x.dstSet = r.real(x.dstSet);
    if (!r.missing) goodCopies.push_back(x);
    else ++r.skippedMissing;
  }
  r.fn<id::vkUpdateDescriptorSets>()(r.device, uint32_t(good.size()), good.data(), uint32_t(goodCopies.size()), goodCopies.data());
}

template <class I>
struct SubmitOf;
template <>
struct SubmitOf<VkSubmitInfo> {
  static constexpr uint16_t kId = id::vkQueueSubmit;
  static void strip(R& r, VkSubmitInfo& s, std::vector<VkCommandBuffer>& order) {
    s.waitSemaphoreCount = s.signalSemaphoreCount = 0;
    s.pNext = nullptr;
    uint32_t kept = 0;
    for (uint32_t i = 0; i < s.commandBufferCount; ++i) {
      const VkCommandBuffer cb = r.real(s.pCommandBuffers[i]);
      if (r.dropped(cb)) continue;
      mut(s.pCommandBuffers)[kept++] = cb;
      order.push_back(cb);
    }
    s.commandBufferCount = kept;
  }
};
template <>
struct SubmitOf<VkSubmitInfo2> {
  static constexpr uint16_t kId = id::vkQueueSubmit2;
  static void strip(R& r, VkSubmitInfo2& s, std::vector<VkCommandBuffer>& order) {
    s.waitSemaphoreInfoCount = s.signalSemaphoreInfoCount = 0;
    s.pNext = nullptr;
    uint32_t kept = 0;
    for (uint32_t i = 0; i < s.commandBufferInfoCount; ++i) {
      VkCommandBufferSubmitInfo ci = s.pCommandBufferInfos[i];
      ci.commandBuffer = r.real(ci.commandBuffer);
      if (r.dropped(ci.commandBuffer)) continue;
      mut(s.pCommandBufferInfos)[kept++] = ci;
      order.push_back(ci.commandBuffer);
    }
    s.commandBufferInfoCount = kept;
  }
};

template <class I>
void submit(R& r, const Call& c, const std::tuple<VkQueue, uint32_t, const I*, VkFence>& args) {
  constexpr uint16_t ID = SubmitOf<I>::kId;
  r.missing = false;
  VkQueue q = r.real(std::get<0>(args));
  VkFence fence = r.real(std::get<3>(args));
  std::vector<VkCommandBuffer> order;
  for (uint32_t i = 0; i < std::get<1>(args); ++i) SubmitOf<I>::strip(r, mut(std::get<2>(args))[i], order);
  if (r.missing || !q) return void(++r.skippedMissing);
  if (fence) r.fn<id::vkResetFences>()(r.device, 1, &fence);
  const VkResult res = r.fn<ID>()(q, std::get<1>(args), std::get<2>(args), fence);
  if (res < 0) r.fail(c, res);
  r.fn<id::vkQueueWaitIdle>()(q);
  r.collectDumps(order, c.frame, r.submitInFrame);
  ++r.submitInFrame;
}
template <>
void run<id::vkQueueSubmit>(R& r, const Call& c) {
  auto d = decode<id::vkQueueSubmit>(c);
  if (!d->ok) return r.corrupt(c);
  submit<VkSubmitInfo>(r, c, d->args);
}
template <>
void run<id::vkQueueSubmit2>(R& r, const Call& c) {
  auto d = decode<id::vkQueueSubmit2>(c);
  if (!d->ok) return r.corrupt(c);
  submit<VkSubmitInfo2>(r, c, d->args);
}

// An event wait is a pipeline barrier here (the events are never signalled: vkCmdSetEvent is dropped).
template <>
void run<id::vkCmdWaitEvents>(R& r, const Call& c) {
  auto d = decode<id::vkCmdWaitEvents>(c);
  if (!d->ok) return r.corrupt(c);
  auto& [cb, n, events, src, dst, mn, mem, bn, buf, in, img] = d->args;
  (void)n, (void)events;
  r.missing = false;
  const uint64_t cbId = idOfFake(cb);
  cb = r.real(cb);
  r.fixBufBarriers(buf, bn);
  r.fixImgBarriers(img, in);
  if (r.missing) return r.taint(cbId), void(++r.skippedMissing);
  r.fn<id::vkCmdPipelineBarrier>()(cb, src ? src : VkPipelineStageFlags(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT),
                                   dst ? dst : VkPipelineStageFlags(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT), 0, mn, mem, bn,
                                   buf, in, img);
}
template <>
void run<id::vkCmdWaitEvents2>(R& r, const Call& c) {
  auto d = decode<id::vkCmdWaitEvents2>(c);
  if (!d->ok) return r.corrupt(c);
  auto& [cb, n, events, deps] = d->args;
  (void)events;
  r.missing = false;
  const uint64_t cbId = idOfFake(cb);
  cb = r.real(cb);
  for (uint32_t i = 0; i < n; ++i) r.fixDep(deps + i);
  if (r.missing) return r.taint(cbId), void(++r.skippedMissing);
  for (uint32_t i = 0; i < n; ++i) r.fn<id::vkCmdPipelineBarrier2>()(cb, deps + i);
}

void R::handle(const Call& c) {
  ++calls;
  if (c.id != id::vkCreateInstance && !instanceMade && c.id >= 0x0100 && !ensureInstance()) std::exit(1);
  switch (c.id) {
#define VPR_X(name, num) \
  case id::name: run<id::name>(*this, c); break;
    VPCAP_VKCALLS(VPR_X)
#undef VPR_X
    case id::vpcapMemoryData: onMemoryData(c); break;
    case id::vpcapImageData: onImageData(c); break;
    case id::vpcapFrameEnd:
      writeIndex();
      break;
    default: ++skipped["unknown call id"]; break;
  }
}

int usage() {
  std::fprintf(stderr,
               "usage: vpreplay <file.vpcap> [--icd vpvk|system] [--lib PATH] [--icd-file ICD.json] [--gpu N]\n"
               "                [--dump DIR] [--dump-at pass|draw] [--raw] [--recapture FILE.vpcap] [--strict] [--allow-truncated]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  R r;
  Options& o = r.opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    std::string* dst = a == "--icd" ? &o.icd : a == "--lib" ? &o.lib : a == "--icd-file" ? &o.icdFile : a == "--dump" ? &o.dumpDir
                       : a == "--dump-at" ? &o.dumpAt : a == "--recapture" ? &o.recapture : nullptr;
    if (dst) {
      if (i + 1 >= argc) return usage();
      *dst = argv[++i];
    } else if (a == "--gpu") {
      if (i + 1 >= argc) return usage();
      o.gpu = std::atoi(argv[++i]);
    } else if (a == "--strict") {
      o.strict = true;
    } else if (a == "--raw") {
      o.raw = true;
    } else if (a == "--allow-truncated") {
      o.allowTruncated = true;
    } else if (a.rfind("--", 0) != 0 && o.file.empty()) {
      o.file = a;
    } else {
      return usage();
    }
  }
  if (o.file.empty() || (o.icd != "vpvk" && o.icd != "system") || (o.dumpAt != "pass" && o.dumpAt != "draw")) return usage();
  if (!o.dumpDir.empty()) std::filesystem::create_directories(o.dumpDir);
  Reader rd;
  std::string err;
  if (!rd.open(o.file, &err)) {
    r.note("%s: %s", o.file.c_str(), err.c_str());
    return 1;
  }
  if (!r.load()) return 1;
  Call c;
  while (rd.next(c)) {
    if (c.section == Section::Frame && c.frame != r.curFrame) r.curFrame = c.frame, r.submitInFrame = 0;
    if (c.section == Section::Frame && !r.recaptureArmed && !o.recapture.empty()) r.armRecapture();
    r.handle(c);
  }
  if (!rd.error().empty()) {
    r.note("%s", rd.error().c_str());
    ++r.errors;
  }
  // A capture without its END block was cut short (or damaged): an error, unless that is what --allow-truncated is for.
  bool badFile = !rd.complete();
  if (badFile) {
    const bool cut = rd.error().empty() || rd.error().rfind("truncated", 0) == 0;
    r.note("%s: capture is incomplete (%s)%s", o.file.c_str(), rd.error().empty() ? "no END block" : rd.error().c_str(),
           o.allowTruncated && cut ? ", accepted (--allow-truncated)" : "");
    badFile = !(o.allowTruncated && cut);
  }
  r.writeIndex();
  {  // what the file says a frame is and whether it is exact (META + END)
    auto key = [&](const char* k) {
      const std::string m = rd.meta(), pat = std::string("\n") + k + "=";
      const size_t at = ("\n" + m).find(pat);
      if (at == std::string::npos) return std::string("unknown");
      const size_t b = at + pat.size() - 1, e = m.find('\n', b);
      return m.substr(b, e == std::string::npos ? std::string::npos : e - b);
    };
    std::fprintf(stderr, "vpreplay: capture frame_unit=%s frames=%s invalid=%s reason=%s unhooked_used=%s other_device_calls=%s img_deferred=%s\n",
                 key("frame_unit").c_str(), key("frames").c_str(), key("invalid").c_str(), key("reason").c_str(),
                 key("unhooked_used").c_str(), key("other_device_calls").c_str(), key("img_deferred").c_str());
  }
  if (r.device) {
    r.fn<id::vkDeviceWaitIdle>()(r.device);
    if (auto d = r.gdpa(r.device, "vkDestroyDevice")) reinterpret_cast<PFN_vkDestroyDevice>(d)(r.device, nullptr);
  }
  if (r.instance)
    if (auto d = r.gipa(r.instance, "vkDestroyInstance")) reinterpret_cast<PFN_vkDestroyInstance>(d)(r.instance, nullptr);
  std::fprintf(stderr,
               "vpreplay: calls=%llu errors=%llu unknown_ids=%llu skipped_missing=%llu unsupported=%llu | memory writes=%llu bytes=%llu lost=%llu "
               "grown=%llu remapped=%llu misaligned=%llu | image uploads=%llu failed=%llu skipped=%llu | cbs skipped=%llu | dumps png=%llu raw=%llu\n",
               (unsigned long long)r.calls, (unsigned long long)r.errors, (unsigned long long)r.unknownIds,
               (unsigned long long)r.skippedMissing, (unsigned long long)r.unsupported, (unsigned long long)r.memWrites,
               (unsigned long long)r.memBytes, (unsigned long long)r.memLost, (unsigned long long)r.memGrown, (unsigned long long)r.memRemapped,
               (unsigned long long)r.misaligned, (unsigned long long)r.imgUploads, (unsigned long long)r.imgUploadFail,
               (unsigned long long)r.imgUploadSkipped, (unsigned long long)r.cbsSkipped, (unsigned long long)r.dumpFiles, (unsigned long long)r.dumpRaw);
  for (const auto& [k, n] : r.skipped) std::fprintf(stderr, "vpreplay:   dropped %-28s %llu\n", k.c_str(), (unsigned long long)n);
  for (const auto& [k, n] : r.failed) std::fprintf(stderr, "vpreplay:   failed  %-28s %llu\n", k.c_str(), (unsigned long long)n);
  return badFile || (o.strict && (r.errors || r.skippedMissing || r.imgUploadSkipped || r.cbsSkipped)) ? 1 : 0;
}
