// Tests of the second audit of gpu/vulkan (notas/auditoria-vk2.md): foveation / config inputs, PNG + JSON readers of the replay
// tools, the pass-dump counters, the translate pool at shutdown, vpreplay on a hostile capture, the capture's END text and file
// names. Usage: vpvk_audit2 <group|all> [path to libVPVulkan] [path to vpreplay].
// The sources it needs are compiled in (foveation, passdump, vpcap); the driver itself is dlopen'ed like a host does it.
#include <dlfcn.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <zlib.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "foveation.h"
#include "passdump.h"
#include "translate_pool.h"
#include "vpimage.h"
#include "vpjson.h"
#include "vpvk/config.h"
#include "vpvk/vpcap.h"
#include "vpvk/vpcap_calls.h"

// passdump.cpp's links into the driver, stubbed (the group "passdump" only needs the counters).
namespace vpvk {
std::atomic<uint64_t> gHideDropped{0};
void log(const std::string&) {}
void logf(const char*, ...) {}
}  // namespace vpvk

namespace {

int failures = 0;
const char* gLib = nullptr;
const char* gReplay = nullptr;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                   \
    }                                                               \
  } while (0)

// Runs `fn` in a child process: 0 = it returned and exited 0, 1 = it crashed or failed, 2 = it did not finish in `ms`.
// `capMb` > 0: the child may map that much more address space than it has now (a hostile size must not eat the machine).
int inChild(const std::function<int()>& fn, int ms = 20000, int capMb = 0) {
  std::fflush(nullptr);
  const pid_t pid = fork();
  if (pid == 0) {
    if (capMb > 0) {
      unsigned long pages = 0;
      if (std::FILE* f = std::fopen("/proc/self/statm", "r")) {
        if (std::fscanf(f, "%lu", &pages) != 1) pages = 0;
        std::fclose(f);
      }
      const rlim_t cap = rlim_t(pages) * rlim_t(sysconf(_SC_PAGESIZE)) + rlim_t(capMb) * 1024 * 1024;
      const rlimit rl{cap, cap};
      setrlimit(RLIMIT_AS, &rl);
    }
    _exit(fn());
  }
  for (int waited = 0; waited < ms; waited += 10) {
    int st = 0;
    if (waitpid(pid, &st, WNOHANG) == pid) return WIFEXITED(st) && WEXITSTATUS(st) == 0 ? 0 : 1;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  kill(pid, SIGKILL);
  waitpid(pid, nullptr, 0);
  return 2;
}

// ---- foveation: NaN / infinite settings and tangents ----
bool finite01(const std::vector<float>& q) {
  for (float v : q)
    if (!std::isfinite(v) || v < 0.1f - 1e-6f || v > 1.0f + 1e-6f) return false;
  return true;
}
void testFovNan() {
  using namespace vpvk::fov;
  const float nan = std::nanf("");
  for (int mode : {1, 2, 3}) {
    const Shape s = shapeOf(mode, nan, nan, nan);
    CHECK(std::isfinite(s.sharp) && std::isfinite(s.periphery) && std::isfinite(s.height));
    Eyes e;
    e.known = true;
    e.tangents[0] = {INFINITY, INFINITY, 1, 1};
    e.tangents[1] = {nan, 1, 1, nan};
    const auto m = buildRateMap(1000, 1000, 2, s, e);
    CHECK(m != nullptr);
    if (!m) continue;
    CHECK(finite01(m->qx) && finite01(m->qy));
    CHECK(std::isfinite(m->keptFraction) && m->keptFraction > 0.0f && m->keptFraction <= 1.0f);
  }
  const Center c = opticalCenter({INFINITY, INFINITY, INFINITY, 1});
  CHECK(std::isfinite(c.x) && std::isfinite(c.y));
  float q[8];
  axisQuality(q, 8, nan, nan, nan, nan);
  CHECK(finite01(std::vector<float>(q, q + 8)));
  // finite inputs are unchanged: the preset M2 at the centre keeps full rate in the middle zone
  const Shape ok = shapeOf(3, 0.5f, 0.25f, 0.1f);
  CHECK(ok.sharp == 0.5f && ok.periphery == 0.25f && std::fabs(ok.height - 0.1f) < 1e-6f);
}

// ---- config: the setters keep what the host passes finite ----
template <class F>
F sym(void* lib, const char* name) {
  return reinterpret_cast<F>(dlsym(lib, name));
}
void testConfigNan() {
  void* lib = dlopen(gLib, RTLD_NOW | RTLD_LOCAL);
  CHECK(lib != nullptr);
  if (!lib) return;
  auto setFov = sym<void (*)(const vpvk_foveation*)>(lib, "vpvk_config_set_foveation");
  auto setTaa = sym<void (*)(const vpvk_vpktaa*)>(lib, "vpvk_config_set_vpktaa");
  auto setScale = sym<void (*)(float)>(lib, "vpvk_config_set_render_scale");
  auto get = sym<void (*)(vpvk_config*)>(lib, "vpvk_config_get");
  CHECK(setFov && setTaa && setScale && get);
  if (!setFov || !setTaa || !setScale || !get) return;
  const float nan = std::nanf("");
  vpvk_foveation f{VPVK_FOVEATION_CUSTOM, nan, nan, nan};
  setFov(&f);
  const vpvk_vpktaa t{1, nan, nan};
  setTaa(&t);
  setScale(nan);
  vpvk_config c{};
  get(&c);
  CHECK(std::isfinite(c.foveation.sharp_zone) && c.foveation.sharp_zone >= 0.05f && c.foveation.sharp_zone <= 1.0f);
  CHECK(std::isfinite(c.foveation.periphery) && c.foveation.periphery >= 0.1f && c.foveation.periphery <= 1.0f);
  CHECK(std::isfinite(c.foveation.height) && std::fabs(c.foveation.height) <= 0.5f);
  CHECK(std::isfinite(c.vpktaa.percent) && c.vpktaa.percent >= 0.0f && c.vpktaa.percent <= 100.0f);
  CHECK(std::isfinite(c.vpktaa.sharpness) && c.vpktaa.sharpness >= 0.0f && c.vpktaa.sharpness <= 1.0f);
  CHECK(std::isfinite(c.render_scale) && c.render_scale >= 0.5f && c.render_scale <= 1.5f);
  // finite values pass through
  vpvk_foveation g{VPVK_FOVEATION_CUSTOM, 0.6f, 0.3f, 0.1f};
  setFov(&g);
  setScale(1.25f);
  get(&c);
  CHECK(c.foveation.sharp_zone == 0.6f && c.foveation.periphery == 0.3f && std::fabs(c.foveation.height - 0.1f) < 1e-6f && c.render_scale == 1.25f);
  setFov(&f);  // leave nothing odd behind
  dlclose(lib);
}

// ---- the image readers of the replay tools ----
std::vector<uint8_t> hostilePng(uint32_t w, uint32_t h) {
  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<uint8_t> ihdr;
  vpimg::putBE32(ihdr, w), vpimg::putBE32(ihdr, h);
  for (uint8_t b : {8, 6, 0, 0, 0}) ihdr.push_back(b);
  vpimg::pngChunk(out, "IHDR", ihdr);
  const uint8_t raw[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uLongf zn = compressBound(sizeof raw);
  std::vector<uint8_t> z(zn);
  compress2(z.data(), &zn, raw, sizeof raw, 6);
  z.resize(zn);
  vpimg::pngChunk(out, "IDAT", z);
  vpimg::pngChunk(out, "IEND", {});
  return out;
}
void testPngHostile() {
  for (const auto& wh : {std::pair<uint32_t, uint32_t>{0xFFFFFFFFu, 0xFFFFFFFFu}, {65536u, 65536u}, {0x7FFFFFFFu, 3u}, {1u << 20, 1u << 20}, {40000u, 40000u}}) {
    const int r = inChild([&] {
      vpimg::Image im;
      std::string err;
      try {
        return vpimg::pngDecode(hostilePng(wh.first, wh.second), im, &err) ? 1 : 0;  // must be refused, not thrown, not decoded
      } catch (...) {
        return 1;
      }
    }, 20000, 1500);
    CHECK(r == 0);
  }
  vpimg::Image a;  // a real image still round trips
  a.w = 3, a.h = 2, a.bits = 8, a.px.assign(24, 0);
  for (size_t i = 0; i < a.px.size(); ++i) a.px[i] = uint16_t(i * 9);
  vpimg::Image b;
  CHECK(vpimg::pngDecode(vpimg::pngEncode(a), b) && b.w == 3 && b.h == 2 && b.px == a.px);
}

void testJson() {
  const int r = inChild([] {  // 300000 nested arrays: refused, not a stack overflow
    const std::string deep(300000, '[');
    vpjson::Value v;
    vpjson::Parser p(deep);
    return p.parse(v) ? 1 : 0;
  });
  CHECK(r == 0);
  const int r2 = inChild([] {
    std::string deep;
    for (int i = 0; i < 100000; ++i) deep += "{\"a\":";
    deep += "1";
    vpjson::Value v;
    vpjson::Parser p(deep);
    return p.parse(v) ? 1 : 0;
  });
  CHECK(r2 == 0);
  {  // escapes the writer's own data can hold, and the ones any JSON can
    const std::string text = "{\"s\":\"a\\rb\\bc\\fd\\/e\\u00e9\\u20ac\\ud83d\\ude00\",\"n\":[1,2.5,true,null]}";
    vpjson::Value v;
    vpjson::Parser p(text);
    CHECK(p.parse(v));
    CHECK(v.text("s") == "a\rb\bc\fd/e\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80");
    const vpjson::Value* n = v.get("n");
    CHECK(n && n->arr.size() == 4 && n->arr[1].num == 2.5);
  }
  {  // a name with control characters round trips through escape()
    const std::string name = std::string("x\r\n\t\"\\\x01y");
    vpjson::Value v;
    const std::string doc = "{\"k\":\"" + vpjson::escape(name) + "\"}";
    vpjson::Parser p(doc);
    CHECK(p.parse(v) && v.text("k") == name);
  }
  {
    vpjson::Value v;
    const std::string bad = "{\"k\":\"\\uZZZZ\"}";
    vpjson::Parser p(bad);
    CHECK(!p.parse(v));
  }
}

// ---- pass dump counters used from two devices' submit threads ----
void testPassDumpRace() {
  vpvk::PassDump p;
  p.on = true, p.start = 0, p.frames = 1;
  constexpr int kThreads = 4, kIters = 1000000;
  std::atomic<bool> go{false};
  std::vector<std::thread> th;
  for (int t = 0; t < kThreads; ++t)
    th.emplace_back([&] {
      while (!go) std::this_thread::yield();
      for (int i = 0; i < kIters; ++i) vpvk::passDumpNext(p, 1);
    });
  go = true;
  for (auto& t : th) t.join();
  CHECK(p.seen == uint64_t(kThreads) * kIters);
}

// ---- translate pool: a thread waiting for a job that never ran is released when the pool goes ----
void testPoolShutdown() {
  const int r = inChild([] {
    auto* pool = new vpvk::TranslatePool(1);
    std::atomic<bool> release{false}, waited{false};
    pool->submit([&] {
      while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
    auto job = pool->submit([] {});
    std::thread waiter([&, job] {
      std::unique_lock<std::mutex> l(job->m);
      job->cv.wait(l, [&] { return job->finished(); });
      waited = job->state.load() == 3 && !job->done();  // dropped, not "done": wait() reports false for it
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::thread rel([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      release = true;
    });
    delete pool;
    rel.join();
    waiter.join();
    return waited ? 0 : 1;
  }, 5000);
  CHECK(r == 0);
  vpvk::TranslatePool pool(1);  // a job that ran: wait() true; one that is still queued past the timeout: false
  std::atomic<bool> release{false};
  auto ran = pool.submit([] {});
  CHECK(pool.wait(ran, UINT32_MAX) && ran->done());
  pool.submit([&] {
    while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  });
  auto late = pool.submit([] {});
  CHECK(!pool.wait(late, 20));
  release = true;
  CHECK(pool.wait(late, UINT32_MAX));
}

// ---- a hostile capture in vpreplay ----
struct FakeIds final : vpcap::HandleIds {
  uint64_t idOf(const void* h) override { return reinterpret_cast<uintptr_t>(h); }
};
template <uint16_t ID, class... A>
std::vector<uint8_t> record(A... args) {
  vpcap::ArgsOf<ID> a{args...};
  FakeIds ids;
  vpcap::Cod c(&ids);
  vpcap::CallDef<ID>::cod(c, a);
  std::vector<uint8_t> v(sizeof(vpcap::CallHeader) + c.out.size());
  const vpcap::CallHeader h{ID, 0, 1, uint32_t(c.out.size())};
  std::memcpy(v.data(), &h, sizeof h);
  if (!c.out.empty()) std::memcpy(v.data() + sizeof h, c.out.data(), c.out.size());
  return v;
}
struct ImgRecord {
  uint32_t mip, layer, width, height;
  uint64_t total, offset, size;
};
// A capture of an instance, a device and an RGBA8 4x4 image (1 mip, 1 layer; id 4) whose FRAME holds one vpcapImageData record.
std::string writeHostileCapture(const std::string& path, const ImgRecord& r) {
  vpcap::Writer w;
  if (!w.open(path)) return "cannot write";
  w.meta("format=VPCAP\nversion=1\nframe_unit=submit\ninstance_in_state=1\n");
  w.begin(vpcap::Section::State);
  auto add = [&](const std::vector<uint8_t>& rec) { w.append(rec.data(), rec.size()); };
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  VkInstance inst = reinterpret_cast<VkInstance>(uintptr_t(1));
  const VkAllocationCallbacks* none = nullptr;
  add(record<vpcap::id::vkCreateInstance>(&ici, none, &inst));
  float prio = 1;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueCount = 1, qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &qci;
  VkDevice dev = reinterpret_cast<VkDevice>(uintptr_t(2));
  add(record<vpcap::id::vkCreateDevice>(reinterpret_cast<VkPhysicalDevice>(uintptr_t(3)), &dci, none, &dev));
  VkImageCreateInfo imc{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imc.imageType = VK_IMAGE_TYPE_2D, imc.format = VK_FORMAT_R8G8B8A8_UNORM, imc.extent = {4, 4, 1}, imc.mipLevels = 1, imc.arrayLayers = 1;
  imc.samples = VK_SAMPLE_COUNT_1_BIT, imc.tiling = VK_IMAGE_TILING_OPTIMAL;
  imc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, imc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImage img = reinterpret_cast<VkImage>(uintptr_t(4));
  add(record<vpcap::id::vkCreateImage>(dev, &imc, none, &img));
  w.begin(vpcap::Section::Frame);
  std::vector<uint8_t> p(vpcap::kImageDataHeader + r.size, 0xAB);
  const uint64_t image = 4;
  const uint32_t f[6] = {r.mip, r.layer, 1, r.width, r.height, 1};  // mip, layer, aspect COLOR, w, h, d
  const uint32_t store = 0;
  std::memcpy(p.data(), &image, 8);
  std::memcpy(p.data() + 8, f, sizeof f);
  std::memcpy(p.data() + 32, &store, 4);
  std::memcpy(p.data() + 40, &r.total, 8);
  std::memcpy(p.data() + 48, &r.offset, 8);
  std::memcpy(p.data() + 56, &r.size, 8);
  auto addCall = [&](uint16_t id, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> v(sizeof(vpcap::CallHeader) + payload.size());
    const vpcap::CallHeader h{id, 0, 1, uint32_t(payload.size())};
    std::memcpy(v.data(), &h, sizeof h);
    if (!payload.empty()) std::memcpy(v.data() + sizeof h, payload.data(), payload.size());
    add(v);
  };
  addCall(vpcap::id::vpcapImageData, p);
  addCall(vpcap::id::vpcapFrameEnd, std::vector<uint8_t>(16, 0));
  return w.close("frames=1\nend=frames_done\n") ? "" : "close failed";
}
void testReplayImageData() {
  if (!gReplay) {
    std::puts("  skipped: needs the vpreplay path");
    return;
  }
  struct Case {
    const char* name;
    ImgRecord r;
    bool counted;  // vpreplay must say it skipped (or failed) the upload
  };
  // A: a total no image has (the chunks would be gathered in a buffer of that size); B: offset + size wraps around 2^64;
  // C: a chunk that ends past the total; D: a mip level, a layer and a region the image does not have (their size is the image's).
  const Case cases[] = {{"huge_total", {0, 0, 4, 4, uint64_t(1) << 62, 0, 16}, true},
                        {"wrapping_offset", {0, 0, 4, 4, 64, ~uint64_t(0) - 7, 16}, true},
                        {"offset_past_total", {0, 0, 4, 4, 64, 60, 16}, true},
                        {"mip_missing", {5, 0, 4, 4, 64, 0, 64}, true},
                        {"layer_missing", {0, 3, 4, 4, 64, 0, 64}, true},
                        {"region_larger_than_mip", {0, 0, 8, 8, 256, 0, 256}, true}};
  const std::string dir = (std::filesystem::temp_directory_path() / ("audit_vk2_replay_" + std::to_string(getpid()))).string();
  std::filesystem::create_directories(dir);
  for (const Case& c : cases) {
    const std::string cap = dir + "/" + c.name + ".vpcap", log = dir + "/" + c.name + ".log";
    CHECK(writeHostileCapture(cap, c.r).empty());
    const std::string cmd = std::string("'") + gReplay + "' '" + cap + "' --icd vpvk --lib '" + gLib + "' > '" + log + "' 2>&1";
    const int st = std::system(cmd.c_str());
    std::string text;
    if (std::FILE* f = std::fopen(log.c_str(), "rb")) {
      char buf[4096];
      size_t n;
      while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
      std::fclose(f);
    }
    const bool clean = WIFEXITED(st) && WEXITSTATUS(st) == 0 && text.find("Sanitizer") == std::string::npos && text.find("terminate") == std::string::npos;
    const bool counted = text.find("image uploads=0 failed=0 skipped=1 ") != std::string::npos;
    if (!clean || (c.counted && !counted)) {
      std::printf("  vpreplay %s: status 0x%x\n", c.name, st);
      for (size_t at = 0; at < text.size();) {  // its own lines (the driver's log is long)
        const size_t e = std::min(text.find('\n', at), text.size());
        if (text.compare(at, 9, "vpreplay:") == 0 || text.find("Error", at) < e || text.find("what()", at) < e) std::printf("    %s\n", text.substr(at, std::min<size_t>(e - at, 300)).c_str());
        at = e + 1;
      }
    }
    CHECK(clean);
    CHECK(!c.counted || counted);
  }
  std::filesystem::remove_all(dir);
}

// ---- the capture's END text and the names of its files ----
struct Api {
  PFN_vkGetInstanceProcAddr gipa = nullptr;
  VkInstance inst = nullptr;
  VkDevice dev = nullptr;
  VkQueue queue = nullptr;
  VkCommandPool pool = nullptr;
  std::vector<VkCommandBuffer> cbs;
  size_t next = 0;
  PFN_vkBeginCommandBuffer begin = nullptr;
  PFN_vkEndCommandBuffer end = nullptr;
  PFN_vkQueueSubmit submitFn = nullptr;
  template <class F>
  F fn(const char* n) const {
    return reinterpret_cast<F>(gipa(inst, n));
  }
  bool init(void* lib, uint32_t cbCount) {
    gipa = sym<PFN_vkGetInstanceProcAddr>(lib, "vkGetInstanceProcAddr");
    if (!gipa) return false;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "audit2", app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"))(&ici, nullptr, &inst) != VK_SUCCESS) return false;
    VkPhysicalDevice pd = nullptr;
    uint32_t n = 1;
    fn<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices")(inst, &n, &pd);
    float prio = 1;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueCount = 1, q.pQueuePriorities = &prio;
    VkDeviceCreateInfo d{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    d.queueCreateInfoCount = 1, d.pQueueCreateInfos = &q;
    if (fn<PFN_vkCreateDevice>("vkCreateDevice")(pd, &d, nullptr, &dev) != VK_SUCCESS) return false;
    fn<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(dev, 0, 0, &queue);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    if (fn<PFN_vkCreateCommandPool>("vkCreateCommandPool")(dev, &pi, nullptr, &pool) != VK_SUCCESS) return false;
    cbs.resize(cbCount);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool, ai.commandBufferCount = cbCount, ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    if (fn<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(dev, &ai, cbs.data()) != VK_SUCCESS) return false;
    begin = fn<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer"), end = fn<PFN_vkEndCommandBuffer>("vkEndCommandBuffer");
    submitFn = fn<PFN_vkQueueSubmit>("vkQueueSubmit");
    return true;
  }
  VkCommandBuffer open() {
    VkCommandBuffer cb = cbs[next++];
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin(cb, &bi);
    return cb;
  }
  void submit(VkCommandBuffer cb) {
    end(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &cb;
    submitFn(queue, 1, &si, nullptr);
  }
  void destroy() {
    if (dev) fn<PFN_vkDestroyDevice>("vkDestroyDevice")(dev, nullptr);
    if (inst) fn<PFN_vkDestroyInstance>("vkDestroyInstance")(inst, nullptr);
    dev = nullptr, inst = nullptr;
  }
};
std::string readEnd(const std::string& path, bool* ok = nullptr) {
  vpcap::Reader rd;
  vpcap::Call c;
  if (!rd.open(path)) return {};
  while (rd.next(c)) {
  }
  if (ok) *ok = rd.complete();
  return rd.meta();
}

// Calls with no recording hook, made inside a capture: the END text names every one of them, whatever its length.
void testCaptureEndText() {
  void* lib = dlopen(gLib, RTLD_NOW | RTLD_LOCAL);
  CHECK(lib != nullptr);
  if (!lib) return;
  auto track = sym<int (*)(int)>(lib, "vpvk_capture_track");
  auto request = sym<int (*)(uint32_t, const char*)>(lib, "vpvk_capture_request");
  auto state = sym<int (*)()>(lib, "vpvk_capture_state");
  CHECK(track && request && state);
  if (!track || !request || !state) return;
  const std::string path = (std::filesystem::temp_directory_path() / ("audit_vk2_end_" + std::to_string(getpid()) + ".vpcap")).string();
  std::remove(path.c_str());
  track(0);
  CHECK(track(1) == 0);
  Api a;
  CHECK(a.init(lib, 3));
  if (!a.dev) return;
  a.submit(a.open());
  CHECK(request(1, path.c_str()) == 1);
  a.submit(a.open());  // the boundary: the capture begins when it returns
  CHECK(state() == 2);
  VkCommandBuffer cb = a.open();
  int called = 0;
  auto count = [&](const char* name, auto&& call) {
    if (auto f = a.gipa(a.inst, name)) call(f), ++called;
  };
  using F = PFN_vkVoidFunction;
  const VkDevice dev = a.dev;
  count("vkCmdSetLogicOpEXT", [&](F f) { reinterpret_cast<PFN_vkCmdSetLogicOpEXT>(f)(cb, VK_LOGIC_OP_COPY); });
  count("vkCmdSetPatchControlPointsEXT", [&](F f) { reinterpret_cast<PFN_vkCmdSetPatchControlPointsEXT>(f)(cb, 3); });
  count("vkCmdSetDeviceMask", [&](F f) { reinterpret_cast<PFN_vkCmdSetDeviceMask>(f)(cb, 1); });
  count("vkCmdSetColorWriteMaskEXT", [&](F f) { reinterpret_cast<PFN_vkCmdSetColorWriteMaskEXT>(f)(cb, 0, 0, nullptr); });
  count("vkCmdDrawMultiEXT", [&](F f) { reinterpret_cast<PFN_vkCmdDrawMultiEXT>(f)(cb, 0, nullptr, 1, 0, sizeof(VkMultiDrawInfoEXT)); });
  count("vkCmdDrawMultiIndexedEXT", [&](F f) { reinterpret_cast<PFN_vkCmdDrawMultiIndexedEXT>(f)(cb, 0, nullptr, 1, 0, sizeof(VkMultiDrawIndexedInfoEXT), nullptr); });
  count("vkCmdSetRenderingAttachmentLocationsKHR", [&](F f) {
    VkRenderingAttachmentLocationInfoKHR li{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR};
    reinterpret_cast<PFN_vkCmdSetRenderingAttachmentLocationsKHR>(f)(cb, &li);
  });
  count("vkCmdSetRenderingInputAttachmentIndicesKHR", [&](F f) {
    VkRenderingInputAttachmentIndexInfoKHR ii{VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO_KHR};
    reinterpret_cast<PFN_vkCmdSetRenderingInputAttachmentIndicesKHR>(f)(cb, &ii);
  });
  count("vkTransitionImageLayoutEXT", [&](F f) { reinterpret_cast<PFN_vkTransitionImageLayoutEXT>(f)(dev, 0, nullptr); });
  count("vkCreatePipelineCache", [&](F f) {
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache pc = nullptr;
    reinterpret_cast<PFN_vkCreatePipelineCache>(f)(dev, &ci, nullptr, &pc);
    if (auto d = a.gipa(a.inst, "vkDestroyPipelineCache")) reinterpret_cast<PFN_vkDestroyPipelineCache>(d)(dev, pc, nullptr), ++called;
  });
  count("vkCreatePrivateDataSlot", [&](F f) {
    VkPrivateDataSlotCreateInfo ci{VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO};
    VkPrivateDataSlot slot = nullptr;
    reinterpret_cast<PFN_vkCreatePrivateDataSlot>(f)(dev, &ci, nullptr, &slot);
    if (auto s = a.gipa(a.inst, "vkSetPrivateData")) reinterpret_cast<PFN_vkSetPrivateData>(s)(dev, VK_OBJECT_TYPE_COMMAND_BUFFER, uint64_t(reinterpret_cast<uintptr_t>(cb)), slot, 7), ++called;
    if (auto d = a.gipa(a.inst, "vkDestroyPrivateDataSlot")) reinterpret_cast<PFN_vkDestroyPrivateDataSlot>(d)(dev, slot, nullptr), ++called;
  });
  count("vkCreateSamplerYcbcrConversion", [&](F f) {
    VkSamplerYcbcrConversionCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO};
    VkSamplerYcbcrConversion yc = nullptr;
    reinterpret_cast<PFN_vkCreateSamplerYcbcrConversion>(f)(dev, &ci, nullptr, &yc);
    if (auto d = a.gipa(a.inst, "vkDestroySamplerYcbcrConversion")) reinterpret_cast<PFN_vkDestroySamplerYcbcrConversion>(d)(dev, yc, nullptr), ++called;
  });
  count("vkWaitSemaphores", [&](F f) {
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    reinterpret_cast<PFN_vkWaitSemaphores>(f)(dev, &wi, 0);
  });
  count("vkCreateEvent", [&](F f) {
    VkEventCreateInfo ci{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
    VkEvent ev = nullptr;
    reinterpret_cast<PFN_vkCreateEvent>(f)(dev, &ci, nullptr, &ev);
    if (auto s = a.gipa(a.inst, "vkSetEvent")) reinterpret_cast<PFN_vkSetEvent>(s)(dev, ev), ++called;
    if (auto s = a.gipa(a.inst, "vkResetEvent")) reinterpret_cast<PFN_vkResetEvent>(s)(dev, ev), ++called;
  });
  std::printf("  unhooked calls made: %d\n", called);
  a.submit(cb);  // its one frame: the capture ends
  CHECK(state() == 0);
  track(0);
  a.destroy();
  bool complete = false;
  const std::string meta = readEnd(path, &complete);
  CHECK(complete);
  std::printf("  META + END text: %zu bytes\n", meta.size());
  CHECK(meta.find("\nend=frames_done\n") != std::string::npos);  // the last key of the END text
  CHECK(meta.find("\ninvalid=1\n") != std::string::npos && meta.find("\nreason=") != std::string::npos);
  CHECK(meta.find("unhooked_used_calls=") != std::string::npos);
  std::remove(path.c_str());
  dlclose(lib);
}

// Two captures into one directory within the same second: two files.
void testCaptureFileNames() {
  void* lib = dlopen(gLib, RTLD_NOW | RTLD_LOCAL);
  CHECK(lib != nullptr);
  if (!lib) return;
  auto track = sym<int (*)(int)>(lib, "vpvk_capture_track");
  auto request = sym<int (*)(uint32_t, const char*)>(lib, "vpvk_capture_request");
  CHECK(track && request);
  if (!track || !request) return;
  const std::string dir = (std::filesystem::temp_directory_path() / ("audit_vk2_dir_" + std::to_string(getpid()))).string();
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  track(0);
  CHECK(track(1) == 0);
  Api a;
  CHECK(a.init(lib, 6));
  if (!a.dev) return;
  a.submit(a.open());
  for (int i = 0; i < 2; ++i) {
    CHECK(request(1, (dir + "/").c_str()) == 1);
    a.submit(a.open());  // the boundary
    a.submit(a.open());  // its frame
  }
  track(0);
  a.destroy();
  int files = 0, complete = 0;
  for (auto& e : std::filesystem::directory_iterator(dir)) {
    ++files;
    bool ok = false;
    readEnd(e.path().string(), &ok);
    complete += ok;
  }
  CHECK(files == 2 && complete == 2);
  std::filesystem::remove_all(dir);
  dlclose(lib);
}

// One command buffer that binds the same big descriptor set 20000 times: the submit of a capture resolves the set once, not 20000 times.
void testCaptureResolve() {
  void* lib = dlopen(gLib, RTLD_NOW | RTLD_LOCAL);
  CHECK(lib != nullptr);
  if (!lib) return;
  auto track = sym<int (*)(int)>(lib, "vpvk_capture_track");
  auto request = sym<int (*)(uint32_t, const char*)>(lib, "vpvk_capture_request");
  auto state = sym<int (*)()>(lib, "vpvk_capture_state");
  CHECK(track && request && state);
  if (!track || !request || !state) return;
  const std::string path = (std::filesystem::temp_directory_path() / ("audit_vk2_res_" + std::to_string(getpid()) + ".vpcap")).string();
  std::remove(path.c_str());
  track(0);
  CHECK(track(1) == 0);
  Api a;
  CHECK(a.init(lib, 3));
  if (!a.dev) return;
  VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 65536, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  lci.bindingCount = 1, lci.pBindings = &b;
  VkDescriptorSetLayout layout = nullptr;
  CHECK(a.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(a.dev, &lci, nullptr, &layout) == VK_SUCCESS);
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 65536};
  VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pci.maxSets = 1, pci.poolSizeCount = 1, pci.pPoolSizes = &ps;
  VkDescriptorPool dpool = nullptr;
  CHECK(a.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(a.dev, &pci, nullptr, &dpool) == VK_SUCCESS);
  VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  sai.descriptorPool = dpool, sai.descriptorSetCount = 1, sai.pSetLayouts = &layout;
  VkDescriptorSet set = nullptr;
  CHECK(a.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(a.dev, &sai, &set) == VK_SUCCESS);
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1, plci.pSetLayouts = &layout;
  VkPipelineLayout pl = nullptr;
  CHECK(a.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(a.dev, &plci, nullptr, &pl) == VK_SUCCESS);
  a.submit(a.open());
  CHECK(request(1, path.c_str()) == 1);
  a.submit(a.open());  // the boundary
  CHECK(state() == 2);
  VkCommandBuffer cb = a.open();
  auto bind = a.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
  for (int i = 0; i < 20000; ++i) bind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, nullptr);
  const auto t0 = std::chrono::steady_clock::now();
  a.submit(cb);  // its one frame: the capture ends
  const long ms = long(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
  std::printf("  submit of 20000 binds of one set: %ld ms\n", ms);
  CHECK(state() == 0);
  CHECK(ms < 500);
  track(0);
  a.destroy();
  std::remove(path.c_str());
  dlclose(lib);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string group = argc > 1 ? argv[1] : "all";
  gLib = argc > 2 ? argv[2] : nullptr;
  gReplay = argc > 3 ? argv[3] : nullptr;
  const std::map<std::string, std::pair<std::function<void()>, bool>> groups = {
      {"fov_nan", {testFovNan, false}},       {"config_nan", {testConfigNan, true}},   {"png_hostile", {testPngHostile, false}},
      {"json", {testJson, false}},            {"passdump_race", {testPassDumpRace, false}}, {"pool_shutdown", {testPoolShutdown, false}},
      {"replay_imagedata", {testReplayImageData, true}}, {"capture_endtext", {testCaptureEndText, true}},
      {"capture_filenames", {testCaptureFileNames, true}}, {"capture_resolve", {testCaptureResolve, true}}};
  int ran = 0;
  for (const auto& [name, g] : groups) {
    if (group != "all" && group != name) continue;
    if (g.second && !gLib) {
      std::printf("[%s] skipped: needs libVPVulkan\n", name.c_str());
      continue;
    }
    std::printf("[%s]\n", name.c_str());
    const int before = failures;
    g.first();
    std::printf("[%s] %s\n", name.c_str(), failures == before ? "ok" : "FAILED");
    ++ran;
  }
  if (!ran) {
    std::printf("unknown group %s\n", group.c_str());
    return 2;
  }
  std::printf("%s\n", failures ? "FAILURES" : "all ok");
  return failures ? 1 : 0;
}
