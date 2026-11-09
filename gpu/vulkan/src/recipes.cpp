#include "recipes.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <thread>

#include "taa_native.h"
#include "vpvk.h"
#include "vpmsl/msl.h"
#include "vpvk/config.h"

namespace vpvk {
namespace {

constexpr char kMagic[4] = {'V', 'Q', 'R', 'C'};
// File format. v3: stage inputs ('W' SPIR-V, 'I' input), pipelines with their inputs ('Q'), drift
// marker ('D'); MSL blobs ('S') as before. Translator changes bump kTranslatorVersion, not this.
constexpr uint32_t kRecipeVersion = 3;
constexpr uint8_t kTagShader = 'S', kTagSpirv = 'W', kTagInput = 'I', kTagPipeline = 'Q', kTagDrift = 'D', kTagIface = 'F';
constexpr size_t kHeaderBytes = 12;
constexpr size_t kRecordHeader = 5;  // tag + length

struct Writer {
  std::vector<uint8_t> b;
  template <class T>
  void pod(T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof v);
  }
  void str(const std::string& s) {
    pod(uint32_t(s.size()));
    b.insert(b.end(), s.begin(), s.end());
  }
  void raw(const void* p, size_t n) {
    const auto* c = static_cast<const uint8_t*>(p);
    b.insert(b.end(), c, c + n);
  }
};

struct Reader {
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;
  template <class T>
  T pod() {
    T v{};
    if (end - p < ptrdiff_t(sizeof v)) return ok = false, v;
    std::memcpy(&v, p, sizeof v);
    p += sizeof v;
    return v;
  }
  std::string str() {
    const uint32_t n = pod<uint32_t>();
    if (!ok || end - p < ptrdiff_t(n)) return ok = false, std::string();
    std::string s(reinterpret_cast<const char*>(p), n);
    p += n;
    return s;
  }
  // Element count with a sanity bound (a corrupt count fails the record instead of allocating).
  uint32_t count(uint32_t max) {
    const uint32_t n = pod<uint32_t>();
    if (n > max) ok = false;
    return ok ? n : 0;
  }
};

uint32_t headerFields() { return uint32_t(vpgpu::PipelineCache::encodedSize()); }

uint64_t fnvBytes(const uint8_t* p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
  return h ? h : 1;
}

// Offset of the MSL text inside a shader record's payload: id, entry (len + bytes), source length.
uint64_t sourceOffsetInPayload(const ShaderBlob& b) { return 8 + 4 + b.entry.size() + 4; }

void encodeShader(Writer& w, uint64_t id, const ShaderBlob& b) {
  w.pod(id);
  w.str(b.entry);
  w.str(b.source);
  for (auto t : b.threadsPerGroup) w.pod(uint32_t(t));
  w.pod(uint8_t(b.colorOutputs));
  w.pod(uint32_t(b.arguments.size()));
  for (const auto& a : b.arguments) w.pod(uint8_t(a.kind)), w.pod(uint16_t(a.slot)), w.pod(uint32_t(a.offset));
}

bool decodeShader(Reader& r, uint64_t& id, ShaderBlob& b) {
  id = r.pod<uint64_t>();
  b.entry = r.str();
  b.source = r.str();
  for (auto& t : b.threadsPerGroup) t = r.pod<uint32_t>();
  b.colorOutputs = r.pod<uint8_t>();
  const uint32_t n = r.count(4096);
  for (uint32_t i = 0; r.ok && i < n; ++i) {
    ShaderBlob::Arg a;
    a.kind = static_cast<vpgpu::BindKind>(r.pod<uint8_t>());
    a.slot = r.pod<uint16_t>();
    a.offset = r.pod<uint32_t>();
    b.arguments.push_back(a);
  }
  return r.ok;
}

// Every field of SpirvLowerOptions, fixed widths (the hash of these bytes identifies the input).
void encodeInput(Writer& w, const StageInput& in) {
  const spirv::SpirvLowerOptions& o = in.opt;
  w.pod(in.spirv);
  w.str(o.entry);
  w.pod(uint8_t(o.stage));
  w.pod(uint32_t(o.specConstants.size()));
  for (const auto& [id, v] : o.specConstants) w.pod(uint32_t(id)), w.pod(uint64_t(v));
  w.pod(uint32_t(o.viewCount));
  w.pod(uint32_t(o.layout.size()));
  for (const spirv::SpirvBinding& b : o.layout) {
    w.pod(uint32_t(b.set)), w.pod(uint32_t(b.binding)), w.pod(uint32_t(b.arraySize));
    w.pod(uint8_t(b.kind)), w.pod(uint16_t(b.slot)), w.pod(uint16_t(b.samplerSlot));
    w.pod(uint8_t(b.written)), w.pod(uint32_t(b.texelFormat));
  }
  w.pod(int32_t(o.pushConstantSlot));
  w.pod(uint32_t(o.vertexAttributes.size()));
  for (const spirv::VertexAttribute& a : o.vertexAttributes)
    w.pod(uint32_t(a.location)), w.pod(uint32_t(a.binding)), w.pod(uint32_t(a.offset)), w.pod(uint32_t(a.format)),
        w.pod(uint8_t(a.components));
  w.pod(uint32_t(o.vertexBuffers.size()));
  for (const spirv::VertexBufferBinding& v : o.vertexBuffers) {
    const bool divided = v.perInstance && v.divisor != 1;  // bit 0 per instance, bit 1 a divisor (uint32) follows the slot
    w.pod(uint32_t(v.binding)), w.pod(uint32_t(v.stride)), w.pod(uint8_t((v.perInstance ? 1 : 0) | (divided ? 2 : 0))),
        w.pod(uint16_t(v.slot));
    if (divided) w.pod(uint32_t(v.divisor));
  }
  // Trailing and only when set: inputs recorded before these fields keep their bytes (and hashes).
  // Flags byte: bit 0 multisampleAsSingle, bit 1 a viewMask (uint32) follows, bit 2 colorIntKinds
  // (uint16) follows, bit 3 the Y'CbCr conversions follow (count, then per binding its layout
  // index and fields), bit 4 uiMaskSlot (uint8) and uiMaskAlpha (uint8) follow, bit 5 alphaTest (no payload), bit 6
  // viewportIndex, bit 7 rateMapPosition (no payload). After all payloads, one more byte only when robustBuffers is off or a sampleMask is set (bit 1: its uint32 follows).
  uint32_t nYcbcr = 0;
  for (const spirv::SpirvBinding& b : o.layout) nYcbcr += b.ycbcr.planes != 0;
  const uint8_t ext =
      uint8_t((o.multisampleAsSingle ? 1 : 0) | (o.viewMask ? 2 : 0) | (o.colorIntKinds ? 4 : 0) | (nYcbcr ? 8 : 0) |
                (o.uiMaskSlot >= 0 ? 16 : 0) | (o.alphaTest ? 32 : 0) | (o.viewportIndex ? 64 : 0) | (o.rateMapPosition ? 128 : 0));
  if (ext || !o.robustBuffers || o.sampleMask != ~0u || o.d24Emul) w.pod(ext);
  if (o.viewMask) w.pod(uint32_t(o.viewMask));
  if (o.colorIntKinds) w.pod(uint16_t(o.colorIntKinds));
  if (nYcbcr) {
    w.pod(nYcbcr);
    for (size_t i = 0; i < o.layout.size(); ++i) {
      const spirv::YcbcrBinding& y = o.layout[i].ycbcr;
      if (!y.planes) continue;
      w.pod(uint32_t(i)), w.pod(y.planes), w.pod(uint8_t(y.model)), w.pod(uint8_t(y.range));
      w.pod(uint8_t(y.cositedX | y.cositedY << 1 | y.subX << 2 | y.subY << 3));
      for (uint8_t c : y.swizzle) w.pod(c);
      w.pod(y.planeSlot[0]), w.pod(y.planeSlot[1]);
    }
  }
  if (o.uiMaskSlot >= 0) w.pod(uint8_t(o.uiMaskSlot)), w.pod(uint8_t(o.uiMaskAlpha ? 1 : 0));
  // last: emit flags, bit 0 robustBuffers off, bit 1 a sampleMask (uint32) follows, bit 2 d24Emul (no payload)
  if (!o.robustBuffers || o.sampleMask != ~0u || o.d24Emul)
    w.pod(uint8_t((o.robustBuffers ? 0 : 1) | (o.sampleMask != ~0u ? 2 : 0) | (o.d24Emul ? 4 : 0)));
  if (o.sampleMask != ~0u) w.pod(uint32_t(o.sampleMask));
}

bool decodeInput(Reader& r, StageInput& in) {
  spirv::SpirvLowerOptions& o = in.opt;
  in.spirv = r.pod<uint64_t>();
  o.entry = r.str();
  o.stage = static_cast<vpir::Stage>(r.pod<uint8_t>());
  for (uint32_t i = 0, n = r.count(4096); r.ok && i < n; ++i) {
    const uint32_t id = r.pod<uint32_t>();
    o.specConstants.push_back({id, r.pod<uint64_t>()});
  }
  o.viewCount = r.pod<uint32_t>();
  for (uint32_t i = 0, n = r.count(65536); r.ok && i < n; ++i) {
    spirv::SpirvBinding b;
    b.set = r.pod<uint32_t>(), b.binding = r.pod<uint32_t>(), b.arraySize = r.pod<uint32_t>();
    b.kind = static_cast<vpgpu::BindKind>(r.pod<uint8_t>());
    b.slot = r.pod<uint16_t>(), b.samplerSlot = r.pod<uint16_t>();
    b.written = r.pod<uint8_t>() != 0;
    b.texelFormat = static_cast<vpgpu::Format>(r.pod<uint32_t>());
    o.layout.push_back(b);
  }
  o.pushConstantSlot = r.pod<int32_t>();
  for (uint32_t i = 0, n = r.count(4096); r.ok && i < n; ++i) {
    spirv::VertexAttribute a;
    a.location = r.pod<uint32_t>(), a.binding = r.pod<uint32_t>(), a.offset = r.pod<uint32_t>();
    a.format = static_cast<vpgpu::Format>(r.pod<uint32_t>());
    a.components = r.pod<uint8_t>();
    o.vertexAttributes.push_back(a);
  }
  for (uint32_t i = 0, n = r.count(4096); r.ok && i < n; ++i) {
    spirv::VertexBufferBinding v;
    v.binding = r.pod<uint32_t>(), v.stride = r.pod<uint32_t>();
    const uint8_t rate = r.pod<uint8_t>();  // bit 0 per instance, bit 1 a divisor (uint32) follows the slot
    v.perInstance = (rate & 1) != 0;
    v.slot = r.pod<uint16_t>();
    if (rate & 2) v.divisor = r.pod<uint32_t>();
    o.vertexBuffers.push_back(v);
  }
  if (r.ok && r.p < r.end) {
    const uint8_t ext = r.pod<uint8_t>();
    o.multisampleAsSingle = ext & 1;
    if (ext & 2) o.viewMask = r.pod<uint32_t>();
    if (ext & 4) o.colorIntKinds = r.pod<uint16_t>();
    if (ext & 8) {
      for (uint32_t i = 0, n = r.count(65536); r.ok && i < n; ++i) {
        const uint32_t at = r.pod<uint32_t>();
        spirv::YcbcrBinding y;
        y.planes = r.pod<uint8_t>();
        y.model = static_cast<spirv::YcbcrModel>(r.pod<uint8_t>());
        y.range = static_cast<spirv::YcbcrRange>(r.pod<uint8_t>());
        const uint8_t f = r.pod<uint8_t>();
        y.cositedX = f & 1, y.cositedY = f & 2, y.subX = f & 4, y.subY = f & 8;
        for (uint8_t& c : y.swizzle) c = r.pod<uint8_t>();
        y.planeSlot[0] = r.pod<uint16_t>(), y.planeSlot[1] = r.pod<uint16_t>();
        if (!r.ok || at >= o.layout.size()) return false;
        o.layout[at].ycbcr = y;
      }
    }
    if (ext & 16) {
      o.uiMaskSlot = int8_t(r.pod<uint8_t>());
      o.uiMaskAlpha = r.pod<uint8_t>() != 0;
    }
    o.alphaTest = (ext & 32) != 0;
    o.viewportIndex = (ext & 64) != 0;
    o.rateMapPosition = (ext & 128) != 0;
  }
  if (r.ok && r.p < r.end) {
    const uint8_t f = r.pod<uint8_t>();
    o.robustBuffers = !(f & 1);
    o.d24Emul = (f & 4) != 0;
    if (f & 2) o.sampleMask = r.pod<uint32_t>();
  }
  return r.ok && in.spirv != 0;
}

}  // namespace

namespace {
std::atomic<uint32_t> gTranslatorMode{~0u};  // ~0: not read yet
}

uint32_t refreshTranslatorMode() {
  auto flag = [](const char* name) {
    const char* e = std::getenv(name);
    return e && *e && *e != '0';
  };
  // VPVK_FETCH_NATIVE is a mask (1 half attributes, 2 unorm/snorm unpack); a value that is not a number counts as 1.
  uint32_t fetch = 0;
  if (const char* e = std::getenv("VPVK_FETCH_NATIVE"); e && *e) {
    char* end = nullptr;
    const unsigned long v = std::strtoul(e, &end, 10);
    fetch = end == e ? 1u : uint32_t(v & 3);
  }
  // VPSPV_KILL_EXIT (default on): only 0 turns it off. The bit is set for "off" so the default stays mode 0.
  const char* kill = std::getenv("VPSPV_KILL_EXIT");
  // VPVK_INVARIANT_POS (default on): only "0" / "off" / "false" (surrounding blanks ignored) turn it off (bit set for
  // "off", the default stays mode 0); any other value, empty included, leaves it on. Every vertex
  // position becomes [[position, invariant]] (Metal preserveInvariance for that MSL), so a depth prepass and the main
  // pass get the same z. Cost: the compiler may not reassociate / contract the math that feeds the position; expected
  // negligible next to the rest of a vertex shader (not measured on the headset: compare vk.hitch / gpu_ms with =0).
  bool invOff = false;
  if (const char* e = std::getenv("VPVK_INVARIANT_POS")) {
    std::string v(e);
    const size_t b = v.find_first_not_of(" \t\r\n"), l = v.find_last_not_of(" \t\r\n");
    v = b == std::string::npos ? std::string() : v.substr(b, l - b + 1);
    invOff = v == "0" || v == "off" || v == "false";
  }
  const uint32_t m = (fetch & 1) | (flag("VPVK_AMPLIFY") ? 2u : 0u) | ((fetch & 2) << 1) | (kill && *kill == '0' ? 8u : 0u) |
                     (invOff ? 16u : 0u) | (flag("VPVK_AMPLIFY_CLIP") ? 32u : 0u);
  if (gTranslatorMode.exchange(m) != m) {
    logf("vk.translator mode=%u native_fetch=%u amplify=%u unpack_fetch=%u kill_exit=%u amplify_clip=%u id=%u", m, m & 1,
         (m >> 1) & 1, (m >> 2) & 1, (~m >> 3) & 1, (m >> 5) & 1, kTranslatorVersion | m << 16);
    // math=fast is metal_backend.mm's MTLMathModeFast. Whether a library really compiled with preserveInvariance
    // (and how many fell back without it) is in the gpu.shader.lib lines: invariant= / invariant_retries=.
    logf("gpu.shaderfx invariant_pos=%u math=fast", (~m >> 4) & 1);
  }
  return m;
}

uint32_t translatorMode() {
  const uint32_t m = gTranslatorMode.load();
  return m != ~0u ? m : refreshTranslatorMode();
}

uint32_t translatorId() { return kTranslatorVersion | translatorMode() << 16; }

void applyTranslatorMode(spirv::SpirvLowerOptions& o) {
  const uint32_t m = translatorMode();
  o.nativeFetch = (m & 1) != 0;
  o.amplify = (m & 2) != 0;
  o.unpackFetch = (m & 4) != 0;
  o.killExit = (m & 8) == 0;
  o.invariantPos = (m & 16) == 0;
  o.ampClip = (m & 32) != 0;
}

uint8_t translatorDiff(uint64_t stored, uint64_t current) {
  uint8_t m = 0;
  if ((stored & 0xFFFF) != (current & 0xFFFF)) m |= kDiffBaseline;
  if ((stored >> 16 & 0xFFFF) != (current >> 16 & 0xFFFF)) m |= kDiffLower;
  if ((stored >> 32 & 0xFFFF) != (current >> 32 & 0xFFFF)) m |= kDiffEmit;
  if ((stored >> 48 & 0xFFFF) != (current >> 48 & 0xFFFF)) m |= kDiffDriver;
  return m;
}

std::string translatorTag(uint64_t code) {
  char b[48];
  std::snprintf(b, sizeof b, "b%u.l%u.e%u.d%u", unsigned(code & 0xFFFF), unsigned(code >> 16 & 0xFFFF),
                unsigned(code >> 32 & 0xFFFF), unsigned(code >> 48 & 0xFFFF));
  return b;
}

namespace {

// Interface of a translated stage ('F' record, after the input hash and translator version).
void encodeIface(Writer& w, const StageIface& f) {
  w.pod(uint8_t(f.usesPushConstants)), w.pod(f.colorOutputs), w.pod(f.mslColorOutputs);
  for (uint32_t x : f.localSize) w.pod(x);
  w.pod(f.varyingComponents);
  w.pod(uint32_t(f.bindings.size()));
  for (const spirv::SpirvBinding& b : f.bindings)
    w.pod(uint32_t(b.set)), w.pod(uint32_t(b.binding)), w.pod(uint8_t(b.written)), w.pod(uint32_t(b.texelFormat));
  // trailing and only when a flag is set (bit 0 amplified, bit 1 dualSource): records without it keep their bytes
  if (f.amplified || f.dualSource) w.pod(uint8_t((f.amplified ? 1 : 0) | (f.dualSource ? 2 : 0)));
}

bool decodeIface(Reader& r, StageIface& f) {
  f.usesPushConstants = r.pod<uint8_t>() != 0, f.colorOutputs = r.pod<uint8_t>(), f.mslColorOutputs = r.pod<uint8_t>();
  for (uint32_t& x : f.localSize) x = r.pod<uint32_t>();
  f.varyingComponents = r.pod<uint32_t>();
  for (uint32_t i = 0, n = r.count(65536); r.ok && i < n; ++i) {
    spirv::SpirvBinding b;
    b.set = r.pod<uint32_t>(), b.binding = r.pod<uint32_t>(), b.written = r.pod<uint8_t>() != 0;
    b.texelFormat = static_cast<vpgpu::Format>(r.pod<uint32_t>());
    f.bindings.push_back(b);
  }
  if (r.ok && r.p < r.end) {
    const uint8_t flags = r.pod<uint8_t>();
    f.amplified = flags & 1, f.dualSource = flags & 2;
  }
  return r.ok;
}

void encodePipeline(Writer& w, const vpgpu::PipelineState& s, const std::array<uint64_t, 3>& inputs) {
  for (uint64_t x : inputs) w.pod(x);
  std::vector<uint64_t> v;
  vpgpu::PipelineCache::encode(s, v);
  for (uint64_t x : v) w.pod(x);
}

uint64_t* shaderField(vpgpu::PipelineState& s, int stage) {
  return stage == 0 ? &s.vertexShader : stage == 1 ? &s.fragmentShader : &s.computeShader;
}

}  // namespace

uint64_t spirvHash(const uint32_t* words, size_t count) {
  uint64_t h = 0xcbf29ce484222325ull ^ (uint64_t(count) << 1);
  for (size_t i = 0; i < count; ++i) h = (h ^ words[i]) * 0x100000001b3ull;
  h ^= h >> 29;
  return h ? h : 1;
}

bool spirvUsesMultisample(const uint32_t* w, size_t n) {
  constexpr uint32_t kOpTypeImage = 25;
  size_t i = 5;  // header: magic, version, generator, bound, schema
  while (i < n) {
    const uint32_t op = w[i] & 0xFFFF, len = w[i] >> 16;
    if (!len || i + len > n) return true;  // malformed: assume the worst (the flag stays in the key)
    if (op == kOpTypeImage && len >= 7 && w[i + 6] == 1) return true;  // result, sampled type, dim, depth, arrayed, MS
    i += len;
  }
  return false;
}

uint64_t StageInput::hash() const {
  Writer w;
  encodeInput(w, *this);
  return fnvBytes(w.b.data(), w.b.size());
}

std::string RecipeStore::defaultDir() {
  const char* d = std::getenv("VP_PIPELINE_CACHE_DIR");
  return d && *d ? d : "";
}

RecipeStore::RecipeStore(std::string dir, uint32_t translatorVersion, uint64_t translatorCode)
    : translator_(translatorVersion), code_(translatorCode) {
  if (dir.empty()) return;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  path_ = dir + "/recipes.bin";
  load();
  if (!openForAppend(size_ == 0)) return;
  size_t stale = 0;
  for (const auto& [h, rec] : inputs_) stale += rec.translator != translator_;
  logf("gpu.pso.recipe read shaders=%zu pipelines=%zu inputs=%zu spirv=%zu stale_inputs=%zu drift=%d dead=%zu "
       "translator=%u bytes=%llu",
       shaders_.size(), pipelines_.size(), inputs_.size(), spirv_.size(), stale, drift_ ? 1 : 0, deadRecords_,
       translator_, (unsigned long long)size_);
}

RecipeStore::~RecipeStore() {
  if (!file_) return;
  logf("gpu.pso.recipe stored=%zu total=%zu", storedThisRun_, pipelines_.size());
  std::fclose(file_);
}

bool RecipeStore::openForAppend(bool fresh) {
  file_ = std::fopen(path_.c_str(), fresh ? "wb" : "ab");
  if (!file_) {
    logf("gpu.pso.recipe what=open_failed path=%s", path_.c_str());
    return false;
  }
  if (fresh) {
    std::fwrite(kMagic, 1, 4, file_);
    const uint32_t hdr[2] = {kRecipeVersion, headerFields()};
    std::fwrite(hdr, sizeof hdr, 1, file_);
    std::fflush(file_);
    size_ = kHeaderBytes;
    records_ = deadRecords_ = 0;
  }
  return true;
}

void RecipeStore::load() {
  FILE* f = std::fopen(path_.c_str(), "rb");
  if (!f) return;
  std::vector<uint8_t> data;
  uint8_t buf[65536];
  for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) data.insert(data.end(), buf, buf + n);
  std::fclose(f);
  Reader r{data.data(), data.data() + data.size()};
  const uint32_t magic = r.pod<uint32_t>();
  const uint32_t ver = r.pod<uint32_t>(), fields = r.pod<uint32_t>();
  if (!r.ok || std::memcmp(&magic, kMagic, 4) || ver != kRecipeVersion || fields != headerFields()) {
    // Older files (v2 and before) hold MSL only, without the translator inputs: nothing to rebuild from.
    logf("gpu.pso.recipe discarded=stale version=%u fields=%u", ver, fields);
    return;  // size_ stays 0: the constructor rewrites the file from scratch
  }
  size_t bad = 0;
  size_t goodEnd = size_t(r.p - data.data());
  std::vector<PipeRec> pending;
  while (r.ok && r.p < r.end) {
    const uint8_t tag = r.pod<uint8_t>();
    const uint32_t len = r.pod<uint32_t>();
    if (!r.ok || r.end - r.p < ptrdiff_t(len)) break;  // truncated tail (crash while appending)
    const uint64_t payload = uint64_t(r.p - data.data());
    Reader rec{r.p, r.p + len};
    r.p += len;
    goodEnd = size_t(r.p - data.data());
    ++records_;
    if (tag == kTagSpirv) {
      const uint64_t h = rec.pod<uint64_t>();
      const uint32_t n = rec.pod<uint32_t>();
      if (!rec.ok || uint64_t(rec.end - rec.p) != uint64_t(n) * 4 || !n) {
        ++bad;
        continue;
      }
      if (!spirv_.emplace(h, SpirvRec{payload + 12, n}).second) ++deadRecords_;
    } else if (tag == kTagShader) {
      uint64_t id;
      ShaderBlob b;
      if (!decodeShader(rec, id, b)) {
        ++bad;
        continue;
      }
      // The MSL stays in the file (srcRef_) and is read back on demand: the store keeps no second copy of
      // 100-250 MB that the backend's shader table holds anyway.
      const SrcRef ref{payload + sourceOffsetInPayload(b), uint32_t(b.source.size()), vpmsl::sourceHash(b.source)};
      auto [sit, added] = shaders_.emplace(id, std::move(b));
      if (!added) {
        ++deadRecords_;
      } else {
        srcRef_[id] = ref;
        std::string().swap(sit->second.source);
      }
    } else if (tag == kTagInput) {
      InputRec in;
      const uint64_t h = rec.pod<uint64_t>();
      in.translator = rec.pod<uint32_t>();
      in.msl = rec.pod<uint64_t>();
      if (!decodeInput(rec, in.input) || !in.msl) {
        ++bad;
        continue;
      }
      auto [it, added] = inputs_.try_emplace(h);
      if (!added) ++deadRecords_;
      it->second = std::move(in);  // the latest record of an input wins
    } else if (tag == kTagIface) {
      const uint64_t h = rec.pod<uint64_t>();
      IfaceRec f;
      f.translator = rec.pod<uint32_t>();
      f.code = rec.pod<uint64_t>();
      if (!decodeIface(rec, f.iface)) {
        ++bad;
        continue;
      }
      if (!ifaces_.insert_or_assign(h, std::move(f)).second) ++deadRecords_;
    } else if (tag == kTagPipeline) {
      PipeRec p;
      for (auto& x : p.inputs) x = rec.pod<uint64_t>();
      std::vector<uint64_t> v(headerFields());
      for (auto& x : v) x = rec.pod<uint64_t>();
      if (!rec.ok || !vpgpu::PipelineCache::decode(v.data(), v.size(), p.state)) {
        ++bad;
        continue;
      }
      p.state.label = "vk.prewarm";
      pending.push_back(std::move(p));
    } else if (tag == kTagDrift) {
      drift_ = true;
    } else {
      ++bad;
    }
  }
  size_ = goodEnd;
  // Pipelines name their shaders through the stage inputs: a later 'I' record (same input, other
  // MSL) moves them along.
  size_t orphans = 0;
  for (PipeRec& p : pending) {
    if (!remapLocked(p)) {
      ++orphans, ++deadRecords_;
      continue;
    }
    insertPipelineLocked(std::move(p));
  }
  if (bad || orphans) logf("gpu.pso.recipe bad_records=%zu orphan_pipelines=%zu", bad, orphans);
  if (goodEnd < data.size()) {  // drop a torn tail so later appends stay framed
    std::error_code ec;
    std::filesystem::resize_file(path_, goodEnd, ec);
    logf("gpu.pso.recipe truncated_tail=%zu", data.size() - goodEnd);
  }
}

void degradeSrc1Blend(vpgpu::PipelineState& st, bool dualSource) {
  using B = vpgpu::BlendFactor;
  for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c) {
    if (dualSource && c == 0) continue;
    auto fix = [](B& f) {
      if (f == B::Src1Color) f = B::SrcColor;
      else if (f == B::OneMinusSrc1Color) f = B::OneMinusSrcColor;
      else if (f == B::Src1Alpha) f = B::SrcAlpha;
      else if (f == B::OneMinusSrc1Alpha) f = B::OneMinusSrcAlpha;
    };
    vpgpu::BlendState& b = st.blend[c];
    fix(b.srcColor), fix(b.dstColor), fix(b.srcAlpha), fix(b.dstAlpha);
  }
}

vpgpu::PipelineState msaaVariantState(const vpgpu::PipelineState& base, uint8_t samples) {
  vpgpu::PipelineState st = base;
  st.samples = samples;
  st.label += ".ms" + std::to_string(samples);
  return st;
}

void applyUiMaskTarget(vpgpu::PipelineState& st, bool writes) {
  constexpr int slot = taa::kUiMaskSlot;
  st.colorFormats[slot] = vpgpu::Format::R8Unorm;
  vpgpu::BlendState& b = st.blend[slot];
  b = vpgpu::BlendState{};
  if (writes) {  // coverage accumulates: the largest value over the draws that touch the pixel
    b.enabled = true;
    b.colorOp = b.alphaOp = vpgpu::BlendOp::Max;
    b.srcColor = b.srcAlpha = b.dstColor = b.dstAlpha = vpgpu::BlendFactor::One;
    b.writeMask = 1;
  } else {
    b.writeMask = 0;
  }
  st.label += writes ? ".umu" : ".ums";
}

void forcedMsaaVariants(const vpgpu::PipelineState& s, int n, bool masks, std::vector<vpgpu::PipelineState>& out) {
  constexpr int slot = taa::kUiMaskSlot;
  if (n < 2 || s.computeShader || s.samples >= n) return;
  if (s.colorFormats[slot] != vpgpu::Format::Undefined) {  // a mask variant of a recorded run: the runtime wants it at n samples
    if (!masks) return;
    out.push_back(s);
    out.back().samples = uint8_t(n);
    return;
  }
  // forceMsaa serves colour 0 + depth passes only (no other colour attachment, no integer / depth colour 0).
  if (s.colorFormats[0] == vpgpu::Format::Undefined || s.depthFormat == vpgpu::Format::Undefined) return;
  for (int i = 1; i < vpgpu::kMaxColorAttachments; ++i)
    if (s.colorFormats[i] != vpgpu::Format::Undefined) return;
  const vpgpu::FormatInfo& fc = vpgpu::formatInfo(s.colorFormats[0]);
  if (fc.integer() || fc.depth()) return;
  out.push_back(msaaVariantState(s, uint8_t(n)));
  if (masks) {
    vpgpu::PipelineState m = s;
    m.samples = uint8_t(n);
    applyUiMaskTarget(m, false);
    out.push_back(std::move(m));
  }
}

bool RecipeStore::remapLocked(PipeRec& p) const {
  for (int s = 0; s < 3; ++s) {
    uint64_t* id = shaderField(p.state, s);
    if (!*id && !p.inputs[s]) continue;
    if (!p.inputs[s]) return false;
    auto it = inputs_.find(p.inputs[s]);
    if (it == inputs_.end()) return false;
    *id = it->second.msl;
    if (s == 1) {  // outputs the fragment shader does not write keep their contents (as createGraphics)
      auto sh = shaders_.find(*id);
      if (sh != shaders_.end())
        for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c)
          if (!(sh->second.colorOutputs & (1u << c))) p.state.blend[c].writeMask = 0;
      // Second-source factors follow the stage's current interface (an old record says nothing about it).
      const auto fi = ifaces_.find(p.inputs[1]);
      if (fi != ifaces_.end() && fi->second.translator == translator_ && fi->second.code == code_)
        degradeSrc1Blend(p.state, fi->second.iface.dualSource);
    }
  }
  return true;
}

void RecipeStore::insertPipelineLocked(PipeRec p) {
  const uint64_t key = p.state.key().value;
  const uint64_t id = recipeId(p.state, p.inputs);
  if (!pipelines_.emplace(key, std::move(p)).second) ++deadRecords_;
  else byId_[id] = key;
}

void RecipeStore::write(uint8_t tag, const std::vector<uint8_t>& payload, uint64_t* payloadOffset) {
  const uint32_t len = uint32_t(payload.size());
  if (payloadOffset) *payloadOffset = size_ + kRecordHeader;
  std::fwrite(&tag, 1, 1, file_);
  std::fwrite(&len, sizeof len, 1, file_);
  if (len) std::fwrite(payload.data(), 1, payload.size(), file_);
  std::fflush(file_);
  size_ += kRecordHeader + len;
  ++records_;
}

bool RecipeStore::readWords(const SpirvRec& s, std::vector<uint32_t>& out) const {
  out.clear();
  FILE* f = std::fopen(path_.c_str(), "rb");
  if (!f) return false;
  bool ok = std::fseek(f, long(s.offset), SEEK_SET) == 0;
  if (ok) {
    out.resize(s.count);
    ok = std::fread(out.data(), 4, out.size(), f) == out.size();
  }
  std::fclose(f);
  if (!ok) out.clear();
  return ok;
}

void RecipeStore::addStage(const StageInput& in, uint64_t h, const std::vector<uint32_t>& words, uint64_t msl,
                           const ShaderBlob& blob, const StageIface* iface) {
  if (words.empty() || !msl) return;
  std::lock_guard lock(mutex_);
  if (!file_) return;
  auto it = inputs_.find(h);
  const bool known = it != inputs_.end();
  auto fi = ifaces_.find(h);
  const bool ifaceKnown = !iface || (fi != ifaces_.end() && fi->second.translator == translator_ && fi->second.code == code_);
  if (known && it->second.msl == msl && it->second.translator == translator_ && shaders_.count(msl) &&
      spirv_.count(in.spirv) && ifaceKnown)
    return;  // the common case after the first run: nothing to record
  if (!spirv_.count(in.spirv)) {
    Writer w;
    w.pod(in.spirv);
    w.pod(uint32_t(words.size()));
    w.raw(words.data(), words.size() * 4);
    uint64_t off = 0;
    write(kTagSpirv, w.b, &off);
    spirv_[in.spirv] = {off + 12, uint32_t(words.size())};
  }
  if (!shaders_.count(msl)) {
    Writer w;
    encodeShader(w, msl, blob);
    uint64_t off = 0;
    write(kTagShader, w.b, &off);  // flushed: readable back at once
    ShaderBlob& kept = shaders_[msl] = blob;
    srcRef_[msl] = SrcRef{off + sourceOffsetInPayload(blob), uint32_t(blob.source.size()), vpmsl::sourceHash(blob.source)};
    std::string().swap(kept.source);
  }
  if (known && it->second.translator == translator_ && it->second.msl != msl) {
    // Same input, same translator version, other MSL: the translator changed without a version
    // bump (or is not deterministic). The next load retranslates everything.
    static std::atomic<uint32_t> lines{0};
    if (lines++ < 8)
      logf("gpu.pso.recipe drift input=%016llx old=%016llx new=%016llx (translator changed without kTranslatorVersion bump?)",
           (unsigned long long)h, (unsigned long long)it->second.msl, (unsigned long long)msl);
    if (!driftWritten_) write(kTagDrift, {}), driftWritten_ = true;
  }
  if (known) ++deadRecords_;
  Writer w;
  w.pod(h);
  w.pod(translator_);
  w.pod(msl);
  encodeInput(w, in);
  write(kTagInput, w.b);
  inputs_[h] = InputRec{in, msl, translator_};
  if (iface) {
    Writer f;
    f.pod(h);
    f.pod(translator_);
    f.pod(code_);
    encodeIface(f, *iface);
    write(kTagIface, f.b);
    if (fi != ifaces_.end()) ++deadRecords_;
    ifaces_[h] = IfaceRec{translator_, code_, *iface};
  }
}

bool RecipeStore::lookupStage(uint64_t h, StageHit& out, MissInfo* miss, uint64_t spirv) const {
  MissInfo local;
  MissInfo& why = miss ? *miss : local;
  why = {};
  std::lock_guard lock(mutex_);
  if (!file_ || drift_) return why.why = StageMiss::Disabled, false;
  auto in = inputs_.find(h);
  auto fi = ifaces_.find(h);
  if (in == inputs_.end() || fi == ifaces_.end()) {
    // The module is known but this input is not (other layout, vertex format, view mask...), or nothing of it is.
    why.why = spirv && spirv_.count(spirv) ? StageMiss::NewOptions : StageMiss::NewContent;
    // An input recorded without its interface ('I' without 'F', files from before the stage cache) is a version miss.
    if (in != inputs_.end()) why.why = StageMiss::Version, why.components = kDiffBaseline;
    return false;
  }
  if (in->second.translator != translator_ || fi->second.translator != translator_ || fi->second.code != code_) {
    why.why = StageMiss::Version;
    why.components = translatorDiff(fi->second.code, code_);
    if (!why.components) why.components = kDiffBaseline;  // only the stored number differs (tests, old files, a VPVK_AMPLIFY / VPVK_FETCH_NATIVE change: no kDiff bit for modes)
    return false;
  }
  auto sh = shaders_.find(in->second.msl);
  if (sh == shaders_.end()) return why.why = StageMiss::NoShader, false;
  out.mslId = in->second.msl;
  const auto ref = srcRef_.find(in->second.msl);
  out.sourceHash = ref != srcRef_.end() ? ref->second.hash : vpmsl::sourceHash(sh->second.source);
  out.iface = fi->second.iface;
  return true;
}

bool RecipeStore::shaderBlob(uint64_t id, ShaderBlob& out) const {
  std::lock_guard lock(mutex_);
  auto sh = shaders_.find(id);
  if (sh == shaders_.end()) return false;
  out = sh->second;
  return fillSource(id, out);
}

// The MSL of a shader whose text lives only in the file (see load). `f`: an open read handle, or null to open one.
bool RecipeStore::fillSource(uint64_t id, ShaderBlob& b, FILE* f) const {
  const auto r = srcRef_.find(id);
  if (r == srcRef_.end()) return true;  // text still in memory
  FILE* own = f ? nullptr : std::fopen(path_.c_str(), "rb");
  f = f ? f : own;
  bool ok = f && std::fseek(f, long(r->second.off), SEEK_SET) == 0;
  if (ok) {
    b.source.resize(r->second.len);
    ok = std::fread(b.source.data(), 1, b.source.size(), f) == b.source.size();
  }
  if (own) std::fclose(own);
  if (!ok) b.source.clear();
  return ok;
}

void RecipeStore::addPipeline(const vpgpu::PipelineState& s, const std::array<uint64_t, 3>& inputs) {
  const uint64_t key = s.key().value;
  std::lock_guard lock(mutex_);
  if (!file_ || pipelines_.count(key)) return;
  for (int k = 0; k < 3; ++k)  // a stage without a recorded input cannot be rebuilt later
    if ((k == 0 ? s.vertexShader : k == 1 ? s.fragmentShader : s.computeShader) && !inputs[k]) return;
  Writer w;
  encodePipeline(w, s, inputs);
  write(kTagPipeline, w.b);
  pipelines_[key] = PipeRec{s, inputs};
  byId_[recipeId(s, inputs)] = key;
  ++storedThisRun_;
}

bool RecipeStore::needsRefresh() const {
  std::lock_guard lock(mutex_);
  if (drift_ || compactionDueLocked()) return true;
  for (const auto& [h, rec] : inputs_)
    if (rec.translator != translator_) return true;
  return false;
}

RecipeStore::RefreshResult RecipeStore::refresh(unsigned threads) {
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  RefreshResult r;
  struct Job {
    uint64_t spirv = 0;
    SpirvRec rec;
    std::vector<std::pair<uint64_t, spirv::SpirvLowerOptions>> inputs;
  };
  struct Out {
    uint64_t input = 0, msl = 0;
    ShaderBlob blob;
    StageIface iface;
    bool ok = false;
  };
  std::vector<Job> jobs;
  std::vector<uint64_t> unbuildable;
  // What existed when the refresh started: the final clean-up drops only those. The game may be adding stages and
  // pipelines while the background prewarm refreshes (a stage is recorded before the pipeline that uses it).
  std::unordered_set<uint64_t> oldInputs, oldShaders, oldSpirv;
  {
    std::lock_guard lock(mutex_);
    if (!file_) return r;
    for (const auto& kv : inputs_) oldInputs.insert(kv.first);
    for (const auto& kv : shaders_) oldShaders.insert(kv.first);
    for (const auto& kv : spirv_) oldSpirv.insert(kv.first);
    std::unordered_map<uint64_t, size_t> jobOf;
    for (const auto& [h, rec] : inputs_) {
      if (!drift_ && rec.translator == translator_) continue;
      auto s = spirv_.find(rec.input.spirv);
      if (s == spirv_.end()) {
        unbuildable.push_back(h);
        continue;
      }
      auto [j, added] = jobOf.emplace(rec.input.spirv, jobs.size());
      if (added) jobs.push_back(Job{rec.input.spirv, s->second, {}});
      jobs[j->second].inputs.push_back({h, rec.input.opt});
    }
    if (jobs.empty() && unbuildable.empty() && !drift_ && !compactionDueLocked()) return r;
  }

  // Translation runs unlocked, one SPIR-V module per job (parsed once for all its inputs).
  std::vector<std::vector<Out>> outs(jobs.size());
  std::atomic<size_t> next{0};
  std::atomic<uint32_t> failLines{0};
  auto work = [&] {
    for (size_t j; (j = next++) < jobs.size();) {
      const Job& job = jobs[j];
      std::vector<uint32_t> words;
      std::string error;
      spirv::SpirvModule* m = nullptr;
      if (!readWords(job.rec, words)) error = "read";
      else if (spirvHash(words.data(), words.size()) != job.spirv) error = "spirv hash mismatch";
      else m = spirv::parse(words.data(), words.size(), error);
      for (const auto& [h, opt] : job.inputs) {
        Out o;
        o.input = h;
        std::string e = error;
        o.ok = m && translateRecipeStage(*m, opt, o.msl, o.blob, e, &o.iface);
        if (!o.ok && failLines++ < 16)
          logf("gpu.pso.recipe retranslate_fail input=%016llx stage=%s error=\"%s\"", (unsigned long long)h,
               vpir::toString(opt.stage), e.c_str());
        outs[j].push_back(std::move(o));
      }
      spirv::release(m);
    }
  };
  const unsigned n = unsigned(std::min<size_t>(std::clamp(threads, 1u, 16u), jobs.size()));
  if (n <= 1) {
    work();
  } else {
    std::vector<std::thread> pool;
    for (unsigned k = 0; k < n; ++k) pool.emplace_back(work);
    for (auto& t : pool) t.join();
  }

  std::lock_guard lock(mutex_);
  if (!file_) return r;
  for (uint64_t h : unbuildable) inputs_.erase(h), ifaces_.erase(h), ++r.failed;
  for (auto& job : outs)
    for (Out& o : job) {
      auto it = inputs_.find(o.input);
      if (it == inputs_.end()) continue;
      if (!o.ok) {
        ifaces_.erase(it->first);
        inputs_.erase(it), ++r.failed;
        continue;
      }
      r.sameMsl += it->second.msl == o.msl;
      it->second.msl = o.msl;
      it->second.translator = translator_;
      ifaces_[o.input] = IfaceRec{translator_, code_, std::move(o.iface)};
      shaders_.try_emplace(o.msl, std::move(o.blob));
      ++r.retranslated;
    }
  // Pipelines follow their inputs to the new MSL ids (new keys); those that lost an input go.
  std::unordered_map<uint64_t, PipeRec> old = std::move(pipelines_);
  pipelines_.clear();
  byId_.clear();
  for (auto& [key, p] : old) {
    if (!remapLocked(p)) {
      ++r.droppedPipelines;
      continue;
    }
    insertPipelineLocked(std::move(p));
  }
  // Keep only what some pipeline still needs.
  std::unordered_set<uint64_t> usedInputs, usedShaders, usedSpirv;
  for (const auto& [key, p] : pipelines_) {
    for (uint64_t h : p.inputs)
      if (h) usedInputs.insert(h);
    for (uint64_t id : {p.state.vertexShader, p.state.fragmentShader, p.state.computeShader})
      if (id) usedShaders.insert(id);
  }
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    if (usedInputs.count(it->first) || !oldInputs.count(it->first)) {
      usedSpirv.insert(it->second.input.spirv);
      ++it;
    } else {
      ifaces_.erase(it->first);
      it = inputs_.erase(it);
    }
  }
  for (auto it = shaders_.begin(); it != shaders_.end();) {
    if (usedShaders.count(it->first) || !oldShaders.count(it->first)) {
      ++it;
    } else {
      srcRef_.erase(it->first);
      it = shaders_.erase(it);
    }
  }
  for (auto it = spirv_.begin(); it != spirv_.end();)
    it = usedSpirv.count(it->first) || !oldSpirv.count(it->first) ? std::next(it) : spirv_.erase(it);
  drift_ = false;
  r.rewritten = rewriteLocked();
  r.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  logf("gpu.pso.recipe refresh retranslated=%u same_msl=%u failed=%u dropped_pipelines=%u pipelines=%zu shaders=%zu "
       "spirv=%zu rewritten=%d translator=%u (%s) workers=%u ms=%.0f",
       r.retranslated, r.sameMsl, r.failed, r.droppedPipelines, pipelines_.size(), shaders_.size(), spirv_.size(),
       r.rewritten ? 1 : 0, translator_, translatorTag(code_).c_str(), n, r.ms);
  return r;
}

// Writes every live record to recipes.bin.tmp and renames it over the file (a crash mid-way leaves
// the old file intact). SPIR-V words are copied from the old file.
bool RecipeStore::rewriteLocked() {
  const std::string tmp = path_ + ".tmp";
  FILE* out = std::fopen(tmp.c_str(), "wb");
  if (!out) {
    logf("gpu.pso.recipe what=rewrite_open_failed path=%s", tmp.c_str());
    return false;
  }
  uint64_t size = 0;
  size_t records = 0;
  auto emit = [&](uint8_t tag, const std::vector<uint8_t>& payload) {
    const uint32_t len = uint32_t(payload.size());
    const uint64_t off = size + kRecordHeader;
    std::fwrite(&tag, 1, 1, out);
    std::fwrite(&len, sizeof len, 1, out);
    if (len) std::fwrite(payload.data(), 1, len, out);
    size += kRecordHeader + len;
    ++records;
    return off;
  };
  std::fwrite(kMagic, 1, 4, out);
  const uint32_t hdr[2] = {kRecipeVersion, headerFields()};
  std::fwrite(hdr, sizeof hdr, 1, out);
  size = kHeaderBytes;
  std::unordered_map<uint64_t, SpirvRec> spirv;
  std::vector<uint32_t> words;
  for (const auto& [h, s] : spirv_) {
    if (!readWords(s, words)) continue;  // unreadable: its inputs stay usable while the version matches
    Writer w;
    w.pod(h);
    w.pod(uint32_t(words.size()));
    w.raw(words.data(), words.size() * 4);
    spirv[h] = {emit(kTagSpirv, w.b) + 12, uint32_t(words.size())};
  }
  std::unordered_map<uint64_t, SrcRef> refs;
  for (const auto& [id, b] : shaders_) {
    ShaderBlob full = b;
    if (!fillSource(id, full)) continue;  // unreadable text: the recipe goes (its pipelines are orphaned on load)
    Writer w;
    encodeShader(w, id, full);
    const uint64_t off = emit(kTagShader, w.b);
    refs[id] = SrcRef{off + sourceOffsetInPayload(full), uint32_t(full.source.size()), vpmsl::sourceHash(full.source)};
  }
  for (const auto& [h, rec] : inputs_) {
    Writer w;
    w.pod(h);
    w.pod(rec.translator);
    w.pod(rec.msl);
    encodeInput(w, rec.input);
    emit(kTagInput, w.b);
    if (auto f = ifaces_.find(h); f != ifaces_.end()) {
      Writer fw;
      fw.pod(h);
      fw.pod(f->second.translator);
      fw.pod(f->second.code);
      encodeIface(fw, f->second.iface);
      emit(kTagIface, fw.b);
    }
  }
  for (const auto& [key, p] : pipelines_) {
    Writer w;
    encodePipeline(w, p.state, p.inputs);
    emit(kTagPipeline, w.b);
  }
  const bool ok = std::fflush(out) == 0 && !std::ferror(out);
  std::fclose(out);
  std::error_code ec;
  if (!ok) {
    std::filesystem::remove(tmp, ec);
    logf("gpu.pso.recipe what=rewrite_failed");
    return false;
  }
  std::fclose(file_);
  file_ = nullptr;
  std::filesystem::rename(tmp, path_, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    logf("gpu.pso.recipe what=rewrite_rename_failed");
    openForAppend(false);  // the old file (and the old offsets) stay valid
    return false;
  }
  spirv_ = std::move(spirv);
  srcRef_ = std::move(refs);  // new file offsets; the texts leave memory again
  for (auto it = shaders_.begin(); it != shaders_.end();) {  // a text that could not be read back is not in the new file
    if (srcRef_.count(it->first)) {
      std::string().swap(it->second.source);
      ++it;
    } else {
      it = shaders_.erase(it);
    }
  }
  size_ = size;
  records_ = records;
  deadRecords_ = 0;
  driftWritten_ = false;
  openForAppend(false);
  return file_ != nullptr;
}

RecipeStore::Snapshot RecipeStore::snapshot() const {
  std::lock_guard lock(mutex_);
  Snapshot out;
  if (FILE* f = std::fopen(path_.c_str(), "rb")) {
    for (const auto& [id, b] : shaders_) {
      ShaderBlob full = b;
      if (fillSource(id, full, f)) out.shaders.emplace_back(id, std::move(full));
    }
    std::fclose(f);
  }
  auto has = [&](uint64_t id) { return !id || shaders_.count(id); };
  for (const auto& [k, p] : pipelines_)
    if (has(p.state.vertexShader) && has(p.state.fragmentShader) && has(p.state.computeShader))
      out.pipelines.push_back(p.state);
  return out;
}

std::vector<uint64_t> RecipeStore::keys() const {
  std::lock_guard lock(mutex_);
  std::vector<uint64_t> k;
  for (const auto& [key, p] : pipelines_) k.push_back(key);
  return k;
}

uint64_t RecipeStore::recipeId(const vpgpu::PipelineState& s, const std::array<uint64_t, 3>& inputs) {
  vpgpu::PipelineState bare = s;
  bare.vertexShader = bare.fragmentShader = bare.computeShader = 0;
  const uint64_t parts[4] = {bare.key().value, inputs[0], inputs[1], inputs[2]};
  return fnvBytes(reinterpret_cast<const uint8_t*>(parts), sizeof parts);
}

std::vector<uint64_t> RecipeStore::ids() const {
  std::lock_guard lock(mutex_);
  std::vector<uint64_t> k;
  k.reserve(byId_.size());
  for (const auto& [id, key] : byId_) k.push_back(id);
  return k;
}

bool RecipeStore::resolve(uint64_t idOrKey, uint64_t& key) const {
  std::lock_guard lock(mutex_);
  if (auto it = byId_.find(idOrKey); it != byId_.end()) return key = it->second, true;
  return pipelines_.count(idOrKey) != 0 && (key = idOrKey, true);
}

bool RecipeStore::stateFor(uint64_t key, vpgpu::PipelineState& out,
                           std::vector<std::pair<uint64_t, ShaderBlob>>* shaders) const {
  std::lock_guard lock(mutex_);
  auto it = pipelines_.find(key);
  if (it == pipelines_.end()) return false;
  out = it->second.state;
  if (shaders) {
    shaders->clear();
    for (int st = 0; st < 3; ++st) {
      const uint64_t id = *shaderField(out, st);
      if (!id) continue;
      auto sh = shaders_.find(id);
      if (sh == shaders_.end()) return false;
      ShaderBlob full = sh->second;
      if (!fillSource(id, full)) return false;
      shaders->emplace_back(id, std::move(full));
    }
  }
  return true;
}

bool RecipeStore::has(uint64_t key) const {
  std::lock_guard lock(mutex_);
  return pipelines_.count(key) != 0;
}

size_t RecipeStore::pipelineCount() const {
  std::lock_guard lock(mutex_);
  return pipelines_.size();
}

size_t RecipeStore::storedThisRun() const {
  std::lock_guard lock(mutex_);
  return storedThisRun_;
}

size_t RecipeStore::shaderCount() const {
  std::lock_guard lock(mutex_);
  return shaders_.size();
}

uint64_t RecipeStore::memoryBytes() const {
  std::lock_guard lock(mutex_);
  uint64_t n = size_;
  for (const auto& [id, b] : shaders_)
    n += sizeof(b) + b.source.capacity() + b.entry.capacity() + b.arguments.capacity() * sizeof(ShaderBlob::Arg);
  n += inputs_.size() * sizeof(InputRec) + ifaces_.size() * sizeof(IfaceRec) + spirv_.size() * sizeof(SpirvRec);
  for (const auto& [id, p] : pipelines_) n += sizeof(p) + p.state.label.capacity();
  return n;
}

std::vector<StageInput> RecipeStore::stageInputs() const {
  std::lock_guard lock(mutex_);
  std::vector<StageInput> v;
  for (const auto& [h, rec] : inputs_) v.push_back(rec.input);
  return v;
}

PrewarmResult prewarm(Platform& platform, vpgpu::PipelineCache& cache, RecipeStore& store,
                      const std::function<void(uint32_t, uint32_t)>& progress, double maxSeconds,
                      const PrewarmHooks& hooks) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  PrewarmResult r;
  auto cancelled = [&] { return hooks.cancel && hooks.cancel->load(); };
  // Recipes recorded by another translator version: retranslated from their SPIR-V first.
  const RecipeStore::RefreshResult rf =
      store.refresh(hooks.refreshThreads ? hooks.refreshThreads : std::max(2u, std::thread::hardware_concurrency()));
  r.retranslated = rf.retranslated;
  if (hooks.afterRefresh) hooks.afterRefresh();  // refresh() rekeys the pipelines of a changed translator: keys read earlier are stale
  if (cancelled()) return r;
  RecipeStore::Snapshot snap = store.snapshot();
  logf("gpu.pso.recipe loaded shaders=%zu pipelines=%zu retranslated=%u same_msl=%u failed=%u ms=%.0f",
       snap.shaders.size(), snap.pipelines.size(), rf.retranslated, rf.sameMsl, rf.failed, rf.ms);
  r.recipes = uint32_t(snap.pipelines.size());
  if (snap.pipelines.empty()) return r;
  {
    std::vector<uint64_t> src;
    src.reserve(snap.shaders.size());
    for (const auto& [id, b] : snap.shaders) src.push_back(vpmsl::sourceHash(b.source));
    addShaderLibRecipes(src);
  }
  for (auto& [id, b] : snap.shaders) {
    if (cancelled()) return r;
    if (hooks.addShader) hooks.addShader(id, std::move(b));
    else platform.addShader(id, std::move(b));
  }
  std::vector<uint64_t> keys;
  for (const auto& s : snap.pipelines) {
    const vpgpu::PipelineKey k = s.key();
    keys.push_back(k.value);
    if (cache.request(s, k, vpgpu::Priority::Prefetch) == vpgpu::PipelineCache::Status::Ready) ++r.hits;
  }
  // Forced MSAA: the variants the runtime will ask for on top of the recipes (pipelines.cpp pipelineFor / maskPipelineKey build
  // the same states through msaaVariantState / applyUiMaskTarget, so the keys agree). Without forcing nothing is added.
  uint32_t variantHits = 0;
  if (const int force = msaaForce(); force >= 2) {
    const bool masks = taaUiMaskOn();
    std::unordered_set<uint64_t> seen(keys.begin(), keys.end());
    std::vector<vpgpu::PipelineState> extra;
    for (const auto& s : snap.pipelines) forcedMsaaVariants(s, force, masks, extra);
    // Every variant stays resident with no owner: at most `cap`, in recipe order (the rest is only counted).
    const size_t cap = hooks.maxVariants ? hooks.maxVariants : kPrewarmMaxVariants;
    size_t total = 0;
    for (const auto& v : extra) {
      const vpgpu::PipelineKey k = v.key();
      if (!seen.insert(k.value).second) continue;
      if (++total > cap) continue;
      keys.push_back(k.value);
      if (cache.request(v, k, vpgpu::Priority::Prefetch) == vpgpu::PipelineCache::Status::Ready) ++variantHits;
    }
    r.variants = uint32_t(keys.size() - snap.pipelines.size());
    if (total > cap) logf("gpu.prewarm variants_capped=%u of %zu", r.variants, total);
    logf("gpu.prewarm variants=%u msaa=%d masks=%d hits=%u", r.variants, force, masks ? 1 : 0, variantHits);
  }
  for (;;) {
    uint32_t done = 0, failed = 0, doneV = 0, failedV = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      const bool variant = i >= r.recipes;
      uint32_t& d = variant ? doneV : done;
      uint32_t& f = variant ? failedV : failed;
      const auto st = cache.status({keys[i]});
      if (st == vpgpu::PipelineCache::Status::Ready) ++d;
      else if (st == vpgpu::PipelineCache::Status::Failed) ++d, ++f;
      else if (st == vpgpu::PipelineCache::Status::Missing) ++d;  // compiled, then released by the game: no longer ours to wait for
    }
    if (progress) progress(done, r.recipes);
    r.ready = done - failed;
    r.variantsFailed = failedV;
    if (done == r.recipes && doneV == r.variants) {
      r.failed = failed;
      break;
    }
    if (cancelled()) {
      r.failed = failed;
      return r;
    }
    // Bounded wait (game boot / watchdog): the rest keeps compiling on the workers.
    if (maxSeconds > 0 && std::chrono::duration<double>(Clock::now() - start).count() > maxSeconds) {
      logf("gpu.prewarm timeout done=%u of %u after %.0f s (rest continues in background)", done + doneV, r.recipes + r.variants,
           maxSeconds);
      r.failed = failed;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(std::max<uint32_t>(1, hooks.pollMs)));
  }
  r.compiled = r.recipes - r.failed - r.hits;
  r.ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  // hits = already compiled in memory; ar_* = the Metal binary archive (hits there skip the compile).
  logf("gpu.prewarm recipes=%u compiled=%u hits=%u failed=%u retranslated=%u ms=%.0f %s", r.recipes, r.compiled, r.hits,
       r.failed, r.retranslated, r.ms, platform.archiveSummary().c_str());
  if (r.variants) logf("gpu.prewarm variants_failed=%u of %u", r.variantsFailed, r.variants);
  return r;
}

}  // namespace vpvk

extern "C" {

__attribute__((visibility("default"))) uint32_t vpvk_prewarm_count(void) {
  const std::string dir = vpvk::RecipeStore::defaultDir();
  if (dir.empty()) return 0;
  vpvk::refreshTranslatorMode();
  vpvk::refreshRobustMode();
  vpvk::RecipeStore store(dir);
  return uint32_t(store.snapshot().pipelines.size());
}

// Standalone platform + device with every core as a pipeline worker (the game is not running).
// Destroying the platform saves the Metal binary archive; the game's device then loads it.
// Host (dlsym): the version tag of the translator's output (recipes.h kTranslatorBaseline ... kDriverVersion). The
// host stamps its first-run / AOT caches with it: it changes only when a component's output changes.
__attribute__((visibility("default"))) const char* vpvk_translator_tag() {
  static const std::string tag = vpvk::translatorTag();
  return tag.c_str();
}

namespace {
std::atomic<uint32_t> gPrewarmPhase{0};
}
// 0 = compiling the recipes, 1 = saving the pipeline archive (the progress callback of vpvk_prewarm counts entries archived).
__attribute__((visibility("default"))) uint32_t vpvk_prewarm_phase(void) { return gPrewarmPhase.load(); }

__attribute__((visibility("default"))) uint32_t vpvk_prewarm(vpvk_prewarm_progress progress, void* user) {
  const std::string dir = vpvk::RecipeStore::defaultDir();
  if (dir.empty()) {
    vpvk::log("gpu.prewarm skipped=no_cache_dir");
    return 0;
  }
  vpvk::refreshTranslatorMode();  // the environment the host set for this run (it may change between runs)
  vpvk::refreshRobustMode();
  vpvk::RecipeStore store(dir);
  if (!store.pipelineCount()) {
    vpvk::log("gpu.prewarm skipped=no_recipes");
    return 0;
  }
  std::unique_ptr<vpvk::Platform> platform = vpvk::createPlatform();
  const unsigned workers = std::max(2u, std::thread::hardware_concurrency());
  vpvk::PrewarmResult r;
  {
    vpgpu::Device gpu(platform->backend(), workers, [](const std::string& s) { vpvk::log(s); });
    r = vpvk::prewarm(*platform, gpu.pipelines(), store, [&](uint32_t done, uint32_t total) {
      if (progress) progress(done, total, user);
    });
  }
  // Workers are gone: the harvest (which never yields here, no frame waits for a pipeline) is flushed in full (`complete`: not
  // the 8 s of the end of a process, ~100 entries, which left ~2400 misses to compile again at every launch) and the archive
  // written before the platform goes, so the line below shows what became of every miss.
  const auto tf = std::chrono::steady_clock::now();
  gPrewarmPhase = 1;  // the progress callback now counts archived entries (vpvk_prewarm_phase)
  platform->flushCaches(true, [&](uint32_t done, uint32_t total) {
    if (progress) progress(done, total, user);
  });
  gPrewarmPhase = 0;
  vpvk::logf("gpu.prewarm archive_saved ms=%.0f %s",
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count(),
             platform->archiveSummary().c_str());
  platform.reset();
  vpvk::logf("gpu.prewarm app done recipes=%u workers=%u", r.recipes, workers);
  return r.recipes;
}

}  // extern "C"
