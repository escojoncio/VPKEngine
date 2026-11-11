#include "passdump.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>

#include "commands.h"

namespace vpvk {

PassDump parsePassDump(const char* v) {
  PassDump p;
  if (!v || !*v || (v[0] == '0' && !v[1])) return p;
  p.on = true;
  char* end = nullptr;
  const unsigned long long a = std::strtoull(v, &end, 10);
  if (a > 1) p.start = a;  // 0 and 1 mean the default
  if (end && *end == ',')
    if (const unsigned long long n = std::strtoull(end + 1, nullptr, 10)) p.frames = uint32_t(std::min<unsigned long long>(n, kPassDumpMaxFrames));
  return p;
}

std::vector<uint32_t> parseSkipPasses(const char* v) {
  std::set<uint32_t> out;  // (duplicates go before the cap counts)
  while (v && *v) {
    char* end = nullptr;
    const unsigned long a = std::strtoul(v, &end, 10);
    if (end == v) {
      ++v;
      continue;
    }
    unsigned long b = a;
    if (*end == '-') {
      char* end2 = nullptr;
      const unsigned long x = std::strtoul(end + 1, &end2, 10);
      if (end2 != end + 1) b = x, end = end2;
    }
    for (unsigned long i = a; i <= b && i < 100000 && out.size() < kMaxSkipPasses; ++i) out.insert(uint32_t(i));
    v = end;
  }
  return {out.begin(), out.end()};
}

PassDump& passDump() {
  static PassDump p = [] {
    PassDump d = parsePassDump(std::getenv("VPVK_PASS_DUMP"));
    if (d.on) logf("vk.pdump armed start=%llu frames=%u (VPVK_PASS_DUMP: render frames from number start; one vk.pdump.sum every 5 s)",
                   (unsigned long long)d.start, d.frames);
    return d;
  }();
  return p;
}

std::vector<uint32_t>& skipPasses() {
  static std::vector<uint32_t> v = [] {
    const char* env = std::getenv("VPVK_SKIP_PASS");
    std::vector<uint32_t> l = parseSkipPasses(env);
    if (l.empty() && env && *env) {
      std::string shown(env);
      shown = shown.substr(0, 64);
      for (char& c : shown)
        if (c == ' ' || c == '=' || c == '\n') c = '_';
      log("vk.skip_pass ignored value=" + shown + " (VPVK_SKIP_PASS: no valid pass index; expected e.g. 5,6 or 4-7)");
    }
    if (!l.empty()) {
      std::string s;
      for (uint32_t i : l) s += (s.empty() ? "" : ",") + std::to_string(i);
      log("vk.skip_pass active passes=[" + s + "] (VPVK_SKIP_PASS: every draw of those pass indices is dropped, in every frame)");
    }
    return l;
  }();
  return v;
}

bool passDumpNext(PassDump& p) {
  const uint64_t n = p.seen++;
  return n >= p.start && n < p.start + p.frames;
}

uint8_t texCensusFlags(const Image& im) {
  uint8_t f = 0;
  if (std::max(im.desc.width, im.desc.height) <= 64) f |= kTexLe64;
  if (std::max(im.info.extent.width, im.info.extent.height) <= 64) f |= kTexDeclLe64;
  if (im.mipDrop) f |= kTexShrunk;
  if (!(im.info.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) && !im.linear) {
    const uint32_t written = im.mipsWritten.load(std::memory_order_relaxed);
    if (!written) f |= kTexUnwritten;
    else if (!(written & 1u)) f |= kTexTop0Missing;
  }
  return f;
}

std::vector<uint32_t> passDrawCounts(const vpgpu::Frame& frame) {
  std::vector<uint32_t> out;
  for (const vpgpu::Command& c : frame.commands()) {
    const auto* rp = std::get_if<vpgpu::RenderPass>(&c);
    out.push_back(rp ? uint32_t(rp->draws.size()) : 0u);
  }
  return out;
}

namespace {
// "name=n,name=n" of the non-zero counters, "-" when all are zero.
template <class T>
std::string nonZero(const T* counts, const char* const* names, size_t n) {
  std::string s;
  for (size_t i = 0; i < n; ++i)
    if (counts[i]) s += (s.empty() ? "" : ",") + std::string(names[i]) + "=" + std::to_string(counts[i]);
  return s.empty() ? "-" : s;
}
std::string passName(uint32_t pass) { return pass == kNoPass ? "-" : std::to_string(pass); }
}  // namespace

std::vector<std::string> passDumpLines(uint64_t frame, const std::vector<PassNote>& notes, const std::vector<uint32_t>& gateDropped) {
  std::vector<std::string> out;
  for (const PassNote& n : notes) {
    uint32_t skipped = 0;
    for (uint32_t s : n.skipped) skipped += s;
    const uint32_t gate = n.pass < gateDropped.size() ? gateDropped[n.pass] : 0;
    const std::string head = " f=" + std::to_string(frame) + " pass=" + passName(n.pass);
    if (skipped || gate || n.fallbackMask || n.fallbackDepthTile)
      out.push_back("vk.pdump.drop" + head + " draws=" + std::to_string(n.draws) + " skipped=" + std::to_string(skipped) +
                    " reasons{" + nonZero(n.skipped, kSkipReasonNames, size_t(SkipReason::Count)) + "} unbound{" +
                    nonZero(n.bindFail, kBindFailNames, size_t(BindFail::Count)) + "} fallback{mask_plain=" + std::to_string(n.fallbackMask) +
                    ",depth_tile_plain=" + std::to_string(n.fallbackDepthTile) + "} gate_dropped=" + std::to_string(gate));
    if (n.tex.empty()) continue;
    uint32_t flagged[kTexFlags] = {}, full = 0;
    for (const auto& [id, flags] : n.tex) {
      full += flags == 0;
      for (size_t b = 0; b < kTexFlags; ++b) flagged[b] += (flags >> b) & 1;
    }
    out.push_back("vk.pdump.tex" + head + " sampled=" + std::to_string(n.tex.size()) + " full=" + std::to_string(full) + " " +
                  nonZero(flagged, kTexFlagNames, kTexFlags));
  }
  return out;
}

namespace {
struct Totals {
  uint64_t frames = 0, passes = 0, draws = 0, gateSkipped = 0, fallbackMask = 0, fallbackDepthTile = 0, sampled = 0;
  uint64_t skipped[kSkipSlots] = {}, bindFail[size_t(BindFail::Count)] = {}, tex[kTexFlags] = {};
};
struct Summary {
  std::mutex m;
  Totals t;
  int64_t startNs = -1, nextNs = 0;
  unsigned lines = 0;
};
Summary& summary() {
  static Summary s;
  return s;
}
}  // namespace

void passDumpAccumulate(const std::vector<PassNote>& notes, uint32_t gateSkipped) {
  Summary& s = summary();
  std::lock_guard<std::mutex> lock(s.m);
  ++s.t.frames;
  s.t.gateSkipped += gateSkipped;
  for (const PassNote& n : notes) {
    ++s.t.passes;
    s.t.draws += n.draws;
    s.t.fallbackMask += n.fallbackMask, s.t.fallbackDepthTile += n.fallbackDepthTile;
    for (size_t i = 0; i < kSkipSlots; ++i) s.t.skipped[i] += n.skipped[i];
    for (size_t i = 0; i < size_t(BindFail::Count); ++i) s.t.bindFail[i] += n.bindFail[i];
    s.t.sampled += n.tex.size();
    for (const auto& [id, flags] : n.tex)
      for (size_t b = 0; b < kTexFlags; ++b) s.t.tex[b] += (flags >> b) & 1;
  }
}

std::string passDumpSummary(int64_t nowNs) {
  if (nowNs < 0) nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  Summary& s = summary();
  std::lock_guard<std::mutex> lock(s.m);
  if (s.startNs < 0) s.startNs = nowNs, s.nextNs = nowNs + 5'000'000'000LL;
  if (nowNs < s.nextNs || s.lines >= 720 || !s.t.frames) return {};
  s.nextNs = nowNs + 5'000'000'000LL;
  ++s.lines;
  const Totals& t = s.t;
  const std::vector<uint32_t>& skip = skipPasses();
  std::string sp;
  for (uint32_t i : skip) sp += (sp.empty() ? "" : ",") + std::to_string(i);
  std::string l = "vk.pdump.sum t=" + std::to_string((nowNs - s.startNs) / 1'000'000'000LL) + "s frames=" + std::to_string(t.frames) +
                  " render_seen=" + std::to_string(passDump().seen) + " passes=" + std::to_string(t.passes) + " draws=" + std::to_string(t.draws) +
                  " gate_dropped=" + std::to_string(t.gateSkipped) + " skipped{" + nonZero(t.skipped, kSkipReasonNames, size_t(SkipReason::Count)) +
                  "} unbound{" + nonZero(t.bindFail, kBindFailNames, size_t(BindFail::Count)) + "} fallback{mask_plain=" +
                  std::to_string(t.fallbackMask) + ",depth_tile_plain=" + std::to_string(t.fallbackDepthTile) + "} tex{sampled=" +
                  std::to_string(t.sampled) + "," + nonZero(t.tex, kTexFlagNames, kTexFlags) + "} hide_mesh=" +
                  std::to_string(gHideDropped.load(std::memory_order_relaxed)) + " skip_pass=" + (sp.empty() ? "off" : "[" + sp + "]");
  s.t = Totals{};
  return l;
}

}  // namespace vpvk
