// Pass diagnostics of the translator (one global mechanism, no per-game code), all behind two environment switches read once:
//   VPVK_PASS_DUMP=1 | <start> | <start>,<frames>
//       Explains a few frames pass by pass. Frames with render work are counted from the first one; the dump covers `frames` of them
//       from number `start` on (defaults 1200 and 4, at most 8 frames), then a "vk.pdump.sum" line every 5 s (the totals since the
//       previous one). A dumped frame logs the core's "gpu.pdump.*" lines (diagnostics.h) plus, per pass index, "vk.pdump.drop"
//       (draws skipped or routed to a fallback, with the reason) and "vk.pdump.tex" (residency of the textures its draws sample).
//   VPVK_SKIP_PASS=<list>   e.g. "5,6" or "4-7" (at most 32 entries): the translator drops every draw of those passes, in every
//       frame (the pass keeps its load / store / clear; only its draws vanish). Off by default; "vk.skip_pass active" once when on.
// A pass index is the index of the pass among the commands of the vpgpu::Frame of one submission (the `pass=` of gpu.pdump.pass).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vpgpu {
class Frame;
}

namespace vpvk {

struct Image;

constexpr uint64_t kPassDumpStart = 1200;  // render frames before the dump window (warm-up)
constexpr uint32_t kPassDumpFrames = 4, kPassDumpMaxFrames = 8;
constexpr size_t kMaxSkipPasses = 32;
constexpr size_t kSkipSlots = 16;  // room for SkipReason::Count (commands.h asserts it)

// Why resolveBindings refused a draw ("Unbound" skips), told apart for the census.
enum class BindFail : uint8_t { None, SetNull, BindingMissing, BufferNull, BufferUnbacked, OffsetOob, TexelNull, SamplerNull, ImageNull,
                                PlaneMissing, Count };
constexpr const char* kBindFailNames[] = {"none",       "set_null",   "binding_missing", "buffer_null", "buffer_unbacked", "offset_oob",
                                          "texel_null", "sampler_null", "image_null",    "plane_missing"};
static_assert(sizeof kBindFailNames / sizeof *kBindFailNames == size_t(BindFail::Count));

// A sampled texture of a draw, as the pass dump sees it (flags of PassNote::tex).
enum TexFlag : uint8_t {
  kTexLe64 = 1,         // the texture the GPU has is 64 px or less on its larger side
  kTexDeclLe64 = 2,     // the game declared it 64 px or less
  kTexShrunk = 4,       // the texture budget dropped mips: smaller than declared
  kTexUnwritten = 8,    // sampled, not an attachment, and no copy into any mip was recorded (never uploaded / not streamed yet)
  kTexTop0Missing = 16  // some mips were uploaded but mip 0 never was
};
constexpr const char* kTexFlagNames[] = {"le64", "decl_le64", "shrunk", "unwritten", "top0_missing"};
constexpr size_t kTexFlags = 5;

// What resolveBindings saw while binding one draw (filled only while the dump is on).
struct BindCensus {
  BindFail fail = BindFail::None;
  std::vector<std::pair<uint64_t, uint8_t>> tex;  // sampled textures: core TextureId (generation << 32 | index), TexFlag bits
};

// The translator's account of one pass (or of the draws outside any pass: pass == kNoPass), by TranslateStats::passNotes.
constexpr uint32_t kNoPass = UINT32_MAX;
struct PassNote {
  uint32_t pass = kNoPass;  // index among the commands of the frame
  uint32_t draws = 0;       // draws that reached the frame
  uint32_t skipped[kSkipSlots] = {};
  uint32_t bindFail[size_t(BindFail::Count)] = {};
  uint32_t fallbackMask = 0, fallbackDepthTile = 0;  // draws that kept the plain pipeline for lack of a UI-mask / depth-in-tile variant
  std::map<uint64_t, uint8_t> tex;                   // distinct sampled textures of the pass -> TexFlag bits
};

struct PassDump {
  bool on = false;
  uint64_t start = kPassDumpStart;
  uint32_t frames = kPassDumpFrames;
  uint64_t seen = 0;  // frames with render work so far
};
// Parses the value of VPVK_PASS_DUMP (null, "", "0" = off).
PassDump parsePassDump(const char* value);
// Parses VPVK_SKIP_PASS: numbers and a-b ranges separated by commas; anything else is ignored.
std::vector<uint32_t> parseSkipPasses(const char* value);

// The process' settings: read from the environment at the first call (tests assign to them).
PassDump& passDump();
std::vector<uint32_t>& skipPasses();
// Counts a frame with render work; true when it falls inside the dump window.
bool passDumpNext(PassDump&);

// TexFlag bits of a sampled image.
uint8_t texCensusFlags(const Image&);
// Draws per command of a frame (0 for what is not a render pass): taken before and after the pipeline gate to see what it dropped.
std::vector<uint32_t> passDrawCounts(const vpgpu::Frame&);
// The "vk.pdump.drop" / "vk.pdump.tex" lines of a dumped frame (`frame` = the core's index of it): one of each per pass that has
// something to say. gateDropped[command] = draws the pipeline gate removed from that pass (may be empty).
std::vector<std::string> passDumpLines(uint64_t frame, const std::vector<PassNote>&, const std::vector<uint32_t>& gateDropped);
// Adds a frame to the totals of the next "vk.pdump.sum" line; gateSkipped = draws the gate removed from the whole frame.
void passDumpAccumulate(const std::vector<PassNote>&, uint32_t gateSkipped);
// The "vk.pdump.sum" line (totals since the previous one) once 5 s passed since it, else "". nowNs < 0 = the steady clock.
std::string passDumpSummary(int64_t nowNs = -1);

}  // namespace vpvk
