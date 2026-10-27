// MSAA on the host, from what Liverpool says about it (pure functions, no GPU objects).
//
// Liverpool has two sample counts per surface: coverage samples (CB_COLORn_ATTRIB.NUM_SAMPLES,
// DB_Z_INFO.NUM_SAMPLES) and colour fragments (NUM_FRAGMENTS, EQAA: more coverage samples than
// stored colours). VPKEngine draws with one count per render pass, never above what the app lets
// the game have, and never above the colour fragments (the host has no EQAA: the extra coverage
// samples of the depth buffer are dropped).
#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "vpgpu/core/commands.h"

namespace vpps4 {

struct SamplePlan {
    uint8_t host = 1;      // samples per pixel of the host attachments (power of two, 1..4 on Apple GPUs)
    uint8_t guest = 1;     // the most any bound surface has (fragments for colour, samples for depth)
    bool eqaa = false;     // a depth surface has more samples than the colour fragments
    bool reduced = false;  // host < guest
};

// `colorFragments`: one entry per bound colour target (0 = slot unused, else its fragment count);
// `depthSamples`: 0 when there is no depth target; `cap`: the app's limit (>= 1, not necessarily a
// power of two). The pass count is the smallest of the surfaces' counts and the cap, rounded down
// to a power of two: Metal needs one count for the whole pass, and a depth buffer with more samples
// than the colour keeps the colour's (EQAA).
SamplePlan planSamples(std::span<const uint8_t> colorFragments, uint8_t depthSamples, uint8_t cap);

// What an FMASK read returns under the host's scheme: the host stores sample i of a pixel in slot i
// of the multisampled texture (no compression), so the mask is the identity map. Nibble s of the
// 64-bit value (lo = samples 0-7, hi = 8-15) is the fragment sample s lives in: s % fragments, which
// for fragments == coverage samples is s itself and for EQAA spreads the extra coverage samples over
// the stored fragments. Samples at or beyond `coverage` read 0.
struct FmaskValue {
    uint32_t lo = 0, hi = 0;
    bool operator==(const FmaskValue&) const = default;
};
FmaskValue fmaskIdentity(uint8_t coverage, uint8_t fragments);

// Data formats of FMASK T#s (IMG_DATA_FORMAT 47..56).
inline constexpr bool isFmaskFormat(uint8_t dataFormat) { return dataFormat >= 47 && dataFormat <= 56; }

// Depth resolve under the host's rule. The title's own resolve shader decides in the guest; when the
// hardware does it (DB depth copy) it takes the sample named by DB_RENDER_CONTROL.COPY_SAMPLE.
// Metal can keep sample 0, the minimum or the maximum: a copy of sample 0 is sample 0; no choice
// made by the title keeps the sample nearest to the camera, which is the minimum for a conventional
// depth test (LESS, LEQUAL) and the maximum for a reversed one (GREATER, GEQUAL).
vpgpu::ResolveMode depthResolveMode(bool hardwareCopy, uint8_t copySample, uint8_t depthFunc);

// The average of `samples` texels (colour resolve, per channel). Reference for the backend's
// MTLStoreActionMultisampleResolve, used by the tests; integer formats keep sample 0 on the host.
std::array<float, 4> resolveColor(std::span<const std::array<float, 4>> samples);

// The colour a fast clear (CB_COLORn_CLEAR_WORD0/1, in the surface's own format and component order in
// memory) clears to, as the logical RGBA a render pass clears with (Metal converts it to the pixel
// format; sRGB surfaces hold the encoded value, converted to linear here). `exact` is cleared for formats
// whose words are not decoded (10_11_11, 5_6_5 family, 32_32_32_32: only two words exist).
std::array<double, 4> fastClearColor(uint8_t cbFormat, uint8_t numberType, uint8_t compSwap,
                                     const std::array<uint32_t, 2>& words, bool& exact);

// A colour resolve done by the render pass that wrote the target: `inPass` when the last pass that
// drew into the multisampled source can store-and-resolve it (the source is still the pass's
// attachment and the formats match), else a resolve pass of its own (load the samples, store the
// resolve). Pure decision, tested.
enum class ResolveWay : uint8_t { InPass, OwnPass, Copy, Skip };
struct ResolveCase {
    bool sourceMultisampled = false;
    bool formatsMatch = true;
    bool formatsViewable = false;        // formats differ but have one texel size: the resolve goes through a view
    bool lastPassOpenForSource = false;  // the recording's last pass has the source as attachment, no resolve target yet
    bool destinationIsSource = false;
    bool destinationMultisampled = false;
};
ResolveWay planResolve(const ResolveCase&);

}  // namespace vpps4
