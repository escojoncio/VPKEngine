// Module-level transformations. Both passes take and return verified modules.
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "vpir/ir.h"

namespace vpir {

// Constant folding (integer and exact f32 arithmetic, compares, conversions, vector building),
// algebraic identities that are exact (x+0, x*1, x|0, select on a constant…), common-subexpression
// elimination over pure ops (a value from an enclosing block is reused inside nested blocks) and
// removal of pure values left unused. Never touches `precise` float math beyond exact identities.
void optimize(Module& module);

struct FetchRejection {
    std::string function;
    ValueId value = kNoValue;  // result of the image op that was left alone
    std::string reason;
};

// Rewrites reads of G-buffer textures into framebuffer_fetch so Apple GPUs read them from tile
// memory. `attachments` maps a texture slot (Resource::index of a `texture` resource) to the colour
// attachment index (0..7) that holds the same image in the current render pass; `nearestSamplers`
// lists sampler slots (Resource::index) known to use nearest filtering. A use is rewritten only when
// it provably reads this fragment's own pixel:
//   * image_fetch on a non-array 2d colour texture, coordinate = convert<vec2<u32>>(position.xy)
//     (optionally through floor/trunc), no lod or a constant lod of 0;
//   * image_sample on such a texture through a nearest sampler, coordinate = position.xy divided by
//     (or multiplied by rcp of) convert<vec2<f32>>(image_size of the same texture, lod 0), with no
//     offset, bias or gradient and no lod or a constant lod of 0. Position is the pixel centre (or a
//     sample position inside the pixel), so nearest filtering picks this pixel's texel.
// Every other use of a mapped slot is left untouched and reported. Textures and samplers in the maps
// that end up unreferenced are removed from the module.
std::vector<FetchRejection> toFramebufferFetch(Module& module, const std::map<uint32_t, uint32_t>& attachments,
                                               const std::set<uint32_t>& nearestSamplers);

}  // namespace vpir
