// Foveated rendering of a two-layer eye pass (an array colour target of two slices, both eyes drawn by
// one pass): the rate map the backend turns into an MTLRasterizationRateMap (gpu/metal: RenderPassDesc::
// rateMap). The fovea is fixed at the centre of each eye (no eye tracking in the PS4 path); quality falls
// with the square of the distance to it, separately along x and y (rate maps are separable).
#pragma once

#include <cstdint>
#include <memory>

#include "vp4gpu/core/commands.h"

namespace vpps4 {

struct FoveationParams {
    float strength = 0.6f;    // how much quality is lost at the edge (0: none, 1: down to minRate)
    float minRate = 0.25f;    // the lowest rate of a zone
    uint16_t zones = 17;      // zones per axis (odd: one is centred)
};

// Null when the pass cannot be foveated (not two layers, or a size that rate maps cannot take).
std::shared_ptr<const vp4gpu::RateMap> makeFoveationMap(uint32_t width, uint32_t height, uint16_t layers,
                                                        const FoveationParams& = {});

// Rates of `zones` zones across one axis (1 at the centre).
void foveationRates(uint16_t zones, const FoveationParams&, float* out);

}  // namespace vpps4
