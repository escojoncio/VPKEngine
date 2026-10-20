#include "vpps4/foveation.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <cstdlib>

namespace vpps4 {

void foveationRates(uint16_t zones, const FoveationParams& p, float* out) {
    for (uint16_t i = 0; i < zones; ++i) {
        // Distance of the zone's centre to the middle of the axis: 0 at the middle, 1 at the edge.
        const float d = zones > 1 ? std::abs((float(i) + 0.5f) / float(zones) - 0.5f) * 2.0f : 0.0f;
        const float rate = 1.0f - std::clamp(p.strength, 0.0f, 1.0f) * d * d;
        out[i] = std::clamp(rate, std::clamp(p.minRate, 0.05f, 1.0f), 1.0f);
    }
}

std::shared_ptr<const vp4gpu::RateMap> makeFoveationMap(uint32_t width, uint32_t height, uint16_t layers,
                                                        const FoveationParams& p) {
    if (layers != 2 || width < 64 || height < 64 || width > 16384 || height > 16384 || p.zones < 3) return nullptr;
    auto m = std::make_shared<vp4gpu::RateMap>();
    m->width = width;
    m->height = height;
    m->layers = layers;
    m->zonesX = m->zonesY = p.zones;
    m->qx.resize(size_t(layers) * p.zones);
    m->qy.resize(size_t(layers) * p.zones);
    float rates[64];
    const uint16_t zones = std::min<uint16_t>(p.zones, 64);
    m->zonesX = m->zonesY = zones;
    m->qx.resize(size_t(layers) * zones);
    m->qy.resize(size_t(layers) * zones);
    foveationRates(zones, p, rates);
    double mean = 0;
    for (uint16_t i = 0; i < zones; ++i) mean += rates[i];
    mean /= zones;
    for (uint16_t l = 0; l < layers; ++l)
        for (uint16_t i = 0; i < zones; ++i) m->qx[size_t(l) * zones + i] = m->qy[size_t(l) * zones + i] = rates[i];
    m->keptFraction = float(mean * mean);
    // Identity: the shape and the parameters.
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&](uint64_t v) { h = (h ^ v) * 0x100000001b3ull; };
    mix(width), mix(height), mix(layers), mix(zones);
    mix(uint64_t(std::lround(p.strength * 1000))), mix(uint64_t(std::lround(p.minRate * 1000)));
    m->key = h ? h : 1;
    m->label = "ps4 eye " + std::to_string(width) + "x" + std::to_string(height);
    return m;
}

}  // namespace vpps4
