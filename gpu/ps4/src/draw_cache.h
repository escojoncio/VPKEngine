// The draw cache: what Renderer::draw resolves from the state of a draw, kept by its content, so that a later draw with
// the same content can take it instead of resolving it again. Two blocks per draw: the fixed one (targets, pass, fixed
// state: all of the DrawState but the programs, what they read and the geometry) and one per stage (its translation and
// what its bindings resolved to). The addresses of the data a draw binds are in neither key: those are bound in every draw.
// A block is good while its witness is: the counters of everything that can make it stale.
// Nothing here knows the renderer (tests/draw_cache_tests.cpp covers it alone).
#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "vpps4/state.h"

namespace vpps4::dc {

inline uint64_t hashBytes(const void* p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const auto mix = [&h](uint64_t w) {
        h = (h ^ w) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 32;
    };
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t w;
    for (; n >= 8; n -= 8, b += 8) std::memcpy(&w, b, 8), mix(w);
    if (n) {
        w = 0;
        std::memcpy(&w, b, n);
        mix(w ^ (uint64_t(n) << 56));
    }
    return h;
}

// ---- The witness ---------------------------------------------------------------------------------------------------
// What a block was resolved under: the frame, the tracker's intervals (MemoryTracker::layout), the surfaces that exist
// (targets and textures made, destroyed, replaced), the open render pass, the clears recorded. A block is good while
// the parts it depends on are what they were.
struct Witness {
    enum Part : unsigned { Frame, Layout, Surface, Pass, Clear, Parts };
    static constexpr unsigned kAll = (1u << Parts) - 1;
    std::array<uint64_t, Parts> part{};
    // The parts of `mask` (a bit each) that are not what they are `now`; 0: good.
    unsigned stale(const Witness& now, unsigned mask = kAll) const {
        unsigned out = 0;
        for (unsigned p = 0; p < Parts; ++p)
            if ((mask >> p & 1) && part[p] != now.part[p]) out |= 1u << p;
        return out;
    }
};
inline constexpr const char* kWitnessParts[Witness::Parts] = {"frame", "layout", "surface", "pass", "clear"};

// ---- The fixed key -------------------------------------------------------------------------------------------------
// The DrawState without what changes from one draw to the next of a run: the programs (code, fetch, address), their user
// data, the geometry (indices, counts, first vertex, indirect arguments). Its fields one after the other, as bytes (no
// padding gets in), and the host's sample limit, which the targets follow.
struct FixedKey {
    static constexpr size_t kCapacity = sizeof(DrawState) + 1;  // (its fields are part of the DrawState's own bytes: they fit)
    std::array<uint8_t, kCapacity> bytes;  // (only the first `size`)
    uint32_t size = 0;
    bool operator==(const FixedKey& o) const { return size == o.size && std::memcmp(bytes.data(), o.bytes.data(), size) == 0; }
    uint64_t hash() const { return hashBytes(bytes.data(), size); }
};

namespace detail {
// Writes the fields of the state structs. Each struct is taken apart whole (a field added to it does not compile until
// it is given a place here).
#define VPPS4_DC_FIELDS(object, ...)                 \
    {                                                \
        const auto& [__VA_ARGS__] = object;          \
        all(__VA_ARGS__);                            \
    }
struct Put {
    FixedKey& k;
    template <class... T>
    void all(const T&... v) {
        (put(v), ...);
    }
    template <class T>
    void put(const T& v) {
        static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);
        std::memcpy(k.bytes.data() + k.size, &v, sizeof v);
        k.size += uint32_t(sizeof v);
    }
    template <class T, size_t N>
    void put(const std::array<T, N>& a) {
        for (const T& x : a) put(x);
    }
    void put(const ColorTarget& c) {
        VPPS4_DC_FIELDS(c, address, width, height, pitch, format, numberType, compSwap, tileMode, sliceStart, sliceCount, samples, writeMask,
                        forceDstAlpha1, blendClamp, blendBypass, coverage, fastClear, clearWord)
    }
    void put(const DepthTarget& z) {
        VPPS4_DC_FIELDS(z, zAddress, stencilAddress, width, height, zFormat, stencilFormat, tileMode, sliceStart, sliceCount, samples, depthClear,
                        stencilClear, clearDepth, clearStencil, zWriteAddress, stencilWriteAddress, zReadOnly, stencilReadOnly, forceZDirty,
                        forceZValid, forceStencilDirty, forceStencilValid, boundsMin, boundsMax, depthCopy, copySample, pitch, rows)
    }
    void put(const Blend& b) {
        VPPS4_DC_FIELDS(b, enable, separateAlpha, colorSrc, colorDst, colorFunc, alphaSrc, alphaDst, alphaFunc, disableRop3)
    }
    void put(const DepthStencil::Face& f) { VPPS4_DC_FIELDS(f, func, fail, zPass, zFail, ref, readMask, writeMask, opVal) }
    void put(const DepthStencil& s) {
        VPPS4_DC_FIELDS(s, depthTest, depthWrite, depthFunc, stencilTest, backfaceEnable, depthBounds, colorWritesOnDepthFail,
                        noColorWritesOnDepthPass, front, back)
    }
    void put(const Viewport& v) { VPPS4_DC_FIELDS(v, xScale, xOffset, yScale, yOffset, zScale, zOffset, xScreen, yScreen) }
    void put(const Scissor& s) { VPPS4_DC_FIELDS(s, left, top, right, bottom) }
    void put(const ScissorRegs& s) {
        VPPS4_DC_FIELDS(s, screen, window, generic, viewportScissor, windowOffsetX, windowOffsetY, windowOffsetDisable, genericOffsetDisable,
                        viewportScissorEnable, present)
    }
    void put(const Raster& r) {
        VPPS4_DC_FIELDS(r, cullFront, cullBack, frontFaceClockwise, offsetFront, offsetBack, offsetPara, offsetClamp, frontScale, frontOffset,
                        backScale, backOffset, dbFmtValid, dbIsFloat, dbBits, clipDisable, zclipNearDisable, zclipFarDisable, clipSpaceZeroToW,
                        userClipPlanes, vsOutputsViewportIndex, vsOutputsClipDistances, polygonModeEnable, polygonModeFront, polygonModeBack,
                        provokingLast, lineWidth, primitiveRestart, restartIndex)
    }
    void put(const GeometryState& g) { VPPS4_DC_FIELDS(g, enabled, esgsItemDwords, gsvsVertDwords, maxVertOut, outPrim, gsInstances) }
    // (a stage's shape: what the registers say of its inputs. Its program and its user data are the stage key's)
    void put(const Stage& s) {
        [[maybe_unused]] const auto& [code, fetch, address, userData, userSgprs, vgprCompCnt, vsOutCntl, geom, geomInputPrim, rectList,
                                      psInputAddr, psIterSample, exportFormat, psInputCntl, psNumInterp, dualSource, clipSpaceGL,
                                      instanceStepRate, workgroupSize, tgidEnable, tgSizeEnable, tidigCompCnt, ldsBytes] = s;
        all(userSgprs, vgprCompCnt, vsOutCntl, geom, geomInputPrim, rectList, psInputAddr, psIterSample, exportFormat, psInputCntl,
            psNumInterp, dualSource, clipSpaceGL, instanceStepRate, workgroupSize, tgidEnable, tgSizeEnable, tidigCompCnt, ldsBytes);
    }
    void put(const DrawState& d) {
        [[maybe_unused]] const auto& [color, blend, blendColor, depth, depthStencil, viewport, scissor, scissorRegs, raster, rop3, degamma,
                                      alphaToMask, alphaToMaskOffsets, earlyZ, depthBeforeShader, psKill, psExportsZ, psExportsStencil,
                                      psExportsMask, primitive, indexAddress, index32, indexSwap, count, instances, firstVertex,
                                      indirectAddress, indirectCountAddress, drawIndex, vs, ps, es, gs, geom] = d;
        all(color, blend, blendColor, depth, depthStencil, viewport, scissor, scissorRegs, raster, rop3, degamma, alphaToMask,
            alphaToMaskOffsets, earlyZ, depthBeforeShader, psKill, psExportsZ, psExportsStencil, psExportsMask, primitive, vs, ps, es, gs,
            geom);
    }
};
#undef VPPS4_DC_FIELDS
}  // namespace detail

inline void fixedKey(const DrawState& d, uint8_t maxSamples, FixedKey& k) {
    k.size = 0;
    detail::Put put{k};
    put.put(d);
    put.put(maxSamples);
}

// ---- The stage key -------------------------------------------------------------------------------------------------
// A stage of a draw: its program (the renderer's key of it: the words and what the registers say of its inputs), what
// else its translation follows (the colour exports, the view index, the alpha test) and the descriptors its bindings
// read, binding after binding. A buffer's base address is left out: the buffer is bound in every draw.
struct StageKey {
    uint64_t program = 0, env = 0;
    uint8_t stage = 0;
    std::vector<uint32_t> words;
    void clear() { program = env = 0, stage = 0, words.clear(); }
    void texture(const uint32_t* w, unsigned dwords) { head(1, dwords), words.insert(words.end(), w, w + dwords); }
    // (V#: the base is dword 0 and the low 16 bits of dword 1: 12 of address, 4 the hardware ignores)
    void buffer(const uint32_t* w, unsigned dwords) {
        head(2, dwords);
        for (unsigned i = 0; i < dwords; ++i) words.push_back(i == 0 ? 0 : i == 1 ? w[i] & 0xFFFF0000u : w[i]);
    }
    void sampler(uint64_t id) { head(3, 2), words.push_back(uint32_t(id)), words.push_back(uint32_t(id >> 32)); }
    void other() { head(0, 0); }  // a binding with nothing of its own in the key (bound in every draw)
    bool operator==(const StageKey& o) const { return program == o.program && env == o.env && stage == o.stage && words == o.words; }
    uint64_t hash() const {
        const uint64_t fixed[3] = {program, env, stage};
        return hashBytes(words.data(), words.size() * 4, hashBytes(fixed, sizeof fixed));
    }

private:
    void head(uint32_t kind, unsigned dwords) { words.push_back(kind << 28 | dwords); }
};

// ---- Blocks --------------------------------------------------------------------------------------------------------
// (`hash`, `used` and `live` are the table's: the key's hash, the frame it was last claimed or found good in, in use)
struct FixedBlock {
    FixedKey key;
    uint64_t hash = 0, used = 0;
    bool live = false;
    Witness seal;
    std::vector<uint8_t> resolved;  // what the draw's prologue resolved, field after field
};

// What a binding of a stage resolved to. Texture: a texture of guest memory no GPU writer has an interval in, good under
// the block's seal; Sampler: the same for the same descriptor; Buffer, Live (a texture an image backs, the user data):
// resolved in every draw, only their place in the stage's bindings is kept.
struct BindItem {
    enum Kind : uint8_t { Texture, Sampler, Buffer, Live };
    Kind kind = Live;
    uint8_t bind = 0, stages = 0;  // (vpgpu::BindKind, vpgpu::StageMask)
    bool write = false;
    uint16_t slot = 0;
    uint64_t id = 0, view = 0;  // Texture, Sampler: the resource bound and the view of it
    // (the same place and kind of binding, whatever is bound)
    bool sameShape(const BindItem& o) const { return kind == o.kind && bind == o.bind && stages == o.stages && write == o.write && slot == o.slot; }
};
struct StageBlock {
    StageKey key;
    uint64_t hash = 0, used = 0;
    bool live = false;
    Witness seal;
    uint64_t translated = 0;  // the translation the stage got (its shader id)
    std::vector<BindItem> items;
};

// ---- Tables --------------------------------------------------------------------------------------------------------
// Open addressing: a key lives in one of the kRun slots from its hash's on. When those are all taken by other keys the
// one used longest ago (Block::used, a frame) gives its place. No memory until the first block is made.
template <class Block>
class Table {
public:
    static constexpr size_t kRun = 8;
    explicit Table(size_t slots) : mask_(slots - 1) { assert(slots >= kRun && (slots & mask_) == 0); }
    template <class Key>
    Block* find(const Key& key, uint64_t hash) {
        if (blocks_.empty()) return nullptr;
        for (size_t i = 0; i < kRun; ++i) {
            Block& b = blocks_[(hash + i) & mask_];
            if (b.live && b.hash == hash && b.key == key) return &b;
        }
        return nullptr;
    }
    // A block for a key find() does not have: a free slot of its run, else the one of the run used longest ago (what it
    // held is lost). The caller fills the rest of it.
    template <class Key>
    Block& make(const Key& key, uint64_t hash, uint64_t frame) {
        if (blocks_.empty()) blocks_.resize(mask_ + 1);
        Block* at = nullptr;
        for (size_t i = 0; i < kRun; ++i) {
            Block& b = blocks_[(hash + i) & mask_];
            if (!b.live) {
                at = &b;
                break;
            }
            if (!at || b.used < at->used) at = &b;
        }
        if (at->live) ++evictions_;
        else ++live_;
        at->live = true, at->hash = hash, at->used = frame, at->key = key;
        return *at;
    }
    size_t slots() const { return mask_ + 1; }
    size_t live() const { return live_; }
    uint64_t evictions() const { return evictions_; }

private:
    std::vector<Block> blocks_;
    size_t mask_;
    size_t live_ = 0;
    uint64_t evictions_ = 0;
};

}  // namespace vpps4::dc
