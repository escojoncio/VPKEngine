// The draw cache's keys, witness and tables (src/draw_cache.h), without a renderer.
#include <cstdio>
#include <cstring>
#include <new>
#include <string_view>
#include <vector>

#include "../src/draw_cache.h"

namespace {
int failures = 0;
#define CHECK(c)                                                                       \
    do {                                                                               \
        if (!(c)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c); \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

namespace dc = vpps4::dc;
using vpps4::DrawState;
using vpps4::Stage;

dc::FixedKey keyOf(const DrawState& d, uint8_t maxSamples = 4) {
    dc::FixedKey k;
    dc::fixedKey(d, maxSamples, k);
    return k;
}

// Every field of the DrawState, changed alone: the key follows it, or (the programs, their user data, the geometry) does not.
void fixedKey() {
    static const uint32_t words[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    DrawState base;
    base.color[0] = {0x1000, 64, 32, 64, 10, 0, 0, 14};
    base.depth.zAddress = 0x2000, base.depth.zFormat = 3, base.depth.width = 64, base.depth.height = 32;
    base.count = 3;
    const dc::FixedKey k0 = keyOf(base);
    CHECK(k0.size > 0 && k0.size <= dc::FixedKey::kCapacity);
    CHECK(keyOf(base) == k0 && keyOf(base).hash() == k0.hash());
#define DIFFERS(change)                                                \
    do {                                                               \
        DrawState d = base;                                            \
        change;                                                        \
        const dc::FixedKey k = keyOf(d);                               \
        CHECK(k.size == k0.size && !(k == k0) && k.hash() != k0.hash()); \
    } while (0)
#define SAME(change)                                  \
    do {                                              \
        DrawState d = base;                           \
        change;                                       \
        const dc::FixedKey k = keyOf(d);              \
        CHECK(k == k0 && k.hash() == k0.hash());      \
    } while (0)
    // Colour targets (every field of one, and the last slot).
    DIFFERS(d.color[0].address += 0x100);
    DIFFERS(d.color[0].width += 1);
    DIFFERS(d.color[0].height += 1);
    DIFFERS(d.color[0].pitch += 1);
    DIFFERS(d.color[0].format += 1);
    DIFFERS(d.color[0].numberType += 1);
    DIFFERS(d.color[0].compSwap += 1);
    DIFFERS(d.color[0].tileMode += 1);
    DIFFERS(d.color[0].sliceStart += 1);
    DIFFERS(d.color[0].sliceCount += 1);
    DIFFERS(d.color[0].samples += 1);
    DIFFERS(d.color[0].writeMask ^= 1);
    DIFFERS(d.color[0].forceDstAlpha1 ^= true);
    DIFFERS(d.color[0].blendClamp ^= true);
    DIFFERS(d.color[0].blendBypass ^= true);
    DIFFERS(d.color[0].coverage += 1);
    DIFFERS(d.color[0].fastClear ^= true);
    DIFFERS(d.color[0].clearWord[0] += 1);
    DIFFERS(d.color[0].clearWord[1] += 1);
    DIFFERS(d.color[7].format += 1);
    DIFFERS(d.color[7].address += 0x100);
    // Blend.
    DIFFERS(d.blend[0].enable ^= true);
    DIFFERS(d.blend[0].separateAlpha ^= true);
    DIFFERS(d.blend[0].colorSrc += 1);
    DIFFERS(d.blend[0].colorDst += 1);
    DIFFERS(d.blend[0].colorFunc += 1);
    DIFFERS(d.blend[0].alphaSrc += 1);
    DIFFERS(d.blend[0].alphaDst += 1);
    DIFFERS(d.blend[0].alphaFunc += 1);
    DIFFERS(d.blend[0].disableRop3 ^= true);
    DIFFERS(d.blend[7].enable ^= true);
    for (unsigned i = 0; i < 4; ++i) DIFFERS(d.blendColor[i] += 0.5f);
    // Depth target.
    DIFFERS(d.depth.zAddress += 0x100);
    DIFFERS(d.depth.stencilAddress += 0x100);
    DIFFERS(d.depth.width += 1);
    DIFFERS(d.depth.height += 1);
    DIFFERS(d.depth.zFormat += 1);
    DIFFERS(d.depth.stencilFormat += 1);
    DIFFERS(d.depth.tileMode += 1);
    DIFFERS(d.depth.sliceStart += 1);
    DIFFERS(d.depth.sliceCount += 1);
    DIFFERS(d.depth.samples += 1);
    DIFFERS(d.depth.depthClear ^= true);
    DIFFERS(d.depth.stencilClear ^= true);
    DIFFERS(d.depth.clearDepth += 0.5f);
    DIFFERS(d.depth.clearStencil += 1);
    DIFFERS(d.depth.zWriteAddress += 0x100);
    DIFFERS(d.depth.stencilWriteAddress += 0x100);
    DIFFERS(d.depth.zReadOnly ^= true);
    DIFFERS(d.depth.stencilReadOnly ^= true);
    DIFFERS(d.depth.forceZDirty ^= true);
    DIFFERS(d.depth.forceZValid ^= true);
    DIFFERS(d.depth.forceStencilDirty ^= true);
    DIFFERS(d.depth.forceStencilValid ^= true);
    DIFFERS(d.depth.boundsMin += 0.5f);
    DIFFERS(d.depth.boundsMax += 0.5f);
    DIFFERS(d.depth.depthCopy ^= true);
    DIFFERS(d.depth.copySample += 1);
    DIFFERS(d.depth.pitch += 1);
    DIFFERS(d.depth.rows += 1);
    // Depth / stencil state.
    DIFFERS(d.depthStencil.depthTest ^= true);
    DIFFERS(d.depthStencil.depthWrite ^= true);
    DIFFERS(d.depthStencil.depthFunc += 1);
    DIFFERS(d.depthStencil.stencilTest ^= true);
    DIFFERS(d.depthStencil.backfaceEnable ^= true);
    DIFFERS(d.depthStencil.depthBounds ^= true);
    DIFFERS(d.depthStencil.colorWritesOnDepthFail ^= true);
    DIFFERS(d.depthStencil.noColorWritesOnDepthPass ^= true);
    for (const bool back : {false, true}) {
        const auto face = [back](DrawState& d) -> vpps4::DepthStencil::Face& { return back ? d.depthStencil.back : d.depthStencil.front; };
        DIFFERS(face(d).func += 1);
        DIFFERS(face(d).fail += 1);
        DIFFERS(face(d).zPass += 1);
        DIFFERS(face(d).zFail += 1);
        DIFFERS(face(d).ref += 1);
        DIFFERS(face(d).readMask += 1);
        DIFFERS(face(d).writeMask += 1);
        DIFFERS(face(d).opVal += 1);
    }
    // Viewport and scissors.
    DIFFERS(d.viewport.xScale += 1);
    DIFFERS(d.viewport.xOffset += 1);
    DIFFERS(d.viewport.yScale += 1);
    DIFFERS(d.viewport.yOffset += 1);
    DIFFERS(d.viewport.zScale += 1);
    DIFFERS(d.viewport.zOffset += 1);
    DIFFERS(d.viewport.xScreen ^= true);
    DIFFERS(d.viewport.yScreen ^= true);
    const auto scissors = [](DrawState& d, unsigned i) -> vpps4::Scissor& {
        vpps4::Scissor* all[5] = {&d.scissor, &d.scissorRegs.screen, &d.scissorRegs.window, &d.scissorRegs.generic, &d.scissorRegs.viewportScissor};
        return *all[i];
    };
    for (unsigned i = 0; i < 5; ++i) {
        DIFFERS(scissors(d, i).left += 1);
        DIFFERS(scissors(d, i).top += 1);
        DIFFERS(scissors(d, i).right += 1);
        DIFFERS(scissors(d, i).bottom += 1);
    }
    DIFFERS(d.scissorRegs.windowOffsetX += 1);
    DIFFERS(d.scissorRegs.windowOffsetY += 1);
    DIFFERS(d.scissorRegs.windowOffsetDisable ^= true);
    DIFFERS(d.scissorRegs.genericOffsetDisable ^= true);
    DIFFERS(d.scissorRegs.viewportScissorEnable ^= true);
    DIFFERS(d.scissorRegs.present ^= true);
    // Raster.
    DIFFERS(d.raster.cullFront ^= true);
    DIFFERS(d.raster.cullBack ^= true);
    DIFFERS(d.raster.frontFaceClockwise ^= true);
    DIFFERS(d.raster.offsetFront ^= true);
    DIFFERS(d.raster.offsetBack ^= true);
    DIFFERS(d.raster.offsetPara ^= true);
    DIFFERS(d.raster.offsetClamp += 1);
    DIFFERS(d.raster.frontScale += 1);
    DIFFERS(d.raster.frontOffset += 1);
    DIFFERS(d.raster.backScale += 1);
    DIFFERS(d.raster.backOffset += 1);
    DIFFERS(d.raster.dbFmtValid ^= true);
    DIFFERS(d.raster.dbIsFloat ^= true);
    DIFFERS(d.raster.dbBits += 1);
    DIFFERS(d.raster.clipDisable ^= true);
    DIFFERS(d.raster.zclipNearDisable ^= true);
    DIFFERS(d.raster.zclipFarDisable ^= true);
    DIFFERS(d.raster.clipSpaceZeroToW ^= true);
    DIFFERS(d.raster.userClipPlanes += 1);
    DIFFERS(d.raster.vsOutputsViewportIndex ^= true);
    DIFFERS(d.raster.vsOutputsClipDistances ^= true);
    DIFFERS(d.raster.polygonModeEnable ^= true);
    DIFFERS(d.raster.polygonModeFront += 1);
    DIFFERS(d.raster.polygonModeBack += 1);
    DIFFERS(d.raster.provokingLast ^= true);
    DIFFERS(d.raster.lineWidth += 1);
    DIFFERS(d.raster.primitiveRestart ^= true);
    DIFFERS(d.raster.restartIndex += 1);
    // The rest of the draw's own state.
    DIFFERS(d.rop3 += 1);
    DIFFERS(d.degamma ^= true);
    DIFFERS(d.alphaToMask ^= true);
    DIFFERS(d.alphaToMaskOffsets += 1);
    DIFFERS(d.earlyZ ^= true);
    DIFFERS(d.depthBeforeShader ^= true);
    DIFFERS(d.psKill ^= true);
    DIFFERS(d.psExportsZ ^= true);
    DIFFERS(d.psExportsStencil ^= true);
    DIFFERS(d.psExportsMask ^= true);
    DIFFERS(d.primitive = vpps4::PrimitiveType::TriangleStrip);
    DIFFERS(d.geom.enabled ^= true);
    DIFFERS(d.geom.esgsItemDwords += 1);
    DIFFERS(d.geom.gsvsVertDwords += 1);
    DIFFERS(d.geom.maxVertOut += 1);
    DIFFERS(d.geom.outPrim += 1);
    DIFFERS(d.geom.gsInstances += 1);
    // The stages: their shape is in the key; their program, where it lives and their user data are not.
    for (Stage DrawState::* st : {&DrawState::vs, &DrawState::ps, &DrawState::es, &DrawState::gs}) {
        DIFFERS((d.*st).userSgprs += 1);
        DIFFERS((d.*st).vgprCompCnt += 1);
        DIFFERS((d.*st).vsOutCntl += 1);
        DIFFERS((d.*st).geom.enabled ^= true);
        DIFFERS((d.*st).geom.esgsItemDwords += 1);
        DIFFERS((d.*st).geom.gsvsVertDwords += 1);
        DIFFERS((d.*st).geom.maxVertOut += 1);
        DIFFERS((d.*st).geom.outPrim += 1);
        DIFFERS((d.*st).geom.gsInstances += 1);
        DIFFERS((d.*st).geomInputPrim += 1);
        DIFFERS((d.*st).rectList ^= true);
        DIFFERS((d.*st).psInputAddr += 1);
        DIFFERS((d.*st).psIterSample ^= true);
        DIFFERS((d.*st).exportFormat[0] += 1);
        DIFFERS((d.*st).exportFormat[7] += 1);
        DIFFERS((d.*st).psInputCntl[0] += 1);
        DIFFERS((d.*st).psInputCntl[31] += 1);
        DIFFERS((d.*st).psNumInterp += 1);
        DIFFERS((d.*st).dualSource ^= true);
        DIFFERS((d.*st).clipSpaceGL ^= true);
        DIFFERS((d.*st).instanceStepRate[0] += 1);
        DIFFERS((d.*st).instanceStepRate[1] += 1);
        DIFFERS((d.*st).workgroupSize[0] += 1);
        DIFFERS((d.*st).workgroupSize[2] += 1);
        DIFFERS((d.*st).tgidEnable[0] ^= true);
        DIFFERS((d.*st).tgidEnable[2] ^= true);
        DIFFERS((d.*st).tgSizeEnable ^= true);
        DIFFERS((d.*st).tidigCompCnt += 1);
        DIFFERS((d.*st).ldsBytes += 1);
        SAME((d.*st).code = words);
        SAME((d.*st).fetch = words);
        SAME((d.*st).address += 0x100);
        SAME((d.*st).userData[0] += 1);
        SAME((d.*st).userData[15] += 1);
    }
    // The geometry of the draw is not in the key.
    SAME(d.indexAddress += 0x100);
    SAME(d.index32 ^= true);
    SAME(d.indexSwap += 1);
    SAME(d.count += 1);
    SAME(d.instances += 1);
    SAME(d.firstVertex += 1);
    SAME(d.indirectAddress += 0x100);
    SAME(d.indirectCountAddress += 0x100);
    SAME(d.drawIndex += 1);
    SAME(d.stateSerial += 1);  // (the host's count of the writes: it is not part of what it counts)
#undef DIFFERS
#undef SAME
    // The host's sample limit is.
    CHECK(!(keyOf(base, 1) == k0) && keyOf(base, 4) == k0);
    // Two fields that change places do not give the same key (the fields are written in their order, with their sizes).
    {
        DrawState a = base, b = base;
        a.color[0].width = 7, a.color[0].height = 9, b.color[0].width = 9, b.color[0].height = 7;
        CHECK(!(keyOf(a) == keyOf(b)));
    }
    // What lies between the fields of the DrawState is not in the key: the same state over memory that held other bytes.
    {
        alignas(DrawState) static unsigned char raw[2][sizeof(DrawState)];
        std::memset(raw[0], 0xAA, sizeof raw[0]);
        std::memset(raw[1], 0x55, sizeof raw[1]);
        const DrawState* a = new (raw[0]) DrawState(base);
        const DrawState* b = new (raw[1]) DrawState(base);
        CHECK(keyOf(*a) == keyOf(*b) && keyOf(*a) == k0);
        a->~DrawState(), b->~DrawState();
    }
}

void witness() {
    using W = dc::Witness;
    W a;
    a.part = {7, 100, 3, 40, 5};
    CHECK(a.stale(a) == 0);
    for (unsigned p = 0; p < W::Parts; ++p) {
        W now = a;
        ++now.part[p];
        CHECK(a.stale(now) == 1u << p);                    // that part, and only it
        CHECK(a.stale(now, W::kAll & ~(1u << p)) == 0);    // (a block that does not depend on it is still good)
        CHECK(a.stale(now, 1u << p) == 1u << p);
        CHECK(now.stale(a) == 1u << p);                    // (older or newer: not the same)
    }
    W all = a;
    for (uint64_t& v : all.part) v += 2;
    CHECK(a.stale(all) == W::kAll);
    CHECK(a.stale(all, 1u << W::Surface | 1u << W::Clear) == (1u << W::Surface | 1u << W::Clear));
    CHECK(W::Frame == 0 && W::Layout == 1 && W::Surface == 2 && W::Pass == 3 && W::Clear == 4 && W::Parts == 5);
    CHECK(std::string_view(dc::kWitnessParts[W::Frame]) == "frame" && std::string_view(dc::kWitnessParts[W::Layout]) == "layout");
    CHECK(std::string_view(dc::kWitnessParts[W::Surface]) == "surface" && std::string_view(dc::kWitnessParts[W::Pass]) == "pass");
    CHECK(std::string_view(dc::kWitnessParts[W::Clear]) == "clear");
}

void stageKey() {
    const uint32_t t[8] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    const uint32_t v[4] = {0x80001000, 0x00100012, 64, 0x7000};  // base 0x12_80001000, stride 16
    const auto make = [&](uint64_t program, uint64_t env, uint8_t stage) {
        dc::StageKey k;
        k.program = program, k.env = env, k.stage = stage;
        k.texture(t, 8);
        k.buffer(v, 4);
        k.sampler(0x1234567890ull);
        k.other();
        return k;
    };
    const dc::StageKey k0 = make(1, 2, 3);
    CHECK(make(1, 2, 3) == k0 && make(1, 2, 3).hash() == k0.hash());
    const auto differs = [&](const dc::StageKey& k) { return !(k == k0) && k.hash() != k0.hash(); };
    CHECK(differs(make(9, 2, 3)) && differs(make(1, 9, 3)) && differs(make(1, 2, 4)));
    // Every dword of a T# counts.
    for (unsigned i = 0; i < 8; ++i) {
        uint32_t t2[8];
        std::memcpy(t2, t, sizeof t);
        t2[i] ^= 0x100;
        dc::StageKey k;
        k.program = 1, k.env = 2, k.stage = 3;
        k.texture(t2, 8), k.buffer(v, 4), k.sampler(0x1234567890ull), k.other();
        CHECK(differs(k));
    }
    // A V#: its base address (dword 0, the low 16 bits of dword 1) does not; its stride, records and format do.
    const auto withBuffer = [&](const uint32_t* w) {
        dc::StageKey k;
        k.program = 1, k.env = 2, k.stage = 3;
        k.texture(t, 8), k.buffer(w, 4), k.sampler(0x1234567890ull), k.other();
        return k;
    };
    {
        uint32_t moved[4] = {0x90002000, 0x0010F0FF, 64, 0x7000};
        CHECK(withBuffer(moved) == k0 && withBuffer(moved).hash() == k0.hash());
        uint32_t stride[4] = {v[0], 0x00200012, 64, 0x7000}, records[4] = {v[0], v[1], 65, 0x7000}, format[4] = {v[0], v[1], 64, 0x8000};
        uint32_t swizzle[4] = {v[0], v[1] | 0x80000000u, 64, 0x7000};
        CHECK(differs(withBuffer(stride)) && differs(withBuffer(records)) && differs(withBuffer(format)) && differs(withBuffer(swizzle)));
    }
    // The sampler bound, the kind of each binding and their order.
    {
        dc::StageKey k = make(1, 2, 3);
        k.words.clear();
        k.texture(t, 8), k.buffer(v, 4), k.sampler(0x1234567891ull), k.other();
        CHECK(differs(k));
        k.words.clear();
        k.texture(t, 8), k.buffer(v, 4), k.sampler(0x1234567890ull | 1ull << 40), k.other();
        CHECK(differs(k));
        k.words.clear();
        k.buffer(v, 4), k.texture(t, 8), k.sampler(0x1234567890ull), k.other();
        CHECK(differs(k));
        k.words.clear();
        k.texture(t, 8), k.buffer(v, 4), k.sampler(0x1234567890ull);
        CHECK(differs(k));  // (one binding less)
        dc::StageKey a, b;
        a.texture(v, 4), b.buffer(v, 4);
        CHECK(!(a == b));  // (the same dwords read as a T# and as a V#)
        uint32_t zeroBase[4] = {0, v[1] & 0xFFFF0000u, v[2], v[3]};
        dc::StageKey c;
        c.texture(zeroBase, 4);
        CHECK(!(c == b));
    }
    dc::StageKey cleared = make(1, 2, 3);
    cleared.clear();
    CHECK(cleared == dc::StageKey() && cleared.words.empty());
}

// A block as the tables see one.
struct Block {
    uint64_t key = 0, hash = 0, used = 0;
    bool live = false;
    int payload = 0;
};

void tables() {
    using T = dc::Table<Block>;
    // Nothing in it.
    {
        T t(64);
        CHECK(t.slots() == 64 && t.live() == 0 && t.evictions() == 0);
        CHECK(!t.find(uint64_t(1), 1));
    }
    // A key is found again, with what was put in its block; another key is not.
    {
        T t(64);
        Block& a = t.make(uint64_t(10), 5, 1);
        a.payload = 100;
        CHECK(a.live && a.key == 10 && a.hash == 5 && a.used == 1 && t.live() == 1);
        CHECK(t.find(uint64_t(10), 5) == &a && t.find(uint64_t(10), 5)->payload == 100);
        CHECK(!t.find(uint64_t(11), 5));  // (the same hash, another key)
        CHECK(!t.find(uint64_t(10), 6));  // (a key is looked for by its hash)
    }
    // Collisions: keys with the same hash, and keys whose hashes fall on the same slot, each in a block of its own.
    {
        T t(64);
        for (uint64_t k = 0; k < 4; ++k) t.make(k, 7, 1).payload = int(k);                 // one hash
        for (uint64_t k = 4; k < 8; ++k) t.make(k, 7 + 64 * (k - 3), 1).payload = int(k);  // one slot, other hashes
        CHECK(t.live() == 8 && t.evictions() == 0);
        for (uint64_t k = 0; k < 8; ++k) {
            Block* b = t.find(k, k < 4 ? 7 : 7 + 64 * (k - 3));
            CHECK(b && b->key == k && b->payload == int(k));
        }
    }
    // A run that is full gives up the block used longest ago, whatever its place in the run; the others stay.
    {
        T t(64);
        const uint64_t hash = 20;
        for (uint64_t k = 0; k < T::kRun; ++k) t.make(k, hash, 10 + k);  // used at frames 10..17
        t.find(uint64_t(0), hash)->used = 30;  // key 0 is used again: key 1 (frame 11) is now the oldest
        Block& fresh = t.make(uint64_t(100), hash, 31);
        CHECK(t.evictions() == 1 && t.live() == T::kRun);
        CHECK(fresh.key == 100 && fresh.used == 31 && fresh.live);
        CHECK(!t.find(uint64_t(1), hash));
        for (uint64_t k = 0; k < T::kRun; ++k)
            if (k != 1) CHECK(t.find(k, hash) && t.find(k, hash)->key == k);
        CHECK(t.find(uint64_t(100), hash) == &fresh);
        // (the next one out is key 2, frame 12)
        t.make(uint64_t(101), hash, 32);
        CHECK(t.evictions() == 2 && !t.find(uint64_t(2), hash) && t.find(uint64_t(3), hash) && t.find(uint64_t(100), hash));
        // A key of a neighbouring run takes a free slot beyond this one: nothing is lost.
        t.make(uint64_t(200), hash + 1, 33);
        CHECK(t.evictions() == 2 && t.live() == T::kRun + 1 && t.find(uint64_t(200), hash + 1));
        for (uint64_t k : {0ull, 3ull, 4ull, 5ull, 6ull, 7ull, 100ull, 101ull}) CHECK(t.find(k, hash));
    }
    // A run at the end of the table goes on at its start.
    {
        T t(16);
        for (uint64_t k = 0; k < T::kRun; ++k) t.make(k, 13, 1).payload = int(k);  // slots 13, 14, 15, 0, 1, 2, 3, 4
        CHECK(t.live() == T::kRun && t.evictions() == 0);
        for (uint64_t k = 0; k < T::kRun; ++k) CHECK(t.find(k, 13) && t.find(k, 13)->payload == int(k));
    }
    // Filled over and over: never more blocks than slots, every key is there right after it is made, and each one made
    // past the first slots' worth took the place of another.
    {
        T t(16);
        uint64_t made = 0;
        for (uint64_t k = 0; k < 400; ++k) {
            const uint64_t hash = k * 0x9E3779B97F4A7C15ull >> 20;
            if (t.find(k, hash)) continue;
            Block& b = t.make(k, hash, k);
            b.payload = int(k);
            ++made;
            Block* again = t.find(k, hash);
            CHECK(again == &b && again->payload == int(k));
            CHECK(t.live() <= t.slots());
        }
        CHECK(made == 400 && t.live() == 16 && t.evictions() == made - t.live());
        // (the newest keys are the ones still there)
        size_t recent = 0;
        for (uint64_t k = 392; k < 400; ++k) recent += t.find(k, k * 0x9E3779B97F4A7C15ull >> 20) != nullptr;
        CHECK(recent == 8);
    }
    // The blocks of the draw cache: found by their keys' content, apart when the hashes collide.
    {
        dc::Table<dc::FixedBlock> fixed(1024);
        DrawState d;
        d.color[0].format = 10;
        dc::FixedKey a = keyOf(d);
        d.color[0].format = 11;
        dc::FixedKey b = keyOf(d);
        dc::FixedBlock& ba = fixed.make(a, 77, 1);
        ba.resolved = {1, 2, 3};
        dc::FixedBlock& bb = fixed.make(b, 77, 1);  // (a forced collision)
        bb.resolved = {4};
        CHECK(&ba != &bb && fixed.find(a, 77) == &ba && fixed.find(b, 77) == &bb && fixed.live() == 2);
        CHECK(fixed.find(a, 77)->resolved == std::vector<uint8_t>({1, 2, 3}));
        dc::Table<dc::StageBlock> stages(8192);
        dc::StageKey s1, s2;
        const uint32_t w[4] = {1, 2, 3, 4};
        s1.program = 5, s1.texture(w, 4);
        s2.program = 5, s2.texture(w, 3);
        dc::StageBlock& b1 = stages.make(s1, 9, 1);
        b1.translated = 0xA;
        dc::StageBlock& b2 = stages.make(s2, 9, 1);
        b2.translated = 0xB;
        CHECK(stages.find(s1, 9) == &b1 && stages.find(s2, 9) == &b2 && stages.find(s1, 9)->translated == 0xA);
        CHECK(!stages.find(s1, s1.hash()) || s1.hash() == 9);
    }
}

void items() {
    dc::BindItem a;
    a.kind = dc::BindItem::Texture, a.bind = 2, a.stages = 1, a.slot = 3, a.id = 10, a.view = 11;
    dc::BindItem b = a;
    b.id = 20, b.view = 0;
    CHECK(a.sameShape(b));  // (what is bound is not its shape)
    for (int f = 0; f < 5; ++f) {
        dc::BindItem c = a;
        if (f == 0) c.kind = dc::BindItem::Live;
        if (f == 1) c.bind = 3;
        if (f == 2) c.stages = 2;
        if (f == 3) c.write = true;
        if (f == 4) c.slot = 4;
        CHECK(!a.sameShape(c));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string_view n = argc > 1 ? argv[1] : "";
    if (n == "fixed_key") fixedKey();
    else if (n == "witness") witness();
    else if (n == "stage_key") stageKey();
    else if (n == "tables") tables();
    else if (n == "items") items();
    else return std::fprintf(stderr, "unknown test\n"), 2;
    return failures ? 1 : 0;
}
