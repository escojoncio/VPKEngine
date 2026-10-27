// GpuTiler on the null backend. Metal cannot run here, so the recorded commands are executed by a
// small emulator: each dispatch runs the kernel's thread body in C++ (the same vpt_* functions the
// MSL pastes from tiling_addr.h, with the parameter block the GPU would read from the scratch
// memory), each copy moves bytes between the scratch buffer and a model of the texture. What the
// emulator produces must equal the CPU path (detileLevels / tileLevels).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpps4/bc_block.h"
#include "vpps4/bc_decode.h"
#include "vpps4/gpu_tiler.h"

extern const char* const kVpps4TilingAddrSource;

using namespace vpps4;
using namespace vpgpu;

static int failures = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c);      \
            ++failures;                                                  \
        }                                                                \
    } while (0)

namespace {

struct World {
    NullBackend backend;
    std::unique_ptr<Device> device;
    std::map<uint32_t, uint8_t*> memory;  // buffer index -> host memory
    std::deque<std::vector<uint8_t>> blocks;
    std::unique_ptr<GpuTiler> tiler;
    unsigned scratchAllocs = 0;
    std::vector<std::string> sinkSources;

    explicit World(unsigned workers = 0) {
        device = std::make_unique<Device>(backend, workers, [](const std::string&) {});
        tiler = std::make_unique<GpuTiler>(
            *device, [this](uint64_t, const vpmsl::MslShader& s) { sinkSources.push_back(s.source); },
            [this](uint64_t bytes, Scratch& out) {
                blocks.emplace_back(bytes + 256);
                uint8_t* p = blocks.back().data();
                p = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(p) + 255) & ~uintptr_t(255));
                out.buffer = device->createBuffer({bytes, p, "scratch"});
                out.offset = 0;
                out.memory = p;
                memory[out.buffer.index] = p;
                ++scratchAllocs;
                return bool(out.buffer);
            });
    }

    BufferId guest(std::vector<uint8_t>& bytes) {
        const BufferId id = device->createBuffer({bytes.size(), bytes.data(), "guest"});
        memory[id.index] = bytes.data();
        return id;
    }
};

// Texture contents as the emulator keeps them: per mip, linear, slice after slice.
using TextureModel = std::map<uint32_t, std::vector<uint8_t>>;

uint8_t* at(World& w, const ResourceRange& r) { return w.memory.at(r.index) + r.begin; }

// Runs the recorded commands. `detile`: the dispatches read tiled memory (binding 0) and write
// linear (binding 1); retile the other way round.
void execute(World& w, const Frame& f, const SurfaceLayout& lay, bool detile, TextureModel& tex) {
    const uint32_t bpe = lay.surface.bytesPerElement;
    for (const Command& c : f.commands()) {
        if (const auto* d = std::get_if<Dispatch>(&c)) {
            const uint8_t *b0 = nullptr, *params = nullptr;
            uint8_t* b1 = nullptr;
            for (const Access& a : d->bindings) {
                if (a.kind == BindKind::Buffer && a.slot == 0) b0 = at(w, a.range);
                if (a.kind == BindKind::Buffer && a.slot == 1) b1 = at(w, a.range);
                if (a.kind == BindKind::ConstantBuffer && a.slot == 0) params = at(w, a.range);
            }
            CHECK(b0 && b1 && params);
            if (!b0 || !b1 || !params) return;
            VptParams P;
            std::memcpy(&P, params, sizeof(P));
            CHECK(P.level.bpe == bpe);
            // The grid: groups of 8x8x1 threads.
            for (uint32_t gz = 0; gz < d->groups[2]; ++gz)
                for (uint32_t gy = 0; gy < d->groups[1] * 8; ++gy)
                    for (uint32_t gx = 0; gx < d->groups[0] * 8; ++gx) {
                        if (!vpt_in_range(P, gx, gy, gz)) continue;
                        const uint32_t t = vpt_tiled_offset(P, gx, gy, gz), l = vpt_linear_offset(P, gx, gy, gz);
                        if (detile) std::memcpy(b1 + l, b0 + t, bpe);
                        else std::memcpy(b1 + t, b0 + l, bpe);
                    }
        } else if (const auto* cp = std::get_if<Copy>(&c)) {
            const bool toTexture = cp->kind == CopyKind::BufferToTexture;
            const BufferSpan& bs = toTexture ? cp->srcBuffer : cp->dstBuffer;
            const TextureSpan& ts = toTexture ? cp->dstTexture : cp->srcTexture;
            const LevelLayout& l = lay.levels[ts.mip];
            std::vector<uint8_t>& m = tex[ts.mip];
            m.resize(l.linearBytes);
            const bool volume = ts.depth > 1 || ts.z > 0;
            const uint32_t first = volume ? ts.z : ts.layer, count = volume ? ts.depth : ts.layerCount;
            CHECK(bs.bytesPerRow == l.width * bpe && bs.rowsPerImage == l.height);
            uint8_t* buf = w.memory.at(bs.buffer.index) + bs.offset;
            const uint64_t image = uint64_t(l.width) * l.height * bpe;
            for (uint32_t i = 0; i < count; ++i) {
                if (toTexture) std::memcpy(m.data() + (first + i) * image, buf + i * image, image);
                else std::memcpy(buf + i * image, m.data() + (first + i) * image, image);
            }
        }
    }
}

std::vector<uint8_t> randomBytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(n);
    for (uint8_t& b : v) b = uint8_t(rng());
    return v;
}

Format formatFor(uint32_t bpe, bool block) {
    if (block) return bpe == 8 ? Format::BC1 : Format::BC3;
    switch (bpe) {
        case 1: return Format::R8Unorm;
        case 2: return Format::RG8Unorm;
        case 4: return Format::RGBA8Unorm;
        case 8: return Format::RGBA16Float;
        default: return Format::RGBA32Float;
    }
}

TiledSurface surface(uint8_t mode, uint32_t bpe, uint32_t w, uint32_t h, uint32_t slices, uint32_t levels, bool volume) {
    TiledSurface s;
    s.tileIndex = mode, s.bytesPerElement = bpe, s.width = w, s.height = h, s.slices = slices, s.levels = levels, s.volume = volume;
    return s;
}

TextureId makeTexture(World& w, const TiledSurface& s, Format f) {
    TextureDesc td;
    td.format = f;
    td.type = s.volume ? TextureType::Tex3D : TextureType::Tex2D;
    td.width = s.width, td.height = s.height;
    td.depth = s.volume ? s.slices : 1;
    td.layers = uint16_t(s.volume ? 1 : s.slices);
    td.mips = uint16_t(s.levels);
    td.usage = usage::sampled;
    return w.device->createTexture(td);
}

// The BC kernel: the recorded dispatches run through bc_kernel_thread (the function the MSL entry point calls), the copies
// into a model of the texture; the result must equal the CPU decoder on the CPU-detiled blocks.
void executeBc(World& w, const Frame& f, const SurfaceLayout& lay, uint32_t texelBytes, TextureModel& tex) {
    for (const Command& c : f.commands()) {
        if (const auto* d = std::get_if<Dispatch>(&c)) {
            const uint8_t* b0 = nullptr;
            uint8_t* b1 = nullptr;
            VpBcParams P{};
            for (const Access& a : d->bindings) {
                if (a.kind == BindKind::Buffer && a.slot == 0) b0 = at(w, a.range);
                if (a.kind == BindKind::Buffer && a.slot == 1) b1 = at(w, a.range);
                if (a.kind == BindKind::ConstantBuffer && a.slot == 0) std::memcpy(&P, at(w, a.range), sizeof(P));
            }
            CHECK(b0 && b1 && P.t.level.bpe == lay.surface.bytesPerElement);
            if (!b0 || !b1) return;
            for (uint32_t gz = 0; gz < d->groups[2]; ++gz)
                for (uint32_t gy = 0; gy < d->groups[1] * 8; ++gy)
                    for (uint32_t gx = 0; gx < d->groups[0] * 8; ++gx) bc_kernel_thread(b0, b1, P, gx, gy, gz);
        } else if (const auto* cp = std::get_if<Copy>(&c)) {
            CHECK(cp->kind == CopyKind::BufferToTexture);
            const TextureSpan& ts = cp->dstTexture;
            const uint64_t image = uint64_t(ts.width) * ts.height * texelBytes;
            CHECK(cp->srcBuffer.bytesPerRow == ts.width * texelBytes && cp->srcBuffer.rowsPerImage == ts.height);
            std::vector<uint8_t>& m = tex[ts.mip];
            const uint32_t first = ts.layer, count = ts.layerCount;
            m.resize((first + count) * image);
            std::memcpy(m.data() + first * image, w.memory.at(cp->srcBuffer.buffer.index) + cp->srcBuffer.offset, count * image);
        }
    }
    (void)lay;
}

void bcKernel(bc::Kind kind, bool isSigned, uint8_t mode, uint32_t w, uint32_t h, uint32_t slices, uint32_t levels) {
    World world;
    const uint32_t bpe = kind == bc::Kind::BC1 || kind == bc::Kind::BC4 ? 8 : 16, tb = bc::texelBytes(kind);
    TiledSurface s = surface(mode, bpe, w, h, slices, levels, false);
    s.blockSize = 4;
    const SurfaceLayout lay = computeLayout(s);
    CHECK(lay.ok);
    if (!lay.ok) return;
    const Format hostFormat = bc::hostFormat(kind, false, isSigned);
    const TextureId tex = makeTexture(world, s, hostFormat);
    CHECK(tex);
    if (!tex) return;
    const std::vector<uint8_t> blocks = randomBytes(lay.linearBytes, uint32_t(kind) * 17u + mode + w);
    std::vector<uint8_t> tiled(lay.bytes, 0xCD);
    CHECK(tileLevels(lay, 0, uint32_t(lay.levels.size()), 0, blocks.data(), tiled.data()));
    const BufferId guest = world.guest(tiled);
    Frame f;
    Subresource all;
    all.levelCount = levels;
    all.sliceCount = slices;
    CHECK(world.tiler->decodeBc(f, {guest, 0, tiled.size()}, tex, lay, all, {uint32_t(kind), isSigned}) == GpuTiler::Result::Recorded);
    TextureModel model;
    executeBc(world, f, lay, tb, model);
    CHECK(model.size() == levels);
    for (uint32_t lv = 0; lv < levels; ++lv) {
        const LevelLayout& l = lay.levels[lv];
        const uint32_t tw = std::max(w >> lv, 1u), th = std::max(h >> lv, 1u);
        std::vector<uint8_t> want(uint64_t(tw) * th * tb);
        const std::vector<uint8_t>& got = model[lv];
        CHECK(got.size() == want.size() * slices);
        if (got.size() != want.size() * slices) continue;
        for (uint32_t k = 0; k < slices; ++k) {
            const uint8_t* level = blocks.data() + l.linearOffset + uint64_t(k) * l.width * l.height * bpe;
            bc::decodeImage(kind, isSigned, level, uint64_t(l.width) * bpe, tw, th, want.data(), uint64_t(tw) * tb);
            CHECK(std::memcmp(got.data() + k * want.size(), want.data(), want.size()) == 0);
        }
    }
}

// One surface through both kernels, against the CPU path.
void roundTrip(uint8_t mode, uint32_t bpe, uint32_t w, uint32_t h, uint32_t slices, uint32_t levels, bool volume, bool block) {
    World world;
    TiledSurface s = surface(mode, bpe, w, h, slices, levels, volume);
    if (block) s.blockSize = 4;
    const SurfaceLayout lay = computeLayout(s);
    CHECK(lay.ok);
    if (!lay.ok) return;
    // For block-compressed formats the texture is in texels, the layout in blocks.
    const Format f = formatFor(bpe, block);
    const TextureId tex = makeTexture(world, s, f);
    CHECK(tex);
    if (!tex) return;

    const std::vector<uint8_t> linear = randomBytes(lay.linearBytes, mode * 31u + bpe);
    std::vector<uint8_t> tiled(lay.bytes, 0xCD);
    CHECK(tileLevels(lay, 0, uint32_t(lay.levels.size()), 0, linear.data(), tiled.data()));
    const BufferId guest = world.guest(tiled);

    // Detile on the "GPU".
    Frame f1;
    Subresource all;
    all.levelCount = uint32_t(lay.levels.size());
    const GpuTiler::Result r = world.tiler->detile(f1, {guest, 0, tiled.size()}, tex, lay, all);
    CHECK(r == GpuTiler::Result::Recorded);
    TextureModel model;
    execute(world, f1, lay, true, model);
    for (size_t i = 0; i < lay.levels.size(); ++i) {
        const LevelLayout& l = lay.levels[i];
        const std::vector<uint8_t> want(linear.begin() + l.linearOffset, linear.begin() + l.linearOffset + l.linearBytes);
        if (model[uint32_t(i)] != want) {
            std::printf("  detile differs: mode %u bpe %u %ux%u slices %u level %zu\n", mode, bpe, w, h, slices, i);
            ++failures;
        }
    }

    // Retile the model into fresh guest memory: must read back as the same linear data.
    std::vector<uint8_t> tiled2(lay.bytes, 0);
    const BufferId guest2 = world.guest(tiled2);
    Frame f2;
    CHECK(world.tiler->retile(f2, tex, {guest2, 0, tiled2.size()}, lay, all) == GpuTiler::Result::Recorded);
    TextureModel model2 = model;
    execute(world, f2, lay, false, model2);
    std::vector<uint8_t> back(lay.linearBytes, 0);
    CHECK(detileLevels(lay, 0, uint32_t(lay.levels.size()), 0, tiled2.data(), back.data()));
    if (back != linear) {
        std::printf("  retile differs: mode %u bpe %u %ux%u slices %u\n", mode, bpe, w, h, slices);
        ++failures;
    }
}

}  // namespace

int main() {
    // Kernel text: the addressing header is embedded verbatim; the argument table matches VpArgs.
    {
        std::ifstream in(VPPS4_ADDR_HEADER);
        std::stringstream ss;
        ss << in.rdbuf();
        CHECK(!ss.str().empty() && ss.str() == kVpps4TilingAddrSource);
        for (const vpmsl::MslShader& sh : {GpuTiler::detileShader(), GpuTiler::retileShader()}) {
            CHECK(sh.source.find("kernel void vp_main") != std::string::npos);
            CHECK(sh.source.find("vpt_offset") != std::string::npos);
            CHECK(sh.source.find("#pragma once") == std::string::npos);
            CHECK(sh.arguments.size() == 3 && sh.arguments[0].offset == 0 && sh.arguments[1].offset == 8 && sh.arguments[2].offset == 16);
            CHECK(sh.arguments[0].kind == vpir::ResourceKind::Buffer && sh.arguments[2].kind == vpir::ResourceKind::ConstantBuffer);
            CHECK(sh.workgroupSize[0] == 8 && sh.workgroupSize[1] == 8 && sh.workgroupSize[2] == 1);
            int depth = 0, paren = 0;
            for (char c : sh.source) depth += c == '{' ? 1 : c == '}' ? -1 : 0, paren += c == '(' ? 1 : c == ')' ? -1 : 0;
            CHECK(depth == 0 && paren == 0);
        }
    }

    // Both directions across tile modes and element sizes: 2D with mips, arrays, cube faces, volumes.
    const uint8_t modes[] = {8, 9, 10, 13, 14, 0, 4, 12, 15, 17, 20, 21, 25, 26};
    for (uint8_t m : modes)
        for (uint32_t bpe : {1u, 2u, 4u, 8u, 16u}) {
            roundTrip(m, bpe, 70, 45, 2, 4, false, false);
            roundTrip(m, bpe, 64, 64, 6, 7, false, false);  // cube faces down to 1x1
            roundTrip(m, bpe, 40, 24, 9, 4, true, false);   // volume
        }
    roundTrip(10, 8, 100, 60, 1, 5, false, true);   // BC1
    roundTrip(14, 16, 100, 60, 1, 5, false, true);  // BC3

    // The BC kernel: every format, tiled and linear, sizes that are not a multiple of 4, mips and slices.
    for (uint32_t k = 0; k < 7; ++k) {
        const auto kind = bc::Kind(k);
        const bool hasSigned = kind == bc::Kind::BC4 || kind == bc::Kind::BC5 || kind == bc::Kind::BC6H;
        for (bool sgn : {false, true}) {
            if (sgn && !hasSigned) continue;
            bcKernel(kind, sgn, 10, 37, 22, 1, 4);
            bcKernel(kind, sgn, 8, 13, 7, 1, 3);
            bcKernel(kind, sgn, 14, 64, 40, 3, 2);
        }
    }

    // Part of a surface: mips 1-2, slices 1-2, one sample plane of a multisampled one.
    {
        World world;
        TiledSurface s = surface(10, 4, 128, 96, 4, 4, false);
        const SurfaceLayout lay = computeLayout(s);
        const TextureId tex = makeTexture(world, s, Format::RGBA8Unorm);
        const std::vector<uint8_t> linear = randomBytes(lay.linearBytes, 5);
        std::vector<uint8_t> tiled(lay.bytes, 0);
        CHECK(tileLevels(lay, 0, 4, 0, linear.data(), tiled.data()));
        const BufferId guest = world.guest(tiled);
        Frame f;
        Subresource sub;
        sub.firstLevel = 1, sub.levelCount = 2, sub.firstSlice = 1, sub.sliceCount = 2;
        CHECK(world.tiler->detile(f, {guest, 0, tiled.size()}, tex, lay, sub) == GpuTiler::Result::Recorded);
        TextureModel model;
        execute(world, f, lay, true, model);
        CHECK(model.size() == 2 && model.count(1) && model.count(2));
        for (uint32_t lv = 1; lv <= 2; ++lv) {
            const LevelLayout& l = lay.levels[lv];
            const uint64_t image = uint64_t(l.width) * l.height * 4;
            for (uint32_t z = 1; z <= 2; ++z)
                CHECK(std::memcmp(model[lv].data() + z * image, linear.data() + l.linearOffset + z * image, image) == 0);
        }
        // Slice 0 and 3 of those mips were not touched.
        CHECK(world.tiler->dispatches() == 2);
    }
    {
        World world;
        TiledSurface s = surface(10, 4, 80, 48, 1, 1, false);
        s.samples = 4;
        const SurfaceLayout lay = computeLayout(s);
        CHECK(lay.ok);
        TextureDesc td;
        td.format = Format::RGBA8Unorm, td.width = 80, td.height = 48, td.usage = usage::sampled;
        const TextureId tex = world.device->createTexture(td);
        std::vector<uint8_t> tiled(lay.bytes, 0);
        for (uint32_t sm = 0; sm < 4; ++sm) {
            const std::vector<uint8_t> plane = randomBytes(lay.linearBytes, 100 + sm);
            CHECK(tileLevels(lay, 0, 1, sm, plane.data(), tiled.data()));
        }
        const BufferId guest = world.guest(tiled);
        for (uint32_t sm = 0; sm < 4; ++sm) {
            Frame f;
            Subresource sub;
            sub.sample = sm;
            CHECK(world.tiler->detile(f, {guest, 0, tiled.size()}, tex, lay, sub) == GpuTiler::Result::Recorded);
            TextureModel model;
            execute(world, f, lay, true, model);
            std::vector<uint8_t> want(lay.linearBytes);
            CHECK(detileLevels(lay, 0, 1, sm, tiled.data(), want.data()));
            CHECK(model[0] == want);
        }
    }

    // Unsupported requests are refused, not recorded.
    {
        World world;
        TiledSurface s = surface(10, 4, 64, 64, 1, 1, false);
        const SurfaceLayout lay = computeLayout(s);
        const TextureId tex = makeTexture(world, s, Format::RGBA8Unorm);
        std::vector<uint8_t> tiled(lay.bytes);
        const BufferId guest = world.guest(tiled);
        Frame f;
        Subresource sub;
        CHECK(world.tiler->detile(f, {guest, 0, lay.bytes - 1}, tex, lay, sub) == GpuTiler::Result::Unsupported);  // too small
        sub.levelCount = 2;
        CHECK(world.tiler->detile(f, {guest, 0, lay.bytes}, tex, lay, sub) == GpuTiler::Result::Unsupported);      // more levels than the layout
        sub.levelCount = 1;
        sub.firstSlice = 1;
        CHECK(world.tiler->detile(f, {guest, 0, lay.bytes}, tex, lay, sub) == GpuTiler::Result::Unsupported);      // slice out of range
        sub.firstSlice = 0;
        const SurfaceLayout bad = computeLayout(surface(27, 4, 64, 64, 1, 1, false));
        CHECK(world.tiler->detile(f, {guest, 0, lay.bytes}, tex, bad, sub) == GpuTiler::Result::Unsupported);
        CHECK(world.tiler->detile(f, {guest, 0, lay.bytes}, TextureId{}, lay, sub) == GpuTiler::Result::Unsupported);
        TiledSurface vol = s;
        vol.volume = true;
        CHECK(world.tiler->detile(f, {guest, 0, lay.bytes}, tex, computeLayout(vol), sub) == GpuTiler::Result::Unsupported);
        CHECK(f.commands().empty() && world.scratchAllocs == 0);
    }

    // While the kernel's pipeline compiles, the caller is told to use the CPU path.
    {
        std::atomic<bool> hold{true};
        World world(1);
        world.backend.onCompile = [&] {
            while (hold) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        };
        TiledSurface s = surface(10, 4, 64, 64, 1, 1, false);
        const SurfaceLayout lay = computeLayout(s);
        const TextureId tex = makeTexture(world, s, Format::RGBA8Unorm);
        std::vector<uint8_t> tiled(lay.bytes);
        const BufferId guest = world.guest(tiled);
        Frame f;
        CHECK(world.tiler->detile(f, {guest, 0, lay.bytes}, tex, lay, {}) == GpuTiler::Result::NotReady);
        CHECK(f.commands().empty());
        hold = false;
        world.device->pipelines().waitIdle();
        CHECK(world.tiler->detile(f, {guest, 0, lay.bytes}, tex, lay, {}) == GpuTiler::Result::Recorded);
        CHECK(!f.commands().empty());
        CHECK(world.sinkSources.size() == 1);  // registered once
    }

    // While the BC kernel's pipeline compiles, the caller is told to use the CPU decoder.
    {
        std::atomic<bool> hold{true};
        World world(1);
        world.backend.onCompile = [&] {
            while (hold) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        };
        TiledSurface s = surface(10, 8, 16, 16, 1, 1, false);
        s.blockSize = 4;
        const SurfaceLayout lay = computeLayout(s);
        const TextureId tex = makeTexture(world, s, Format::RGBA8Unorm);
        std::vector<uint8_t> tiled(lay.bytes);
        const BufferId guest = world.guest(tiled);
        Frame f;
        const BcJob job{uint32_t(bc::Kind::BC1), false};
        CHECK(world.tiler->decodeBc(f, {guest, 0, lay.bytes}, tex, lay, {}, job) == GpuTiler::Result::NotReady);
        CHECK(f.commands().empty() && world.scratchAllocs == 0);
        hold = false;
        world.device->pipelines().waitIdle();
        CHECK(world.tiler->decodeBc(f, {guest, 0, lay.bytes}, tex, lay, {}, job) == GpuTiler::Result::Recorded);
        CHECK(world.sinkSources.size() == 1 && world.sinkSources[0].find("bc_kernel_thread") != std::string::npos);
        // A layout that is not 8 / 16 byte blocks, or a kind that does not exist, is refused.
        CHECK(world.tiler->decodeBc(f, {guest, 0, lay.bytes}, tex, lay, {}, {9, false}) == GpuTiler::Result::Unsupported);
    }

    // The kernels' MSL as checked-in goldens (tests/kernels/*.metal), which compile_all.sh compiles with Apple's compiler on
    // macOS CI. VPPS4_UPDATE_KERNELS=1 rewrites them.
    {
        const struct { const char* name; vpmsl::MslShader shader; } kernels[] = {
            {"detile", GpuTiler::detileShader()}, {"retile", GpuTiler::retileShader()}, {"bcdecode", GpuTiler::bcShader()}};
        for (const auto& k : kernels) {
            const std::string path = std::string(VPPS4_KERNEL_DIR) + "/" + k.name + ".metal";
            if (std::getenv("VPPS4_UPDATE_KERNELS")) {
                std::ofstream(path, std::ios::binary) << k.shader.source;
                continue;
            }
            std::ifstream in(path, std::ios::binary);
            std::stringstream ss;
            ss << in.rdbuf();
            if (ss.str() != k.shader.source) std::printf("kernel golden %s differs (VPPS4_UPDATE_KERNELS=1 rewrites it)\n", path.c_str());
            CHECK(ss.str() == k.shader.source);
        }
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("gpu_tiler: ok\n");
    return 0;
}
