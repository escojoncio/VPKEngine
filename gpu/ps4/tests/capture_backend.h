// The backend of the per-draw equivalence tests (perdraw_tests.cpp) and of the replay of a call capture (tools/vp4replay.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpgpu/core/pipeline_cache.h"

// ---- Capture ---------------------------------------------------------------------------------------------------------
// Everything the renderer hands to the backend and to its host, written down in order: resources made and destroyed,
// pipelines, every command of every frame with all its fields, the bytes of the renderer's own buffers that the commands
// bind (upload rings: constants, tables, indices), the compiled frame, the shaders and the watches. Two runs that leave
// the same bytes here drew the same thing.
class Capture final : public vpgpu::Backend {
public:
    std::string bytes;
    std::vector<std::pair<size_t, const char*>> marks;  // (where each record starts: names the first difference)
    const uint8_t* guestHost = nullptr;
    uint64_t inFlight = 2;  // frames the "GPU" is behind

    void mark(const char* what) { marks.push_back({bytes.size(), what}); }
    void raw(const void* p, size_t n) { bytes.append(static_cast<const char*>(p), n); }
    template <class T>
    void v(const T& x) {
        static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);
        raw(&x, sizeof x);
    }
    void str(const std::string& x) { v(uint64_t(x.size())), raw(x.data(), x.size()); }
    template <class Tag>
    void id(vpgpu::Handle<Tag> h) { v(h.index), v(h.generation); }

    void createBuffer(vpgpu::BufferId b, const vpgpu::BufferDesc& d) override {
        mark("createBuffer");
        id(b), v(d.size), v(d.residentByUse), str(d.label.rfind("guest ", 0) == 0 ? d.label : std::string("own"));
        const bool guest = d.label.rfind("guest ", 0) == 0;
        v(guest ? uint64_t(static_cast<const uint8_t*>(d.guestMemory) - guestHost) : uint64_t(d.guestMemory != nullptr));
        buffers[b.index] = {static_cast<const uint8_t*>(d.guestMemory), d.size, !guest};
    }
    void createTexture(vpgpu::TextureId t, const vpgpu::TextureDesc& d) override {
        mark("createTexture");
        id(t), v(d.type), v(d.format), v(d.width), v(d.height), v(d.depth), v(d.mips), v(d.layers), v(d.samples), v(d.usage);
        v(d.persistent), v(d.transient), v(d.samplesReadable), str(d.label);
    }
    void createSampler(vpgpu::SamplerId sm, const vpgpu::SamplerDesc& d) override {
        mark("createSampler");
        id(sm), v(d.minFilter), v(d.magFilter), v(d.mipFilter), v(d.u), v(d.v), v(d.w), v(d.maxAnisotropy);
        v(d.compare.has_value()), v(d.compare.value_or(vpgpu::CompareOp::Never)), v(d.border), v(d.lodMin), v(d.lodMax), v(d.lodBias);
        v(d.shaderLodBias), v(d.normalized);
    }
    void destroyBuffer(vpgpu::BufferId b) override { mark("destroyBuffer"), id(b); }
    void destroyTexture(vpgpu::TextureId t) override { mark("destroyTexture"), id(t); }
    void destroySampler(vpgpu::SamplerId sm) override { mark("destroySampler"), id(sm); }
    bool compilePipeline(const vpgpu::PipelineState& ps, std::string&) override {
        mark("pipeline");
        vpgpu::PipelineState unlabelled = ps;
        unlabelled.label.clear();
        v(ps.key().value), str(vpgpu::describePipelineState(unlabelled)), str(ps.label);
        return true;
    }
    bool takeTimings(uint64_t&, vpgpu::GpuTimings&) override { return false; }
    uint64_t completedFrames() const override { return completed; }
    bool waitFramesBelow(uint64_t frames, uint64_t) override {
        completed = std::max(completed, frames);
        return true;
    }
    bool waitAll(uint64_t) override {
        completed = submitted;
        return true;
    }
    // The host knows when the GPU finished each frame (a call capture replayed, see vpps4/callcap.h): the frames below `frames`
    // are done. `inFlight` past every frame leaves that to these calls alone.
    void finishFramesBelow(uint64_t frames) { completed = std::max(completed, std::min(frames, submitted)); }

    void range(const vpgpu::ResourceRange& r) {
        v(r.kind), v(r.index), v(r.generation), v(r.begin), v(r.end), v(r.mipBegin), v(r.mipEnd), v(r.layerBegin), v(r.layerEnd);
        v(r.rect), v(r.x0), v(r.y0), v(r.x1), v(r.y1);
    }
    // The bytes of one of the renderer's own buffers (upload rings and the like) a command reads: what the CPU put there.
    // Not the ranges an earlier command of the frame fills on the GPU (a kernel's output, a copy's destination): the host
    // memory behind those holds nothing defined when the frame is handed over.
    void contents(uint32_t index, uint64_t begin, uint64_t end) {
        const auto it = buffers.find(index);
        if (it == buffers.end() || !it->second.own || !it->second.memory) return;
        end = std::min(end, it->second.size);
        for (const auto& [i, b, e] : gpuFilled)
            if (i == index && b < end && begin < e) return;
        if (begin < end) raw(it->second.memory + begin, end - begin);
    }
    void filled(uint32_t index, uint64_t begin, uint64_t end) { gpuFilled.push_back({index, begin, end}); }
    void accesses(const std::vector<vpgpu::Access>& list) {
        v(uint64_t(list.size()));
        for (const vpgpu::Access& a : list) {
            range(a.range), v(a.stages), v(a.write), v(a.pixelLocal), v(a.kind), v(a.slot);
            if (a.range.kind != vpgpu::ResourceRange::Kind::Buffer) continue;
            if (a.write) filled(a.range.index, a.range.begin, a.range.end);
            else contents(a.range.index, a.range.begin, a.range.end);
        }
    }
    void views(const std::vector<vpgpu::ViewBinding>& list) {
        v(uint64_t(list.size()));
        for (const vpgpu::ViewBinding& b : list) v(b.kind), v(b.slot), id(b.view);
    }
    void span(const vpgpu::BufferSpan& b) { id(b.buffer), v(b.offset), v(b.size), v(b.bytesPerRow), v(b.rowsPerImage); }
    void span(const vpgpu::TextureSpan& t) {
        id(t.texture), v(t.mip), v(t.layer), v(t.layerCount), v(t.x), v(t.y), v(t.z), v(t.width), v(t.height), v(t.depth), v(t.aspect);
    }
    void attachment(const vpgpu::Attachment& a) {
        id(a.texture), v(a.mip), v(a.layer), v(a.load), v(a.store);
        for (double c : a.clear.color) v(c);
        v(a.clear.depth), v(a.clear.stencil), id(a.resolve), v(a.resolveMip), v(a.resolveLayer), v(a.resolveRequired);
        v(a.depthResolve), v(a.stencilResolve), v(a.format), v(a.keepSamples), v(a.autoMsaa), v(a.needsSeed), id(a.seedSrc);
        v(a.seedMip), v(a.seedLayer);
    }
    void pass(const vpgpu::RenderPassDesc& d) {
        for (const vpgpu::Attachment& a : d.colors) attachment(a);
        attachment(d.depth);
        v(d.layerCount), v(d.width), v(d.height), v(d.samples), v(d.renderWidth), v(d.renderHeight);
        v(d.rateMap ? d.rateMap->key : uint64_t(0)), str(d.label);
    }
    void draw(const vpgpu::Draw& d) {
        mark("draw");
        v(bool(d.clear));
        if (d.clear) {
            v(uint64_t(d.clear->rects.size()));
            for (const auto& rc : d.clear->rects) v(rc.x), v(rc.y), v(rc.width), v(rc.height), v(rc.layer), v(rc.layerCount);
            v(d.clear->colorMask), raw(d.clear->colorBits.data(), sizeof d.clear->colorBits);
            v(d.clear->depth), v(d.clear->stencil), v(d.clear->depthValue), v(d.clear->stencilValue);
        }
        v(d.pipeline.value), accesses(d.bindings), v(d.primitive), v(d.first), v(d.count), v(d.instances), v(d.baseVertex);
        v(d.baseInstance), v(d.instanceViews), v(d.amplified), v(d.drawIndex), v(d.index.has_value());
        if (d.index) {
            id(d.index->buffer), v(d.index->offset), v(d.index->wide);
            const uint64_t stride = d.index->wide ? 4 : 2;
            contents(d.index->buffer.index, d.index->offset + d.first * stride, d.index->offset + (uint64_t(d.first) + d.count) * stride);
        }
        v(d.stencilRef), v(d.stencilRefBack);
        for (int k = 0; k < 2; ++k) v(d.stencilReadMask[size_t(k)]), v(d.stencilWriteMask[size_t(k)]);
        for (float c : d.blendColor) v(c);
        v(d.viewport.x), v(d.viewport.y), v(d.viewport.width), v(d.viewport.height), v(d.viewport.znear), v(d.viewport.zfar);
        v(d.scissor.x), v(d.scissor.y), v(d.scissor.width), v(d.scissor.height), v(d.frontFace), v(d.depthBias), v(d.depthBiasSlope);
        v(d.depthBiasClamp), v(d.depthClamp), v(d.wireframe), views(d.views), v(d.visibility.has_value());
        if (d.visibility) span(*d.visibility), filled(d.visibility->buffer.index, d.visibility->offset, d.visibility->offset + 8);
        v(d.indirect.has_value());
        if (d.indirect) id(d.indirect->buffer), v(d.indirect->offset);
        v(d.expandedBy[0].value), v(d.expandedBy[1].value);
    }
    void submit(uint64_t index, const vpgpu::Frame& f, const vpgpu::CompiledFrame& c) override {
        mark("frame");
        gpuFilled.clear();
        v(index), v(uint64_t(f.commands().size()));
        for (const vpgpu::Command& cmd : f.commands()) {
            if (const auto* rp = std::get_if<vpgpu::RenderPass>(&cmd)) {
                mark("pass");
                pass(rp->desc), v(uint64_t(rp->draws.size()));
                for (const vpgpu::Draw& d : rp->draws) draw(d);
            } else if (const auto* dp = std::get_if<vpgpu::Dispatch>(&cmd)) {
                mark("dispatch");
                v(dp->pipeline.value), accesses(dp->bindings), raw(dp->groups.data(), sizeof dp->groups), str(dp->label), views(dp->views);
                v(dp->indirect.has_value()), v(dp->required);
                if (dp->indirect) id(dp->indirect->buffer), v(dp->indirect->offset);
            } else {
                const vpgpu::Copy& cp = std::get<vpgpu::Copy>(cmd);
                mark("copy");
                v(cp.kind), span(cp.srcBuffer), span(cp.dstBuffer), span(cp.srcTexture), span(cp.dstTexture), v(cp.fillByte);
                v(cp.blitFilter), v(cp.blitFlipX), v(cp.blitFlipY), v(cp.blitFormat), v(cp.blitSwapRB), v(cp.mipChain), str(cp.label);
                if (cp.kind == vpgpu::CopyKind::BufferToBuffer || cp.kind == vpgpu::CopyKind::BufferToTexture)
                    contents(cp.srcBuffer.buffer.index, cp.srcBuffer.offset, cp.srcBuffer.offset + cp.srcBuffer.size);
                if (cp.kind == vpgpu::CopyKind::BufferToBuffer || cp.kind == vpgpu::CopyKind::TextureToBuffer || cp.kind == vpgpu::CopyKind::FillBuffer)
                    filled(cp.dstBuffer.buffer.index, cp.dstBuffer.offset, cp.dstBuffer.size ? cp.dstBuffer.offset + cp.dstBuffer.size : ~uint64_t(0));
            }
        }
        mark("compiled");
        v(uint64_t(c.encoders.size()));
        for (const vpgpu::Encoder& e : c.encoders) {
            v(e.kind), v(e.reason), pass(e.pass), v(uint64_t(e.commands.size()));
            for (uint32_t k : e.commands) v(k);
            v(e.passes), raw(e.tile.data(), sizeof e.tile), raw(e.memBacked.data(), sizeof e.memBacked), v(e.heavy), v(e.drawBegin);
            v(e.drawEnd), v(e.view), str(e.splitWhy ? e.splitWhy : ""), v(e.splitSig), v(e.splitViewVerts), v(e.splitHeavyAfter);
        }
        v(uint64_t(c.barriers.size()));
        for (const vpgpu::Barrier& b : c.barriers) v(b.encoder), v(b.item), v(b.after), v(b.before), v(b.hazards);
        v(uint64_t(c.badDraws.size()));
        for (const vpgpu::DrawRef& d : c.badDraws) v(d.command), v(d.draw);
        v(uint64_t(c.memAfter.size()));
        for (const auto& [t, valid] : c.memAfter) id(t), v(valid);
        v(uint64_t(c.autoAfter.size()));
        for (const auto& a : c.autoAfter) id(a.texture), id(a.src), v(a.mip), v(a.layer), v(a.valid);
        v(uint64_t(c.transientViolations.size())), v(uint64_t(c.msaaRejects.size())), v(uint64_t(c.storeObs.size()));
        const vpgpu::CompileStats& st = c.stats;
        v(st.commands), v(st.passesMerged), v(st.copiesHoisted), v(st.copiesSunk), v(st.dropped), raw(st.cuts, sizeof st.cuts);
        raw(st.hazards, sizeof st.hazards), v(st.intraBarriers), v(st.barriersComputeDraw), v(st.tileLocal), v(st.tileRejected);
        v(st.badFetch), v(st.tileBytes), v(st.storesKept), v(st.msaaStaleReads), v(st.heavyEncoders), v(st.loadsDropped);
        v(st.maxEncoderVertices), v(st.seedDraws), v(st.seedMissing);
        submitted = index + 1;
        if (submitted > inFlight) completed = std::max(completed, submitted - inFlight);
    }

private:
    struct Memory {
        const uint8_t* memory;
        uint64_t size;
        bool own;
    };
    std::map<uint32_t, Memory> buffers;
    std::vector<std::tuple<uint32_t, uint64_t, uint64_t>> gpuFilled;  // (buffer index, begin, end) of the frame being written down
    uint64_t submitted = 0, completed = 0;
};
