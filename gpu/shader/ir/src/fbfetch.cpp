#include <algorithm>

#include "pass_util.h"
#include "vpir/passes.h"

namespace vpir {

namespace {

class FetchRewriter {
public:
    FetchRewriter(Module& m, const std::map<uint32_t, uint32_t>& att, const std::set<uint32_t>& nearest, bool msDepth = false)
        : m_(m), att_(att), nearest_(nearest), msDepth_(msDepth) {}

    std::vector<FetchRejection> run() {
        std::vector<size_t> before = references();
        for (Function& f : m_.functions) {
            fn_ = &f;
            defs_ = definitions(f.body);
            forEachInst(f.body, [&](Inst& in) { visit(in); });
            if (changed_) removeDeadValues(f);
            changed_ = false;
        }
        dropUnreferenced(before);
        // The per-sample fetch only reads this invocation's sample if the function runs per sample: the module must take
        // [[sample_id]] (the emitter declares every builtin input as an entry parameter, used or not).
        if (msDepth_ && fetched_ && std::none_of(m_.inputs.begin(), m_.inputs.end(), [](const IoVar& v) { return v.builtin == Builtin::SampleId; })) {
            IoVar v;
            v.builtin = Builtin::SampleId, v.name = "sample_id", v.type = kU32;
            m_.inputs.push_back(v);
        }
        return std::move(rejected_);
    }

private:
    static bool readsImage(Op op) { return op == Op::ImageFetch || op == Op::ImageSample || op == Op::ImageGather; }

    void visit(Inst& in) {
        if (!readsImage(in.op) || in.sym >= m_.resources.size()) return;
        const Resource& r = m_.resources[in.sym];
        auto it = att_.find(r.index);
        if (r.kind != ResourceKind::Texture || it == att_.end()) return;
        std::string why = msDepth_ ? checkMsDepth(in, r, it->second) : check(in, r, it->second);
        if (!why.empty()) {
            rejected_.push_back({fn_->name, in.result, std::move(why)});
            return;
        }
        Type t = in.type;
        ValueId res = in.result;
        in = Inst{};
        in.op = Op::FramebufferFetch;
        in.type = t;
        in.result = res;
        in.lits = {it->second};
        changed_ = true;
        fetched_ = true;
    }

    std::string check(const Inst& in, const Resource& r, uint32_t attachment) const {
        if (m_.stage != Stage::Fragment) return "framebuffer fetch is fragment-only";
        if (r.depth) return "depth texture: Apple GPUs only read colour attachments from tile memory";
        if (attachment >= 8) return "attachment index must be 0..7";
        if (r.dim != TexDim::D2) return "only non-array 2d textures map to an attachment";
        if (in.op == Op::ImageGather) return "gather reads a 2x2 footprint, not one pixel";
        if (in.image & kImageOffset) return "texel offset reads a neighbouring pixel";
        if (in.image & (kImageBias | kImageGrad)) return "bias or gradient may select another mip level";
        if (in.image & kImageCompare) return "compare sampling";
        size_t lodArg = 1;
        if ((in.image & kImageLod) && !isZero(in.args[lodArg])) return "lod is not a constant 0";
        if (in.op == Op::ImageFetch) {
            if (!isPixelIndex(in.args[0])) return "coordinate is not this fragment's integer pixel position";
            return {};
        }
        if (!isPixelCentreUv(in.args[0], in.sym)) return "coordinate is not this fragment's pixel centre";
        if (!nearest_.contains(m_.resources[in.sampler].index)) return "sampler is not known to filter nearest";
        return {};
    }

    // A multisampled input attachment (subpassLoad of a depth image, served from a colour attachment holding its depth): the
    // fetch of this invocation's own pixel, sample and layer. The sample operand (lod) must be the SampleId builtin, possibly
    // through convert / bitcast: any other sample is not what the tile holds for this invocation.
    std::string checkMsDepth(const Inst& in, const Resource& r, uint32_t attachment) const {
        if (m_.stage != Stage::Fragment) return "framebuffer fetch is fragment-only";
        if (attachment >= 8) return "attachment index must be 0..7";
        if (in.op != Op::ImageFetch) return "only image_fetch reads a multisampled input attachment";
        if (!r.multisample || (r.dim != TexDim::D2 && r.dim != TexDim::D2Array)) return "not a multisampled 2d input attachment";
        if (in.type != (r.depth ? kF32 : Type(Scalar::F32, 4))) return "the fetch is not an f32 depth value";
        if (!(in.image & kImageLod) || in.args.size() < 2 || !isSampleId(in.args[1])) return "the sample is not gl_SampleID";
        if (r.dim == TexDim::D2) return isPixelIndex(in.args[0]) ? std::string() : "coordinate is not this fragment's pixel";
        const Inst* c = def(in.args[0]);
        if (!c || c->op != Op::Construct || c->args.size() != 3) return "array coordinate is not (x, y, layer)";
        for (uint32_t k = 0; k < 2; ++k) {
            const Inst* e = def(c->args[k]);
            if (!e || e->op != Op::Extract || e->lits[0] != k || !isPixelIndex(e->args[0])) return "coordinate is not this fragment's pixel";
        }
        return {};
    }

    bool isSampleId(ValueId v) const {
        const Inst* d = def(v);
        for (int hop = 0; d && hop < 4; ++hop) {
            if (d->op == Op::LoadInput) return d->sym < m_.inputs.size() && m_.inputs[d->sym].builtin == Builtin::SampleId;
            if (d->op != Op::Convert && d->op != Op::Bitcast) return false;
            d = def(d->args[0]);
        }
        return false;
    }

    const Inst* def(ValueId v) const {
        auto it = defs_.find(v);
        return it == defs_.end() ? nullptr : it->second;
    }

    bool isZero(ValueId v) const {
        const Inst* d = def(v);
        if (!d || d->op != Op::Const) return false;
        // -0.0 is a float zero; the same bits as an integer are INT_MIN.
        const uint32_t mask = d->type.isFloat() ? 0x7FFFFFFFu : 0xFFFFFFFFu;
        return std::all_of(d->lits.begin(), d->lits.end(), [&](uint32_t l) { return (l & mask) == 0; });
    }

    bool isPosition(ValueId v) const {
        const Inst* d = def(v);
        return d && d->op == Op::LoadInput && m_.inputs[d->sym].builtin == Builtin::Position;
    }

    // position.xy as a vec2<f32>: a shuffle picking x, y or a construct of the two extracts.
    bool isPositionXy(ValueId v) const {
        const Inst* d = def(v);
        if (!d) return false;
        if (d->op == Op::Shuffle)
            return d->lits == std::vector<uint32_t>{0, 1} && isPosition(d->args[0]);
        if (d->op == Op::Construct && d->args.size() == 2) {
            for (uint32_t c = 0; c < 2; ++c) {
                const Inst* e = def(d->args[c]);
                if (!e || e->op != Op::Extract || e->lits[0] != c || !isPosition(e->args[0])) return false;
            }
            return true;
        }
        return false;
    }

    // vec2<u32>(position.xy), optionally floored or truncated first: the pixel's integer address.
    bool isPixelIndex(ValueId v) const {
        const Inst* d = def(v);
        if (!d || d->op != Op::Convert || d->type != Type(Scalar::U32, 2)) return false;
        ValueId src = d->args[0];
        if (const Inst* r = def(src); r && (r->op == Op::Floor || r->op == Op::Trunc)) src = r->args[0];
        return isPositionXy(src);
    }

    // vec2<f32>(image_size(texture)) of mip 0.
    bool isTextureSize(ValueId v, uint32_t texture) const {
        const Inst* d = def(v);
        if (!d || d->op != Op::Convert || d->type != Type(Scalar::F32, 2)) return false;
        const Inst* s = def(d->args[0]);
        return s && s->op == Op::ImageSize && s->sym == texture && (!s->image || isZero(s->args[0]));
    }

    // position.xy / size or position.xy * rcp(size): the centre of this pixel in normalized space.
    bool isPixelCentreUv(ValueId v, uint32_t texture) const {
        const Inst* d = def(v);
        if (!d || !isPositionXy(d->args.empty() ? kNoValue : d->args[0])) return false;
        if (d->op == Op::Div) return isTextureSize(d->args[1], texture);
        if (d->op != Op::Mul) return false;
        const Inst* r = def(d->args[1]);
        return r && r->op == Op::Rcp && isTextureSize(r->args[0], texture);
    }

    // Reference count per resource over all functions.
    std::vector<size_t> references() {
        std::vector<size_t> n(m_.resources.size());
        for (Function& f : m_.functions)
            forEachInst(f.body, [&](Inst& in) {
                std::string_view sig = opInfo(in.op).signature;
                if ((sig.find('r') != sig.npos || sig.find('m') != sig.npos) && in.sym < n.size()) ++n[in.sym];
                if (sig.find('s') != sig.npos && in.sampler < n.size()) ++n[in.sampler];
            });
        return n;
    }

    // Resources this pass made unreferenced leave the module (and the argument buffer).
    void dropUnreferenced(const std::vector<size_t>& before) {
        std::vector<size_t> after = references();
        std::vector<uint32_t> remap(m_.resources.size());
        std::vector<Resource> kept;
        for (uint32_t i = 0; i < m_.resources.size(); ++i) {
            remap[i] = uint32_t(kept.size());
            if (!(before[i] && !after[i])) kept.push_back(std::move(m_.resources[i]));
        }
        if (kept.size() == m_.resources.size()) {
            m_.resources = std::move(kept);
            return;
        }
        m_.resources = std::move(kept);
        for (Function& f : m_.functions)
            forEachInst(f.body, [&](Inst& in) {
                std::string_view sig = opInfo(in.op).signature;
                if ((sig.find('r') != sig.npos || sig.find('m') != sig.npos) && in.sym < remap.size())
                    in.sym = remap[in.sym];
                if (sig.find('s') != sig.npos) in.sampler = remap[in.sampler];
            });
    }

    Module& m_;
    const std::map<uint32_t, uint32_t>& att_;
    const std::set<uint32_t>& nearest_;
    Function* fn_ = nullptr;
    std::unordered_map<ValueId, const Inst*> defs_;
    std::vector<FetchRejection> rejected_;
    bool changed_ = false;
    bool msDepth_ = false, fetched_ = false;
};

}  // namespace

std::vector<FetchRejection> toFramebufferFetch(Module& module, const std::map<uint32_t, uint32_t>& attachments,
                                               const std::set<uint32_t>& nearestSamplers) {
    return FetchRewriter(module, attachments, nearestSamplers).run();
}

std::vector<FetchRejection> toDepthFramebufferFetch(Module& module, const std::map<uint32_t, uint32_t>& attachments) {
    const std::set<uint32_t> none;
    return FetchRewriter(module, attachments, none, true).run();
}

}  // namespace vpir
