// Plays a call capture (vpps4/callcap.h: callcap-u<N>.vp4, written by a host in its diagnostic mode) back into a renderer of
// its own, on a backend that writes down everything it is given (tests/capture_backend.h), and compares the paths of the
// renderer over it: the same calls with each per-draw shortcut switched off must leave the same bytes.
//
// Usage: vp4replay FILE [--time R] [--off NAME[=VALUE]]... [--in-flight N] [--dump] [--digest] [--rates]
//   (default)      replays FILE with every shortcut on and with each switch of the list off (R passes each, default 5), and
//                  prints: draws, us/draw of every path per picture (the least of the passes), whether the paths left the
//                  same bytes, and where the first difference is. Exit 1 if they differ, 2 if FILE cannot be replayed.
//   --off N[=V]    one more switch for the list (an environment variable; V defaults to 1)
//   --in-flight N  the "GPU" finishes a frame N submits after it got it, instead of when the capture says (its Done records)
//   --dump         what FILE holds, record by record counts and per picture; nothing is replayed
//   --digest       one replay with every shortcut on: a hash of the bytes it left
//   --rates        one replay with the probe on: the RATES family of the renderer (hit rates of its caches)
//
// The guest memory of the capture is mapped at its own addresses (the capturing host's regions, pages of zeros where the
// capture holds nothing): the addresses must be free in this process.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "vpgpu/core/device.h"
#include "vpgpu/core/probe.h"
#include "vpmsl/msl.h"
#include "vpps4/callcap.h"
#include "vpps4/formats.h"
#include "vpps4/renderer.h"

#include "../tests/capture_backend.h"

namespace vp4replay {

using vpps4::callcap::Call;
using vpps4::callcap::Type;

// The switches of the per-draw shortcuts (tests/perdraw_tests.cpp kOptOuts) and of the draw cache.
struct OptOut {
    std::string name, value;
};
inline std::vector<OptOut> defaultOptOuts() {
    return {{"VPKENGINE_BIND_MEMO_OFF", "1"}, {"VPKENGINE_PROGRAM_MEMO_OFF", "1"}, {"VPKENGINE_HAZARD_MEMO_OFF", "1"}, {"VPVK_FRAME_FOLD", "0"},
            {"VPKENGINE_DRAW_CACHE_OFF", "1"}};
}

struct Options {
    uint64_t inFlight = ~uint64_t(0);     // Capture::inFlight (past every frame: the Done records alone say when a frame finished)
    const uint8_t* guestHost = nullptr;  // Capture::guestHost (the guest buffers are named by their distance from it)
    bool rates = false;                  // the probe on, and the RATES lines of its summary kept
};

struct Picture {
    uint64_t draws = 0, drawNs = 0, submits = 0, submitNs = 0;
};
// What a replay left.
struct Run {
    std::string error;  // why the file could not be replayed ("" if it was)
    std::string bytes;
    std::vector<std::pair<size_t, const char*>> marks;
    std::vector<Picture> pictures;  // (calls after the last Picture record: one more, not whole)
    uint64_t whole = 0;             // pictures the file holds whole
    uint64_t calls = 0, draws = 0, drawn = 0, memBytes = 0, regions = 0, madeUp = 0;
    int end = -1;  // EndRecord::cut; -1: the file has no End record
    std::string skipped;
    std::vector<std::string> rates;
};

// Guest memory at its own addresses: anonymous pages, made when a region or a piece of memory of the capture names them.
class Memory {
public:
    ~Memory() {
        for (const auto& [begin, end] : mapped_) munmap(reinterpret_cast<void*>(begin), end - begin);
    }
    bool ensure(uint64_t address, uint64_t bytes) {
        const uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
        uint64_t lo = address & ~(page - 1);
        const uint64_t hi = (address + std::max<uint64_t>(bytes, 1) + page - 1) & ~(page - 1);
        if (!lo || hi < lo) return false;
        while (lo < hi) {
            auto next = mapped_.upper_bound(lo);
            if (next != mapped_.begin() && std::prev(next)->second > lo) {  // (inside a mapped piece: on to its end)
                lo = std::prev(next)->second;
                continue;
            }
            const uint64_t to = next == mapped_.end() ? hi : std::min(hi, next->first);
            void* want = reinterpret_cast<void*>(lo);
            void* got = mmap(want, to - lo, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (got != want) {  // (the address is taken, or not one a process can have)
                if (got != MAP_FAILED) munmap(got, to - lo);
                return false;
            }
            mapped_[lo] = to;
            lo = to;
        }
        return true;
    }

private:
    std::map<uint64_t, uint64_t> mapped_;  // [begin, end), not merged
};

inline Run replay(const std::string& path, const Options& options) {
    using Clock = std::chrono::steady_clock;
    namespace cc = vpps4::callcap;
    Run out;
    cc::Reader reader(path);
    if (!reader.ok()) return out.error = "not a call capture: " + path, out;
    cc::Reader::Record rec;
    cc::Config cfg;
    bool haveConfig = false;
    while (!haveConfig && reader.next(rec))
        if (rec.type == Type::Config && rec.payload.size() == sizeof cfg) std::memcpy(&cfg, rec.payload.data(), sizeof cfg), haveConfig = true;
    if (!haveConfig) return out.error = "no Config record", out;
    if (cfg.version != cc::Config{}.version || cfg.drawStateBytes != sizeof(vpps4::DrawState) || cfg.dispatchStateBytes != sizeof(vpps4::DispatchState) ||
        cfg.colorTargetBytes != sizeof(vpps4::ColorTarget) || cfg.imageDescBytes != sizeof(vpps4::ImageDesc))
        return out.error = "captured by a build with another layout of the call structs (or another version of the file)", out;

    Memory memory;
    struct Region {
        uint64_t base, size, generation;
        bool mirrored;
    };
    std::vector<Region> regions;
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> misses;  // (address, size, generation) the host had no region for
    uint64_t generation = 0;
    bool mirroredLast = true;
    uint64_t wantedAnswer = 0, takenAnswer = 0;                  // the host's answers inside the next call
    std::map<std::array<uint64_t, 7>, bool> viewAnswers;         // by the shape of the request

    vpps4::setFormatCaps({cfg.bc != 0});
    if (options.rates) vpgpu::probe::setEnabled(true);
    Capture capture;
    capture.guestHost = options.guestHost;
    capture.inFlight = options.inFlight;
    std::vector<std::string> log;
    vpgpu::Device device{capture, 0, [&](const std::string& s) { log.push_back(s); }};
    device.setRefuseFailedBuffers(true);
    device.setHoldSteadyMs(12);
    vpps4::Hooks hooks;
    hooks.region = [&](uint64_t address, uint64_t size, vpps4::GuestRegion& g) {
        const uint64_t end = address + std::max<uint64_t>(size, 1);
        const Region* best = nullptr;
        for (const Region& r : regions)
            if (r.generation == generation && address >= r.base && end <= r.base + r.size && (!best || r.size < best->size)) best = &r;
        if (!best) {
            // The capturing host said there is none; or it was never asked (its renderer was warm): then a window of zeros.
            constexpr uint64_t kUnit = 16ull << 20;
            const uint64_t base = address & ~(kUnit - 1), top = (end + kUnit - 1) & ~(kUnit - 1);
            if (address < 0x10000 || misses.count({address, size, generation}) || !memory.ensure(base, top - base)) return false;
            regions.push_back({base, top - base, generation, mirroredLast});
            ++out.madeUp;
            best = &regions.back();
        }
        g = {best->base, best->size, reinterpret_cast<void*>(best->base), best->mirrored};
        return true;
    };
    if (cfg.hooks & cc::kHookRegionGeneration) hooks.regionGeneration = [&] { return generation; };
    hooks.shader = [&](uint64_t id, vpir::Stage stage, const vpmsl::MslShader& sh) {
        capture.mark("shader"), capture.v(id), capture.v(stage), capture.str(sh.source);
    };
    hooks.log = [&](const std::string& s) { log.push_back(s); };
    if (cfg.hooks & cc::kHookWatchWrites) hooks.watchWrites = [&](uint64_t a, uint64_t n) { capture.mark("watchWrites"), capture.v(a), capture.v(n); };
    if (cfg.hooks & cc::kHookWatchOwned)
        hooks.watchOwned = [&](uint64_t a, uint64_t n, bool on) { capture.mark("watchOwned"), capture.v(a), capture.v(n), capture.v(on); };
    if (cfg.hooks & cc::kHookView) {
        hooks.view = [&](const vpps4::ViewRequest& q) -> vpgpu::TextureId {
            const auto answer = viewAnswers.find({uint64_t(q.format), uint64_t(q.type), q.array, q.baseLayer, q.layers, q.baseMip, q.mips});
            if (answer != viewAnswers.end() && !answer->second) return {};
            vpgpu::TextureDesc d;  // (a stand-in for the host's view: the capture backend only writes its id down)
            d.format = q.format == vpgpu::Format::Undefined ? vpgpu::Format::RGBA8Unorm : q.format;
            d.width = d.height = 4;
            return device.createTexture(d);
        };
        hooks.destroyView = [&](vpgpu::TextureId v) { device.destroy(v); };
    }
    if (cfg.hooks & cc::kHookComputeBuffersWanted) hooks.computeBuffersWanted = [&] { return wantedAnswer != 0; };
    if (cfg.hooks & cc::kHookComputeBuffers)
        hooks.computeBuffers = [&](std::span<const vpps4::ComputeBuffer>, bool, bool) { return takenAnswer != 0; };
    {
        vpps4::Renderer renderer{device, std::move(hooks)};
        renderer.setMaxSamples(cfg.maxSamples);
        renderer.setSummaryEvery(0);
        renderer.setGpuTiling(cfg.gpuTiling), renderer.setGuestReadOnly(cfg.guestReadOnly), renderer.setImageAlias(cfg.imageAlias);
        renderer.setFoveation(cfg.foveation);
        renderer.setMemoryLimits(cfg.stagingLimit, cfg.scratchLimit, cfg.bcFrameBytes);
        renderer.setResidency(cfg.idleSubmits, cfg.byteBudget);

        Picture picture;
        // A stage's program, where the capture says it is.
        const auto stage = [&](vpps4::Stage& st, const cc::StageRef& ref) {
            if (!ref.codeWords) return true;
            if (!memory.ensure(ref.code, ref.codeWords * 4) || (ref.fetchWords && !memory.ensure(ref.fetch, ref.fetchWords * 4))) return false;
            st.code = {reinterpret_cast<const uint32_t*>(ref.code), size_t(ref.codeWords)};
            if (ref.fetchWords) st.fetch = {reinterpret_cast<const uint32_t*>(ref.fetch), size_t(ref.fetchWords)};
            return true;
        };
        while (out.error.empty() && out.end < 0 && reader.next(rec)) {
            const uint8_t* p = rec.payload.data();
            const size_t n = rec.payload.size();
            const uint64_t* a = reinterpret_cast<const uint64_t*>(p);  // (payloads of u64s; the vector's storage is aligned)
            const auto bad = [&](const char* what) { out.error = std::string("damaged record: ") + what; };
            switch (rec.type) {
                case Type::Gen:
                    if (n != 8) bad("Gen");
                    else generation = a[0];
                    break;
                case Type::Region: {
                    cc::RegionRecord r;
                    if (n != sizeof r) {
                        bad("Region");
                        break;
                    }
                    std::memcpy(&r, p, sizeof r);
                    if (rec.flags & 1) {
                        misses.insert({r.base, r.size, r.generation});
                    } else if (!memory.ensure(r.base, r.size)) {
                        out.error = "guest region " + std::to_string(r.base) + " +" + std::to_string(r.size) + " cannot be mapped at its address";
                    } else {
                        regions.push_back({r.base, r.size, r.generation, r.mirrored != 0});
                        mirroredLast = r.mirrored != 0;
                        ++out.regions;
                    }
                    break;
                }
                case Type::Mem:
                    if (n <= 8) bad("Mem");
                    else if (!memory.ensure(a[0], n - 8)) out.error = "guest memory at " + std::to_string(a[0]) + " cannot be mapped at its address";
                    else std::memcpy(reinterpret_cast<void*>(a[0]), p + 8, n - 8), out.memBytes += n - 8;
                    break;
                case Type::Done:
                    if (n != 8) {
                        bad("Done");
                        break;
                    }
                    capture.finishFramesBelow(a[0]);
                    device.pumpLabels(true);  // (the labels of the finished frames have run, as the host's completion thread does)
                    break;
                case Type::HookAns:
                    if (cc::Hook(rec.flags) == cc::Hook::View && n == sizeof(cc::ViewAnswer)) viewAnswers[{a[0], a[1], a[2], a[3], a[4], a[5], a[6]}] = a[7] != 0;
                    else if (cc::Hook(rec.flags) == cc::Hook::ComputeBuffersWanted && n == 8) wantedAnswer = a[0];
                    else if (cc::Hook(rec.flags) == cc::Hook::ComputeBuffers && n == 8) takenAnswer = a[0];
                    else bad("HookAns");
                    break;
                case Type::Picture:
                    out.pictures.push_back(picture), picture = {};
                    ++out.whole;
                    break;
                case Type::End: {
                    cc::EndRecord e;
                    if (n != sizeof e) {
                        bad("End");
                        break;
                    }
                    std::memcpy(&e, p, sizeof e);
                    out.end = int(e.cut);
                    break;
                }
                case Type::Call: {
                    const auto args = [&](size_t count) { return n == count * 8 || (bad("Call"), false); };
                    ++out.calls;
                    switch (Call(rec.flags)) {
                        case Call::Draw: {
                            cc::StageRef refs[4];
                            vpps4::DrawState d;
                            if (n != sizeof d + sizeof refs) {
                                bad("Draw");
                                break;
                            }
                            std::memcpy(static_cast<void*>(&d), p, sizeof d), std::memcpy(refs, p + sizeof d, sizeof refs);
                            if (!stage(d.vs, refs[0]) || !stage(d.ps, refs[1]) || !stage(d.es, refs[2]) || !stage(d.gs, refs[3])) {
                                out.error = "a program of draw " + std::to_string(out.draws) + " cannot be mapped at its address";
                                break;
                            }
                            const auto t0 = Clock::now();
                            renderer.draw(d);
                            picture.drawNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
                            ++picture.draws, ++out.draws;
                            break;
                        }
                        case Call::Dispatch: {
                            cc::StageRef ref;
                            vpps4::DispatchState d;
                            if (n != sizeof d + sizeof ref) {
                                bad("Dispatch");
                                break;
                            }
                            std::memcpy(static_cast<void*>(&d), p, sizeof d), std::memcpy(&ref, p + sizeof d, sizeof ref);
                            if (!stage(d.cs, ref)) out.error = "a compute program cannot be mapped at its address";
                            else renderer.dispatch(d);
                            break;
                        }
                        case Call::Resolve: {
                            vpps4::ColorTarget t[2];
                            if (n != sizeof t) {
                                bad("Resolve");
                                break;
                            }
                            std::memcpy(static_cast<void*>(t), p, sizeof t);
                            renderer.resolve(t[0], t[1]);
                            break;
                        }
                        case Call::WriteBack: {
                            vpps4::ColorTarget t;
                            if (n != sizeof t) {
                                bad("WriteBack");
                                break;
                            }
                            std::memcpy(static_cast<void*>(&t), p, sizeof t);
                            renderer.writeBack(t);
                            break;
                        }
                        case Call::TextureFor: {
                            vpps4::ImageDesc d;
                            if (n != sizeof d + 8) {
                                bad("TextureFor");
                                break;
                            }
                            uint64_t integerOnly;
                            std::memcpy(static_cast<void*>(&d), p, sizeof d), std::memcpy(&integerOnly, p + sizeof d, 8);
                            renderer.textureFor(d, integerOnly != 0);
                            break;
                        }
                        case Call::Fill: if (args(3)) renderer.fill(a[0], a[1], uint32_t(a[2])); break;
                        case Call::Copy: if (args(3)) renderer.copy(a[0], a[1], a[2]); break;
                        case Call::DepthCleared: if (args(2)) renderer.depthCleared(a[0], std::bit_cast<float>(uint32_t(a[1]))); break;
                        case Call::Invalidate: if (args(2)) renderer.invalidate(a[0], a[1]); break;
                        case Call::CoherencePoint: if (args(2)) renderer.coherencePoint(a[0], a[1]); break;
                        case Call::SyncToMemory: if (args(2)) renderer.syncToMemory(a[0], a[1]); break;
                        case Call::AbandonOwners: if (args(2)) renderer.abandonOwners(a[0], a[1]); break;
                        case Call::NoteGpuWrite: if (args(2)) renderer.noteGpuWrite(a[0], a[1]); break;
                        case Call::CpuAccessed: if (args(2)) renderer.cpuAccessed(a[0], a[1]); break;
                        case Call::AfterGpu: if (args(3)) renderer.afterGpu(a[0], uint32_t(a[1]), [] {}, a[2] != 0); break;
                        case Call::GpuIdle: if (args(0)) renderer.gpuIdle(); break;
                        case Call::SubmittedIdle: if (args(0)) renderer.submittedIdle(); break;
                        // (a frame index of the capturing Device: this one's frames count from where the capture began)
                        case Call::WaitedBelow: if (args(1)) renderer.waitedBelow(a[0] > cfg.frameBase ? a[0] - cfg.frameBase : 0); break;
                        case Call::ZpassDone: if (args(2)) renderer.zpassDone(a[0], uint32_t(a[1])); break;
                        case Call::DrainOcclusion: if (args(1)) renderer.drainOcclusion(a[0] != 0); break;
                        case Call::DropGuestBuffers: if (args(0)) renderer.dropGuestBuffers(); break;
                        case Call::Submit: {
                            if (!args(0)) break;
                            capture.mark("submit");
                            const auto t0 = Clock::now();
                            renderer.submit();
                            picture.submitNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
                            ++picture.submits;
                            break;
                        }
                        case Call::ColorTargetAt: if (args(1)) renderer.colorTargetAt(a[0]); break;
                        case Call::ColorTargetSlice: {
                            uint16_t layer = 0;
                            if (args(3)) renderer.colorTargetSlice(a[0], uint32_t(a[1]), uint32_t(a[2]), layer);
                            break;
                        }
                        case Call::Gds: if (args(0)) renderer.gds(); break;
                        case Call::MarkLabelOrdered: if (args(2)) renderer.markLabelOrdered(a[0], uint32_t(a[1])); break;
                        case Call::SupersedeLabels: if (args(4)) renderer.supersedeLabels(a[0], a[1], a[2], a[3]); break;
                        default: out.error = "a call this build does not know: " + std::to_string(rec.flags); break;
                    }
                    wantedAnswer = takenAnswer = 0;
                    break;
                }
                default: break;  // (Meta, a second Config, a type of a later version)
            }
        }
        if (picture.draws || picture.submits) out.pictures.push_back(picture);
        const vpps4::Stats& st = renderer.stats();
        out.drawn = st.drawn;
        for (const auto& [why, count] : st.skipped) out.skipped += " skip[" + why + "]=" + std::to_string(count);
        if (options.rates) {
            renderer.publishProbe(out.whole);
            for (const std::string& line : vpgpu::probe::registry().sample())
                if (line.find("RATES") != std::string::npos) out.rates.push_back(line);
        }
    }
    if (options.rates) vpgpu::probe::setEnabled(false);
    out.bytes = std::move(capture.bytes);
    out.marks = std::move(capture.marks);
    return out;
}

inline uint64_t fnv(const std::string& x) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : x) h = (h ^ c) * 0x100000001b3ull;
    return h;
}

// Where the bytes two replays left first differ ("" if they are the same): the record of `a`, the frames and draws before it.
inline std::string difference(const Run& a, const Run& b) {
    if (a.bytes == b.bytes) return "";
    size_t at = 0;
    while (at < a.bytes.size() && at < b.bytes.size() && a.bytes[at] == b.bytes[at]) ++at;
    size_t record = 0;
    while (record + 1 < a.marks.size() && a.marks[record + 1].first <= at) ++record;
    size_t frames = 0, draws = 0;
    for (size_t k = 0; k <= record && k < a.marks.size(); ++k) {
        frames += std::strcmp(a.marks[k].second, "frame") == 0;
        draws += std::strcmp(a.marks[k].second, "draw") == 0;
    }
    char line[256];
    std::snprintf(line, sizeof line, "byte %zu of %zu / %zu: record %zu (%s, +%zu), after %zu frames and %zu draws", at, a.bytes.size(),
                  b.bytes.size(), record, a.marks.empty() ? "?" : a.marks[record].second, a.marks.empty() ? size_t(0) : at - a.marks[record].first,
                  frames, draws);
    return line;
}

// The capture replayed with every shortcut on and with each switch of `offs` set, `passes` times each (the paths in turn
// within a pass, so that none has the machine's quiet moments to itself): the figures of every path to `to`.
// 0: every replay left the same bytes; 1: one did not; 2: the file cannot be replayed.
inline int compare(const std::string& path, const std::vector<OptOut>& offs, unsigned passes, const Options& options, std::FILE* to) {
    for (const OptOut& o : offs) unsetenv(o.name.c_str());
    struct Path {
        std::string name, diff;
        std::vector<Picture> least;
    };
    std::vector<Path> paths(offs.size() + 1);
    Run base;
    std::string first;
    for (unsigned pass = 0; pass < passes; ++pass)
        for (size_t k = 0; k < paths.size(); ++k) {
            Path& p = paths[k];
            p.name = k ? offs[k - 1].name + "=" + offs[k - 1].value : std::string("every shortcut on");
            if (k) setenv(offs[k - 1].name.c_str(), offs[k - 1].value.c_str(), 1);
            Run run = replay(path, options);
            if (k) unsetenv(offs[k - 1].name.c_str());
            if (!run.error.empty()) return std::fprintf(to, "%s: %s\n", path.c_str(), run.error.c_str()), 2;
            if (!pass) p.least = run.pictures;
            for (size_t i = 0; i < p.least.size() && i < run.pictures.size(); ++i) p.least[i].drawNs = std::min(p.least[i].drawNs, run.pictures[i].drawNs);
            if (!pass && !k) {
                base = std::move(run);
                continue;
            }
            const std::string diff = difference(base, run);
            if (diff.empty()) continue;
            if (p.diff.empty()) p.diff = diff;
            if (first.empty()) first = p.name + (k ? ": " : " does not repeat itself: ") + diff;
        }
    std::fprintf(to, "%s: %llu pictures whole, %llu calls, %llu draws (%llu drawn), %llu bytes of memory, %llu regions (+%llu made up), %s\n",
                 path.c_str(), (unsigned long long)base.whole, (unsigned long long)base.calls, (unsigned long long)base.draws,
                 (unsigned long long)base.drawn, (unsigned long long)base.memBytes, (unsigned long long)base.regions, (unsigned long long)base.madeUp,
                 base.end < 0 ? "no End record (the writer died)" : base.end == 0 ? "complete" : base.end == 1 ? "cut at the byte limit" :
                 base.end == 2 ? "cut by a failed write" : "ended by the host");
    if (!base.skipped.empty()) std::fprintf(to, "  left out:%s\n", base.skipped.c_str());
    std::fprintf(to, "  digest %016llx (%zu bytes); us/draw by picture, least of %u passes (the first picture is cold):\n",
                 (unsigned long long)fnv(base.bytes), base.bytes.size(), passes);
    for (const Path& p : paths) {
        std::fprintf(to, "  %-34s", p.name.c_str());
        uint64_t steadyNs = 0, steadyDraws = 0;
        for (size_t i = 0; i < p.least.size(); ++i) {
            std::fprintf(to, " %6.2f", p.least[i].draws ? double(p.least[i].drawNs) / 1e3 / double(p.least[i].draws) : 0.0);
            if (i >= 2 || p.least.size() < 3) steadyNs += p.least[i].drawNs, steadyDraws += p.least[i].draws;
        }
        std::fprintf(to, " | from the third %6.2f | %s\n", steadyDraws ? double(steadyNs) / 1e3 / double(steadyDraws) : 0.0,
                     !p.diff.empty() ? "DIFFERENT" : &p == &paths[0] ? (passes > 1 ? "repeats itself" : "-") : "same bytes");
    }
    std::fprintf(to, "  equal: %s\n", first.empty() ? "yes" : "no");
    if (!first.empty()) std::fprintf(to, "  first difference: %s\n", first.c_str());
    return first.empty() ? 0 : 1;
}

// What the file holds, without replaying it.
inline bool dump(const std::string& path, std::FILE* to) {
    namespace cc = vpps4::callcap;
    cc::Reader reader(path);
    if (!reader.ok()) return std::fprintf(to, "not a call capture: %s\n", path.c_str()), false;
    static const char* const types[] = {"?", "Meta", "Config", "Gen", "Region", "Mem", "Done", "HookAns", "Call", "Picture", "End"};
    std::map<uint16_t, uint64_t> records, calls;
    uint64_t bytes = sizeof cc::kMagic, mem = 0, pictureMem = 0, pictureCalls = 0, pictureDraws = 0, pictures = 0;
    bool ended = false;
    for (cc::Reader::Record rec; reader.next(rec);) {
        bytes += 8 + rec.payload.size();
        ++records[uint16_t(rec.type)];
        if (rec.type == Type::Meta) std::fprintf(to, "meta:\n%.*s\n", int(rec.payload.size()), reinterpret_cast<const char*>(rec.payload.data()));
        if (rec.type == Type::Config && rec.payload.size() == sizeof(cc::Config)) {
            cc::Config c;
            std::memcpy(&c, rec.payload.data(), sizeof c);
            std::fprintf(to, "config: version %u samples %u gpuTiling %u guestReadOnly %u imageAlias %u foveation %u bc %u hooks %#x staging %llu "
                             "scratch %llu bcFrame %llu idleSubmits %llu byteBudget %llu frameBase %llu rendererFrame %llu\n",
                         c.version, c.maxSamples, c.gpuTiling, c.guestReadOnly, c.imageAlias, c.foveation, c.bc, c.hooks,
                         (unsigned long long)c.stagingLimit, (unsigned long long)c.scratchLimit, (unsigned long long)c.bcFrameBytes,
                         (unsigned long long)c.idleSubmits, (unsigned long long)c.byteBudget, (unsigned long long)c.frameBase,
                         (unsigned long long)c.rendererFrame);
        }
        if (rec.type == Type::Mem) mem += rec.payload.size() - 8, pictureMem += rec.payload.size() - 8;
        if (rec.type == Type::Call) ++calls[rec.flags], ++pictureCalls, pictureDraws += Call(rec.flags) == Call::Draw;
        if (rec.type == Type::Picture) {
            std::fprintf(to, "picture %llu: %llu calls, %llu draws, %llu bytes of memory\n", (unsigned long long)++pictures,
                         (unsigned long long)pictureCalls, (unsigned long long)pictureDraws, (unsigned long long)pictureMem);
            pictureMem = pictureCalls = pictureDraws = 0;
        }
        if (rec.type == Type::End && rec.payload.size() == sizeof(cc::EndRecord)) {
            cc::EndRecord e;
            std::memcpy(&e, rec.payload.data(), sizeof e);
            std::fprintf(to, "end: %llu pictures, %llu calls, %llu draws, cut %llu\n", (unsigned long long)e.pictures, (unsigned long long)e.calls,
                         (unsigned long long)e.draws, (unsigned long long)e.cut);
            ended = true;
        }
    }
    if (pictureCalls) std::fprintf(to, "after the last picture: %llu calls, %llu draws\n", (unsigned long long)pictureCalls, (unsigned long long)pictureDraws);
    if (!ended) std::fprintf(to, "no End record (the writer died)\n");
    std::fprintf(to, "records (%llu bytes, %llu of memory):", (unsigned long long)bytes, (unsigned long long)mem);
    for (const auto& [type, count] : records) std::fprintf(to, " %s=%llu", type < std::size(types) ? types[type] : "?", (unsigned long long)count);
    std::fprintf(to, "\ncalls by kind:");
    for (const auto& [kind, count] : calls) std::fprintf(to, " %u=%llu", kind, (unsigned long long)count);
    std::fprintf(to, "\n");
    return true;
}

}  // namespace vp4replay

#ifndef VP4REPLAY_NO_MAIN
int main(int argc, char** argv) {
    std::string file;
    std::vector<vp4replay::OptOut> offs = vp4replay::defaultOptOuts();
    vp4replay::Options options;
    unsigned passes = 5;
    bool dump = false, digest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dump") dump = true;
        else if (arg == "--digest") digest = true;
        else if (arg == "--rates") options.rates = true;
        else if (arg == "--time" && i + 1 < argc) passes = unsigned(std::max(1, std::atoi(argv[++i])));
        else if (arg == "--in-flight" && i + 1 < argc) options.inFlight = std::strtoull(argv[++i], nullptr, 0);
        else if (arg == "--off" && i + 1 < argc) {
            const std::string v = argv[++i];
            const size_t eq = v.find('=');
            offs.push_back({v.substr(0, eq), eq == std::string::npos ? std::string("1") : v.substr(eq + 1)});
        } else if (!arg.empty() && arg[0] != '-' && file.empty()) file = arg;
        else file.clear(), i = argc;
    }
    if (file.empty()) {
        std::fprintf(stderr, "usage: vp4replay FILE [--time R] [--off NAME[=VALUE]]... [--in-flight N] [--dump] [--digest] [--rates]\n");
        return 2;
    }
    if (dump) return vp4replay::dump(file, stdout) ? 0 : 2;
    if (digest || options.rates) {
        const vp4replay::Run run = vp4replay::replay(file, options);
        if (!run.error.empty()) return std::fprintf(stderr, "%s: %s\n", file.c_str(), run.error.c_str()), 2;
        std::printf("%s: commands %016llx (%zu bytes) draws %llu drawn %llu\n", file.c_str(), (unsigned long long)vp4replay::fnv(run.bytes),
                    run.bytes.size(), (unsigned long long)run.draws, (unsigned long long)run.drawn);
        for (const std::string& line : run.rates) std::printf("%s\n", line.c_str());
        return 0;
    }
    return vp4replay::compare(file, offs, passes, options, stdout);
}
#endif
