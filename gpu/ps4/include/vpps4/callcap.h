// A call capture: what a host asked of vpps4::Renderer over a few pictures, with the guest memory the renderer read on the
// CPU meanwhile, written to a file that another process plays back into a renderer of its own (gpu/ps4/tools/vp4replay.cpp)
// without the host. Renderer::beginCallCapture starts one; nothing here costs anything while none is on.
//
// The file (little endian): kMagic, then records {u16 type, u16 flags, u32 bytes} + `bytes` of payload, not compressed.
//   Meta     text: the renderer's translator fingerprint, the switches of its per-draw paths that are set in the environment,
//            and what the host added (CallCapOptions::meta).
//   Config   struct Config: the renderer's settings, which optional hooks the host gave, the sizes of the structs in the calls.
//   Gen      u64: Hooks::regionGeneration changed (the regions before it are void).
//   Region   RegionRecord: an answer of Hooks::region not seen before under this generation; flags 1: "no region" for
//            (base, size) as asked.
//   Mem      u64 address + bytes: guest memory the renderer is about to read on the CPU (descriptors, tables, indices, program
//            code, constant copies), where it differs from what the file said last (all zeros at the start). Not texture
//            sources (copied by the GPU, or filled, decoded and sampled for changes on the CPU) nor the buffers bound in
//            place, which only the GPU reads: a replay finds zeros there.
//   Done     u64: Device::completedFrames() as the next call finds it, counted from the frame the capture began at.
//   HookAns  flags = Hook: an answer the host gave inside the next call (computeBuffersWanted, computeBuffers: u64 0 / 1;
//            view: ViewAnswer).
//   Call     flags = Call: one call of the public interface and its arguments, after the Mem / Region / HookAns it made.
//            Calls made from inside another one (the runaway submit, a host hook) are part of it and not written.
//   Picture  PictureRecord: the host presented a picture (Renderer::callCapturePicture).
//   End      EndRecord: the capture closed (a file without it was cut short: the writer died).
// The captured renderer was warm and the one that replays starts cold, with other ids and zeros where the memory was not
// captured: a replay does not reproduce the pictures. It is deterministic, which is what comparing two builds of the
// renderer (or two of its paths) over the same calls needs.
#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vpps4 {

struct CallCapOptions {
    std::string path;
    uint32_t pictures = 6;             // the capture closes after this many (the first is cold in a replay, the second fills its caches)
    uint64_t maxBytes = 120ull << 20;  // hard limit of the file: at it the capture closes, cut back to its last whole picture
    std::string meta;                  // lines of the host for the Meta record (its build, the environment)
};

// What a capture left (Renderer::endCallCapture).
struct CallCapResult {
    std::string path;
    uint64_t bytes = 0, pictures = 0, calls = 0, draws = 0;  // (of the whole pictures in the file)
    std::string cut;  // why it closed before its pictures were done ("" if it did not): "bytes", "write", "ended"
};

namespace callcap {

inline constexpr char kMagic[8] = {'V', 'P', '4', 'C', 'A', 'L', 'L', '1'};

enum class Type : uint16_t { Meta = 1, Config, Gen, Region, Mem, Done, HookAns, Call, Picture, End };
enum class Hook : uint16_t { ComputeBuffersWanted = 1, ComputeBuffers, View };
// The calls and their payloads. Unless said otherwise: the arguments in order, each a u64.
enum class Call : uint16_t {
    Draw = 1,          // DrawState (its code / fetch spans emptied) + StageRef of vs, ps, es, gs
    Dispatch,          // DispatchState (idem) + StageRef of cs
    Resolve,           // ColorTarget src, ColorTarget dst
    WriteBack,         // ColorTarget
    Fill, Copy,
    DepthCleared,      // address, the bits of the float
    Invalidate, CoherencePoint, SyncToMemory, AbandonOwners, NoteGpuWrite, CpuAccessed,
    AfterGpu,          // address, size, pureWrite (the action is not kept: a replay queues an empty one)
    GpuIdle, SubmittedIdle, WaitedBelow, ZpassDone, DrainOcclusion,
    TextureFor,        // ImageDesc + u64 integerOnly
    DropGuestBuffers, Submit,
    ColorTargetAt, ColorTargetSlice,  // (they keep what a flip presents stored)
    Gds,               // (makes the GDS buffer)
    MarkLabelOrdered, SupersedeLabels,  // the Device's, through Renderer::markLabelOrdered / supersedeLabels
};

// Hooks the host gave (Config::hooks): a replay gives the same ones.
inline constexpr uint32_t kHookRegionGeneration = 1, kHookWatchWrites = 2, kHookWatchOwned = 4, kHookView = 8, kHookComputeBuffers = 16,
                          kHookComputeBuffersWanted = 32;

struct Config {
    uint32_t version = 1;
    uint32_t drawStateBytes = 0, dispatchStateBytes = 0, colorTargetBytes = 0, imageDescBytes = 0;  // sizeof, as the capturing build had them
    uint32_t hooks = 0;
    uint8_t maxSamples = 1, gpuTiling = 1, guestReadOnly = 0, imageAlias = 1, foveation = 0, bc = 1, pad[2] = {};
    uint64_t stagingLimit = 0, scratchLimit = 0, bcFrameBytes = 0, idleSubmits = 0, byteBudget = 0;
    uint64_t frameBase = 0;      // Device::submittedFrames() when the capture began (Done counts from it)
    uint64_t rendererFrame = 0;  // submits the renderer had made
};
// Where a stage's program is: guest addresses and words (0: none). The fetch subroutine is taken to lie in the same host
// mapping as the code (its address is the code's plus the distance between the two spans).
struct StageRef {
    uint64_t code = 0, codeWords = 0, fetch = 0, fetchWords = 0;
};
struct RegionRecord {
    uint64_t base = 0, size = 0, generation = 0, mirrored = 0;
};
struct ViewAnswer {
    uint64_t format = 0, type = 0, array = 0, baseLayer = 0, layers = 0, baseMip = 0, mips = 0, ok = 0;
};
struct PictureRecord {
    uint64_t pictures = 0, calls = 0, draws = 0;  // so far, this picture included
};
struct EndRecord {
    uint64_t pictures = 0, calls = 0, draws = 0, cut = 0;  // cut: 0 no, 1 "bytes", 2 "write", 3 "ended"
};

// Writes the file. Made and driven by the Renderer (renderer.cpp); every method does nothing once the capture has closed.
class Writer {
public:
    Writer(const CallCapOptions&, uint64_t frameBase);
    ~Writer();
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    bool on() const { return file_ != nullptr; }

    void meta(const std::string& text);
    void config(const Config&);
    void generation(uint64_t g);
    void region(uint64_t base, uint64_t size, const void* host, bool mirrored);
    void noRegion(uint64_t address, uint64_t size);
    // The host pointer of guest [address, +n) in a mirrored region seen under this generation; null if there is none.
    const void* mirror(uint64_t address, uint64_t n) const;
    // Guest [address, +n), readable at `host`, is about to be read.
    void mem(uint64_t address, const void* host, uint64_t n);
    void hookAnswer(Hook, const void* payload, uint32_t n);
    // A call of the public interface begins (`done`: Device::completedFrames(), `generation`: the host's now) ... its
    // arguments (kept until it ends) ... and it ends. Only the outermost call is written.
    void enter(uint64_t done, uint64_t generation);
    void call(Call, const void* payload, size_t n, const void* more = nullptr, size_t moreBytes = 0);
    void leave();
    // The host presented a picture. False: the capture has closed (its pictures are done, or it had closed before).
    bool picture();
    // Closes the capture if it is still on (`cut`: why, before its pictures were done) and says what the file holds.
    CallCapResult close(const char* cut = "ended");

private:
    static constexpr uint64_t kPage = 4096;
    bool begin(Type, uint16_t flags, uint64_t n);
    void bytes(const void* p, size_t n);
    void put(Type type, uint16_t flags, const void* p, size_t n) {
        if (begin(type, flags, n)) bytes(p, n);
    }
    void finish(const char* cut);

    std::FILE* file_ = nullptr;
    CallCapResult result_;
    uint64_t maxBytes_ = 0, pictureLimit_ = 0, frameBase_ = 0;
    uint64_t written_ = 0, calls_ = 0, draws_ = 0;
    uint64_t generation_ = 0, done_ = 0;
    bool haveGeneration_ = false, haveDone_ = false, failed_ = false;
    unsigned depth_ = 0;
    Call pendingKind_ = Call::Draw;
    bool pending_ = false;
    std::vector<uint8_t> pendingBytes_;
    std::unordered_map<uint64_t, std::unique_ptr<uint8_t[]>> shadow_;  // by page address: what the file says the page holds
    struct Mirror {
        uint64_t base, size;
        const void* host;
    };
    std::vector<Mirror> mirrors_;
    std::set<std::tuple<uint64_t, uint64_t, bool>> regions_;  // written under this generation
    std::set<std::pair<uint64_t, uint64_t>> misses_;
};

// Reads the file back.
class Reader {
public:
    struct Record {
        Type type = Type::End;
        uint16_t flags = 0;
        std::vector<uint8_t> payload;
    };
    explicit Reader(const std::string& path);
    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    bool ok() const { return file_ != nullptr; }  // the file opened and begins with kMagic
    // The next record; false at the end of the file or at a record cut short.
    bool next(Record&);

private:
    std::FILE* file_ = nullptr;
};

}  // namespace callcap
}  // namespace vpps4
