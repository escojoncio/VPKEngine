// Persistent store of the renderer's shader work: for every program the bindings the lifter found
// (level 1) and for every program + descriptor state the MSL with its argument / binding metadata
// (level 2), so a later run translates nothing it has translated before; programs the lifter could
// not translate are remembered too (a negative cache: they are not retried every start). Append-only
// file under the cache root, stamped with a fingerprint of everything that decides the translation:
// a file from another build is dropped. Loading and writing happen on a thread of their own; the
// renderer merges what was loaded on its own thread (take()).
//
// The file is never read whole: a first pass indexes the records by their keys (seeking over the
// payloads), a record that repeats a key counts as a duplicate, and when duplicates (and records
// nobody needs) waste more than a set slack, or the file is over its size cap, the file is rewritten
// at start with each key once, in order of last use (a translation a run used appends a small touch
// record) and, over the cap, without the least recently used translations. The records are then
// decoded one at a time. Keys already in the file are not appended again.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "vp4gpu/core/pipeline_cache.h"
#include "vpgcn/lift.h"
#include "vp4ir/ir.h"
#include "vp4msl/msl.h"

namespace vpps4::tcache {

struct ProgramRec {  // level 1: what the program reads
    uint64_t key1 = 0;
    std::vector<vpgcn::Binding> bindings;
};

struct TranslationRec {  // level 2
    uint64_t key1 = 0, key2 = 0, id = 0;
    uint8_t stage = 0;  // vpgcn::ShaderStage
    vp4msl::MslShader shader;
    std::vector<vpgcn::Binding> bindings;
    std::vector<vp4ir::Resource> resources;
    std::optional<uint32_t> userDataSlot;
};

struct FailureRec {  // a program that does not translate (negative cache)
    uint8_t level = 1;   // 1: the lifter's first pass (key = the program's key), 2: the whole chain (key = key2)
    uint64_t key = 0;
    uint8_t stage = 0;
    std::string error;
};

struct Loaded {
    std::vector<FailureRec> failures;
    std::vector<ProgramRec> programs;
    std::vector<TranslationRec> translations;
    std::vector<vp4gpu::PipelineState> pipelines;  // states seen by earlier runs (prefetch list)
};

// Record encoding, exposed for tests. A record is [type u8][length u32][payload].
std::string encode(const ProgramRec&);
std::string encode(const TranslationRec&);
std::string encode(const FailureRec&);
std::string encodeTouch(uint64_t key2);

// Fingerprint of this build's translation: the compile-time hash of the sources that decide the MSL
// plus the emitter's own (runtime) one.
std::string buildFingerprint();

struct Limits {
    uint64_t maxBytes = 256ull << 20;       // over this at start, the least recently used translations go
    double keepFraction = 0.75;             // ... down to this fraction of it
    uint64_t slackBytes = 8ull << 20;       // reclaimable bytes (duplicates, touches) that trigger a rewrite ...
    double slackFraction = 0.2;             // ... or this fraction of the file, whichever is larger
};

// What the start-up pass did (valid once the load is done).
struct LoadStats {
    uint64_t fileBytes = 0, afterBytes = 0;  // before / after the rewrite
    uint64_t programs = 0, translations = 0, failures = 0;  // records kept
    uint64_t duplicates = 0;                 // records that repeated a key
    uint64_t evicted = 0;                    // translations dropped for the size cap
    bool compacted = false;
};

class Worker {
public:
    // `pipelines`: whose Ready states are saved (and the earlier ones read back) under root/pipelines.keys.
    Worker(std::filesystem::path root, std::string fingerprint, vp4gpu::PipelineCache& pipelines, Limits limits = {});
    ~Worker();
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    // Main thread: true once, when the load is done, with what was loaded.
    bool take(Loaded& out);
    // Blocks until the load is done (tests, hosts that prefer to wait).
    void waitLoaded();
    void append(const ProgramRec&);
    void append(const TranslationRec&);
    void append(const FailureRec&);
    // A translation of the file was used by this run (once per key): its place in the order of use.
    void touch(uint64_t key2);
    // Writes what is queued now (tests); the thread also does it by itself. Serialised with it.
    void flush();
    LoadStats loadStats();

private:
    enum Kind : uint8_t { KProgram, KTranslation, KFailure1, KFailure2, KTouch, KKinds };
    struct Item {
        Kind kind;
        uint64_t key;
        std::string bytes;
    };
    void run();
    void queue(Kind kind, uint64_t key, std::string bytes);
    void writeQueued();

    std::filesystem::path root_, file_;
    std::string fingerprint_;
    vp4gpu::PipelineCache& pipelines_;
    std::mutex mutex_;
    std::condition_variable wake_, loadedCv_;
    std::vector<Item> queued_;
    std::mutex writeMutex_;                          // one writer at a time (the thread, flush())
    std::unordered_set<uint64_t> known_[KKinds];     // keys in the file; the load fills it, then writeMutex_
    LoadStats stats_;
    Limits limits_;
    bool stop_ = false, loaded_ = false, taken_ = false;
    Loaded result_;
    uint32_t savedCompiled_ = ~0u;
    std::thread thread_;
};

}  // namespace vpps4::tcache
