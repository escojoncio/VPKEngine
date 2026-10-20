// Persistent store of the renderer's shader work: for every program the bindings the lifter found
// (level 1) and for every program + descriptor state the MSL with its argument / binding metadata
// (level 2), so a later run translates nothing it has translated before. Append-only file under
// the cache root, stamped with a fingerprint of everything that decides the translation: a file
// from another build is dropped. Loading and writing happen on a thread of their own; the
// renderer merges what was loaded on its own thread (take()).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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

struct Loaded {
    std::vector<ProgramRec> programs;
    std::vector<TranslationRec> translations;
    std::vector<vp4gpu::PipelineState> pipelines;  // states seen by earlier runs (prefetch list)
};

// Record encoding, exposed for tests. A record is [type u8][length u32][payload].
std::string encode(const ProgramRec&);
std::string encode(const TranslationRec&);

// Fingerprint of this build's translation: the compile-time hash of the sources that decide the MSL
// plus the emitter's own (runtime) one.
std::string buildFingerprint();

class Worker {
public:
    // `pipelines`: whose Ready states are saved (and the earlier ones read back) under root/pipelines.keys.
    Worker(std::filesystem::path root, std::string fingerprint, vp4gpu::PipelineCache& pipelines);
    ~Worker();
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    // Main thread: true once, when the load is done, with what was loaded.
    bool take(Loaded& out);
    // Blocks until the load is done (tests, hosts that prefer to wait).
    void waitLoaded();
    void append(const ProgramRec&);
    void append(const TranslationRec&);
    // Writes what is queued now (tests); the thread also does it by itself.
    void flush();

private:
    void run();
    void writeQueued(std::unique_lock<std::mutex>& lock);

    std::filesystem::path root_, file_;
    std::string fingerprint_;
    vp4gpu::PipelineCache& pipelines_;
    std::mutex mutex_;
    std::condition_variable wake_, loadedCv_;
    std::string queued_;
    bool stop_ = false, loaded_ = false, taken_ = false, flushNow_ = false;
    Loaded result_;
    uint32_t savedCompiled_ = ~0u;
    std::thread thread_;
};

}  // namespace vpps4::tcache
