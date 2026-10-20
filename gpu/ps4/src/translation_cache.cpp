#include "translation_cache.h"

#include <chrono>
#include <cstring>
#include <fstream>

namespace vpps4::tcache {

#ifndef VPPS4_TRANSLATOR_HASH
#define VPPS4_TRANSLATOR_HASH "unknown"
#endif

namespace {

constexpr char kMagic[8] = {'V', 'P', 'S', '4', 'T', 'C', 0, 2};
constexpr uint8_t kProgram = 1, kTranslation = 2;
constexpr uint64_t kMaxFile = 512ull << 20;  // a bigger file is dropped rather than loaded

struct Out {
    std::string s;
    void u8(uint64_t v) { s.push_back(char(v)); }
    void u16(uint64_t v) { for (int i = 0; i < 2; ++i) s.push_back(char(v >> (8 * i))); }
    void u32(uint64_t v) { for (int i = 0; i < 4; ++i) s.push_back(char(v >> (8 * i))); }
    void u64(uint64_t v) { for (int i = 0; i < 8; ++i) s.push_back(char(v >> (8 * i))); }
    void str(const std::string& v) { u32(v.size()), s += v; }
};

struct In {
    const uint8_t* p;
    const uint8_t* end;
    bool bad = false;
    bool need(size_t n) {
        if (size_t(end - p) < n) bad = true;
        return !bad;
    }
    uint64_t take(unsigned bytes) {
        if (!need(bytes)) return 0;
        uint64_t v = 0;
        for (unsigned i = 0; i < bytes; ++i) v |= uint64_t(p[i]) << (8 * i);
        p += bytes;
        return v;
    }
    uint8_t u8() { return uint8_t(take(1)); }
    uint16_t u16() { return uint16_t(take(2)); }
    uint32_t u32() { return uint32_t(take(4)); }
    uint64_t u64() { return take(8); }
    std::string str() {
        const uint32_t n = u32();
        if (!need(n)) return {};
        std::string v(reinterpret_cast<const char*>(p), n);
        p += n;
        return v;
    }
    // A count that cannot exceed what is left (every element takes at least `each` bytes).
    uint32_t count(size_t each) {
        const uint32_t n = u32();
        if (size_t(end - p) / each < n) bad = true;
        return bad ? 0 : n;
    }
};

void put(Out& o, const vpgcn::Binding& b) {
    o.u8(uint8_t(b.kind)), o.u32(b.slot), o.u8(b.source.kind), o.u16(b.source.sgpr), o.u8(b.dwords);
    o.u32(b.source.path.size());
    for (uint32_t v : b.source.path) o.u32(v);
}
vpgcn::Binding getBinding(In& in) {
    vpgcn::Binding b;
    b.kind = vp4ir::ResourceKind(in.u8());
    b.slot = in.u32();
    b.source.kind = vpgcn::DescriptorSource::Kind(in.u8());
    b.source.sgpr = in.u16();
    b.dwords = in.u8();
    const uint32_t n = in.count(4);
    for (uint32_t i = 0; i < n; ++i) b.source.path.push_back(in.u32());
    return b;
}
void put(Out& o, const std::vector<vpgcn::Binding>& v) {
    o.u32(v.size());
    for (const auto& b : v) put(o, b);
}
std::vector<vpgcn::Binding> getBindings(In& in) {
    std::vector<vpgcn::Binding> v;
    const uint32_t n = in.count(9);
    for (uint32_t i = 0; i < n && !in.bad; ++i) v.push_back(getBinding(in));
    return v;
}

std::string frame(uint8_t type, const std::string& payload) {
    Out o;
    o.u8(type), o.u32(payload.size());
    return o.s + payload;
}

}  // namespace

std::string encode(const ProgramRec& r) {
    Out o;
    o.u64(r.key1);
    put(o, r.bindings);
    return frame(kProgram, o.s);
}

std::string encode(const TranslationRec& r) {
    Out o;
    o.u64(r.key1), o.u64(r.key2), o.u64(r.id), o.u8(r.stage);
    o.str(r.shader.source);
    o.u32(r.shader.arguments.size());
    for (const auto& a : r.shader.arguments) o.str(a.name), o.u8(uint8_t(a.kind)), o.u32(a.index), o.u32(a.offset);
    for (uint32_t w : r.shader.workgroupSize) o.u32(w);
    o.u8(r.shader.colorOutputs);
    put(o, r.bindings);
    o.u32(r.resources.size());
    for (const auto& x : r.resources)
        o.str(x.name), o.u8(uint8_t(x.kind)), o.u32(x.index), o.u8(uint8_t(x.dim)), o.u8(uint8_t(x.elem)),
            o.u8(x.depth), o.u8(x.multisample), o.u8(uint8_t(x.access));
    o.u8(r.userDataSlot.has_value()), o.u32(r.userDataSlot.value_or(0));
    return frame(kTranslation, o.s);
}

namespace {

bool decodeProgram(In in, ProgramRec& r) {
    r.key1 = in.u64();
    r.bindings = getBindings(in);
    return !in.bad && in.p == in.end;
}

bool decodeTranslation(In in, TranslationRec& r) {
    r.key1 = in.u64(), r.key2 = in.u64(), r.id = in.u64(), r.stage = in.u8();
    r.shader.source = in.str();
    const uint32_t na = in.count(13);
    for (uint32_t i = 0; i < na && !in.bad; ++i) {
        vp4msl::Argument a;
        a.name = in.str();
        a.kind = vp4ir::ResourceKind(in.u8());
        a.index = in.u32();
        a.offset = in.u32();
        r.shader.arguments.push_back(std::move(a));
    }
    for (uint32_t& w : r.shader.workgroupSize) w = in.u32();
    r.shader.colorOutputs = in.u8();
    r.bindings = getBindings(in);
    const uint32_t nr = in.count(14);
    for (uint32_t i = 0; i < nr && !in.bad; ++i) {
        vp4ir::Resource x;
        x.name = in.str();
        x.kind = vp4ir::ResourceKind(in.u8());
        x.index = in.u32();
        x.dim = vp4ir::TexDim(in.u8());
        x.elem = vp4ir::Scalar(in.u8());
        x.depth = in.u8() != 0;
        x.multisample = in.u8() != 0;
        x.access = vp4ir::Access(in.u8());
        r.resources.push_back(std::move(x));
    }
    const bool has = in.u8() != 0;
    const uint32_t slot = in.u32();
    if (has) r.userDataSlot = slot;
    return !in.bad && in.p == in.end && !r.shader.source.empty() && r.id;
}

std::string header(const std::string& fingerprint) {
    Out o;
    o.s.assign(kMagic, sizeof kMagic);
    o.str(fingerprint);
    return o.s;
}

// Reads the file; returns the offset of the end of the last good record (0: not usable).
uint64_t readFile(const std::filesystem::path& path, const std::string& fingerprint, Loaded& out) {
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(path, ec);
    if (ec || size < sizeof kMagic || size > kMaxFile) return 0;
    std::string data(size, '\0');
    {
        std::ifstream f(path, std::ios::binary);
        if (!f.read(data.data(), std::streamsize(size))) return 0;
    }
    const std::string want = header(fingerprint);
    if (data.size() < want.size() || std::memcmp(data.data(), want.data(), want.size()) != 0) return 0;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(data.data());
    const uint8_t* p = base + want.size();
    const uint8_t* end = base + data.size();
    const uint8_t* good = p;
    while (end - p >= 5) {
        In head{p, end};
        const uint8_t type = head.u8();
        const uint32_t len = head.u32();
        if (size_t(end - head.p) < len) break;  // truncated tail: a crash in the middle of a write
        In body{head.p, head.p + len};
        if (type == kProgram) {
            ProgramRec r;
            if (!decodeProgram(body, r)) break;
            out.programs.push_back(std::move(r));
        } else if (type == kTranslation) {
            TranslationRec r;
            if (!decodeTranslation(body, r)) break;
            out.translations.push_back(std::move(r));
        }  // (unknown types: skipped, written by a newer build of the same fingerprint)
        p = head.p + len;
        good = p;
    }
    return uint64_t(good - base);
}

}  // namespace

std::string buildFingerprint() {
    return std::string(VPPS4_TRANSLATOR_HASH) + "/" + vp4msl::translatorFingerprint();
}

Worker::Worker(std::filesystem::path root, std::string fingerprint, vp4gpu::PipelineCache& pipelines)
    : root_(std::move(root)), file_(root_ / "translations.vpsc"), fingerprint_(std::move(fingerprint)), pipelines_(pipelines) {
    thread_ = std::thread([this] { run(); });
}

Worker::~Worker() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
}

bool Worker::take(Loaded& out) {
    std::lock_guard lock(mutex_);
    if (!loaded_ || taken_) return false;
    taken_ = true;
    out = std::move(result_);
    return true;
}

void Worker::waitLoaded() {
    std::unique_lock lock(mutex_);
    loadedCv_.wait(lock, [&] { return loaded_; });
}

void Worker::append(const ProgramRec& r) {
    std::string bytes = encode(r);
    std::lock_guard lock(mutex_);
    queued_ += bytes;
    wake_.notify_one();
}

void Worker::append(const TranslationRec& r) {
    std::string bytes = encode(r);
    std::lock_guard lock(mutex_);
    queued_ += bytes;
    wake_.notify_one();
}

void Worker::flush() {
    std::unique_lock lock(mutex_);
    loadedCv_.wait(lock, [&] { return loaded_; });
    writeQueued(lock);
}

// Appends the queued records to the file (the lock is released while writing).
void Worker::writeQueued(std::unique_lock<std::mutex>& lock) {
    if (queued_.empty()) return;
    std::string bytes;
    bytes.swap(queued_);
    lock.unlock();
    {
        std::ofstream f(file_, std::ios::binary | std::ios::app);
        if (f) f.write(bytes.data(), std::streamsize(bytes.size()));
    }
    lock.lock();
}

void Worker::run() {
    std::error_code ec;
    std::filesystem::create_directories(root_, ec);
    Loaded loaded;
    const uint64_t good = readFile(file_, fingerprint_, loaded);
    if (good) {
        std::filesystem::resize_file(file_, good, ec);  // a torn tail goes before appending after it
    } else {
        loaded = {};
        std::ofstream f(file_, std::ios::binary | std::ios::trunc);  // none, stale or unreadable: start over
        const std::string h = header(fingerprint_);
        if (f) f.write(h.data(), std::streamsize(h.size()));
    }
    loaded.pipelines = vp4gpu::PipelineCache::readKeyList(vp4gpu::DiskCachePath{root_});
    {
        std::lock_guard lock(mutex_);
        result_ = std::move(loaded);
        loaded_ = true;
    }
    loadedCv_.notify_all();

    std::unique_lock lock(mutex_);
    auto nextSave = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!stop_) {
        wake_.wait_for(lock, std::chrono::seconds(2), [&] { return stop_ || !queued_.empty(); });
        writeQueued(lock);
        if (!stop_ && std::chrono::steady_clock::now() < nextSave) continue;
        // The pipeline states seen (for the next run to compile ahead), when new ones were built.
        const uint32_t compiled = pipelines_.totals().compiled;
        if (compiled != savedCompiled_) {
            savedCompiled_ = compiled;
            lock.unlock();
            pipelines_.save(vp4gpu::DiskCachePath{root_});
            lock.lock();
        }
        nextSave = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    }
    writeQueued(lock);
    lock.unlock();
    const uint32_t compiled = pipelines_.totals().compiled;
    if (compiled != savedCompiled_) pipelines_.save(vp4gpu::DiskCachePath{root_});
}

}  // namespace vpps4::tcache
