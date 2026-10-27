#include "translation_cache.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace vpps4::tcache {

#ifndef VPPS4_TRANSLATOR_HASH
#define VPPS4_TRANSLATOR_HASH "unknown"
#endif

namespace {

constexpr char kMagic[8] = {'V', 'P', 'S', '4', 'T', 'C', 0, 2};
constexpr uint8_t kProgram = 1, kTranslation = 2, kFailure = 3, kTouch = 4;
constexpr uint64_t kMaxFile = 4ull << 30;    // a bigger file is dropped (the size cap rewrites long before)
constexpr uint32_t kMaxRecord = 256u << 20;  // a longer record is a torn or foreign file

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
    b.kind = vpir::ResourceKind(in.u8());
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

std::string encode(const FailureRec& r) {
    Out o;
    o.u8(r.level), o.u64(r.key), o.u8(r.stage), o.str(r.error);
    return frame(kFailure, o.s);
}

std::string encodeTouch(uint64_t key2) {
    Out o;
    o.u64(key2);
    return frame(kTouch, o.s);
}

namespace {

bool decodeFailure(In in, FailureRec& r) {
    r.level = in.u8(), r.key = in.u64(), r.stage = in.u8();
    r.error = in.str();
    return !in.bad && in.p == in.end && (r.level == 1 || r.level == 2) && r.key;
}

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
        vpmsl::Argument a;
        a.name = in.str();
        a.kind = vpir::ResourceKind(in.u8());
        a.index = in.u32();
        a.offset = in.u32();
        r.shader.arguments.push_back(std::move(a));
    }
    for (uint32_t& w : r.shader.workgroupSize) w = in.u32();
    r.shader.colorOutputs = in.u8();
    r.bindings = getBindings(in);
    const uint32_t nr = in.count(14);
    for (uint32_t i = 0; i < nr && !in.bad; ++i) {
        vpir::Resource x;
        x.name = in.str();
        x.kind = vpir::ResourceKind(in.u8());
        x.index = in.u32();
        x.dim = vpir::TexDim(in.u8());
        x.elem = vpir::Scalar(in.u8());
        x.depth = in.u8() != 0;
        x.multisample = in.u8() != 0;
        x.access = vpir::Access(in.u8());
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

// ---- The start-up pass ------------------------------------------------------------------------------

// One record of the file as the index pass saw it.
struct Entry {
    uint8_t type = 0;
    uint64_t key = 0;
    uint64_t off = 0;    // of the record (its type byte)
    uint32_t len = 0;    // payload bytes
    uint64_t stamp = 0;  // order of last use: the file offset of its latest creation or touch
    bool live = true;
    uint64_t total() const { return 5ull + len; }
};

struct Index {
    std::vector<Entry> entries;
    uint64_t good = 0;       // end of the last intact record
    uint64_t headerEnd = 0;
    uint64_t duplicates = 0;
};

// Walks the records without reading their payloads (only the leading keys): which keys are there, how
// big, where. False: not usable (missing, other build, too big).
bool indexFile(const std::filesystem::path& path, const std::string& fingerprint, uint64_t size, Index& out) {
    if (size < sizeof kMagic || size > kMaxFile) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    const std::string want = header(fingerprint);
    std::string got(want.size(), '\0');
    if (size < want.size() || !f.read(got.data(), std::streamsize(got.size())) || got != want) return false;
    out.headerEnd = out.good = want.size();
    std::unordered_map<uint64_t, size_t> at[4];  // program, translation, failure level 1 / 2
    uint64_t p = want.size();
    while (size - p >= 5) {
        uint8_t h[5];
        f.seekg(std::streamoff(p));
        if (!f.read(reinterpret_cast<char*>(h), 5)) break;
        uint32_t len = 0;
        for (int i = 0; i < 4; ++i) len |= uint32_t(h[1 + i]) << (8 * i);
        if (len > kMaxRecord || size - p - 5 < len) break;  // truncated tail: a crash in the middle of a write
        uint8_t k[17] = {};
        const size_t lead = std::min<size_t>(len, sizeof k);
        if (lead && !f.read(reinterpret_cast<char*>(k), std::streamsize(lead))) break;
        auto key = [&](size_t from) {
            uint64_t v = 0;
            for (int i = 0; i < 8; ++i) v |= uint64_t(k[from + i]) << (8 * i);
            return v;
        };
        Entry e;
        e.type = h[0];
        e.off = e.stamp = p;
        e.len = len;
        int slot = -1;
        bool ok = true;
        if (e.type == kProgram && len >= 8) e.key = key(0), slot = 0;
        else if (e.type == kTranslation && len >= 16) e.key = key(8), slot = 1;  // (key1, key2: keyed by key2)
        else if (e.type == kFailure && len >= 9 && (k[0] == 1 || k[0] == 2)) e.key = key(1), slot = 1 + k[0];
        else if (e.type == kTouch && len >= 8) {
            auto t = at[1].find(key(0));
            if (t != at[1].end()) out.entries[t->second].stamp = p;
            ok = false;  // (not an entry of its own: its effect is the stamp)
        } else {
            ok = false;  // unknown or short: reclaimable
        }
        if (ok) {
            auto [it, fresh] = at[slot].emplace(e.key, out.entries.size());
            if (fresh) {
                out.entries.push_back(e);
            } else {  // the same key again: kept once, in the place of its latest sighting
                ++out.duplicates;
                out.entries[it->second].stamp = p;
            }
        }
        p += 5 + uint64_t(len);
        out.good = p;
    }
    return true;
}

// Rewrites the file with the live entries once each, oldest use first. Beside and renamed over.
bool rewrite(const std::filesystem::path& path, const std::string& fingerprint, Index& idx) {
    std::vector<const Entry*> keep;
    for (const Entry& e : idx.entries)
        if (e.live) keep.push_back(&e);
    std::stable_sort(keep.begin(), keep.end(), [](const Entry* a, const Entry* b) { return a->stamp < b->stamp; });
    const std::filesystem::path tmp = path.string() + ".tmp";
    {
        std::ifstream in(path, std::ios::binary);
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!in || !out) return false;
        const std::string h = header(fingerprint);
        out.write(h.data(), std::streamsize(h.size()));
        std::vector<char> buf;
        for (const Entry* e : keep) {
            buf.resize(e->total());
            in.seekg(std::streamoff(e->off));
            if (!in.read(buf.data(), std::streamsize(buf.size()))) return false;
            out.write(buf.data(), std::streamsize(buf.size()));
        }
        if (!out) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) std::filesystem::remove(tmp, ec);
    return !ec;
}

// Decodes the file record by record (nothing of it is held but the current payload). A key seen
// before is skipped; a record that does not decode is skipped too (its frame was checked by the index).
void loadRecords(const std::filesystem::path& path, uint64_t from, Loaded& out, std::unordered_set<uint64_t>* known) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    f.seekg(std::streamoff(from));
    std::vector<uint8_t> payload;
    for (;;) {
        uint8_t h[5];
        if (!f.read(reinterpret_cast<char*>(h), 5)) break;
        uint32_t len = 0;
        for (int i = 0; i < 4; ++i) len |= uint32_t(h[1 + i]) << (8 * i);
        if (len > kMaxRecord) break;
        payload.resize(len);
        if (len && !f.read(reinterpret_cast<char*>(payload.data()), len)) break;
        const In body{payload.data(), payload.data() + len};
        // known[0..3]: program, translation, failure 1, failure 2
        if (h[0] == kProgram) {
            ProgramRec r;
            if (decodeProgram(body, r) && known[0].insert(r.key1).second) out.programs.push_back(std::move(r));
        } else if (h[0] == kTranslation) {
            TranslationRec r;
            if (decodeTranslation(body, r) && known[1].insert(r.key2).second) out.translations.push_back(std::move(r));
        } else if (h[0] == kFailure) {
            FailureRec r;
            if (decodeFailure(body, r) && known[1 + r.level].insert(r.key).second) out.failures.push_back(std::move(r));
        }
    }
}

}  // namespace

std::string buildFingerprint() {
    return std::string(VPPS4_TRANSLATOR_HASH) + "/" + vpmsl::translatorFingerprint();
}

Worker::Worker(std::filesystem::path root, std::string fingerprint, vpgpu::PipelineCache& pipelines, Limits limits)
    : root_(std::move(root)), file_(root_ / "translations.vpsc"), fingerprint_(std::move(fingerprint)), pipelines_(pipelines),
      limits_(limits) {
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

LoadStats Worker::loadStats() {
    std::unique_lock lock(mutex_);
    loadedCv_.wait(lock, [&] { return loaded_; });
    return stats_;
}

void Worker::queue(Kind kind, uint64_t key, std::string bytes) {
    std::lock_guard lock(mutex_);
    queued_.push_back({kind, key, std::move(bytes)});
    wake_.notify_one();
}

void Worker::append(const ProgramRec& r) { queue(KProgram, r.key1, encode(r)); }
void Worker::append(const TranslationRec& r) { queue(KTranslation, r.key2, encode(r)); }
void Worker::append(const FailureRec& r) { queue(r.level == 1 ? KFailure1 : KFailure2, r.key, encode(r)); }
void Worker::touch(uint64_t key2) { queue(KTouch, key2, encodeTouch(key2)); }

void Worker::flush() {
    waitLoaded();
    writeQueued();
}

// Appends the queued records to the file: the ones whose key is not in it already. One writer at a
// time (the worker thread and flush() both come here); the queue lock is not held while writing.
void Worker::writeQueued() {
    std::lock_guard writer(writeMutex_);
    std::vector<Item> items;
    {
        std::lock_guard lock(mutex_);
        items.swap(queued_);
    }
    if (items.empty()) return;
    std::string bytes;
    for (Item& it : items) {
        if (it.kind == KTouch && !known_[KTranslation].count(it.key)) continue;  // (nothing of that key to date)
        if (!known_[it.kind].insert(it.key).second) continue;                      // already in the file / this batch
        bytes += it.bytes;
    }
    if (bytes.empty()) return;
    std::ofstream f(file_, std::ios::binary | std::ios::app);
    if (f) f.write(bytes.data(), std::streamsize(bytes.size()));
}

void Worker::run() {
    std::error_code ec;
    std::filesystem::create_directories(root_, ec);
    Loaded loaded;
    LoadStats st;
    {
        std::lock_guard writer(writeMutex_);  // (known_ is filled here, before anything is written)
        std::error_code sec;
        const uint64_t size = std::filesystem::file_size(file_, sec);
        Index idx;
        const bool usable = !sec && indexFile(file_, fingerprint_, size, idx);
        if (usable) {
            st.fileBytes = size;
            st.duplicates = idx.duplicates;
            if (idx.good < size) std::filesystem::resize_file(file_, idx.good, ec);  // a torn tail goes before appending
            uint64_t live = idx.headerEnd;
            for (const Entry& e : idx.entries) live += e.total();
            // Over the size cap: the least recently used translations go (programs and failures are small
            // and cheap to keep).
            bool evicted = false;
            if (live > limits_.maxBytes) {
                std::vector<Entry*> byUse;
                for (Entry& e : idx.entries)
                    if (e.type == kTranslation) byUse.push_back(&e);
                std::sort(byUse.begin(), byUse.end(), [](const Entry* a, const Entry* b) { return a->stamp < b->stamp; });
                const uint64_t target = uint64_t(double(limits_.maxBytes) * limits_.keepFraction);
                for (Entry* e : byUse) {
                    if (live <= target) break;
                    e->live = false;
                    live -= e->total();
                    ++st.evicted;
                    evicted = true;
                }
            }
            const uint64_t reclaimable = idx.good - live;
            const uint64_t slack = std::max<uint64_t>(limits_.slackBytes, uint64_t(double(idx.good) * limits_.slackFraction));
            if (evicted || reclaimable > slack) {
                // (if it fails the old file stays: duplicates are skipped at load, nothing is lost)
                st.compacted = rewrite(file_, fingerprint_, idx);
            }
            loadRecords(file_, idx.headerEnd, loaded, known_);  // (known_: the keys in the file, for the writer)
            std::error_code e2;
            st.afterBytes = std::filesystem::file_size(file_, e2);
        } else {
            loaded = {};
            std::ofstream f(file_, std::ios::binary | std::ios::trunc);  // none, stale or unreadable: start over
            const std::string h = header(fingerprint_);
            if (f) f.write(h.data(), std::streamsize(h.size()));
            st.fileBytes = size_t(sec ? 0 : size);
            st.afterBytes = h.size();
        }
        st.programs = loaded.programs.size();
        st.translations = loaded.translations.size();
        st.failures = loaded.failures.size();
    }
    loaded.pipelines = vpgpu::PipelineCache::readKeyList(vpgpu::DiskCachePath{root_});
    {
        std::lock_guard lock(mutex_);
        result_ = std::move(loaded);
        stats_ = st;
        loaded_ = true;
    }
    loadedCv_.notify_all();

    std::unique_lock lock(mutex_);
    auto nextSave = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!stop_) {
        wake_.wait_for(lock, std::chrono::seconds(2), [&] { return stop_ || !queued_.empty(); });
        lock.unlock();
        writeQueued();
        lock.lock();
        if (!stop_ && std::chrono::steady_clock::now() < nextSave) continue;
        // The pipeline states seen (for the next run to compile ahead), when new ones were built.
        const uint32_t compiled = pipelines_.totals().compiled;
        if (compiled != savedCompiled_) {
            savedCompiled_ = compiled;
            lock.unlock();
            pipelines_.save(vpgpu::DiskCachePath{root_});
            lock.lock();
        }
        nextSave = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    }
    lock.unlock();
    writeQueued();
    const uint32_t compiled = pipelines_.totals().compiled;
    if (compiled != savedCompiled_) pipelines_.save(vpgpu::DiskCachePath{root_});
}

}  // namespace vpps4::tcache
