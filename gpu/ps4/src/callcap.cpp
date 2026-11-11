#include "vpps4/callcap.h"

#include <algorithm>
#include <cstring>
#include <filesystem>

namespace vpps4::callcap {

namespace {
constexpr uint64_t kHeaderBytes = 8;
constexpr uint64_t kEndBytes = kHeaderBytes + sizeof(EndRecord);
constexpr size_t kMisses = 4096;  // "no region" answers written per generation
}  // namespace

Writer::Writer(const CallCapOptions& o, uint64_t frameBase)
    : maxBytes_(o.maxBytes), pictureLimit_(std::max<uint32_t>(o.pictures, 1)), frameBase_(frameBase) {
    result_.path = o.path;
    if (maxBytes_ < sizeof kMagic + kEndBytes) return;
    file_ = std::fopen(o.path.c_str(), "wb");
    if (!file_) return;
    std::setvbuf(file_, nullptr, _IOFBF, 1 << 20);
    bytes(kMagic, sizeof kMagic);
    written_ = sizeof kMagic;
}

Writer::~Writer() { close(); }

bool Writer::begin(Type type, uint16_t flags, uint64_t n) {
    if (!file_) return false;
    if (failed_ || n > ~uint32_t(0) || written_ + kHeaderBytes + n + kEndBytes > maxBytes_) {
        finish("bytes");  // ("write" if a write failed)
        return false;
    }
    const uint16_t head[2] = {uint16_t(type), flags};
    const uint32_t size = uint32_t(n);
    bytes(head, sizeof head), bytes(&size, sizeof size);
    written_ += kHeaderBytes + n;
    return true;
}

void Writer::bytes(const void* p, size_t n) {
    if (n && std::fwrite(p, 1, n, file_) != n) failed_ = true;
}

void Writer::meta(const std::string& text) { put(Type::Meta, 0, text.data(), text.size()); }
void Writer::config(const Config& c) { put(Type::Config, 0, &c, sizeof c); }

void Writer::generation(uint64_t g) {
    if (!file_ || (haveGeneration_ && g == generation_)) return;
    generation_ = g, haveGeneration_ = true;
    mirrors_.clear(), regions_.clear(), misses_.clear();
    put(Type::Gen, 0, &g, sizeof g);
}

void Writer::region(uint64_t base, uint64_t size, const void* host, bool mirrored) {
    if (!file_) return;
    if (!regions_.insert({base, size, mirrored}).second) return;
    if (mirrored) mirrors_.push_back({base, size, host});
    const RegionRecord r{base, size, generation_, mirrored};
    put(Type::Region, 0, &r, sizeof r);
}

void Writer::noRegion(uint64_t address, uint64_t size) {
    if (!file_ || misses_.size() >= kMisses || !misses_.insert({address, size}).second) return;
    const RegionRecord r{address, size, generation_, 0};
    put(Type::Region, 1, &r, sizeof r);
}

const void* Writer::mirror(uint64_t address, uint64_t n) const {
    for (const Mirror& m : mirrors_)
        if (address >= m.base && address + n <= m.base + m.size) return static_cast<const uint8_t*>(m.host) + (address - m.base);
    return nullptr;
}

void Writer::mem(uint64_t address, const void* host, uint64_t n) {
    if (!file_ || !host || !n) return;
    const uint8_t* src = static_cast<const uint8_t*>(host);
    // The pieces (one per page) that differ from what the file said last go to the shadow first and are written from it,
    // neighbours in one record: the file and the shadow then agree even if the guest writes the memory meanwhile.
    uint64_t runBegin = 0, runEnd = 0;
    const auto flush = [&] {
        if (runEnd > runBegin && begin(Type::Mem, 0, 8 + (runEnd - runBegin))) {
            bytes(&runBegin, 8);
            for (uint64_t at = runBegin; at < runEnd;) {
                const uint64_t page = at & ~(kPage - 1), piece = std::min(runEnd, page + kPage) - at;
                bytes(shadow_[page].get() + (at - page), piece);
                at += piece;
            }
        }
        runBegin = runEnd = 0;
    };
    for (uint64_t at = address; at < address + n && file_;) {
        const uint64_t page = at & ~(kPage - 1), piece = std::min(address + n, page + kPage) - at;
        std::unique_ptr<uint8_t[]>& was = shadow_[page];
        if (!was) was.reset(new uint8_t[kPage]());  // (a replay starts from zeros)
        if (std::memcmp(was.get() + (at - page), src + (at - address), piece) != 0) {
            std::memcpy(was.get() + (at - page), src + (at - address), piece);
            if (runEnd != at) flush(), runBegin = at;
            runEnd = at + piece;
        }
        at += piece;
    }
    flush();
}

void Writer::hookAnswer(Hook h, const void* payload, uint32_t n) { put(Type::HookAns, uint16_t(h), payload, n); }

void Writer::enter(uint64_t done, uint64_t g) {
    if (depth_++ || !file_) return;
    pending_ = false;
    generation(g);
    const uint64_t since = done > frameBase_ ? done - frameBase_ : 0;
    if (!haveDone_ || since != done_) done_ = since, haveDone_ = true, put(Type::Done, 0, &since, sizeof since);
}

void Writer::call(Call kind, const void* payload, size_t n, const void* more, size_t moreBytes) {
    if (depth_ != 1 || !file_) return;
    pendingKind_ = kind, pending_ = true;
    pendingBytes_.resize(n + moreBytes);
    if (n) std::memcpy(pendingBytes_.data(), payload, n);
    if (moreBytes) std::memcpy(pendingBytes_.data() + n, more, moreBytes);
}

void Writer::leave() {
    if (!depth_ || --depth_ || !pending_) return;
    pending_ = false;
    if (begin(Type::Call, uint16_t(pendingKind_), pendingBytes_.size())) {
        bytes(pendingBytes_.data(), pendingBytes_.size());
        ++calls_, draws_ += pendingKind_ == Call::Draw;
    }
}

bool Writer::picture() {
    if (!file_) return false;
    const PictureRecord p{result_.pictures + 1, calls_, draws_};
    if (!begin(Type::Picture, 0, sizeof p)) return false;
    bytes(&p, sizeof p);
    result_.pictures = p.pictures, result_.calls = calls_, result_.draws = draws_, result_.bytes = written_;
    if (result_.pictures >= pictureLimit_) finish("");
    return file_ != nullptr;
}

void Writer::finish(const char* cut) {
    if (!file_) return;
    if (failed_) cut = "write";
    result_.cut = cut;
    if (!*cut || !result_.pictures) {
        // (cut before the first picture was whole: the calls written so far are kept)
        result_.bytes = written_, result_.calls = calls_, result_.draws = draws_;
    } else if (written_ > result_.bytes) {
        // Back to the end of the last whole picture.
        std::fclose(file_);
        std::error_code ec;
        std::filesystem::resize_file(result_.path, result_.bytes, ec);
        file_ = ec ? nullptr : std::fopen(result_.path.c_str(), "ab");
    }
    if (file_) {
        const uint16_t head[2] = {uint16_t(Type::End), 0};
        const uint32_t size = sizeof(EndRecord);
        const EndRecord e{result_.pictures, result_.calls, result_.draws,
                          uint64_t(!*cut ? 0 : !std::strcmp(cut, "bytes") ? 1 : !std::strcmp(cut, "write") ? 2 : 3)};
        bytes(head, sizeof head), bytes(&size, sizeof size), bytes(&e, sizeof e);
        result_.bytes += kEndBytes;
        std::fclose(file_);
        file_ = nullptr;
    }
    shadow_.clear(), mirrors_.clear(), regions_.clear(), misses_.clear();
    pendingBytes_ = {};
}

CallCapResult Writer::close(const char* cut) {
    finish(cut);
    return result_;
}

Reader::Reader(const std::string& path) {
    file_ = std::fopen(path.c_str(), "rb");
    char magic[sizeof kMagic];
    if (file_ && (std::fread(magic, 1, sizeof magic, file_) != sizeof magic || std::memcmp(magic, kMagic, sizeof magic) != 0)) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

Reader::~Reader() {
    if (file_) std::fclose(file_);
}

bool Reader::next(Record& r) {
    uint16_t head[2];
    uint32_t size;
    if (!file_ || std::fread(head, 1, sizeof head, file_) != sizeof head || std::fread(&size, 1, sizeof size, file_) != sizeof size) return false;
    r.type = Type(head[0]), r.flags = head[1];
    if (size > (1u << 30)) return false;  // (no record is that long: the file is damaged)
    r.payload.resize(size);
    return !size || std::fread(r.payload.data(), 1, size, file_) == size;
}

}  // namespace vpps4::callcap
