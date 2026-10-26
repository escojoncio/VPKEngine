// VPCAP file writer / reader (format: include/vpvk/vpcap.h). No dependency on the driver: the vpcap_dump tool and the
// phase-2 replayer link this alone.
#include "vpvk/vpcap.h"

#include <zlib.h>

#include <cstring>

namespace vpcap {

namespace {
constexpr size_t kFlushBytes = 1u << 20;  // raw bytes per FRAME / STATE block
}

const char* callName(uint16_t callId) {
  switch (callId) {
#define VPCAP_X(name, num) \
  case num: return #name;
    VPCAP_CALLS(VPCAP_X)
#undef VPCAP_X
    default: return "unknown";
  }
}

Writer::~Writer() {
  if (f_) std::fclose(f_);
}

bool Writer::open(const std::string& path) {
  f_ = std::fopen(path.c_str(), "wb");
  if (!f_) return ok_ = false;
  FileHeader h{};
  std::memcpy(h.magic, kMagic, sizeof h.magic);
  h.version = kVersion, h.headerBytes = sizeof h, h.flags = 1;
  ok_ = std::fwrite(&h, sizeof h, 1, f_) == 1;
  fileBytes_ = sizeof h;
  return ok_;
}

void Writer::writeBlock(Section s, const uint8_t* data, size_t n) {
  if (!f_ || !ok_ || !n) return;
  std::vector<uint8_t> z;
  BlockHeader b{static_cast<uint32_t>(s), kCodecRaw, static_cast<uint32_t>(n), static_cast<uint32_t>(n)};
  const uint8_t* out = data;
  if (n >= 128) {
    uLongf zn = compressBound(static_cast<uLong>(n));
    z.resize(zn);
    if (compress2(z.data(), &zn, data, static_cast<uLong>(n), 1) == Z_OK && zn < n) {
      b.codec = kCodecZlib, b.storedBytes = static_cast<uint32_t>(zn);
      out = z.data();
    }
  }
  ok_ = std::fwrite(&b, sizeof b, 1, f_) == 1 && std::fwrite(out, 1, b.storedBytes, f_) == b.storedBytes;
  fileBytes_ += sizeof b + b.storedBytes;
}

void Writer::flush() {
  writeBlock(cur_, buf_.data(), buf_.size());
  buf_.clear();
}

void Writer::meta(const std::string& text) {
  flush();
  writeBlock(Section::Meta, reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

void Writer::begin(Section s) {
  if (s != cur_) flush();
  cur_ = s;
}

void Writer::append(const void* record, size_t bytes) {
  const auto* p = static_cast<const uint8_t*>(record);
  buf_.insert(buf_.end(), p, p + bytes);
  rawBytes_ += bytes;
  ++calls_;
  if (buf_.size() >= kFlushBytes) flush();
}

void Writer::sync() {
  flush();
  if (f_ && std::fflush(f_) != 0) ok_ = false;
}

bool Writer::close(const std::string& endText) {
  flush();
  writeBlock(Section::End, reinterpret_cast<const uint8_t*>(endText.data()), endText.size());
  if (f_ && std::fclose(f_) != 0) ok_ = false;
  f_ = nullptr;
  return ok_;
}

Reader::~Reader() {
  if (f_) std::fclose(f_);
}

bool Reader::open(const std::string& path, std::string* error) {
  auto fail = [&](const char* why) {
    error_ = why;
    if (error) *error = why;
    return false;
  };
  f_ = std::fopen(path.c_str(), "rb");
  if (!f_) return fail("cannot open file");
  FileHeader h;
  if (std::fread(&h, sizeof h, 1, f_) != 1 || std::memcmp(h.magic, kMagic, sizeof h.magic) != 0) return fail("not a VPCAP file");
  if (h.version == 0 || h.version > kVersion) return fail("unsupported VPCAP version");
  if (h.headerBytes < sizeof h) return fail("bad header");
  if (h.headerBytes > sizeof h) std::fseek(f_, long(h.headerBytes), SEEK_SET);
  version_ = h.version;
  return true;
}

// Loads the next STATE / FRAME block into block_ (META / END blocks are folded into meta_). False at EOF or on error.
bool Reader::loadBlock() {
  for (;;) {
    BlockHeader b;
    if (std::fread(&b, sizeof b, 1, f_) != 1) return false;
    std::vector<uint8_t> stored(b.storedBytes);
    if (b.storedBytes && std::fread(stored.data(), 1, b.storedBytes, f_) != b.storedBytes) {
      error_ = "truncated block";
      return false;
    }
    std::vector<uint8_t> raw;
    if (b.codec == kCodecRaw) {
      raw = std::move(stored);
    } else if (b.codec == kCodecZlib) {
      raw.resize(b.rawBytes);
      uLongf n = b.rawBytes;
      if (uncompress(raw.data(), &n, stored.data(), b.storedBytes) != Z_OK || n != b.rawBytes) {
        error_ = "corrupt compressed block";
        return false;
      }
    } else {
      error_ = "unknown block codec";
      return false;
    }
    const auto s = static_cast<Section>(b.section);
    if (s == Section::Meta || s == Section::End) {
      meta_.append(reinterpret_cast<const char*>(raw.data()), raw.size());
      continue;
    }
    if (s != Section::State && s != Section::Frame) continue;  // a block kind a later version added
    block_ = std::move(raw), pos_ = 0, section_ = s;
    return true;
  }
}

bool Reader::next(Call& out) {
  if (done_ || !f_) return false;
  while (pos_ + sizeof(CallHeader) > block_.size()) {
    if (!loadBlock()) return done_ = true, false;
  }
  CallHeader h;
  std::memcpy(&h, block_.data() + pos_, sizeof h);
  if (pos_ + sizeof h + h.size > block_.size()) {
    error_ = "truncated call record";
    return done_ = true, false;
  }
  out.id = h.id, out.thread = h.thread, out.size = h.size;
  out.data = block_.data() + pos_ + sizeof h;
  out.section = section_, out.frame = frame_, out.index = index_++;
  pos_ += sizeof h + h.size;
  if (section_ == Section::Frame && h.id == id::vpcapFrameEnd) ++frame_;
  return true;
}

bool decodeMemoryData(const Call& c, MemoryData& out) {
  if (c.id != id::vpcapMemoryData || c.size < 24) return false;
  std::memcpy(&out.memory, c.data, 8);
  std::memcpy(&out.offset, c.data + 8, 8);
  std::memcpy(&out.size, c.data + 16, 8);
  if (out.size > c.size - 24) return false;
  out.data = c.data + 24;
  return true;
}

bool decodeImageData(const Call& c, ImageData& out) {
  if (c.id != id::vpcapImageData || c.size < kImageDataHeader) return false;
  uint32_t f[6];
  std::memcpy(&out.image, c.data, 8);
  std::memcpy(f, c.data + 8, sizeof f);
  out.mip = f[0], out.layer = f[1], out.aspect = f[2], out.width = f[3], out.height = f[4], out.depth = f[5];
  std::memcpy(&out.total, c.data + 40, 8);
  std::memcpy(&out.offset, c.data + 48, 8);
  std::memcpy(&out.size, c.data + 56, 8);
  if (out.size > c.size - kImageDataHeader) return false;
  out.data = c.data + kImageDataHeader;
  return true;
}

bool decodeFrameEnd(const Call& c, uint64_t& frame, uint64_t& submits) {
  if (c.id != id::vpcapFrameEnd || c.size < 16) return false;
  std::memcpy(&frame, c.data, 8);
  std::memcpy(&submits, c.data + 8, 8);
  return true;
}

}  // namespace vpcap
