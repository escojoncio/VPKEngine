// Key index of a pipeline binary archive file: pipelines-<n>.keys next to pipelines-<n>.metalar holds the
// descriptor keys (MetalBackend descriptorKey) of the pipelines in that file. With it a lookup asks the
// one file that has the pipeline instead of every loaded archive. Platform-neutral (testable off-device).
#pragma once
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace vpgpu::metal2::archive_index {

constexpr uint64_t kMagic = 0x3158444b50565056ull;  // "VPVPKDX1" (bump when descriptorKey changes)

inline std::filesystem::path keysPath(const std::filesystem::path& archive) {
  std::filesystem::path p = archive;
  p.replace_extension(".keys");
  return p;
}

// magic, count, keys... (little-endian u64); tmp + rename so a reader never sees a partial file.
inline bool write(const std::filesystem::path& path, const std::vector<uint64_t>& keys) {
  std::filesystem::path tmp = path;
  tmp += ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) return false;
  std::vector<uint64_t> buf;
  buf.reserve(keys.size() + 2);
  buf.push_back(kMagic);
  buf.push_back(keys.size());
  buf.insert(buf.end(), keys.begin(), keys.end());
  const bool ok = std::fwrite(buf.data(), 8, buf.size(), f) == buf.size();
  const bool closed = std::fclose(f) == 0;
  std::error_code ec;
  if (!ok || !closed) {
    std::filesystem::remove(tmp, ec);
    return false;
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    return false;
  }
  return true;
}

// False: missing, truncated or another version (the archive is then searched as a legacy file).
inline bool read(const std::filesystem::path& path, std::vector<uint64_t>& keys) {
  keys.clear();
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  uint64_t head[2] = {0, 0};
  bool ok = std::fread(head, 8, 2, f) == 2 && head[0] == kMagic && head[1] < (1ull << 28);
  if (ok) {
    keys.resize(size_t(head[1]));
    ok = std::fread(keys.data(), 8, keys.size(), f) == keys.size();
    uint8_t extra;
    ok = ok && std::fread(&extra, 1, 1, f) == 0;  // trailing bytes: not ours
  }
  std::fclose(f);
  if (!ok) keys.clear();
  return ok;
}

}  // namespace vpgpu::metal2::archive_index
