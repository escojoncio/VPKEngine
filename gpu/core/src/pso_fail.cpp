#include "vpgpu/core/pso_fail.h"

#include <algorithm>
#include <cstdio>

namespace vpgpu {
namespace {

std::string hex16(uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

// Never cut inside a UTF-8 sequence: back up over continuation bytes.
size_t cutAt(const std::string& s, size_t at) {
  if (at >= s.size()) return s.size();
  while (at > 0 && (static_cast<unsigned char>(s[at]) & 0xC0) == 0x80) --at;
  return at;
}

// One line: newlines become the two characters "\n", other control characters a space.
std::string oneLine(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '\n') out += "\\n";
    else if (static_cast<unsigned char>(c) < 0x20 || c == 0x7F) out += ' ';
    else out += c;
  }
  return out;
}

// One-token form of the label for key=value lines.
std::string token(std::string s) {
  for (char& c : s)
    if (c == ' ' || c == '\n' || c == '=' || c == '\t' || c == '\r') c = '_';
  return s;
}

}  // namespace

std::vector<std::string> PsoFailLog::note(uint64_t key, const std::string& label, uint64_t vs, uint64_t fs, uint64_t cs,
                                          const std::string& error) {
  std::vector<std::string> lines;
  const std::string ids = hex16(vs) + hex16(fs) + hex16(cs);
  unsigned n;
  {
    std::lock_guard lock(mutex_);
    if (seen_.size() >= kMaxErrors) return lines;
    if (!seen_.insert(ids + error).second) return lines;
    n = static_cast<unsigned>(seen_.size());
  }
  const bool trunc = error.size() > kMaxErrorBytes;
  const std::string text = trunc ? error.substr(0, cutAt(error, kMaxErrorBytes)) : error;
  std::vector<std::string> parts;
  for (size_t at = 0;;) {
    size_t end = cutAt(text, at + kPartBytes);
    if (end <= at) end = std::min(text.size(), at + kPartBytes);  // (not valid UTF-8: cut anywhere rather than loop)
    parts.push_back(text.substr(at, end - at));
    at = end;
    if (at >= text.size()) break;
  }
  for (size_t i = 0; i < parts.size(); ++i)
    lines.push_back("VPKENGINE_PROBE PSOFAIL n=" + std::to_string(n) + " part=" + std::to_string(i + 1) + "/" +
                    std::to_string(parts.size()) + " key=" + hex16(key) + " label=" + token(label) + " vs=" + hex16(vs) +
                    " ps=" + hex16(fs) + " cs=" + hex16(cs) + " len=" + std::to_string(error.size()) +
                    " trunc=" + (trunc ? "1" : "0") + " error=" + oneLine(parts[i]));
  return lines;
}

std::string PsoFailLog::writeMsl(const std::string& dir, uint64_t id, const char* stage, const std::string& msl) {
  if (dir.empty() || !id || msl.empty()) return "";
  const std::string name = "shader-" + hex16(id) + "-" + stage + "-FAILED.metal";
  {
    std::lock_guard lock(mutex_);
    if (fileBytes_ + msl.size() > kMaxFileBytes || written_.count(name)) return "";
    written_.insert(name);
  }
  FILE* fp = std::fopen((dir + "/" + name).c_str(), "wb");
  if (!fp) return "";
  const bool ok = std::fwrite(msl.data(), 1, msl.size(), fp) == msl.size();
  std::fclose(fp);
  if (!ok) return "";
  std::lock_guard lock(mutex_);
  fileBytes_ += msl.size();
  return name;
}

unsigned PsoFailLog::reported() const {
  std::lock_guard lock(mutex_);
  return static_cast<unsigned>(seen_.size());
}

}  // namespace vpgpu
