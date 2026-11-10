// Diagnostics (probe on): why a pipeline did not compile. The backend's error text (Metal's, which is long and multi-line and
// does not fit the one-token `gpu.err what=pipeline` line) goes to the log as
//   "VPKENGINE_PROBE PSOFAIL n= part=i/m key= label= vs= ps= cs= len= trunc= error=<text>"
// and the MSL of the shaders of that pipeline to <dir>/shader-<id>-<stage>-FAILED.metal. Capped: kMaxErrors distinct
// failures per session, kMaxErrorBytes of text each, kMaxFileBytes of files in all. Nothing here changes what the engine does.
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace vpgpu {

class PsoFailLog {
 public:
  static constexpr unsigned kMaxErrors = 32;           // distinct (error text, shaders) per session
  static constexpr size_t kMaxErrorBytes = 4096;       // of one error text; the rest is cut (trunc=1)
  static constexpr size_t kPartBytes = 1000;           // raw bytes of the text per log line
  static constexpr uint64_t kMaxFileBytes = 8u << 20;  // MSL written in all

  // The log lines for a failed pipeline, or none when it repeats one already reported or the cap is reached.
  // `vs`/`fs`/`cs`: the shader ids of the pipeline (0 = none).
  std::vector<std::string> note(uint64_t key, const std::string& label, uint64_t vs, uint64_t fs, uint64_t cs,
                                const std::string& error);
  // Writes <dir>/shader-<id>-<stage>-FAILED.metal once per (id, stage); returns the file name, or "" when nothing was written
  // (no dir, already written, size budget spent, file not writable).
  std::string writeMsl(const std::string& dir, uint64_t id, const char* stage, const std::string& msl);

  unsigned reported() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_set<std::string> seen_;
  std::unordered_set<std::string> written_;
  uint64_t fileBytes_ = 0;
};

}  // namespace vpgpu
