// Diagnostics: the MSL of the guest shaders the title draws with most, and a line of statistics for each
// ("VPKENGINE_PROBE SHADERSTAT id=... stage=... ifs= loads= samples= loops= locals= privarrays= lds= mbcnt= lines= draws= file="),
// so a log says what the shaders the GPU runs look like. Fed only while the probe is on (the renderer does not call it
// otherwise); nothing here changes what the renderer does.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vpps4 {

// Counted on the text of the MSL (a heuristic, as good as the emitter's habits): `ifs` "if (" (the emitter's loop guards
// excluded), `loads` reads through constant / device pointers and texture reads, `samples` sample / gather calls, `loops`
// while / for, `locals` the "lN" variables of the entry, `privarrays` per-lane arrays (vp_private / vp_pN: the LDS spills),
// `lds` uses of threadgroup memory. `mbcnt`: 1 when the lifter noted a v_mbcnt it could not resolve, 0 when not, -1 unknown
// (the translation came from the cache).
struct ShaderStat {
    uint32_t ifs = 0, loads = 0, samples = 0, loops = 0, locals = 0, privArrays = 0, lds = 0, lines = 0;
    int mbcnt = -1;
};
ShaderStat analyzeMsl(std::string_view msl);

class ShaderDump {
public:
    struct Limits {
        unsigned maxShaders = 64;                 // shaders given a SHADERSTAT line (and a file) per run
        uint64_t maxFileBytes = 2ull << 20;       // MSL written to the directory per run
        uint64_t maxRetainedBytes = 32ull << 20;  // MSL kept in memory waiting for its shader to be drawn enough
        uint64_t minDraws = 50;                   // a shader is reported once it has been used this many times
        unsigned round = 16;                      // the most used shaders considered by each flush
    } limits;

    // A shader registered with the host (`stage`: static text). Its text is kept while the budget lasts.
    void add(uint64_t id, const char* stage, const std::string& msl, int mbcnt);
    // A draw or dispatch used it.
    void use(uint64_t id) {
        const auto it = map_.find(id);
        if (it != map_.end()) ++it->second.draws;
    }
    // Reports, among the `round` most used shaders, those not reported yet: one log line each, and (with `dir`) the MSL
    // in <dir>/shader-<id>-<stage>.metal while the size budget lasts. Returns the lines (without the "VPKENGINE_PROBE " prefix).
    std::vector<std::string> flush(const std::string& dir);

    size_t shaders() const { return map_.size(); }
    unsigned reported() const { return reported_; }
    uint64_t fileBytes() const { return fileBytes_; }
    uint64_t retainedBytes() const { return retained_; }

private:
    struct Entry {
        const char* stage = "";
        ShaderStat stat;
        std::string source;  // empty once reported or when the retention budget was spent
        bool hasStat = false;
        bool reported = false;
        uint64_t draws = 0;
    };
    std::unordered_map<uint64_t, Entry> map_;
    uint64_t retained_ = 0, fileBytes_ = 0;
    unsigned reported_ = 0;
};

}  // namespace vpps4
