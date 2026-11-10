#include "vpps4/shader_stats.h"

#include <algorithm>
#include <cstdio>

namespace vpps4 {

namespace {

uint32_t countOf(std::string_view line, std::string_view what) {
    uint32_t n = 0;
    for (size_t at = line.find(what); at != std::string_view::npos; at = line.find(what, at + what.size())) ++n;
    return n;
}

bool startsWith(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

// "<type> l<digits>{};" : a local variable of the entry.
bool isLocalDecl(std::string_view t) {
    if (t.size() < 6 || t.substr(t.size() - 3) != "{};") return false;
    t.remove_suffix(3);
    const size_t sp = t.rfind(' ');
    if (sp == std::string_view::npos || sp + 2 > t.size() || t[sp + 1] != 'l') return false;
    for (size_t i = sp + 2; i < t.size(); ++i)
        if (t[i] < '0' || t[i] > '9') return false;
    return sp + 2 < t.size();
}

}  // namespace

ShaderStat analyzeMsl(std::string_view msl) {
    ShaderStat s;
    size_t pos = 0;
    while (pos < msl.size()) {
        size_t end = msl.find('\n', pos);
        if (end == std::string_view::npos) end = msl.size();
        std::string_view t = msl.substr(pos, end - pos);
        pos = end + 1;
        ++s.lines;
        while (!t.empty() && (t.front() == ' ' || t.front() == '\t')) t.remove_prefix(1);
        while (!t.empty() && (t.back() == '\r' || t.back() == ' ')) t.remove_suffix(1);
        if (t.empty()) continue;
        if (!startsWith(t, "if (++vp_guard")) s.ifs += countOf(t, "if (");
        s.loops += countOf(t, "while (") + countOf(t, "for (");
        // (a line that begins with a dereference is a store: the pointer it writes through is not a load)
        const uint32_t derefs = countOf(t, "*(constant ") + countOf(t, "*(const device ") + countOf(t, "*(device ");
        const bool store = startsWith(t, "*(constant ") || startsWith(t, "*(const device ") || startsWith(t, "*(device ");
        s.loads += derefs - (store ? 1 : 0) + countOf(t, ".read(");
        s.samples += countOf(t, ".sample(") + countOf(t, ".sample_compare(") + countOf(t, ".gather(") + countOf(t, ".gather_compare(");
        if (isLocalDecl(t)) ++s.locals;
        if (startsWith(t, "uint4 vp_p") && t.find("] = {}") != std::string_view::npos) ++s.privArrays;
        if (!startsWith(t, "threadgroup ")) s.lds += countOf(t, "vp_shared");
    }
    return s;
}

void ShaderDump::add(uint64_t id, const char* stage, const std::string& msl, int mbcnt) {
    auto [it, fresh] = map_.try_emplace(id);
    if (!fresh) return;
    Entry& e = it->second;
    e.stage = stage;
    e.stat = analyzeMsl(msl);
    e.stat.mbcnt = mbcnt;
    e.hasStat = true;
    if (retained_ + msl.size() > limits.maxRetainedBytes) {
        // (the budget is for the shaders that get drawn: the texts of those not used enough yet make room)
        for (auto& [other, x] : map_)
            if (!x.reported && x.draws < limits.minDraws && !x.source.empty()) retained_ -= x.source.size(), std::string().swap(x.source);
    }
    if (retained_ + msl.size() <= limits.maxRetainedBytes) {
        e.source = msl;
        retained_ += msl.size();
    }
}

std::vector<std::string> ShaderDump::flush(const std::string& dir) {
    std::vector<std::string> lines;
    if (reported_ >= limits.maxShaders) return lines;
    std::vector<std::pair<uint64_t, uint64_t>> order;  // (draws, id)
    for (const auto& [id, e] : map_)
        if (e.hasStat && !e.reported && e.draws >= limits.minDraws) order.push_back({e.draws, id});
    const size_t top = std::min<size_t>(order.size(), limits.round);
    // (the most used among all of them: shaders already reported are not in the ranking, so a round reaches past them)
    std::partial_sort(order.begin(), order.begin() + top, order.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    for (size_t k = 0; k < top && reported_ < limits.maxShaders; ++k) {
        Entry& e = map_.at(order[k].second);
        char id[24];
        std::snprintf(id, sizeof id, "%016llx", static_cast<unsigned long long>(order[k].second));
        std::string file = "none";
        if (dir.empty()) {
            file = "no_dir";
        } else if (e.source.empty()) {
            file = "no_source";
        } else if (fileBytes_ + e.source.size() > limits.maxFileBytes) {
            file = "skipped_size";
        } else {
            const std::string name = std::string("shader-") + id + "-" + e.stage + ".metal";
            if (FILE* fp = std::fopen((dir + "/" + name).c_str(), "wb")) {
                const bool ok = std::fwrite(e.source.data(), 1, e.source.size(), fp) == e.source.size();
                std::fclose(fp);
                if (ok) fileBytes_ += e.source.size(), file = name;
                else file = "write_failed";
            } else {
                file = "open_failed";
            }
        }
        const ShaderStat& s = e.stat;
        lines.push_back(std::string("SHADERSTAT id=") + id + " stage=" + e.stage + " ifs=" + std::to_string(s.ifs) +
                        " loads=" + std::to_string(s.loads) + " samples=" + std::to_string(s.samples) + " loops=" + std::to_string(s.loops) +
                        " locals=" + std::to_string(s.locals) + " privarrays=" + std::to_string(s.privArrays) + " lds=" + std::to_string(s.lds) +
                        " mbcnt=" + std::to_string(s.mbcnt) + " lines=" + std::to_string(s.lines) + " draws=" + std::to_string(e.draws) +
                        " file=" + file);
        e.reported = true;
        ++reported_;
        retained_ -= e.source.size();
        std::string().swap(e.source);
    }
    return lines;
}

}  // namespace vpps4
