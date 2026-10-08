// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// vpaot: ahead-of-time translation of an x86-64 program image into C.
//
//   vpaot --elf eboot.bin --out game.c [--stats report.json]
//   vpaot --raw code.bin --base 0x1000 --entry 0x1000 [--entry ...] --out test.c
//
// `--report` alone prints the coverage statistics (what the image needs that the translator
// does not do yet) without writing C.
#include "image.h"
#include "translate.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace vpaot;

static void usage() {
    fprintf(stderr,
            "usage: vpaot (--elf FILE | --raw FILE --base ADDR) [--entry ADDR]... [--out FILE.c]\n"
            "             [--stats FILE.json] [--rip] [--trace] [--max-functions N] [--split FUNCTIONS_PER_FILE] [--native ADDR]...\n");
}

static void write_stats(const Stats& s, const std::string& path) {
    FILE* f = path.empty() ? stdout : fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + path);
    fprintf(f, "{\n  \"functions\": %" PRIu64 ",\n  \"instructions\": %" PRIu64 ",\n  \"unsupported\": %" PRIu64 ",\n",
            s.functions, s.instructions, s.unsupported);
    fprintf(f, "  \"indirect_calls\": %" PRIu64 ",\n  \"indirect_jumps\": %" PRIu64 ",\n  \"jump_tables\": %" PRIu64 ",\n", s.indirect_calls, s.indirect_jumps, s.jump_tables);
    fprintf(f, "  \"supported_fraction\": %.4f,\n",
            s.instructions ? 1.0 - (double)s.unsupported / (double)s.instructions : 1.0);
    fprintf(f, "  \"unsupported_by_mnemonic\": {\n");
    bool first = true;
    for (auto& [k, v] : s.unsupported_by_mnemonic) {
        fprintf(f, "%s    \"%s\": %" PRIu64, first ? "" : ",\n", k.c_str(), v);
        first = false;
    }
    fprintf(f, "\n  },\n  \"by_mnemonic\": {\n");
    first = true;
    for (auto& [k, v] : s.by_mnemonic) {
        fprintf(f, "%s    \"%s\": %" PRIu64, first ? "" : ",\n", k.c_str(), v);
        first = false;
    }
    fprintf(f, "\n  }\n}\n");
    if (f != stdout) fclose(f);
}

int main(int argc, char** argv) {
    std::string elf, raw, out, stats_path;
    uint64_t base = 0;
    std::vector<uint64_t> entries;
    Options opt;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(); exit(2); }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--elf")) elf = next();
        else if (!strcmp(argv[i], "--raw")) raw = next();
        else if (!strcmp(argv[i], "--base")) base = strtoull(next(), nullptr, 0);
        else if (!strcmp(argv[i], "--entry")) entries.push_back(strtoull(next(), nullptr, 0));
        else if (!strcmp(argv[i], "--out")) out = next();
        else if (!strcmp(argv[i], "--stats")) stats_path = next();
        else if (!strcmp(argv[i], "--rip")) opt.emit_rip_updates = true;
        else if (!strcmp(argv[i], "--trace")) opt.trace = true;
        else if (!strcmp(argv[i], "--max-functions")) opt.max_functions = strtoull(next(), nullptr, 0);
        else if (!strcmp(argv[i], "--regcache")) opt.regcache = true;
        else if (!strcmp(argv[i], "--scan-data")) opt.scan_data = true;
        else if (!strcmp(argv[i], "--no-regcache")) opt.regcache = false;
        else if (!strcmp(argv[i], "--no-lazy-flags")) opt.lazy_flags = false;
        else if (!strcmp(argv[i], "--split")) opt.split = strtoull(next(), nullptr, 0);
        else if (!strcmp(argv[i], "--native")) opt.natives.insert(strtoull(next(), nullptr, 0));
        else { usage(); return 2; }
    }
    if (elf.empty() == raw.empty()) { usage(); return 2; }
    try {
        Image img = raw.empty() ? load_elf_or_self(elf) : load_raw(raw, base);
        if (entries.empty()) entries.push_back(img.entry);
        Stats stats;
        auto functions = discover(img, entries, stats, opt);
        fprintf(stderr, "vpaot: image %s..%s, %zu executable ranges, %zu code pointers, %zu .eh_frame starts, %zu functions\n",
                std::to_string(img.base).c_str(), std::to_string(img.end()).c_str(), img.executable.size(),
                img.code_pointers.size(), img.eh_frame_starts.size(), functions.size());
        const std::string c_path = out.empty() ? "/dev/null" : out;
        emit_c(img, functions, opt, c_path, stats);
        fprintf(stderr, "vpaot: %" PRIu64 " instructions, %" PRIu64 " unsupported (%.2f%% supported), %" PRIu64
                " indirect calls, %" PRIu64 " indirect jumps, %" PRIu64 " jump tables resolved, %" PRIu64 " landing pads\n",
                stats.instructions, stats.unsupported,
                stats.instructions ? 100.0 * (1.0 - (double)stats.unsupported / (double)stats.instructions) : 100.0,
                stats.indirect_calls, stats.indirect_jumps, stats.jump_tables, stats.landing_pads);
        if (!stats_path.empty() || out.empty()) write_stats(stats, stats_path);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "vpaot: %s\n", e.what());
        return 1;
    }
}
