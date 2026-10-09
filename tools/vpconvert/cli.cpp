// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// vpconvert on a computer (the same code the app runs): converts a game folder into a pack.
//   vpconvert-cli GAME_DIR WORK_DIR OUT SDK_DIR [jobs] [triple] [platform...]
#include "vpconvert.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

static void log_line(void*, const char* line) {
    static const auto start = std::chrono::steady_clock::now();
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("[%8.1f] %s\n", t, line);
    std::fflush(stdout);
}

static int stop_requested(void*) { return std::getenv("VPCONVERT_STOP_NOW") != nullptr; }

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s GAME_DIR WORK_DIR OUT SDK_DIR [jobs] [triple] [platform]\n", argv[0]);
        return 2;
    }
    VpConvertConfig c{};
    c.game_dir = argv[1];
    c.work_dir = argv[2];
    c.output = argv[3];
    c.sdk_dir = argv[4];
    c.jobs = argc > 5 ? std::atoi(argv[5]) : 0;
    c.triple = argc > 6 ? argv[6] : nullptr;
    c.platform = argc > 7 ? argv[7] : nullptr;
    c.split = std::getenv("VPCONVERT_SPLIT") ? std::atoi(std::getenv("VPCONVERT_SPLIT")) : 0;
    VpConvertCallbacks cb{};
    cb.log = log_line;
    cb.should_stop = stop_requested;
    const int r = vp_convert(&c, &cb);
    std::printf("vp_convert: %d\n", r);
    return r < 0 ? 1 : r;
}
