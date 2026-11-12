// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// tools/vpconvert/vpconvert_cache.h: a value kept for a file is worked out once while the file is
// the same, again when it changes, and a patch is worked out again when asked to.
//   vpconvert_cache DIR
#include "../../tools/vpconvert/vpconvert_cache.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sys/time.h>
#include <thread>
#include <vector>

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL "); printf(__VA_ARGS__); printf("\n"); ++failures; } } while (0)

static void write(const std::string& path, const char* text) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("cannot write %s\n", path.c_str()); exit(2); }
    fputs(text, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const std::string dir = argv[1], file = dir + "/cache_test.bin";
    write(file, "one");

    // ---- values of a file
    vpcache::FileValues values;
    int computed = 0;
    auto compute = [&] { ++computed; return std::string("v") + std::to_string(computed); };
    CHECK(values.get(file, compute) == "v1", "first value");
    CHECK(values.get(file, compute) == "v1" && computed == 1, "the value of an unchanged file was worked out again (%d)", computed);
    write(file, "three");
    CHECK(values.get(file, compute) == "v2", "a file of another size kept its old value");
    write(file, "THREE"); // same size
    struct timeval later[2] = {{2000000000, 0}, {2000000000, 0}};
    utimes(file.c_str(), later);
    CHECK(values.get(file, compute) == "v3", "a file with another time kept its old value");
    CHECK(values.get(file, compute) == "v3" && computed == 3, "worked out again without a change (%d)", computed);
    int missing = 0;
    values.get(dir + "/no_such_file", [&] { ++missing; return std::string("x"); });
    values.get(dir + "/no_such_file", [&] { ++missing; return std::string("x"); });
    CHECK(missing == 2, "a file that does not exist was kept (%d)", missing);
    { // several threads on one file: every answer is the value
        std::vector<std::thread> threads;
        std::vector<std::string> answers(8);
        for (size_t i = 0; i < answers.size(); ++i) threads.emplace_back([&, i] { answers[i] = values.get(file, [] { return std::string("t"); }); });
        for (auto& t : threads) t.join();
        for (auto& a : answers) CHECK(a == "v3" || a == "t", "a thread got '%s'", a.c_str());
    }

    // ---- patches: by module file and name; the callback and its user data are not part of the key
    // (convert, is_current and the translation on demand each pass their own)
    vpcache::Patches patches;
    int patched = 0;
    auto patch = [&] { ++patched; return std::string("patch") + std::to_string(patched); };
    CHECK(patches.get(file, "eboot", true, patch) == "patch1", "first patch");
    CHECK(patches.get(file, "eboot", false, patch) == "patch1" && patched == 1, "a patch kept was worked out again (%d)", patched);
    CHECK(patches.get(file, "eboot", true, patch) == "patch2", "a patch asked again kept the old one");
    CHECK(patches.get(file, "eboot", false, patch) == "patch2" && patched == 2, "the patch asked again was not kept (%d)", patched);
    CHECK(patches.get(file, "libc", false, patch) == "patch3", "another module shares a patch");
    // a conversion with settings that changed, then the translation on demand: it sees the new patch
    std::string settings = "old";
    auto by_settings = [&] { ++patched; return "patched with " + settings; };
    CHECK(patches.get(file, "settings", true, by_settings) == "patched with old", "patch by settings");
    settings = "new";
    CHECK(patches.get(file, "settings", false, by_settings) == "patched with old", "a patch kept changed without being asked");
    CHECK(patches.get(file, "settings", true, by_settings) == "patched with new", "changed settings were not seen by a conversion or the check whether the pack is current");
    patched = 0;
    CHECK(patches.get(file, "settings", false, by_settings) == "patched with new" && patched == 0, "the translation on demand did not use the patch of the last conversion (%d)", patched);
    // many conversions: one entry per module, not one per conversion
    const size_t before = patches.size();
    for (int i = 0; i < 50; ++i) patches.get(file, "eboot", true, patch);
    CHECK(patches.size() == before, "the patches grew from %zu to %zu entries over 50 conversions", before, patches.size());
    write(file, "a longer file now");
    patched = 0;
    CHECK(patches.get(file, "eboot", false, patch) == "patch1" && patched == 1, "a changed module file kept its patch (%d)", patched);
    try {
        patches.get(file, "boom", true, []() -> std::string { throw std::runtime_error("unreadable"); });
        CHECK(false, "an error was swallowed");
    } catch (const std::runtime_error&) {}
    CHECK(patches.get(file, "boom", false, [] { return std::string("fine"); }) == "fine", "an error left a value");

    remove(file.c_str());
    printf(failures ? "vpconvert_cache: %d FAILED\n" : "vpconvert_cache: ok\n", failures);
    return failures != 0;
}
