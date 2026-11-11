// The call capture (vpps4/callcap.h) and its replay (tools/vp4replay.cpp), over the scripts of the per-draw tests.
// Usage: vpps4_replay_tests TEST
//   callcap_roundtrip   every script run as it is, run again while a call capture writes its calls down, and the capture
//                       replayed: the three leave the same bytes on the backend
//   callcap_limits      the capture closes after its pictures and at its byte limit, cut back to a whole picture; the run
//                       goes on as if nothing had been captured
//   replay_equal        every script's capture replayed with every shortcut on and with each switch off: the same bytes;
//                       and the us/draw of each path over six pictures of the measured load (the least of 5 passes).
//                       VP4_CAPTURE=FILE: a capture of a host too (they are not kept in the repository)
//   write FILE          leaves the capture of those six pictures in FILE (something to try vp4replay on)
#include <filesystem>

// (the scripts, their scene and their runner: the whole of the per-draw tests but their main)
#define main perdraw_main
#include "perdraw_tests.cpp"
#undef main
#define VP4REPLAY_NO_MAIN
#include "../tools/vp4replay.cpp"

namespace {

std::string capturePath(const char* what) {
    std::string name = what;
    for (char& c : name)
        if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
    return (std::filesystem::temp_directory_path() / ("vpps4_callcap_" + std::to_string(getpid()) + "_" + name + ".vp4")).string();
}
// The replay's backend names the guest buffers as the scripts' own does: by their distance from the first guest address.
// `sameGpu`: it also finishes its frames as the scripts' own does, two submits late (inside the submit, where the renderer
// sees it at once), instead of when the capture says a frame had finished (between two calls): what a replay needs to
// leave the very bytes of the run it was captured from.
vp4replay::Options replayOptions(bool sameGpu = false) {
    vp4replay::Options o;
    o.guestHost = reinterpret_cast<const uint8_t*>(kGuest);
    if (sameGpu) o.inFlight = Capture().inFlight;
    return o;
}
// (a host that does not watch memory has the renderer sample its texture sources: a capture does not hold those)
bool replayable(const RunOptions& options) { return options.watches; }

// The measured load as a host presents it: the warm-up, then six pictures, each marked where it ends.
Script sixPictures(Scene& sc) {
    const auto presented = [] { return call([](vpps4::Renderer& r) { r.callCapturePicture(); }); };
    Script s = sc.primeArenas();
    s.push_back(presented());
    for (unsigned n = 0; n < 6; ++n) append(s, sc.picture(n, 2000)), s.push_back(presented());
    return s;
}

void roundtrip(const char* what, const std::function<Script(Scene&)>& make, RunOptions options) {
    if (!replayable(options)) return;
    const Outcome direct = run(make, false, nullptr, options);
    vpps4::CallCapOptions capture;
    capture.path = capturePath(what), capture.pictures = ~0u;
    options.callcap = &capture;
    const Outcome recorded = run(make, false, nullptr, options);
    CHECK(same(recorded, direct, (std::string(what) + ", while captured").c_str()));
    const vp4replay::Run replayed = vp4replay::replay(capture.path, replayOptions(true));
    CHECK(replayed.error.empty());
    CHECK(replayed.end == 3);  // (ended by the host, whole)
    CHECK(replayed.draws > 0 && replayed.drawn == direct.drawn);
    vp4replay::Run asDirect;
    asDirect.bytes = direct.bytes, asDirect.marks = direct.marks;
    const std::string diff = vp4replay::difference(asDirect, replayed);
    CHECK(diff.empty());
    if (!replayed.error.empty() || !diff.empty())
        std::fprintf(stderr, "%s: the replay of its capture differs: %s%s (drawn %llu / %llu, regions %llu +%llu made up)\n", what,
                     replayed.error.c_str(), diff.c_str(), (unsigned long long)replayed.drawn, (unsigned long long)direct.drawn,
                     (unsigned long long)replayed.regions, (unsigned long long)replayed.madeUp);
    std::remove(capture.path.c_str());
}

void replayEqual(const char* what, const std::function<Script(Scene&)>& make, RunOptions options) {
    if (!replayable(options)) return;
    vpps4::CallCapOptions capture;
    capture.path = capturePath(what), capture.pictures = ~0u;
    options.callcap = &capture;
    run(make, false, nullptr, options);
    std::vector<vp4replay::OptOut> offs;
    for (const OptOut& o : kOptOuts) offs.push_back({o.name, o.value});
    CHECK(vp4replay::compare(capture.path, offs, 1, replayOptions(), stdout) == 0);
    std::remove(capture.path.c_str());
}

struct Ending {
    uint64_t fileBytes = 0, pictures = 0, calls = 0, draws = 0, cut = ~uint64_t(0);
    bool endsWithEnd = false;
};
Ending endingOf(const std::string& path) {
    Ending e;
    std::error_code ec;
    e.fileBytes = std::filesystem::file_size(path, ec);
    vpps4::callcap::Reader reader(path);
    vpps4::callcap::Reader::Record rec;
    while (reader.next(rec)) {
        e.endsWithEnd = rec.type == vpps4::callcap::Type::End && rec.payload.size() == sizeof(vpps4::callcap::EndRecord);
        if (!e.endsWithEnd) continue;
        vpps4::callcap::EndRecord end;
        std::memcpy(&end, rec.payload.data(), sizeof end);
        e.pictures = end.pictures, e.calls = end.calls, e.draws = end.draws, e.cut = end.cut;
    }
    return e;
}

void limits() {
    const Outcome direct = run(sixPictures, false);
    vpps4::CallCapOptions capture;
    capture.path = capturePath("limits");
    RunOptions options;
    options.callcap = &capture;
    // Its pictures done: closed there, and the rest of the run is not in the file.
    capture.pictures = 3;
    CHECK(same(run(sixPictures, false, nullptr, options), direct, "limits, three pictures"));
    const Ending three = endingOf(capture.path);
    CHECK(three.endsWithEnd && three.pictures == 3 && three.cut == 0 && three.draws > 2000 && three.draws < 5000);
    const vp4replay::Run replayed = vp4replay::replay(capture.path, replayOptions());
    CHECK(replayed.error.empty() && replayed.whole == 3 && replayed.pictures.size() == 3 && replayed.draws == three.draws && replayed.end == 0);
    CHECK(replayed.calls == three.calls);
    // The byte limit inside the third picture: cut back to the second, never past the limit.
    capture.pictures = 6;
    capture.maxBytes = three.fileBytes - 4096;
    CHECK(same(run(sixPictures, false, nullptr, options), direct, "limits, bytes"));
    const Ending cut = endingOf(capture.path);
    CHECK(cut.endsWithEnd && cut.pictures == 2 && cut.cut == 1 && cut.fileBytes <= capture.maxBytes && cut.draws < three.draws);
    const vp4replay::Run shorter = vp4replay::replay(capture.path, replayOptions());
    CHECK(shorter.error.empty() && shorter.whole == 2 && shorter.pictures.size() == 2 && shorter.draws == cut.draws && shorter.end == 1);
    // ... and inside the first one: what was written of it stays (whole records), under the limit.
    capture.maxBytes = 1 << 20;
    CHECK(same(run(sixPictures, false, nullptr, options), direct, "limits, few bytes"));
    const Ending few = endingOf(capture.path);
    CHECK(few.endsWithEnd && few.pictures == 0 && few.cut == 1 && few.calls > 0 && few.fileBytes <= capture.maxBytes &&
          few.fileBytes > capture.maxBytes - (128 << 10));
    CHECK(vp4replay::replay(capture.path, replayOptions()).error.empty());
    // No room at all, a path that cannot be made: no capture, and the run is the same.
    capture.maxBytes = 8;
    CHECK(same(run(sixPictures, false, nullptr, options), direct, "limits, no room"));
    capture.maxBytes = 120ull << 20;
    const std::string file = capture.path;
    capture.path = "/nonexistent-directory/callcap.vp4";
    CHECK(same(run(sixPictures, false, nullptr, options), direct, "limits, no file"));
    std::remove(file.c_str());
}

void write(const std::string& path) {
    vpps4::CallCapOptions capture;
    capture.path = path, capture.pictures = 7;  // (the warm-up counts as one)
    RunOptions options;
    options.callcap = &capture;
    run(sixPictures, false, nullptr, options);
}

// The figures the plan of the draw cache asks of a capture, every path, 5 passes: over six pictures of the measured load,
// and over a host's capture if one is given.
void timed() {
    const std::string path = capturePath("six_pictures");
    write(path);
    for (const char* file : {path.c_str(), static_cast<const char*>(std::getenv("VP4_CAPTURE"))}) {
        if (!file) continue;
        const vp4replay::Options options = file == path.c_str() ? replayOptions() : vp4replay::Options();
        CHECK(vp4replay::compare(file, vp4replay::defaultOptOuts(), 5, options, stdout) == 0);
        vp4replay::Options rates = options;
        rates.rates = true;
        for (const std::string& line : vp4replay::replay(file, rates).rates) std::printf("  %s\n", line.c_str());
    }
    std::remove(path.c_str());
}

void everyScript() { equalPicture(), equalBuffers(), equalTextures(), equalReuse(), equalCaches(), equalPrograms(), equalTargets(); }

}  // namespace

int main(int argc, char** argv) {
    const std::string t = argc > 1 ? argv[1] : "";
    if (t == "callcap_roundtrip") scriptHook = roundtrip, everyScript();
    else if (t == "callcap_limits") limits();
    else if (t == "replay_equal") scriptHook = replayEqual, everyScript(), timed();
    else if (t == "write" && argc > 2) write(argv[2]);
    else return std::fprintf(stderr, "unknown test %s\n", t.c_str()), 2;
    return failures ? 1 : 0;
}
