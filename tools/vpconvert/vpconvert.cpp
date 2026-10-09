// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// vpconvert (see vpconvert.h): translate with vpaot, compile with clang and link with lld, all in
// this process. On visionOS a process may not start another one, so the compiler and the linker
// are libraries here, called the way their own command-line tools call themselves.

#include "vpconvert.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "lld/Common/Driver.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

LLD_HAS_DRIVER(macho)

// vpaot's command line (tools/vpaot/main.cpp, built with -Dmain=vpaot_main).
int vpaot_main(int argc, char** argv);

namespace fs = std::filesystem;

namespace {

// vpaot keeps no state between runs, but it is not made for two at once.
std::mutex g_vpaot_mutex;
// lld is not reentrant.
std::mutex g_lld_mutex;

struct Logger {
    const VpConvertCallbacks* cb;
    std::mutex mutex;
    void line(const char* f, ...) {
        char buf[2048];
        va_list ap;
        va_start(ap, f);
        vsnprintf(buf, sizeof buf, f, ap);
        va_end(ap);
        std::scoped_lock lock{mutex};
        if (cb && cb->log) cb->log(cb->user, buf); else fprintf(stderr, "vpconvert: %s\n", buf);
    }
    void progress(int phase, long done, long total) {
        if (cb && cb->progress) cb->progress(cb->user, phase, done, total);
    }
    bool stop() { return cb && cb->should_stop && cb->should_stop(cb->user); }
};

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

bool write_file(const fs::path& p, const std::string& text) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << text;
    return bool(f);
}

// The module name the scripts give a file: eboot, libc_prx...
std::string module_name(const fs::path& file) {
    std::string stem = file.stem().string();
    std::string name;
    for (char c : stem) {
        c = (char)std::tolower((unsigned char)c);
        name += (std::isalnum((unsigned char)c) || c == '_') ? c : '_';
    }
    if (!name.empty() && std::isdigit((unsigned char)name[0])) name = "m_" + name;
    std::string ext = file.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (ext != ".bin") name += "_prx";
    return name;
}

// What a module's translation depends on: if this is the same, its C (and objects) are current.
std::string module_stamp(const fs::path& file, const VpConvertConfig& c, int split) {
    struct stat st {};
    stat(file.string().c_str(), &st);
    std::string roots = c.missing_log && fs::exists(c.missing_log) ? read_file(c.missing_log) : "";
    char buf[512];
    snprintf(buf, sizeof buf, "vpconvert 1\nsize %lld\nmtime %lld\nsplit %d\nroots %zu %zx\nopt %s\ntriple %s\n",
             (long long)st.st_size, (long long)st.st_mtime, split, roots.size(), std::hash<std::string>{}(roots),
             c.opt_level ? c.opt_level : "-O2", c.triple ? c.triple : "");
    return buf;
}

// ---- the compiler -------------------------------------------------------------------------

void init_llvm_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmPrinter();
        LLVMInitializeAArch64AsmParser();
    });
}

// clang -c `in` -o `out` with `args`, in this process: the driver turns the command line into the
// compiler's own (-cc1) arguments, which run as a CompilerInstance here instead of a new process.
bool compile(const std::vector<std::string>& args, std::string& diagnostics) {
    init_llvm_once();
    std::vector<const char*> argv;
    argv.push_back("clang");
    for (const auto& a : args) argv.push_back(a.c_str());

    llvm::raw_string_ostream diag_stream(diagnostics);
    clang::DiagnosticOptions diag_options;
    auto* printer = new clang::TextDiagnosticPrinter(diag_stream, diag_options);
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> ids(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine diags(ids, diag_options, printer);

    clang::driver::Driver driver("/vpconvert/clang", llvm::sys::getDefaultTargetTriple(), diags);
    driver.setCheckInputsExist(false);
    std::unique_ptr<clang::driver::Compilation> compilation(driver.BuildCompilation(argv));
    if (!compilation || diags.hasErrorOccurred()) return false;
    const clang::driver::JobList& jobs = compilation->getJobs();
    if (jobs.size() != 1 || !llvm::isa<clang::driver::Command>(*jobs.begin())) {
        diag_stream << "vpconvert: the driver made " << jobs.size() << " jobs, not one compile\n";
        return false;
    }
    const auto& command = llvm::cast<clang::driver::Command>(*jobs.begin());
    const llvm::opt::ArgStringList& cc1 = command.getArguments();

    auto instance = std::make_unique<clang::CompilerInstance>();
    if (!clang::CompilerInvocation::CreateFromArgs(instance->getInvocation(), cc1, diags)) return false;
    instance->createDiagnostics(*llvm::vfs::getRealFileSystem(), printer, /*ShouldOwnClient=*/false);
    clang::EmitObjAction action;
    const bool ok = instance->ExecuteAction(action);
    diag_stream.flush();
    return ok && !instance->getDiagnostics().hasErrorOccurred();
}

// ---- the linker ---------------------------------------------------------------------------

bool link(const std::vector<std::string>& args, std::string& output) {
    std::scoped_lock lock{g_lld_mutex};
    std::vector<const char*> argv;
    argv.push_back("ld64.lld");
    for (const auto& a : args) argv.push_back(a.c_str());
    llvm::raw_string_ostream out(output);
    lld::Result r = lld::lldMain(argv, out, out, {{lld::Darwin, &lld::macho::link}});
    out.flush();
    return r.retCode == 0;
}

struct Piece {
    fs::path c, o;
    uintmax_t bytes = 0;
};

} // namespace

extern "C" int vp_convert(const VpConvertConfig* config, const VpConvertCallbacks* callbacks) {
    Logger log{callbacks, {}};
    const VpConvertConfig& c = *config;
    const auto started = std::chrono::steady_clock::now();
    const int split = c.split > 0 ? c.split : 300;
    const int jobs = c.jobs > 0 ? c.jobs : 4;
    const std::string triple = c.triple ? c.triple : "arm64-apple-xros2.0";
    const std::string opt = c.opt_level ? c.opt_level : "-O2";
    const fs::path game(c.game_dir), work(c.work_dir), sdk(c.sdk_dir);
    std::error_code ec;
    fs::create_directories(work, ec);
    if (ec) { log.line("ERROR: cannot create %s: %s", work.string().c_str(), ec.message().c_str()); return -1; }
    std::string title = c.title ? c.title : game.filename().string();

    // ---- 1. translation ------------------------------------------------------------------
    std::vector<fs::path> files;
    if (fs::exists(game / "eboot.bin")) files.push_back(game / "eboot.bin");
    if (fs::is_directory(game / "sce_module")) {
        std::vector<fs::path> mods;
        for (const auto& e : fs::directory_iterator(game / "sce_module")) {
            std::string ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char x) { return (char)std::tolower(x); });
            if (e.is_regular_file() && (ext == ".prx" || ext == ".sprx")) mods.push_back(e.path());
        }
        std::sort(mods.begin(), mods.end());
        files.insert(files.end(), mods.begin(), mods.end());
    }
    if (files.empty()) { log.line("ERROR: no eboot.bin or sce_module/*.prx in %s", game.string().c_str()); return -1; }
    log.line("game %s: %zu modules, %d pieces of %d functions at once, %s %s", title.c_str(), files.size(), jobs, split,
             triple.c_str(), opt.c_str());

    std::vector<std::string> names;
    std::vector<Piece> pieces;
    long translated = 0;
    for (const auto& file : files) {
        const std::string name = module_name(file);
        if (std::find(names.begin(), names.end(), name) != names.end()) {
            log.line("ERROR: two modules map to the name %s", name.c_str());
            return -1;
        }
        names.push_back(name);
        const fs::path stamp = work / (name + ".stamp");
        const std::string want = module_stamp(file, c, split);
        const fs::path list = work / (name + "_files.txt");
        std::vector<fs::path> parts;
        auto read_parts = [&] {
            parts.clear();
            if (fs::exists(list)) {
                std::istringstream in(read_file(list));
                for (std::string l; std::getline(in, l);) {
                    while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();
                    if (!l.empty()) parts.push_back(work / fs::path(l).filename());
                }
            } else {
                parts.push_back(work / (name + ".c"));
            }
        };
        bool current = fs::exists(stamp) && read_file(stamp) == want;
        if (current) {
            read_parts();
            for (const auto& p : parts) {
                fs::path o = p; o.replace_extension(".o");
                if (!fs::exists(p) && !(fs::exists(o) && fs::exists(o.string() + ".ok"))) { current = false; break; }
            }
        }
        if (!current) {
            // Everything this module had before goes: a new translation may split differently.
            const std::regex mine("^" + name + "(\\.(c|h|json|stamp|o|o\\.ok|o\\.tmp)|_decl\\.h|_files\\.txt|_[0-9]+\\.(c|o|o\\.ok|o\\.tmp))$");
            for (const auto& e : fs::directory_iterator(work)) {
                if (std::regex_match(e.path().filename().string(), mine)) fs::remove(e.path(), ec);
            }
            const auto t = std::chrono::steady_clock::now();
            std::vector<std::string> a = {"vpaot", "--elf", file.string(), "--pic", "--module", name, "--split", std::to_string(split),
                                          "--out", (work / (name + ".c")).string(), "--stats", (work / (name + ".json")).string()};
            if (c.missing_log && fs::exists(c.missing_log)) { a.push_back("--roots"); a.push_back(c.missing_log); }
            std::vector<char*> argv;
            for (auto& s : a) argv.push_back(s.data());
            int r;
            {
                std::scoped_lock lock{g_vpaot_mutex};
                r = vpaot_main((int)argv.size(), argv.data());
            }
            if (r != 0) { log.line("ERROR: translating %s failed (%d)", file.filename().string().c_str(), r); return -1; }
            const std::string stats = read_file(work / (name + ".json"));
            auto field = [&](const char* key) -> std::string {
                const auto at = stats.find(std::string("\"") + key + "\"");
                if (at == std::string::npos) return "?";
                auto v = stats.find(':', at) + 1;
                while (v < stats.size() && stats[v] == ' ') ++v;
                auto end = stats.find_first_of(",\n}", v);
                return stats.substr(v, end - v);
            };
            read_parts();
            uintmax_t bytes = 0;
            for (const auto& p : parts) bytes += fs::file_size(p, ec);
            write_file(stamp, want);
            log.line("translated %s -> %s in %.1f s: %s functions, %s instructions, supported %s, %zu pieces, %.1f MB of C",
                     file.filename().string().c_str(), name.c_str(), seconds_since(t), field("functions").c_str(),
                     field("instructions").c_str(), field("supported_fraction").c_str(), parts.size(), bytes / 1048576.0);
        } else {
            log.line("%s: translation kept from before", name.c_str());
        }
        for (const auto& p : parts) {
            Piece piece;
            piece.c = p;
            piece.o = p; piece.o.replace_extension(".o");
            piece.bytes = fs::exists(p) ? fs::file_size(p, ec) : 0;
            pieces.push_back(piece);
        }
        log.progress(VP_CONVERT_TRANSLATE, ++translated, (long)files.size());
        if (log.stop()) { log.line("stopped after translating %s", name.c_str()); return 1; }
    }
    {
        // The registry with the pack's vp_pack_info (small: compiled every time).
        std::vector<std::string> a = {"vpaot", "--registry", (work / "vpengine_registry.c").string(), "--pack", title};
        for (const auto& n : names) a.push_back(n);
        std::vector<char*> argv;
        for (auto& s : a) argv.push_back(s.data());
        std::scoped_lock lock{g_vpaot_mutex};
        if (vpaot_main((int)argv.size(), argv.data()) != 0) { log.line("ERROR: vpaot --registry failed"); return -1; }
    }
    Piece reg;
    reg.c = work / "vpengine_registry.c";
    reg.o = work / "vpengine_registry.o";
    fs::remove(reg.o.string() + ".ok", ec);
    pieces.push_back(reg);

    // ---- 2. compilation ------------------------------------------------------------------
    std::vector<Piece*> todo;
    long already = 0;
    for (auto& p : pieces) {
        if (fs::exists(p.o) && fs::exists(p.o.string() + ".ok")) { ++already; continue; }
        todo.push_back(&p);
    }
    // Largest first: the long ones start early instead of ending the run alone.
    std::stable_sort(todo.begin(), todo.end(), [](const Piece* a, const Piece* b) { return a->bytes > b->bytes; });
    uintmax_t todo_bytes = 0;
    for (auto* p : todo) todo_bytes += p->bytes;
    log.line("compiling %zu pieces (%.1f MB of C), %ld compiled before", todo.size(), todo_bytes / 1048576.0, already);
    const auto compile_started = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    std::atomic<long> done{already};
    std::atomic<bool> failed{false}, stopped{false};
    std::atomic<uintmax_t> done_bytes{0};
    const long total = (long)pieces.size();
    log.progress(VP_CONVERT_COMPILE, already, total);
    const std::vector<std::string> base_args = {
        "--target=" + triple, opt, "-ffreestanding", "-fno-stack-protector", "-fno-math-errno", "-frounding-math", "-w",
        "-nostdinc", "-resource-dir", (sdk / "clang").string(),
        "-isystem", (sdk / "clang" / "include").string(),
        "-isystem", (sdk / "runtime" / "freestanding").string(), "-I", (sdk / "runtime").string(), "-c"};
    auto worker = [&] {
        for (;;) {
            if (failed || log.stop()) { if (!failed) stopped = true; return; }
            const size_t i = next++;
            if (i >= todo.size()) return;
            Piece& p = *todo[i];
            const auto t = std::chrono::steady_clock::now();
            std::vector<std::string> args = base_args;
            const std::string tmp = p.o.string() + ".tmp";
            args.push_back(p.c.string());
            args.push_back("-o");
            args.push_back(tmp);
            std::string diagnostics;
            if (!compile(args, diagnostics)) {
                failed = true;
                log.line("ERROR: compiling %s failed:\n%s", p.c.filename().string().c_str(), diagnostics.substr(0, 4000).c_str());
                return;
            }
            std::error_code e;
            fs::rename(tmp, p.o, e);
            write_file(p.o.string() + ".ok", "ok\n");
            if (p.c.filename() != "vpengine_registry.c") fs::remove(p.c, e); // compiled: its C goes
            const long d = ++done;
            done_bytes += p.bytes;
            log.line("compiled %s (%.2f MB) in %.1f s [%ld/%ld]", p.c.filename().string().c_str(), p.bytes / 1048576.0,
                     seconds_since(t), d, total);
            log.progress(VP_CONVERT_COMPILE, d, total);
        }
    };
    std::vector<std::thread> threads;
    for (int k = 0; k < jobs; ++k) threads.emplace_back(worker);
    for (auto& th : threads) th.join();
    const double compile_s = seconds_since(compile_started);
    if (!todo.empty()) {
        log.line("compiled %.1f MB of C in %.1f s (%.2f MB/s) with %d jobs", done_bytes.load() / 1048576.0, compile_s,
                 compile_s > 0 ? done_bytes.load() / 1048576.0 / compile_s : 0.0, jobs);
    }
    if (failed) return -1;
    if (stopped) { log.line("stopped: %ld of %ld pieces compiled, the rest next time", done.load(), total); return 1; }

    // ---- 3. link -------------------------------------------------------------------------
    const auto link_started = std::chrono::steady_clock::now();
    std::string platform = c.platform ? c.platform : "xros 2.0 26.0";
    std::vector<std::string> args = {"-arch", "arm64", "-platform_version"};
    {
        std::istringstream in(platform);
        for (std::string w; in >> w;) args.push_back(w);
    }
    const std::string out_tmp = std::string(c.output) + ".tmp";
    for (const std::string& a : {std::string("-dylib"), std::string("-adhoc_codesign"), std::string("-install_name"),
                                 std::string("@rpath/vpengine.vpgame"), std::string("-rpath"), std::string("@executable_path/Frameworks"),
                                 std::string("-o"), out_tmp}) {
        args.push_back(a);
    }
    for (const auto& p : pieces) args.push_back(p.o.string());
    args.push_back((sdk / "tbd" / "libVPRuntime.tbd").string());
    args.push_back((sdk / "tbd" / "libSystem.tbd").string());
    std::string link_output;
    log.progress(VP_CONVERT_LINK, 0, 1);
    if (!link(args, link_output)) {
        log.line("ERROR: linking failed:\n%s", link_output.substr(0, 4000).c_str());
        return -1;
    }
    fs::rename(out_tmp, c.output, ec);
    if (ec) { log.line("ERROR: cannot move the pack to %s: %s", c.output, ec.message().c_str()); return -1; }
    log.progress(VP_CONVERT_LINK, 1, 1);
    log.line("linked %zu objects into %s (%.1f MB) in %.1f s", pieces.size(), c.output, fs::file_size(c.output, ec) / 1048576.0,
             seconds_since(link_started));
    log.line("conversion finished in %.1f s of this run", seconds_since(started));
    return 0;
}
