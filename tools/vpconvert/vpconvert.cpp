// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// vpconvert (see vpconvert.h): translate with vpaot, compile with clang and link with lld, all in
// this process. On visionOS a process may not start another one, so the compiler and the linker
// are libraries here, called the way their own command-line tools call themselves.

#include "vpconvert.h"
#include "../vpaot/image.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
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
#include "clang/Basic/Stack.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "lld/Common/Driver.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/thread.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "llvm/Config/llvm-config.h"
#include <Zydis/Zydis.h>

#include "vp_pack.h"

// What vpaot's output depends on besides its input (set by CMake: a hash of vpaot's sources).
#ifndef VPAOT_SOURCE_ID
#define VPAOT_SOURCE_ID "unknown"
#endif

LLD_HAS_DRIVER(macho)

// LLVM's buffer allocator (Support/MemAlloc.cpp), defined here instead so that its archive member
// is not linked: the original asks the C++ aligned nothrow `operator new`, which inside the app
// returned NULL at once in the clang driver ("LLVM ERROR: out of memory / Buffer allocation
// failed", 450 MB used, 7.7 GB more allowed). This one goes to posix_memalign directly and says
// what it was asked for when even that fails. Memory from it is only ever freed below.
namespace llvm {
void* allocate_buffer(size_t size, size_t alignment) {
    const size_t align = alignment < sizeof(void*) ? sizeof(void*) : alignment;
    void* p = nullptr;
    const int error = posix_memalign(&p, align, size ? size : 1);
    if (error || !p) {
        fprintf(stderr, "vpconvert: allocate_buffer(%zu bytes, alignment %zu) failed: %s\n", size, alignment, strerror(error));
        report_bad_alloc_error("Buffer allocation failed");
    }
    return p;
}
void deallocate_buffer(void* ptr, size_t, size_t) { free(ptr); }
} // namespace llvm

// vpaot's command line (tools/vpaot/main.cpp, built with -Dmain=vpaot_main).
int vpaot_main(int argc, char** argv);

namespace fs = std::filesystem;

namespace {

// vpaot keeps no state between runs, but it is not made for two at once.
std::mutex g_vpaot_mutex;
// lld is not reentrant.
std::mutex g_lld_mutex;
// lld said a link left it unable to run again in this process (lld::Result::canRunAgain).
std::atomic<bool> g_lld_spent{false};

struct Logger {
    const VpConvertCallbacks* cb;
    std::mutex mutex;
    void line(const char* f, ...) {
        va_list ap, ap2;
        va_start(ap, f);
        va_copy(ap2, ap);
        const int n = vsnprintf(nullptr, 0, f, ap);
        va_end(ap);
        std::string buf(n > 0 ? (size_t)n : 0, '\0');
        if (n > 0) vsnprintf(buf.data(), buf.size() + 1, f, ap2);
        va_end(ap2);
        std::scoped_lock lock{mutex};
        if (cb && cb->log) cb->log(cb->user, buf.c_str()); else fprintf(stderr, "vpconvert: %s\n", buf.c_str());
    }
    void progress(int phase, long done, long total) {
        if (cb && cb->progress) cb->progress(cb->user, phase, done, total);
    }
    bool stop() { return cb && cb->should_stop && cb->should_stop(cb->user); }
    int max_jobs(int jobs) {
        if (!cb || !cb->max_jobs) return jobs;
        const int n = cb->max_jobs(cb->user);
        return n < 1 ? 1 : n > jobs ? jobs : n;
    }
    void overall(double f) {
        if (cb && cb->overall) cb->overall(cb->user, f < 0 ? 0 : f > 1 ? 1 : f);
    }
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
// The lines of the missing-entries log that are this module's ("name+0x..."): only those change
// its translation.
std::string module_roots(const VpConvertConfig& c, const std::string& name) {
    if (!c.missing_log || !fs::exists(c.missing_log)) return "";
    std::istringstream in(read_file(c.missing_log));
    std::string out;
    for (std::string l; std::getline(in, l);) {
        if (l.rfind(name + "+", 0) == 0) out += l + "\n";
    }
    return out;
}

// The embedder's changes to a module's image (VpConvertCallbacks.patch_image) as vpaot --patch
// lines ("OFFSET HEXBYTES", runs of changed bytes); empty when it changes nothing.
std::string module_patch(const fs::path& file, const std::string& name, const VpConvertCallbacks* cb) {
    if (!cb || !cb->patch_image) return {};
    vpaot::Image img = vpaot::load_elf_or_self(file.string());
    std::vector<uint8_t> changed = img.memory;
    cb->patch_image(cb->user, name.c_str(), changed.data(), (unsigned long)changed.size());
    std::string out;
    char head[32];
    for (size_t i = 0; i < changed.size();) {
        if (changed[i] == img.memory[i]) { ++i; continue; }
        size_t j = i;
        while (j < changed.size() && changed[j] != img.memory[j] && j - i < 1024) ++j;
        snprintf(head, sizeof head, "%zx ", i);
        out += head;
        for (size_t k = i; k < j; ++k) { snprintf(head, sizeof head, "%02x", changed[k]); out += head; }
        out += "\n";
        i = j;
    }
    return out;
}

// What a module's translation depends on: if this is the same, its C (and objects) are current.
// `build` (see build_key) covers the translator, the runtime's interface and the compiler options.
std::string module_stamp(const fs::path& file, const VpConvertConfig& c, const std::string& name, int split,
                         const std::string& build) {
    struct stat st {};
    stat(file.string().c_str(), &st);
    const std::string roots = module_roots(c, name);
    char buf[512];
    snprintf(buf, sizeof buf, "vpconvert 2\nsize %lld\nmtime %lld\nsplit %d\nroots %zu %zx\n",
             (long long)st.st_size, (long long)st.st_mtime, split, roots.size(), std::hash<std::string>{}(roots));
    return buf + build;
}

// FNV-1a 64 of a file, in hex: an object is kept when the C it was compiled from is the same.
std::string content_hash(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    uint64_t h = 1469598103934665603ull;
    char buf[1 << 16];
    while (f) {
        f.read(buf, sizeof buf);
        for (std::streamsize i = 0; i < f.gcount(); ++i) h = (h ^ (unsigned char)buf[i]) * 1099511628211ull;
    }
    char out[32];
    snprintf(out, sizeof out, "%016" PRIx64, h);
    return out;
}

// The compiler flags of every piece (besides target, level, include paths).
const char* const kPieceFlags[] = {"-ffreestanding", "-fno-stack-protector", "-fno-math-errno", "-frounding-math", "-w"};

// What every object depends on besides its C: the runtime's binary interface and its headers in
// the SDK (an app update may change them), clang (version, headers) and the compiler options. Not
// the translator: a new translator that writes the same C for a piece keeps its object (its C's
// hash is in the key). An object is reused only when its C and this are the same.
std::string compile_key(const fs::path& sdk, const std::string& triple, const std::string& opt) {
    std::vector<fs::path> headers;
    std::error_code ec;
    for (const fs::path dir : {sdk / "runtime", sdk / "clang" / "include"}) {
        for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (it->is_regular_file(ec)) headers.push_back(it->path());
        }
        ec.clear();
    }
    std::sort(headers.begin(), headers.end());
    std::string all;
    for (const auto& h : headers) all += fs::relative(h, sdk, ec).generic_string() + " " + content_hash(h) + "\n";
    char buf[256];
    std::string flags;
    for (const char* f : kPieceFlags) flags += std::string(f) + " ";
    snprintf(buf, sizeof buf, "llvm %s\nabi %d\nsdk %zu %016zx\nopt %s\ntriple %s\nflags %016zx\n", LLVM_VERSION_STRING,
             VP_RUNTIME_ABI, headers.size(), std::hash<std::string>{}(all), opt.c_str(), triple.c_str(),
             std::hash<std::string>{}(flags));
    return buf;
}

// Shares of the whole conversion for the overall progress.
constexpr double kTranslateShare = 0.05, kCompileShare = 0.92;

uintmax_t file_bytes(const fs::path& p) {
    std::error_code ec;
    const uintmax_t n = fs::file_size(p, ec);
    return ec ? 0 : n;
}

// "mnemonic count, ..." of the stats' "unsupported_by_mnemonic", most frequent first.
std::string top_unsupported(const std::string& stats, size_t n) {
    const auto at = stats.find("\"unsupported_by_mnemonic\"");
    if (at == std::string::npos) return "";
    const auto open = stats.find('{', at), close = stats.find('}', at);
    if (open == std::string::npos || close == std::string::npos || close < open) return "";
    std::vector<std::pair<unsigned long long, std::string>> items;
    size_t i = open + 1;
    while (i < close) {
        const auto q1 = stats.find('"', i);
        if (q1 == std::string::npos || q1 >= close) break;
        const auto q2 = stats.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 >= close) break;
        const auto colon = stats.find(':', q2);
        if (colon == std::string::npos || colon >= close) break;
        items.emplace_back(std::strtoull(stats.c_str() + colon + 1, nullptr, 10), stats.substr(q1 + 1, q2 - q1 - 1));
        i = colon + 1;
    }
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string out;
    for (size_t k = 0; k < items.size() && k < n; ++k) {
        if (k) out += ", ";
        out += items[k].second + " " + std::to_string(items[k].first);
    }
    return out;
}

// ---- the compiler -------------------------------------------------------------------------

// A fatal error inside LLVM (report_fatal_error) would end the process, which is the app's. It
// goes back instead to the CrashRecoveryContext the compile or the link runs in, as clang's own
// cc1 does (sys::Process::Exit returns to the current context), with the reason in the log.
thread_local std::string* t_diagnostics = nullptr;
// Where a compile was when it crashed (for the log).
thread_local const char* t_stage = "start";

// Out of memory: nothing that allocates (the message goes to stderr, the console log).
void bad_alloc_handler(void*, const char* reason, bool) {
    fprintf(stderr, "vpconvert: out of memory in LLVM: %s\n", reason);
    llvm::sys::Process::Exit(1);
}

void fatal_error_handler(void*, const char* reason, bool) {
    if (t_diagnostics) *t_diagnostics += std::string("fatal error: ") + reason + "\n";
    else fprintf(stderr, "vpconvert: fatal error: %s\n", reason);
    llvm::sys::Process::Exit(1);
}

// Crash recovery (the setjmp of lld's own safeLldMain and of the compiles here) only while a
// conversion runs: it installs process-wide handlers for SIGSEGV, SIGBUS... and the emulator
// has its own (the game never runs during a conversion). Counted: two conversions may overlap.
std::mutex g_recovery_mutex;
int g_recovery_users = 0;
struct CrashRecovery {
    CrashRecovery() {
        std::scoped_lock lock{g_recovery_mutex};
        ++g_recovery_users;
        // Always (idempotent): LLVM's own signal handler disables recovery when a signal comes on
        // a thread outside any context, and a count alone would never turn it back on.
        llvm::CrashRecoveryContext::Enable();
    }
    ~CrashRecovery() {
        std::scoped_lock lock{g_recovery_mutex};
        if (--g_recovery_users == 0) llvm::CrashRecoveryContext::Disable();
    }
};

void init_llvm_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmPrinter();
        LLVMInitializeAArch64AsmParser();
        llvm::install_fatal_error_handler(fatal_error_handler, nullptr);
        // Out of memory inside LLVM: the same way back (it would otherwise abort the app).
        llvm::install_bad_alloc_error_handler(bad_alloc_handler, nullptr);
    });
}

// clang -c `in` -o `out` with `args`, in this process: the driver turns the command line into the
// compiler's own (-cc1) arguments, which run as a CompilerInstance here instead of a new process.
bool compile_unsafe(const std::vector<std::string>& args, std::string& diagnostics) {
    init_llvm_once();
    std::vector<const char*> argv;
    argv.push_back("clang");
    for (const auto& a : args) argv.push_back(a.c_str());

    llvm::raw_string_ostream diag_stream(diagnostics);
    clang::DiagnosticOptions diag_options;
    auto* printer = new clang::TextDiagnosticPrinter(diag_stream, diag_options);
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> ids(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine diags(ids, diag_options, printer);

    t_stage = "the driver";
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

    t_stage = "the compiler's arguments";
    auto instance = std::make_unique<clang::CompilerInstance>();
    if (!clang::CompilerInvocation::CreateFromArgs(instance->getInvocation(), cc1, diags)) return false;
    // The driver asks cc1 not to free its AST, Sema and target machine (a process ends right
    // after): here every piece would leak them, hundreds of MB over a game. As clang's tooling.
    instance->getFrontendOpts().DisableFree = false;
    instance->getCodeGenOpts().DisableFree = false;
    instance->createDiagnostics(*llvm::vfs::getRealFileSystem(), printer, /*ShouldOwnClient=*/false);
    t_stage = "the compiler";
    clang::EmitObjAction action;
    const bool ok = instance->ExecuteAction(action);
    t_stage = "the end";
    diag_stream.flush();
    return ok && !instance->getDiagnostics().hasErrorOccurred();
}

// compile_unsafe in a CrashRecoveryContext: a crash or a fatal error in the driver or the compiler
// fails this piece instead of ending the app (what it had allocated is not freed then).
// After a crash by a signal inside the compiler, its process-wide state (heap, LLVM's locks) may
// be broken: no more compiles until the app starts again (a fatal error is an orderly exit).
std::atomic<bool> g_compiler_spent{false};

bool compile(const std::vector<std::string>& args, std::string& diagnostics, bool probing = false) {
    if (g_compiler_spent && !probing) {
        diagnostics = "vpconvert: the compiler crashed earlier in this run of the app; close the app and open it "
                      "again to continue (what was compiled is kept)\n";
        return false;
    }
    bool ok = false;
    std::string fatal;
    t_diagnostics = &fatal;
    llvm::CrashRecoveryContext crc;
    // As clang's own driver (Job.cpp): a crash must not leave this thread's pretty stack trace
    // pointing at frames that are gone.
    const void* pretty = llvm::SavePrettyStackState();
    const bool ran = crc.RunSafely([&] { ok = compile_unsafe(args, diagnostics); });
    llvm::RestorePrettyStackState(pretty);
    t_diagnostics = nullptr;
    // A fatal error may also end in one of clang's own nested contexts (its stack switching for
    // deep recursion), which it does not check: `ok` would then be wrong. The message counts.
    if (!ran || !fatal.empty()) {
        if (!ran && crc.RetCode > 128) g_compiler_spent = true; // a signal, not an orderly exit
        if (fatal.empty()) {
            char what[160];
            const int code = crc.RetCode;
            if (code > 128) snprintf(what, sizeof what, "signal %d (%s)", code - 128, strsignal(code - 128));
            else snprintf(what, sizeof what, "exit code %d", code);
            diagnostics += std::string("vpconvert: the compiler crashed in ") + t_stage + ": " + what + "\n";
        } else {
            diagnostics += std::string("in ") + t_stage + ": " + fatal;
        }
        return false;
    }
    return ok;
}

// ---- the linker ---------------------------------------------------------------------------

bool link(const std::vector<std::string>& args, std::string& output) {
    std::scoped_lock lock{g_lld_mutex};
    if (g_lld_spent) {
        output = "vpconvert: an earlier link failed in a way lld cannot recover from in the same process; "
                 "close the app and open it again to link\n";
        return false;
    }
    std::vector<const char*> argv;
    argv.push_back("ld64.lld");
    for (const auto& a : args) argv.push_back(a.c_str());
    llvm::raw_string_ostream out(output);
    std::string fatal;
    t_diagnostics = &fatal;
    lld::Result r = lld::lldMain(argv, out, out, {{lld::Darwin, &lld::macho::link}});
    t_diagnostics = nullptr;
    out.flush();
    output += fatal;
    if (!r.canRunAgain) g_lld_spent = true;
    return r.retCode == 0;
}

struct Piece {
    fs::path c, o;
    std::string module; // empty: the registry
    uintmax_t bytes = 0;
};

int convert(const VpConvertConfig& c, Logger& log) {
    init_llvm_once(); // also the fatal error handler, which the link needs too
    const auto started = std::chrono::steady_clock::now();
    const int split = c.split > 0 ? c.split : 300;
    const int jobs = c.jobs > 0 ? c.jobs : 4;
    const std::string triple = c.triple ? c.triple : "arm64-apple-xros2.0";
    const std::string opt = c.opt_level ? c.opt_level : "-O2";
    const fs::path game(c.game_dir), work(c.work_dir), sdk(c.sdk_dir);
    const std::string compiled_with = compile_key(sdk, triple, opt);
    // A module's translation is current when its file, roots, split, translator and compiler are.
    char translator[96];
    snprintf(translator, sizeof translator, "vpaot %s\nzydis %llx\n", VPAOT_SOURCE_ID, (unsigned long long)ZYDIS_VERSION);
    const std::string build = translator + compiled_with;
    // What a piece's ".o.ok" holds: its object is current when its C and the build are the same.
    auto object_key = [&](const fs::path& c_file) { return content_hash(c_file) + "\n" + compiled_with; };
    std::error_code ec;
    fs::create_directories(work, ec);
    if (ec) { log.line("ERROR: cannot create %s: %s", work.string().c_str(), ec.message().c_str()); return -1; }
    std::string title = c.title ? c.title : game.filename().string();

    // ---- 1. translation ------------------------------------------------------------------
    std::vector<fs::path> files;
    if (fs::exists(game / "eboot.bin", ec)) files.push_back(game / "eboot.bin");
    if (fs::is_directory(game / "sce_module", ec)) {
        std::vector<fs::path> mods;
        for (const auto& e : fs::directory_iterator(game / "sce_module")) { // throws: caught in vp_convert
            std::string ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char x) { return (char)std::tolower(x); });
            if (e.is_regular_file(ec) && (ext == ".prx" || ext == ".sprx")) mods.push_back(e.path());
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
    // Overall progress: translation by the modules' sizes, compilation by MB of C, then the link.
    uintmax_t all_bytes = 0, translated_bytes = 0;
    for (const auto& f : files) all_bytes += file_bytes(f);
    log.overall(0);
    for (const auto& file : files) {
        const std::string name = module_name(file);
        if (std::find(names.begin(), names.end(), name) != names.end()) {
            log.line("ERROR: two modules map to the name %s", name.c_str());
            return -1;
        }
        names.push_back(name);
        const fs::path stamp = work / (name + ".stamp");
        std::string want = module_stamp(file, c, name, split, build);
        // The embedder's code patches: translated as it will run, part of what the module is.
        const fs::path patch_path = work / (name + ".patch");
        std::string patch;
        try {
            patch = module_patch(file, name, log.cb);
        } catch (const std::exception& e) {
            log.line("ERROR: reading %s for its patches: %s", file.filename().string().c_str(), e.what());
            return -1;
        }
        if (!patch.empty()) {
            char h[64];
            snprintf(h, sizeof h, "patch %zu %016llx\n", patch.size(), (unsigned long long)std::hash<std::string>{}(patch));
            want += h;
            if (!write_file(patch_path, patch)) { log.line("ERROR: cannot write %s", patch_path.string().c_str()); return -1; }
            log.line("%s: the emulator changes %zu runs of its code when it loads it: translated as changed", name.c_str(),
                     (size_t)std::count(patch.begin(), patch.end(), '\n'));
        } else {
            fs::remove(patch_path, ec);
        }
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
            // Its objects stay: a piece whose new C is the same as before keeps its object.
            const std::regex mine("^" + name + "(\\.(c|h|json|stamp|o\\.tmp)|_decl\\.h|_files\\.txt|_([0-9]+|u[0-9a-f]+|utables)\\.(c|o\\.tmp))$");
            for (const auto& e : fs::directory_iterator(work)) {
                if (std::regex_match(e.path().filename().string(), mine)) fs::remove(e.path(), ec);
            }
            const auto t = std::chrono::steady_clock::now();
            std::vector<std::string> a = {"vpaot", "--elf", file.string(), "--pic", "--module", name, "--split", std::to_string(split),
                                          "--out", (work / (name + ".c")).string(), "--stats", (work / (name + ".json")).string()};
            if (c.missing_log && fs::exists(c.missing_log)) { a.push_back("--roots"); a.push_back(c.missing_log); }
            if (!patch.empty()) { a.push_back("--patch"); a.push_back(patch_path.string()); }
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
            size_t kept = 0;
            for (const auto& p : parts) {
                bytes += fs::file_size(p, ec);
                fs::path o = p; o.replace_extension(".o");
                const fs::path ok = o.string() + ".ok";
                if (fs::exists(o) && fs::exists(ok) && read_file(ok) == object_key(p)) {
                    fs::remove(p, ec); // compiled before from the same C
                    ++kept;
                } else {
                    fs::remove(ok, ec);
                }
            }
            // Objects of pieces this translation no longer has.
            const std::regex piece_object("^" + name + "(_([0-9]+|u[0-9a-f]+|utables))?\\.o(\\.ok|\\.csize)?$");
            for (const auto& e : fs::directory_iterator(work)) {
                const std::string f = e.path().filename().string();
                if (!std::regex_match(f, piece_object)) continue;
                fs::path base = e.path();
                for (const char* suffix : {".ok", ".csize"}) {
                    const size_t n = strlen(suffix);
                    if (f.size() > n && f.compare(f.size() - n, n, suffix) == 0) base = base.string().substr(0, base.string().size() - n);
                }
                fs::path c_of = base; c_of.replace_extension(".c");
                if (std::find(parts.begin(), parts.end(), c_of) == parts.end()) fs::remove(e.path(), ec);
            }
            if (kept) log.line("%s: %zu of %zu pieces are the same as before (their objects are kept)", name.c_str(), kept, parts.size());
            if (!write_file(stamp, want)) { log.line("ERROR: cannot write %s", stamp.string().c_str()); return -1; }
            log.line("translated %s -> %s in %.1f s: %s functions, %s instructions, supported %s, %zu pieces, %.1f MB of C",
                     file.filename().string().c_str(), name.c_str(), seconds_since(t), field("functions").c_str(),
                     field("instructions").c_str(), field("supported_fraction").c_str(), parts.size(), bytes / 1048576.0);
            const std::string missing = top_unsupported(stats, 15);
            if (!missing.empty()) log.line("%s: most frequent unsupported instructions: %s", name.c_str(), missing.c_str());
        } else {
            log.line("%s: translation kept from before", name.c_str());
        }
        for (const auto& p : parts) {
            Piece piece;
            piece.c = p;
            piece.o = p; piece.o.replace_extension(".o");
            piece.module = name;
            piece.bytes = fs::exists(p) ? fs::file_size(p, ec) : 0;
            pieces.push_back(piece);
        }
        log.progress(VP_CONVERT_TRANSLATE, ++translated, (long)files.size());
        translated_bytes += file_bytes(file);
        log.overall(kTranslateShare * (double)translated_bytes / (double)(all_bytes ? all_bytes : 1));
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
    // By MB of C, pieces compiled before included (their size was kept when they compiled; one
    // compiled by an older version without it counts as an average piece).
    const double average = pieces.empty() ? 0 : (double)(todo_bytes) / (double)std::max<size_t>(1, todo.size());
    double before_bytes = 0;
    for (auto& p : pieces) {
        if (std::find(todo.begin(), todo.end(), &p) != todo.end()) continue;
        const uintmax_t n = std::strtoull(read_file(p.o.string() + ".csize").c_str(), nullptr, 10);
        before_bytes += n ? (double)n : average;
    }
    auto compile_fraction = [&] {
        const double all = before_bytes + (double)todo_bytes;
        return all > 0 ? (before_bytes + (double)done_bytes.load()) / all : 1.0;
    };
    log.overall(kTranslateShare + kCompileShare * compile_fraction());
    std::vector<std::string> base_args = {"--target=" + triple, opt};
    for (const char* f : kPieceFlags) base_args.push_back(f);
    for (const std::string& a : {std::string("-nostdinc"), std::string("-resource-dir"), (sdk / "clang").string(),
                                 std::string("-isystem"), (sdk / "clang" / "include").string(),
                                 std::string("-isystem"), (sdk / "runtime" / "freestanding").string(), std::string("-I"),
                                 (sdk / "runtime").string(), std::string("-c")}) {
        base_args.push_back(a);
    }
    std::atomic<int> limit_logged{jobs};
    auto worker = [&](int k) {
        for (;;) {
            if (failed) return;
            // Heat: workers past the allowed number wait (checked every 2 s).
            for (;;) {
                const int allowed = log.max_jobs(jobs);
                if (int was = limit_logged.exchange(allowed); was != allowed)
                    log.line("pieces at once: %d (was %d)", allowed, was);
                if (k < allowed || failed || log.stop()) break;
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            if (failed) return;
            const size_t i = next++;
            if (i >= todo.size()) return;
            if (log.stop()) { stopped = true; return; }
            Piece& p = *todo[i];
            const auto t = std::chrono::steady_clock::now();
            std::vector<std::string> args = base_args;
            const std::string tmp = p.o.string() + ".tmp";
            args.push_back(p.c.string());
            args.push_back("-o");
            args.push_back(tmp);
            std::string diagnostics;
            std::error_code e;
            if (!compile(args, diagnostics)) {
                failed = true;
                fs::remove(tmp, e);
                // Translated again next time (the C may be cut, e.g. by a full disk), not the
                // same C failing at every run.
                if (!p.module.empty()) fs::remove(work / (p.module + ".stamp"), e);
                log.line("ERROR: compiling %s failed:\n%s", p.c.filename().string().c_str(), diagnostics.substr(0, 8000).c_str());
                return;
            }
            const std::string key = object_key(p.c);
            fs::rename(tmp, p.o, e);
            if (e || !write_file(p.o.string() + ".ok", key)) {
                failed = true;
                fs::remove(p.o.string() + ".ok", e);
                log.line("ERROR: cannot keep the object of %s (%s)", p.c.filename().string().c_str(),
                         e ? e.message().c_str() : "cannot write its .ok");
                return;
            }
            write_file(p.o.string() + ".csize", std::to_string(p.bytes) + "\n"); // for the progress on a resume
            if (!p.module.empty()) fs::remove(p.c, e); // compiled: its C goes
            const long d = ++done;
            done_bytes += p.bytes;
            log.line("compiled %s (%.2f MB) in %.1f s [%ld/%ld]", p.c.filename().string().c_str(), p.bytes / 1048576.0,
                     seconds_since(t), d, total);
            log.progress(VP_CONVERT_COMPILE, d, total);
            log.overall(kTranslateShare + kCompileShare * compile_fraction());
        }
    };
    // Before the game's pieces: one line of C, in a thread like theirs. If clang cannot work in
    // this process at all, this says so (and how) instead of every piece failing alike, and tries
    // it other ways too (on this thread; at -O0) so one run of the app tells what works.
    if (!todo.empty()) {
        const fs::path test_c = work / "vpconvert_selftest.c", test_o = work / "vpconvert_selftest.o";
        if (!write_file(test_c, "int vp_selftest(int x) { return x * 3 + 1; }\n")) {
            log.line("ERROR: cannot write %s", test_c.string().c_str());
            return -1;
        }
        auto self_test = [&](const char* how, bool own_thread, const char* level) {
            std::vector<std::string> args = base_args;
            if (level) std::replace(args.begin(), args.end(), opt, std::string(level));
            args.push_back(test_c.string());
            args.push_back("-o");
            args.push_back(test_o.string());
            std::string diagnostics;
            bool ok = false;
            const auto t = std::chrono::steady_clock::now();
            if (own_thread) {
                llvm::thread test(std::optional<unsigned>(32u << 20), [&] {
                    clang::noteBottomOfStack();
                    ok = compile(args, diagnostics, /*probing=*/true);
                });
                test.join();
            } else {
                ok = compile(args, diagnostics, /*probing=*/true);
            }
            std::error_code e;
            const uintmax_t size = fs::file_size(test_o, e);
            fs::remove(test_o, e);
            if (ok) log.line("compiler self-test (%s): OK in %.2f s (%ju bytes of object)", how, seconds_since(t), size);
            else log.line("compiler self-test (%s): FAILED in %.2f s:\n%s", how, seconds_since(t), diagnostics.substr(0, 4000).c_str());
            return ok;
        };
        if (g_compiler_spent) { // not even the test: the process's state may be broken
            log.line("ERROR: the compiler crashed earlier in this run of the app; close the app and open it again to continue");
            return -1;
        }
        const bool works = self_test("worker thread, as the pieces", true, nullptr);
        if (!works) {
            self_test("the conversion's own thread", false, nullptr);
            self_test("worker thread, -O0", true, "-O0");
        }
        std::error_code e;
        fs::remove(test_c, e);
        if (!works) {
            log.line("ERROR: the compiler does not work in this process (see the self-tests above)");
            return -1;
        }
    }
    // clang recurses deeply on large functions: its own driver gives it 8 MB of stack, more than
    // a secondary thread gets by default (512 KB on Apple systems).
    std::vector<std::unique_ptr<llvm::thread>> threads;
    for (int k = 0; k < jobs; ++k) {
        threads.push_back(std::make_unique<llvm::thread>(std::optional<unsigned>(32u << 20), [&, k] {
            clang::noteBottomOfStack();
            worker(k);
        }));
    }
    for (auto& th : threads) th->join();
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
        log.line("ERROR: linking failed:\n%s", link_output.substr(0, 8000).c_str());
        return -1;
    }
    fs::rename(out_tmp, c.output, ec);
    if (ec) { log.line("ERROR: cannot move the pack to %s: %s", c.output, ec.message().c_str()); return -1; }
    log.progress(VP_CONVERT_LINK, 1, 1);
    log.overall(1);
    log.line("linked %zu objects into %s (%.1f MB) in %.1f s", pieces.size(), c.output, fs::file_size(c.output, ec) / 1048576.0,
             seconds_since(link_started));
    log.line("conversion finished in %.1f s of this run", seconds_since(started));
    return 0;
}

} // namespace

extern "C" int vp_convert(const VpConvertConfig* config, const VpConvertCallbacks* callbacks) {
    Logger log{callbacks, {}};
    // C++ errors (a folder that became unreadable, memory) must not cross into Swift.
    try {
        CrashRecovery recovery;
        return convert(*config, log);
    } catch (const std::exception& e) {
        log.line("ERROR: %s", e.what());
    } catch (...) {
        log.line("ERROR: unexpected exception");
    }
    return -1;
}
