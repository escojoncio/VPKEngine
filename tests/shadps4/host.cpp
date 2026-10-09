// A stand-in for AstroVisionPro's shadPS4 around the AOT guest engine: maps the module, points
// its imports at x86 veneers written at run time, answers HLE calls in a GuestBridge, runs guest
// functions the way Linker::RunGuestFunction does, and compares with the native build.
#include "core/fex/fex_guest_engine.h"
extern "C" {
#include "vp_loader.h"
}
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <thread>
#include <vector>

using namespace Core;
using u64 = unsigned long;
__thread u64 native_tls;
extern "C" u64 guest_main_native(u64 seed);
extern "C" void fail_test_native(u64 x);
extern "C" void on_signal_native(int sig, void* ctx);
extern "C" u64 lib_fn_native(u64 x);
extern "C" volatile u64 fail_out_native;
extern "C" u64 hle_lib(u64 x) { return lib_fn_native(x); }
extern "C" double hle_fmul(double a, double b) { on_signal_native(30, nullptr); return a * b; }
extern "C" u64 hle_fail(u64 (*fn)(u64), u64 x) { (void)fn(x); return static_cast<u64>(-EIO); }
extern "C" u64 hle_mix(u64 a, u64 b) { return a * 3 + b; }
extern "C" u64 hle_callback(u64 (*fn)(u64), u64 x) { return fn(x) + 1; }
static u64 thread_counter;
extern "C" u64 hle_thread(u64 (*fn)(u64), u64 x) {
    u64 r = 0;
    const u64 tls = 0x2000 + __atomic_add_fetch(&thread_counter, 1, __ATOMIC_SEQ_CST);
    std::thread t([&] { native_tls = tls; r = fn(x); });
    t.join();
    return r;
}

static std::unique_ptr<Fex::GuestEngine> engine;
static uint64_t veneer_page, module_base, module_size, lib_base, lib_size, lib_entry, signal_handler;
static uint64_t guest_thread_counter;

// Per guest thread: its TLS block (fs base) with the value at +0x10.
static uint64_t make_tls(uint64_t value) {
    auto* block = static_cast<uint64_t*>(mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    block[2] = value;
    return reinterpret_cast<uint64_t>(block);
}

// Linker::RunGuestFunction: nested CallGuest when a guest thread is active here, else a new
// engine thread with its own stack, the return address pushed, Run, and the halt checked.
static u64 run_guest_function(uint64_t entry, std::vector<u64> args, uint64_t fs_base) {
    auto nested = engine->CallGuest(entry, args);
    if (auto* state = std::get_if<GuestExecutionState>(&nested)) return state->Gpr[0];
    auto* failure = std::get_if<Fex::EngineFailure>(&nested);
    if (!failure || failure->Error != ENXIO) { std::fprintf(stderr, "CallGuest failed\n"); std::exit(1); }
    const size_t stack_size = 1 << 20;
    auto* stack = static_cast<uint8_t*>(mmap(nullptr, stack_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    const uint64_t rsp = ((reinterpret_cast<uint64_t>(stack) + stack_size - 256) & ~uint64_t{15}) - 8;
    const uint64_t ret = engine->ReturnAddress();
    std::memcpy(reinterpret_cast<void*>(rsp), &ret, 8);
    GuestExecutionRequest request;
    request.Rip = entry;
    request.Rsp = rsp;
    request.Rflags = 2;
    request.FsBase = fs_base;
    static const int regs[6] = {7, 6, 2, 1, 8, 9};
    for (size_t i = 0; i < args.size(); ++i) request.Gpr[regs[i]] = args[i];
    auto created = engine->CreateThread(request);
    auto* thread = std::get<Fex::GuestEngine::Thread*>(created);
    auto result = engine->Run(*thread);
    if (auto* f = std::get_if<Fex::EngineFailure>(&result)) { std::fprintf(stderr, "Run failed: stage %d error %d\n", (int)f->Stage, f->Error); std::exit(1); }
    auto& state = std::get<GuestExecutionState>(result);
    if (state.StopReason != GuestStopReason::Halted || state.Rip < ret || state.Rip >= ret + 4096) {
        std::fprintf(stderr, "did not halt at the return page (rip %#" PRIx64 ")\n", (uint64_t)state.Rip);
        std::exit(1);
    }
    engine->DestroyThread(thread);
    munmap(stack, stack_size);
    return state.Gpr[0];
}

class Bridge final : public Fex::GuestBridge {
public:
    Fex::EngineResult<bool> Invoke(GuestCpu::HleCallFrame& frame) override {
        auto& g = frame.gpr;
        switch (frame.operation) {
        case 1: g[0] = g[7] * 3 + g[6]; return true;                                 // hle_mix
        case 2: g[0] = run_guest_function(g[7], {g[6]}, 0) + 1; return true;          // hle_callback
        case 3: {                                                                     // hle_thread
            u64 r = 0;
            const uint64_t fn = g[7], x = g[6];
            const uint64_t tls = make_tls(0x2000 + __atomic_add_fetch(&guest_thread_counter, 1, __ATOMIC_SEQ_CST));
            std::thread t([&] { r = run_guest_function(fn, {x}, tls); });
            t.join();
            g[0] = r;
            return true;
        }
        case 4: g[0] = run_guest_function(lib_entry, {g[7]}, 0); return true;        // hle_lib
        case 5: {                                                                     // hle_fmul
            // The signal arrives while the call is in the host (as a GC stop-the-world kill does).
            Fex::DeliverGuestOrbisSignal(30, nullptr, nullptr, signal_handler);
            double a, b;
            std::memcpy(&a, &frame.xmm[0][0], 8);
            std::memcpy(&b, &frame.xmm[1][0], 8);
            const double r = a * b;
            std::memcpy(&frame.xmm[0][0], &r, 8);
            return true;
        }
        case 7: {                                                                     // hle_fiber_switch
            static const int saved[7] = {3, 4, 5, 12, 13, 14, 15}; // rbx rsp rbp r12-r15
            auto* from = reinterpret_cast<uint64_t*>(g[7]);
            auto* to = reinterpret_cast<uint64_t*>(g[6]);
            for (int i = 0; i < 7; ++i) from[i] = g[saved[i]];
            for (int i = 0; i < 7; ++i) g[saved[i]] = to[i];
            return true;
        }
        case 6:                                                                       // hle_fail
            (void)run_guest_function(g[7], {g[6]}, 0);
            return Fex::EngineFailure{Fex::EngineStage::Bridge, EIO};
        default: return Fex::EngineFailure{Fex::EngineStage::Bridge, ENOSYS};
        }
    }
    std::optional<GuestExecutionRange> QueryExecutableRange(std::uintptr_t address) override {
        if (address >= veneer_page && address < veneer_page + 4096) return GuestExecutionRange{veneer_page, 4096, true, false};
        if (address >= module_base && address < module_base + module_size) return GuestExecutionRange{module_base, module_size, true, false};
        if (address >= lib_base && address < lib_base + lib_size) return GuestExecutionRange{lib_base, lib_size, true, false};
        return std::nullopt;
    }
};

// shadPS4's veneer: mov r10, rcx; mov rax, operation; syscall; ret.
static uint64_t veneer(int index, uint32_t operation) {
    auto* p = reinterpret_cast<uint8_t*>(veneer_page + index * 16);
    const uint8_t code[] = {0x49, 0x89, 0xca, 0x48, 0xc7, 0xc0, 0, 0, 0, 0, 0x0f, 0x05, 0xc3};
    std::memcpy(p, code, sizeof code);
    std::memcpy(p + 6, &operation, 4);
    return reinterpret_cast<uint64_t>(p);
}

static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void*) {
    (void)function; *native = nullptr;
    if (!std::strcmp(name, "hle_mix")) { *data = veneer(0, 1); return 1; }
    if (!std::strcmp(name, "hle_callback")) { *data = veneer(1, 2); return 1; }
    if (!std::strcmp(name, "hle_thread")) { *data = veneer(2, 3); return 1; }
    if (!std::strcmp(name, "hle_lib")) { *data = veneer(3, 4); return 1; }
    if (!std::strcmp(name, "hle_fmul")) { *data = veneer(4, 5); return 1; }
    if (!std::strcmp(name, "hle_fail")) { *data = veneer(5, 6); return 1; }
    if (!std::strcmp(name, "hle_fiber_switch")) { *data = veneer(6, 7); return 1; }
    return 0;
}

int main(int argc, char** argv) {
    veneer_page = reinterpret_cast<uint64_t>(mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    Bridge bridge;
    auto created = Fex::GuestEngine::Create(bridge);
    engine = std::move(std::get<std::unique_ptr<Fex::GuestEngine>>(created));
    // Map the modules where "shadPS4" wants them (not their link addresses), then forget the
    // attachments: the engine must find each translation by fingerprint on first use.
    VpModule* game = vp_module_by_name("game");
    VpModule* lib = vp_module_by_name("lib");
    VpLoadedImage img, lib_img;
    if (vp_load_module(argv[1], game, 0x5000000000ull, resolve, nullptr, &img)) { std::fprintf(stderr, "load: %s\n", img.error); return 1; }
    if (vp_load_module(argv[4], lib, 0x6000000000ull, resolve, nullptr, &lib_img)) { std::fprintf(stderr, "load lib: %s\n", lib_img.error); return 1; }
    module_base = img.base; module_size = img.end - img.base;
    lib_base = lib_img.base; lib_size = lib_img.end - lib_img.base;
    vp_detach_module(game);
    vp_detach_module(lib);
    const uint64_t guest_main = img.base + std::strtoull(argv[2], nullptr, 0) - game->link_base;
    signal_handler = img.base + std::strtoull(argv[3], nullptr, 0) - game->link_base;
    const uint64_t fail_test = img.base + std::strtoull(argv[6], nullptr, 0) - game->link_base;
    const uint64_t fail_out = img.base + std::strtoull(argv[7], nullptr, 0) - game->link_base;
    lib_entry = lib_img.base + std::strtoull(argv[5], nullptr, 0) - lib->link_base;
    const uint64_t main_tls = make_tls(0x1111);
    const u64 translated = run_guest_function(guest_main, {12345}, main_tls);
    native_tls = 0x1111;
    const u64 expected = guest_main_native(12345);
    std::printf("shadPS4-style engine: translated %016lx native %016lx %s\n", translated, expected, translated == expected ? "OK" : "MISMATCH");
    if (translated != expected) return 1;

    // An HLE call that fails after calling back: Run reports the failure, and the guest still
    // returned from the call through its own stack with rax = -EIO.
    {
        GuestExecutionRequest request;
        auto* stack = static_cast<uint8_t*>(mmap(nullptr, 1 << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        const uint64_t rsp = ((reinterpret_cast<uint64_t>(stack) + (1 << 20) - 256) & ~uint64_t{15}) - 8;
        const uint64_t ret = engine->ReturnAddress();
        std::memcpy(reinterpret_cast<void*>(rsp), &ret, 8);
        request.Rip = fail_test;
        request.Rsp = rsp;
        request.Rflags = 2;
        request.FsBase = main_tls;
        request.Gpr[7] = 77;
        auto* thread = std::get<Fex::GuestEngine::Thread*>(engine->CreateThread(request));
        auto result = engine->Run(*thread);
        auto* f = std::get_if<Fex::EngineFailure>(&result);
        u64 got;
        std::memcpy(&got, reinterpret_cast<void*>(fail_out), 8);
        fail_test_native(77);
        const u64 want = fail_out_native;
        std::printf("failing HLE call after a callback: %s, result %#lx native %#lx %s\n", f && f->Error == EIO ? "EIO reported" : "NOT reported",
                    got, want, f && f->Error == EIO && got == want ? "OK" : "MISMATCH");
        if (!f || f->Error != EIO || got != want) return 1;
        engine->DestroyThread(thread);
    }
    if (const char* bench = std::getenv("VP_HLE_BENCH")) {
        const uint64_t fn = img.base + std::strtoull(bench, nullptr, 0) - game->link_base;
        const auto t0 = std::chrono::steady_clock::now();
        const u64 r = run_guest_function(fn, {1000000}, main_tls);
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 1e6;
        std::printf("HLE call through a veneer: %.1f ns (result %lx)\n", ns, r);
    }
    // Fibers: 6 round trips between the main context and a fiber through the host.
    if (const char* fib = std::getenv("VP_FIBER_TEST")) {
        const uint64_t fn = img.base + std::strtoull(fib, nullptr, 0) - game->link_base;
        const u64 got = run_guest_function(fn, {6}, main_tls);
        u64 want = 0, f = 0;
        for (u64 i = 0; i < 6; ++i) { f = f * 5 + (i + 1); want = want * 31 + f + i; }
        std::printf("fibers switched by the HLE bridge: %016lx expected %016lx %s\n", got, want, got == want ? "OK" : "MISMATCH");
        if (got != want) return 1;
    }
    // A second engine after this one (the app starting another game): the return pages still end runs.
    engine.reset();
    for (int i = 0; i < 6; ++i) {
        auto again = Fex::GuestEngine::Create(bridge);
        if (!std::holds_alternative<std::unique_ptr<Fex::GuestEngine>>(again)) { std::fprintf(stderr, "engine %d not created\n", i); return 1; }
        engine = std::move(std::get<std::unique_ptr<Fex::GuestEngine>>(again));
        engine.reset();
    }
    engine = std::move(std::get<std::unique_ptr<Fex::GuestEngine>>(Fex::GuestEngine::Create(bridge)));
    const u64 again = run_guest_function(guest_main, {12345}, main_tls);
    native_tls = 0x1111;
    const u64 again_expected = guest_main_native(12345);
    std::printf("engine re-created 7 times: %s\n", again == again_expected ? "OK" : "MISMATCH");
    return again == again_expected ? 0 : 1;
}
