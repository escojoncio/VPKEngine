#include "vpps4/report.h"

#include <atomic>
#include <cstdint>

namespace {
struct Sink {
    vpps4_report_fn fn;
    void* user;
};
// Replaced sinks are leaked (a handful of bytes): a reporting thread may still be using the old one.
std::atomic<const Sink*> gSink{nullptr};
} // namespace

extern "C" void vpps4_set_report_sink(vpps4_report_fn fn, void* user) {
    gSink.store(fn ? new Sink{fn, user} : nullptr, std::memory_order_release);
}

extern "C" void vpps4_report_count(const char* category, const char* key, const char* detail, uint64_t count) {
    if (const Sink* s = gSink.load(std::memory_order_acquire)) s->fn(s->user, category, key, detail ? detail : "", count);
}

extern "C" void vpps4_report(const char* category, const char* key, const char* detail) {
    vpps4_report_count(category, key, detail, 1);
}

namespace vpps4 {

void report(std::string_view category, std::string_view key, std::string_view detail) {
    const Sink* s = gSink.load(std::memory_order_acquire);
    if (!s) return;
    // The C API takes NUL-terminated strings.
    const std::string c(category), k(key), d(detail);
    s->fn(s->user, c.c_str(), k.c_str(), d.c_str(), 1);
}

} // namespace vpps4
