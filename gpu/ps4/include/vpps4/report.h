// What vpps4 skips, approximates or cannot translate, reported to whoever hosts it. vpps4 does not
// know the emulator: the host installs a callback (AstroVisionPro wires it to its census, which
// counts the events and prints a summary) and every skip / approximation / translation error calls
// it. Without a sink the reports are dropped (one atomic load).
//
// Categories used here: "draw-skip" (a draw or dispatch left out; key = reason), "approx" (done in an
// approximate way), "unsupported" (a format / tile mode / feature the renderer cannot do),
// "shader" (a translation error; key = the first error, detail = the program key).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// `count`: how many times the event happened (the renderer aggregates the per-draw ones and reports once per frame).
typedef void (*vpps4_report_fn)(void* user, const char* category, const char* key, const char* detail, uint64_t count);

// Installs (or, with a null function, removes) the sink. Any thread; the sink must be thread-safe.
void vpps4_set_report_sink(vpps4_report_fn fn, void* user);
// Reports one event through the sink; strings are copied by the sink if it keeps them.
void vpps4_report(const char* category, const char* key, const char* detail);
// The same for `count` events at once.
void vpps4_report_count(const char* category, const char* key, const char* detail, uint64_t count);

#ifdef __cplusplus
}

#include <string>
#include <string_view>

namespace vpps4 {

inline void setReportSink(vpps4_report_fn fn, void* user = nullptr) { vpps4_set_report_sink(fn, user); }
void report(std::string_view category, std::string_view key, std::string_view detail = {});

} // namespace vpps4
#endif
