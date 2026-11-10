// A/B switches of the engine (diagnostic experiments, all off unless the variable is set). They change what the host
// asks of the GPU for one run so a figure can be compared with a run without them; none is a default, none is per title.
//   VPKENGINE_VS_FASTMATH=1|2   (gpu/metal2) vertex libraries whose position is [[invariant]] are compiled with fast math (the
//       engine compiles them with MTLMathModeSafe so that a depth prepass and a colour pass compute the same position bits).
//       1: Safe off, preserveInvariance kept (the compiler's own mechanism for a consistent position). 2: both off. RISK: the
//       position of two programs that compute it with the same operations may differ in the last bits (fast math contracts a
//       multiply and an add differently in each program): a depth test EQUAL / LEQUAL of the colour pass against the prepass then
//       fails triangle by triangle (holes, flicker). The library key changes with the mode so archived pipelines are not mixed.
//   VPKENGINE_MAX_ANISO=<n>     every sampler's anisotropy is capped to n (1 = none). Visible quality change.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace vpgpu::ab {

// 0 = off; the value starts with '1': 1; starts with '2': 2.
inline unsigned parseVsFastMath(const char* v) { return v && v[0] == '1' ? 1u : v && v[0] == '2' ? 2u : 0u; }
// 0 = off (unset, not a number, 0); else the cap, at most 16.
inline unsigned parseMaxAniso(const char* v) {
  if (!v || v[0] < '0' || v[0] > '9') return 0;
  const unsigned long n = std::strtoul(v, nullptr, 10);
  return n == 0 ? 0u : n > 16 ? 16u : unsigned(n);
}
inline uint8_t capAniso(uint8_t aniso, unsigned cap) { return cap && aniso > cap ? uint8_t(cap) : aniso; }

// Read at each call (a new shader library, a new sampler), not once: the variable may be set while the process runs.
inline unsigned vsFastMath() { return parseVsFastMath(std::getenv("VPKENGINE_VS_FASTMATH")); }
inline unsigned maxAniso() { return parseMaxAniso(std::getenv("VPKENGINE_MAX_ANISO")); }

}  // namespace vpgpu::ab
