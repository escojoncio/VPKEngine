// SPDX-License-Identifier: MIT
// Test stand-in for shadPS4's core/guest_cpu/hle_call_adapter.h: only the call frame the guest
// engine exchanges with the HLE bridge (same layout as the real one).
#pragma once
#include "guest_cpu.h"
#include <array>
#include <cstddef>
#include <cstdint>
namespace Core::GuestCpu {
using u64 = std::uint64_t;
struct HleCallFrame final {
    u64 operation{};
    std::array<u64, 16> gpr{};
    std::array<std::array<u64, 2>, 16> xmm{};
    std::uintptr_t rsp{};
    bool (*validate_range)(void* context, std::uintptr_t address, std::size_t size, bool writable){};
    void* validate_context{};
    bool (*publish_host_range)(void* context, std::uintptr_t address, std::size_t size, bool writable){};
    bool (*revoke_host_range)(void* context, std::uintptr_t address){};
    void* host_range_context{};
};
} // namespace Core::GuestCpu
