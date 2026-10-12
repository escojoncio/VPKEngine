// The fetch subroutine of a PS4 vertex shader: the VS calls it with s_swappc_b64 through a pair of
// SGPRs that hold its address (user data, or a copy of user data made with s_mov).
#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace vpps4 {

// Address of the fetch subroutine `vs` calls; 0 if it calls none, or if the address is not one
// the user data gives (`userSgprs` of `userData` are set).
uint64_t fetchAddress(std::span<const uint32_t> vs, const std::array<uint32_t, 16>& userData, uint8_t userSgprs);

// Dwords of the fetch subroutine at the start of `words`, through its s_setpc_b64; 0 if `words`
// ends first or holds something that does not decode.
size_t fetchLength(std::span<const uint32_t> words);

}  // namespace vpps4
