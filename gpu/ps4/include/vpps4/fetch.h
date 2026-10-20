// The fetch subroutine of a PS4 vertex shader: the VS calls it with s_swappc_b64 through a pair of
// SGPRs that hold its address (user data, or a copy of user data made with s_mov).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace vpps4 {

// Address of the fetch subroutine `vs` calls; 0 if it calls none, or if the address is not one
// the user data gives (`userSgprs` of `userData` are set).
uint64_t fetchAddress(std::span<const uint32_t> vs, const std::array<uint32_t, 16>& userData, uint8_t userSgprs);

// Dwords of the fetch subroutine at the start of `words`, through its s_setpc_b64; 0 if `words`
// ends first or holds something that does not decode.
size_t fetchLength(std::span<const uint32_t> words);

// fetchAddress() per draw decodes the vertex shader up to its call; the walk depends on the code
// and on which SGPRs it copies, and only the final value on the user data. This remembers, per
// vertex shader (by the address of its code, checked against the words it walked), where the pair
// comes from, so a draw costs a compare of those words and two loads. Not thread-safe.
class FetchAddressMemo {
public:
    uint64_t address(std::span<const uint32_t> vs, const std::array<uint32_t, 16>& userData, uint8_t userSgprs);

private:
    struct Source {
        enum Kind : uint8_t { Unknown, User, Const } kind = Unknown;
        uint32_t value = 0;  // user data index, or the constant
    };
    struct Entry {
        const uint32_t* code = nullptr;
        uint8_t userSgprs = 0;
        bool valid = false;       // the walk reached the call and knows the pair
        Source lo, hi;
        std::vector<uint32_t> words;  // what the walk looked at
    };
    static constexpr size_t kEntries = 64;
    std::array<Entry, kEntries> entries_{};
};

}  // namespace vpps4
