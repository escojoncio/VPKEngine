// Canonical text form of VPIR. print(parse(text)) == text for any canonically printed module.
//
//   module fragment
//   input uv : vec2<f32> location(0) linear centroid
//   output color : vec4<f32> location(0)
//   texture tex : 2d f32 index(0)
//   sampler smp : index(0)
//
//   func main() {
//     var $0 : f32
//     %0 = load_input uv : vec2<f32>
//     %1 = image_sample tex, smp, %0, bias(%2) : vec4<f32>
//     if %3 {
//       discard
//     } else {
//     }
//   }
//
// `;` starts a comment that runs to the end of the line (comments are not preserved).
#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "vpir/ir.h"

namespace vpir {

struct ParseResult {
    std::optional<Module> module;
    std::string error;  // "line N: message" when parsing failed
};

ParseResult parse(std::string_view text);
std::string print(const Module& module);

}  // namespace vpir
