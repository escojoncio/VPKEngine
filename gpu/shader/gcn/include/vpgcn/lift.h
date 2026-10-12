// GCN 2 (PS4) shader program -> VPIR module.
//
// Execution model: one VPIR invocation per GCN lane. SGPRs, VGPRs, M0 and SCC are function-local
// variables; EXEC, VCC and every SGPR pair used as a lane mask (compare results, saved EXEC) are
// per-lane bools. A wave's scalar work is repeated by each lane, which gives the same values while
// every lane runs it; branches on EXEC/VCC become per-lane conditions, so a lane leaves a region
// its wave would still run while other lanes are active (scalar work done there by GCN is not
// done for that lane: compilers do not keep scalar state alive across such regions in practice).
// VGPR writes are predicated by EXEC once the program writes EXEC; exports and stores run only
// where EXEC is set.
//
// Control flow: an acyclic CFG whose branches rejoin at their immediate post-dominator becomes
// nested `if`s; anything else (loops, irreducible flow) becomes one dispatch loop over the basic
// blocks, which is always correct.
//
// Resources: descriptors (V#, T#, S#) are traced back to where they come from: user data SGPRs,
// or a scalar load at a constant offset from a pointer held in user data SGPRs. Each distinct
// descriptor becomes a VPIR resource slot and a Binding telling the client where to read it.
// Buffer contents are addressed as bytes from the descriptor's base address (the client binds the
// slot at that address); the descriptor words themselves are also available to the shader, read
// from the user data constant buffer, so strides and sizes come from the real descriptor.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "vpir/ir.h"

namespace vpgcn {

enum class ShaderStage : uint8_t { Vertex, Pixel, Compute };

// How a compressed (16-bit per component) colour export is packed: SPI_SHADER_COL_FORMAT.
enum class CompressedExport : uint8_t { Fp16, Unorm16, Snorm16, Uint16, Sint16 };

// Where a descriptor's dwords are read from, at draw time. UserData: user data SGPRs from `sgpr`.
// Pointer: start at the 64-bit address in user data SGPRs sgpr/sgpr+1; every element of `path` but
// the last reads the next 64-bit address at that byte offset; the last is the byte offset of the
// descriptor (or of the memory, for Binding::dwords == 0) from the final address.
// Code: the program's own image at byte offset path[0] from its first instruction (constant tables
// compilers place after the code and address through s_getpc_b64).
struct DescriptorSource {
    // Gds: not a descriptor - the GPU's global data share (64 KiB), a buffer of the client's own.
    enum Kind : uint8_t { UserData, Pointer, Code, Gds } kind = UserData;
    uint16_t sgpr = 0;
    std::vector<uint32_t> path;
    bool operator==(const DescriptorSource&) const = default;
    bool operator<(const DescriptorSource& o) const {
        return std::tie(kind, sgpr, path) < std::tie(o.kind, o.sgpr, o.path);
    }
};

// What the client knows about a descriptor when it builds the pipeline (it reads the real one).
// Fields not known are left at their defaults.
struct DescriptorInfo {
    // T#: dimension of the image (2D when unknown), its texel component type, depth format.
    vpir::TexDim dim = vpir::TexDim::D2;
    vpir::Scalar elem = vpir::Scalar::F32;
    bool depth = false;
    // V#: data and number format of typed (format) buffer access; 0 = take the instruction's.
    uint8_t dfmt = 0, nfmt = 0;
};

struct Binding {
    vpir::ResourceKind kind = vpir::ResourceKind::Buffer;
    uint32_t slot = 0;          // VPIR resource index (Module::resources)
    DescriptorSource source;
    uint8_t dwords = 4;         // descriptor size: V#/S# 4, T# 8 (4 with r128)
};

struct LiftInput {
    ShaderStage stage = ShaderStage::Pixel;
    std::span<const uint32_t> code;    // program, starting at its entry
    std::span<const uint32_t> fetch;   // vertex: the fetch subroutine called with s_swappc_b64
    unsigned userSgprs = 0;            // SPI_SHADER_PGM_RSRC2.USER_SGPR / COMPUTE_PGM_RSRC2.USER_SGPR
    // Pixel: SPI_PS_INPUT_ADDR (the input VGPR layout, in the hardware order; the enabled inputs
    // are a subset of it).
    uint32_t psInputAddr = 0x2;        // default: perspective centre barycentrics only
    // Vertex: VGPR_COMP_CNT (v0 vertex index; 3 adds v1-v3 with v3 the instance id).
    unsigned vgprCompCnt = 3;
    // VS: VGT_INSTANCE_STEP_RATE_0/1, the divisors of the instance id in v1 / v2 (0: unknown, the
    // instance id itself); v3 is the instance id.
    uint32_t instanceStepRate[2] = {0, 0};
    // Compute: COMPUTE_PGM_RSRC2 TGID_X/Y/Z_EN and TG_SIZE_EN, TIDIG_COMP_CNT, threadgroup size.
    bool tgidEnable[3] = {true, true, true};
    bool tgSizeEnable = false;
    unsigned tidigCompCnt = 2;
    uint32_t workgroupSize[3] = {64, 1, 1};
    uint32_t ldsBytes = 0;             // compute: LDS the program uses (0 = none)
    // Pixel: SPI_PS_INPUT_CNTL_0..31 per attribute the program interpolates: OFFSET (bits 0-5) is the
    // vertex shader parameter it reads (bit 5 set: the constant DEFAULT_VAL (bits 8-9: 0000, 0001,
    // 1110, 1111) instead of a parameter), FLAT_SHADE
    // (bit 10) interpolates it flat. Empty: attribute n reads parameter n, interpolated as the code says.
    std::vector<uint32_t> psInputCntl;
    // Pixel: packing of compressed exports per colour target (mrt0-7).
    CompressedExport compressedExport[8] = {};
    // Optional: what is known about a descriptor at pipeline build time.
    std::function<std::optional<DescriptorInfo>(const DescriptorSource&)> describe;
};

struct LiftResult {
    vpir::Module module;
    std::vector<Binding> bindings;
    // Slot of the user data constant buffer (the user SGPRs, 4 bytes each), if the shader reads any.
    std::optional<uint32_t> userDataSlot;
    std::vector<std::string> errors;  // the module is unusable when set
    std::vector<std::string> notes;   // approximations taken (shown in the log)
    bool structured = false;          // nested ifs (true) or the dispatch loop (false)
};

LiftResult lift(const LiftInput& input);

}  // namespace vpgcn
