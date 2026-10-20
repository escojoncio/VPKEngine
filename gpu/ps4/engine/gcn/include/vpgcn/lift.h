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

#include "vp4ir/ir.h"
#include "vpgcn/geom.h"

namespace vpgcn {

// Export, Geometry: the ES and the GS of the geometry pipeline, lifted as compute kernels (one
// invocation per ES vertex / per GS primitive, see geom.h). CopyVertex: the GS's copy shader, a
// vertex shader reading the GS->VS ring.
enum class ShaderStage : uint8_t { Vertex, Pixel, Compute, Export, Geometry, CopyVertex };

// How a compressed (16-bit per component) colour export is packed: SPI_SHADER_COL_FORMAT.
enum class CompressedExport : uint8_t { Fp16, Unorm16, Snorm16, Uint16, Sint16 };

// How a pixel shader's colour export (4 values, float or integer bits) becomes the host render
// target's contents. The client picks the host format; the hardware did the rest (blending aside).
struct ExportConv {
    // Float: the target is a normalised / float host format (the host converts). Uint / Sint: a
    // per-component integer host format, values clamped to `bits`. Packed: the target is a raw
    // integer container; the shader packs the element of data / number format dfmt / nfmt into it.
    enum Kind : uint8_t { Float, Uint, Sint, Packed } kind = Float;
    // Host channel j takes export component select[j] (0..3; 4 constant zero, 5 constant one):
    // CB_COMP_SWAP and the channel order the host format needs, composed.
    uint8_t select[4] = {0, 1, 2, 3};
    bool srgb = false;             // Float: encode components x, y, z (the host format has no sRGB)
    uint8_t bits[4] = {};          // Uint / Sint: field widths
    uint8_t dfmt = 0, nfmt = 0;    // Packed
    bool identity() const {
        return kind == Float && !srgb && select[0] == 0 && select[1] == 1 && select[2] == 2 && select[3] == 3;
    }
};

// Where a descriptor's dwords are read from, at draw time. UserData: user data SGPRs from `sgpr`.
// Pointer: start at the 64-bit address in user data SGPRs sgpr/sgpr+1; every element of `path` but
// the last reads the next 64-bit address at that byte offset; the last is the byte offset of the
// descriptor (or of the memory, for Binding::dwords == 0) from the final address.
// Code: the program's own image at byte offset path[0] from its first instruction (constant tables
// compilers place after the code and address through s_getpc_b64).
// The GuestMap constant buffer: u32 count + 3 pad words, then `count` entries of 8 dwords
// { baseLo, baseHi, sizeLo, sizeHi, devLo, devHi, 0, 0 } sorted by guest base address (dev = the region's
// device address). Always bound at full size (16 + 32 * kGuestMapEntries bytes): the shader reads entries
// past `count` while it searches.
inline constexpr unsigned kGuestMapEntries = 256;
inline constexpr unsigned kGuestMapBytes = 16 + 32 * kGuestMapEntries;

struct DescriptorSource {
    // Gds: not a descriptor - the GPU's global data share (64 KiB), a buffer of the client's own.
    // Ring: a buffer of the client's own for the geometry pipeline; `sgpr` is its GeomRing id.
    // GuestMap: not a descriptor either - the table of guest memory regions FLAT instructions translate their
    // raw GPU virtual addresses through (layout at kGuestMapEntries; filled by the client at draw time).
    enum Kind : uint8_t { UserData, Pointer, Code, Gds, Ring, GuestMap } kind = UserData;
    uint16_t sgpr = 0;
    std::vector<uint32_t> path;
    bool operator==(const DescriptorSource&) const = default;
    bool operator<(const DescriptorSource& o) const {
        return std::tie(kind, sgpr, path) < std::tie(o.kind, o.sgpr, o.path);
    }
};

// How a texture result is converted after the hardware read it (or after the shader decoded it from
// a raw container): the client chose the host texture format, this is what is left to do.
// Order: affine, sRGB decode, then the component select (the T# DST_SEL composed with any swap the
// host format needs).
struct TextureConv {
    // Per output component: 0 constant zero, 1 constant one, 4..7 source component x..w (SQ_SEL_*).
    uint8_t select[4] = {4, 5, 6, 7};
    // value * scale + bias per component (float textures): scaled formats read through a normalised
    // host format, SNORM_NZ from SNORM.
    bool affine = false;
    float scale[4] = {1, 1, 1, 1}, bias[4] = {0, 0, 0, 0};
    // Linearise x, y, z (the host format has no sRGB: 8 / 16 bit, 10-bit, storage images). For a
    // storage texture it is also the encode applied to stores.
    bool srgb = false;
    // The texture is a raw integer container of the data format DescriptorInfo.dfmt / nfmt: the
    // shader fetches the element (nearest, mip 0 or the explicit lod, edge clamped) and decodes it;
    // stores encode. `texelsPerElement` texels share an element along x (GB_GR 2, 1-bit 8).
    bool manual = false;
    uint8_t texelsPerElement = 1;
    bool identity() const {
        return !affine && !srgb && !manual && select[0] == 4 && select[1] == 5 && select[2] == 6 && select[3] == 7;
    }
};

// What the client knows about a descriptor when it builds the pipeline (it reads the real one).
// Fields not known are left at their defaults.
struct DescriptorInfo {
    // T#: conversion of the results (see TextureConv).
    TextureConv conv;
    // T#: dimension of the image (2D when unknown), its texel component type, depth format.
    vp4ir::TexDim dim = vp4ir::TexDim::D2;
    vp4ir::Scalar elem = vp4ir::Scalar::F32;
    bool depth = false;
    // T#: the image is multisampled (texture2d_ms): loads take a sample index as the last address
    // coordinate. The client sets it when it binds the multisampled surface itself; a T# that
    // names an MSAA image the host draws at 1 sample stays a plain texture (the sample index is dropped).
    bool multisample = false;
    // T# of an FMASK (IMG_DATA_FORMAT 47..56): the host keeps no compression metadata, so loads and
    // samples of it return this constant instead (dword 0 and 1 of the result; see vpps4 fmaskIdentity)
    // and no resource is bound.
    bool fmask = false;
    uint32_t fmaskLo = 0, fmaskHi = 0;
    // V#: data and number format of typed (format) buffer access; 0 = take the instruction's.
    // T# with conv.manual: the data / number format of the container.
    uint8_t dfmt = 0, nfmt = 0;
};

struct Binding {
    vp4ir::ResourceKind kind = vp4ir::ResourceKind::Buffer;
    uint32_t slot = 0;          // VPIR resource index (Module::resources)
    DescriptorSource source;
    uint8_t dwords = 4;         // descriptor size: V#/S# 4, T# 8 (4 with r128)
};

// What the lifter needs of the geometry pipeline state (VGT_* registers); geom.h has the model.
struct GeometryInput {
    uint32_t esgsItemDwords = 0;   // VGT_ESGS_RING_ITEMSIZE: dwords per ES vertex
    uint32_t gsvsVertDwords = 0;   // VGT_GS_VERT_ITEMSIZE[0]: dwords per GS output vertex
    uint32_t maxVertOut = 0;       // VGT_GS_MAX_VERT_OUT
    uint32_t inputPrim = inprim::TriList;  // GS input primitive (VGT_PRIMITIVE_TYPE)
    uint32_t outPrim = outprim::TriStrip;  // VGT_GS_OUT_PRIM_TYPE (stream 0)
    uint32_t gsInstances = 1;      // VGT_GS_INSTANCE_CNT
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
    // Pixel: CB_BLEND0 reads the second source (dual-source blending): the mrt1 export becomes the
    // output [[color(0), index(1)]] instead of colour attachment 1.
    bool dualSource = false;
    // Vertex: clip z is -w..w (PA_CL_CLIP_CNTL.DX_CLIP_SPACE_DEF clear): the position export is
    // rewritten to Metal's 0..w (z' = (z + w) / 2).
    bool clipSpaceGL = false;
    // Vertex, CopyVertex: PA_CL_VS_OUT_CNTL. Which of the pos1-pos3 export components mean point
    // size, render target layer, clip distances (0: none do, the exports are ignored).
    uint32_t vsOutCntl = 0;
    // Export, Geometry, CopyVertex: the geometry pipeline's state.
    GeometryInput geom;
    // Vertex: a RectList draw. The program runs for three vertices (their ids are read from the
    // RectTable ring) and the module outputs the rectangle as two triangles: output vertex ids
    // are rect * 6 + 0..5 (the three given, then the other two corners and the derived fourth).
    bool rectList = false;
    // Pixel: packing of compressed exports per colour target (mrt0-7).
    CompressedExport compressedExport[8] = {};
    // Pixel: conversion of each colour target's export to its host format.
    ExportConv exportConv[8] = {};
    // The pass renders two layers (both eyes) through a rasterization rate map: the vertex shader's pos1
    // export of the render target array index becomes the Layer and ViewportIndex outputs (the view), and
    // the pixel shader reads FragCoord back through the rate map (Module::rateMapPosition, Layer input).
    bool viewIndexed = false;
    // Pixel: DB_ALPHA_TO_MASK with one sample. The coverage the hardware derives from the alpha of the
    // colour 0 export becomes, with a single sample, an alpha test: the fragment is discarded below 0.5
    // (with several samples the host's alpha-to-coverage does it, and this stays off).
    bool alphaTest = false;
    // The MODE register at wave start (SPI_SHADER_PGM_RSRC1/COMPUTE_PGM_RSRC1 FLOAT_MODE in bits 0-7, DX10_CLAMP
    // bit 8, IEEE bit 9). The default allows f32/f64 denormals and rounds to nearest even. FP_DENORM[5:4]
    // = 0 / 1 / 2 flush f32 denormal inputs and outputs / outputs only / inputs only (software flush in the
    // lifted code; Metal cannot preserve denormals, so "allow" is best effort). Rounding modes other than
    // nearest-even are not honoured (Metal has no rounding control): the lifter notes it.
    uint32_t modeRegister = 0x1f0;
    // Optional: what is known about a descriptor at pipeline build time.
    std::function<std::optional<DescriptorInfo>(const DescriptorSource&)> describe;
};

struct LiftResult {
    vp4ir::Module module;
    std::vector<Binding> bindings;
    // Slot of the user data constant buffer (the user SGPRs, 4 bytes each), if the shader reads any.
    std::optional<uint32_t> userDataSlot;
    std::vector<std::string> errors;  // the module is unusable when set
    std::vector<std::string> notes;   // approximations taken (shown in the log)
    bool structured = false;          // nested ifs (true) or the dispatch loop (false)
};

LiftResult lift(const LiftInput& input);

}  // namespace vpgcn
