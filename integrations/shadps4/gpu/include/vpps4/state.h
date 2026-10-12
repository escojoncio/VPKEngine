// What a PS4 draw or dispatch asks of the GPU, at the level of Liverpool's registers (field
// values as the hardware encodes them), filled by the emulator's adapter from its register file.
// Nothing here depends on the emulator; vpps4::Renderer turns it into gpu/core work.
#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace vpps4 {

// A colour target (CB_COLORn_*). format 0 = slot unused.
struct ColorTarget {
    uint64_t address = 0;          // byte address of slice 0, mip 0
    uint32_t width = 0, height = 0;
    uint32_t pitch = 0;            // texels per row (as allocated)
    uint8_t format = 0;            // CB_COLORn_INFO.FORMAT (COLOR_8_8_8_8 = 10, ...)
    uint8_t numberType = 0;        // CB_COLORn_INFO.NUMBER_TYPE (0 unorm ... 6 srgb, 7 float)
    uint8_t compSwap = 0;          // CB_COLORn_INFO.COMP_SWAP (0 std, 1 alt = BGRA order, ...)
    uint8_t tileMode = 0;          // CB_COLORn_ATTRIB.TILE_MODE_INDEX (0: linear general)
    uint16_t sliceStart = 0, sliceCount = 1;
    uint8_t samples = 1;
    uint8_t writeMask = 0xF;       // CB_TARGET_MASK nibble (bit 0 red ... 3 alpha)
};

// The depth/stencil target (DB_*). zFormat 0 and stencilFormat 0 = none.
struct DepthTarget {
    uint64_t zAddress = 0, stencilAddress = 0;
    uint32_t width = 0, height = 0;
    uint8_t zFormat = 0;           // DB_Z_INFO.FORMAT: 1 Z_16, 2 Z_24, 3 Z_32_FLOAT
    uint8_t stencilFormat = 0;     // DB_STENCIL_INFO.FORMAT: 1 STENCIL_8
    uint8_t tileMode = 0;
    uint16_t sliceStart = 0, sliceCount = 1;
    uint8_t samples = 1;
    // DB_RENDER_CONTROL depth/stencil clear: the draw writes the clear values over its area.
    bool depthClear = false, stencilClear = false;
    float clearDepth = 1.0f;       // DB_DEPTH_CLEAR
    uint8_t clearStencil = 0;      // DB_STENCIL_CLEAR
};

// CB_BLENDn_CONTROL fields (BLEND_* factor and COMB_* function encodings).
struct Blend {
    bool enable = false, separateAlpha = false;
    uint8_t colorSrc = 1, colorDst = 0, colorFunc = 0;  // BLEND_ONE, BLEND_ZERO, COMB_DST_PLUS_SRC
    uint8_t alphaSrc = 1, alphaDst = 0, alphaFunc = 0;
};

// DB_DEPTH_CONTROL and DB_STENCIL*.
struct DepthStencil {
    bool depthTest = false, depthWrite = false;
    uint8_t depthFunc = 7;         // FRAG_NEVER 0, LESS 1, EQUAL 2, LEQUAL 3, GREATER 4, NOTEQUAL 5, GEQUAL 6, ALWAYS 7
    bool stencilTest = false, backfaceEnable = false;
    struct Face {
        uint8_t func = 7;          // same encoding as depthFunc
        uint8_t fail = 0, zPass = 0, zFail = 0;  // STENCIL_KEEP 0, ZERO 1, ONES 2, REPLACE_TEST 3, ...
        uint8_t ref = 0, readMask = 0xFF, writeMask = 0xFF;
    } front, back;
};

// PA_CL_VPORT_* (screen = ndc * scale + offset) and the screen scissor that bounds it.
struct Viewport {
    float xScale = 0, xOffset = 0, yScale = 0, yOffset = 0, zScale = 1, zOffset = 0;
};
struct Scissor {
    int32_t left = 0, top = 0, right = 0, bottom = 0;  // exclusive right/bottom; right <= left: none
};

// PA_SU_SC_MODE_CNTL.
struct Raster {
    bool cullFront = false, cullBack = false;
    bool frontFaceClockwise = false;  // FACE: 0 CCW is front, 1 CW is front
};

// VGT_PRIMITIVE_TYPE.DI_PT_* values.
enum class PrimitiveType : uint8_t {
    PointList = 1, LineList = 2, LineStrip = 3, TriangleList = 4, TriangleFan = 5, TriangleStrip = 6,
    RectList = 0x11, LineLoop = 0x12, QuadList = 0x13, QuadStrip = 0x14, Polygon = 0x15,
};

// One shader stage: its program and what the SPI registers say about its inputs.
struct Stage {
    std::span<const uint32_t> code;        // the program (from its start; constant tables follow it)
    std::span<const uint32_t> fetch;       // VS: the fetch subroutine its s_swappc calls (empty: none)
    uint64_t address = 0;                  // where `code` lives in guest memory
    std::array<uint32_t, 16> userData{};   // SPI_SHADER_USER_DATA_*_0..15
    uint8_t userSgprs = 0;                 // SPI_SHADER_PGM_RSRC2.USER_SGPR
    uint8_t vgprCompCnt = 0;               // VS: PGM_RSRC1.VGPR_COMP_CNT
    uint32_t psInputAddr = 0;              // PS: SPI_PS_INPUT_ADDR
    std::array<uint8_t, 8> exportFormat{}; // PS: SPI_SHADER_COL_FORMAT per target (0 zero ... 4 FP16_ABGR ...)
    std::array<uint32_t, 2> instanceStepRate{};  // VS: VGT_INSTANCE_STEP_RATE_0/1
    // CS
    std::array<uint32_t, 3> workgroupSize{1, 1, 1};  // COMPUTE_NUM_THREAD_X/Y/Z
    std::array<bool, 3> tgidEnable{true, true, true};
    uint8_t tidigCompCnt = 0;
    uint32_t ldsBytes = 0;
};

struct DrawState {
    std::array<ColorTarget, 8> color{};
    std::array<Blend, 8> blend{};
    std::array<float, 4> blendColor{};
    DepthTarget depth;
    DepthStencil depthStencil;
    Viewport viewport;
    Scissor scissor;
    Raster raster;
    PrimitiveType primitive = PrimitiveType::TriangleList;
    // Indices: 0 address = not indexed.
    uint64_t indexAddress = 0;
    bool index32 = false;
    uint32_t count = 0;            // vertices or indices
    uint32_t instances = 1;
    uint32_t firstVertex = 0;      // VGT_INDX_OFFSET / DrawIndexOffset
    Stage vs, ps;
};

struct DispatchState {
    Stage cs;
    std::array<uint32_t, 3> groups{1, 1, 1};
};

}  // namespace vpps4
