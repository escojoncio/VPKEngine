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
    bool forceDstAlpha1 = false;   // CB_COLORn_ATTRIB.FORCE_DST_ALPHA_1: the destination reads alpha 1
    bool blendClamp = true;        // CB_COLORn_INFO.BLEND_CLAMP
    bool blendBypass = false;      // CB_COLORn_INFO.BLEND_BYPASS: blending off whatever CB_BLENDn says
    // samples = colour fragments (NUM_FRAGMENTS: what the surface stores); coverage = NUM_SAMPLES
    // (EQAA: may exceed it; 0 = same as samples). The host draws with at most `samples`.
    uint8_t coverage = 0;
    // CB_COLORn_INFO.FAST_CLEAR: a rectangle draw with it set clears the target to the colour in
    // CB_COLORn_CLEAR_WORD0/1 instead of drawing (the hardware writes CMASK only). The words hold the
    // colour in the target's own format (see fastClearColor in formats.h).
    bool fastClear = false;
    std::array<uint32_t, 2> clearWord{};
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
    // DB_Z_WRITE_BASE / DB_STENCIL_WRITE_BASE (0: same as the read base). A draw tests against the
    // read surface and writes the write one.
    uint64_t zWriteAddress = 0, stencilWriteAddress = 0;
    bool zReadOnly = false, stencilReadOnly = false;  // DB_DEPTH_VIEW.Z_READ_ONLY / STENCIL_READ_ONLY
    // DB_RENDER_OVERRIDE FORCE_Z_DIRTY / FORCE_Z_VALID (and stencil): with colour writes off, the
    // draw copies the read surface into the write one.
    bool forceZDirty = false, forceZValid = false, forceStencilDirty = false, forceStencilValid = false;
    // DB_DEPTH_BOUNDS_MIN/MAX (tested when DepthStencil::depthBounds).
    float boundsMin = 0.0f, boundsMax = 1.0f;
    // DB_RENDER_CONTROL.DEPTH_COPY / COPY_SAMPLE: the hardware copies the depth of one sample (a
    // depth resolve done by a draw). Without it the host resolves nearest-sample (msaa.h).
    bool depthCopy = false;
    uint8_t copySample = 0;
};

// CB_BLENDn_CONTROL fields (BLEND_* factor and COMB_* function encodings).
struct Blend {
    bool enable = false, separateAlpha = false;
    uint8_t colorSrc = 1, colorDst = 0, colorFunc = 0;  // BLEND_ONE, BLEND_ZERO, COMB_DST_PLUS_SRC
    uint8_t alphaSrc = 1, alphaDst = 0, alphaFunc = 0;
    bool disableRop3 = false;      // CB_BLENDn_CONTROL.DISABLE_ROP3 (blending replaces the logic op)
};

// DB_DEPTH_CONTROL and DB_STENCIL*.
struct DepthStencil {
    bool depthTest = false, depthWrite = false;
    uint8_t depthFunc = 7;         // FRAG_NEVER 0, LESS 1, EQUAL 2, LEQUAL 3, GREATER 4, NOTEQUAL 5, GEQUAL 6, ALWAYS 7
    bool stencilTest = false, backfaceEnable = false;
    bool depthBounds = false;      // DB_DEPTH_CONTROL.DEPTH_BOUNDS_ENABLE
    bool colorWritesOnDepthFail = false;   // ENABLE_COLOR_WRITES_ON_DEPTH_FAIL
    bool noColorWritesOnDepthPass = false; // DISABLE_COLOR_WRITES_ON_DEPTH_PASS
    struct Face {
        uint8_t func = 7;          // same encoding as depthFunc
        uint8_t fail = 0, zPass = 0, zFail = 0;  // STENCIL_KEEP 0, ZERO 1, ONES 2, REPLACE_TEST 3, REPLACE_OP 4, ...
        uint8_t ref = 0, readMask = 0xFF, writeMask = 0xFF;
        uint8_t opVal = 0;         // DB_STENCILREFMASK STENCILOPVAL (the value REPLACE_OP writes)
    } front, back;
};

// PA_CL_VPORT_* (screen = ndc * scale + offset) and the screen scissor that bounds it.
struct Viewport {
    float xScale = 0, xOffset = 0, yScale = 0, yOffset = 0, zScale = 1, zOffset = 0;
};
struct Scissor {
    int32_t left = 0, top = 0, right = 0, bottom = 0;  // exclusive right/bottom; right <= left: none
};

// The scissor registers of a draw (PA_SC_SCREEN/WINDOW/GENERIC/VPORT_SCISSOR, window offset); the
// viewport and clip registers that bound it. combineScissor() (fixed_function.h) reduces them to
// one Scissor.
struct ScissorRegs {
    Scissor screen, window, generic, viewportScissor;
    int32_t windowOffsetX = 0, windowOffsetY = 0;
    bool windowOffsetDisable = true;   // PA_SC_WINDOW_SCISSOR_TL.WINDOW_OFFSET_DISABLE
    bool viewportScissorEnable = false;  // PA_SC_MODE_CNTL_0.VPORT_SCISSOR_ENABLE
    bool present = false;              // false: only DrawState::scissor (the adapter did the work)
};

// PA_SU_SC_MODE_CNTL, PA_SU_POLY_OFFSET_*, PA_CL_CLIP_CNTL, PA_CL_VTE_CNTL, PA_SU_LINE_CNTL.
struct Raster {
    bool cullFront = false, cullBack = false;
    bool frontFaceClockwise = false;  // FACE: 0 CCW is front, 1 CW is front
    // Polygon offset (shadow bias). Scales and offsets as the registers hold them: offset in units
    // of the depth format's resolution, scale applied to the slope divided by 16.
    bool offsetFront = false, offsetBack = false, offsetPara = false;
    float offsetClamp = 0, frontScale = 0, frontOffset = 0, backScale = 0, backOffset = 0;
    // Clipping.
    bool clipDisable = false;           // CLIP_DISABLE
    bool zclipNearDisable = false, zclipFarDisable = false;
    bool clipSpaceZeroToW = true;       // DX_CLIP_SPACE_DEF: z in 0..w (false: -w..w, GL)
    uint8_t userClipPlanes = 0;         // USER_CLIP_PLANE_ENABLE mask
    bool vsOutputsViewportIndex = false;  // PA_CL_VS_OUT_CNTL.USE_VTX_VIEWPORT_INDX
    bool vsOutputsClipDistances = false;
    // PA_SU_SC_MODE_CNTL polygon mode: 0 points, 1 lines, 2 fill.
    bool polygonModeEnable = false;
    uint8_t polygonModeFront = 2, polygonModeBack = 2;
    bool provokingLast = false;         // PROVOKING_VTX_LAST
    float lineWidth = 1.0f;             // PA_SU_LINE_CNTL.WIDTH in pixels
    // VGT_MULTI_PRIM_IB_RESET_EN / VGT_MULTI_PRIM_IB_RESET_INDX.
    bool primitiveRestart = false;
    uint32_t restartIndex = 0xFFFFFFFF;
};

// VGT_PRIMITIVE_TYPE.DI_PT_* values.
enum class PrimitiveType : uint8_t {
    PointList = 1, LineList = 2, LineStrip = 3, TriangleList = 4, TriangleFan = 5, TriangleStrip = 6,
    Patch = 9,  // tessellation: the HS/DS stages are not supported (the draw is left out)
    // With adjacency: meaningful as geometry shader input; without a GS the adjacency vertices are dropped.
    LineListAdj = 10, LineStripAdj = 11, TriangleListAdj = 12, TriangleStripAdj = 13,
    RectList = 0x11, LineLoop = 0x12, QuadList = 0x13, QuadStrip = 0x14, Polygon = 0x15,
};

// The geometry pipeline (ES -> GS -> copy shader, see vpgcn/geom.h). When enabled, DrawState::es and
// ::gs hold the two stages and DrawState::vs is the GS's copy shader (the program in the VS registers).
struct GeometryState {
    bool enabled = false;          // VGT_GS_MODE.MODE != off, with a GS and an ES bound
    uint32_t esgsItemDwords = 0;   // VGT_ESGS_RING_ITEMSIZE
    uint32_t gsvsVertDwords = 0;   // VGT_GS_VERT_ITEMSIZE (stream 0)
    uint32_t maxVertOut = 0;       // VGT_GS_MAX_VERT_OUT
    uint32_t outPrim = 2;          // VGT_GS_OUT_PRIM_TYPE: 0 points, 1 line strip, 2 triangle strip
    uint32_t gsInstances = 1;      // VGT_GS_INSTANCE_CNT (1 when disabled)
    bool operator==(const GeometryState&) const = default;
};

// One shader stage: its program and what the SPI registers say about its inputs.
struct Stage {
    std::span<const uint32_t> code;        // the program (from its start; constant tables follow it)
    std::span<const uint32_t> fetch;       // VS: the fetch subroutine its s_swappc calls (empty: none)
    uint64_t address = 0;                  // where `code` lives in guest memory
    std::array<uint32_t, 16> userData{};   // SPI_SHADER_USER_DATA_*_0..15
    uint8_t userSgprs = 0;                 // SPI_SHADER_PGM_RSRC2.USER_SGPR
    uint8_t vgprCompCnt = 0;               // VS/ES: PGM_RSRC1.VGPR_COMP_CNT
    uint32_t vsOutCntl = 0;                // VS/copy shader: PA_CL_VS_OUT_CNTL (pos1-pos3 exports)
    GeometryState geom;                    // ES/GS/copy shader: set by the renderer from DrawState::geom
    uint32_t geomInputPrim = 0;            // (idem) the draw's primitive type, the GS's input
    bool rectList = false;                 // (idem) VS of a RectList draw
    uint32_t psInputAddr = 0;              // PS: SPI_PS_INPUT_ADDR
    std::array<uint8_t, 8> exportFormat{}; // PS: SPI_SHADER_COL_FORMAT per target (0 zero ... 4 FP16_ABGR ...)
    std::array<uint32_t, 32> psInputCntl{};  // PS: SPI_PS_INPUT_CNTL_0..31 (OFFSET bits 0-5, FLAT_SHADE bit 10)
    uint8_t psNumInterp = 0;               // PS: SPI_PS_IN_CONTROL.NUM_INTERP (entries of psInputCntl in use)
    // PS: mrt1 is the second blend source of target 0 (dual-source blending): exported as
    // [[color(0), index(1)]]. Set by the renderer from CB_BLEND0_CONTROL, not by the adapter.
    bool dualSource = false;
    // VS: PA_CL_CLIP_CNTL.DX_CLIP_SPACE_DEF is 0 (clip z in -w..w): the position export maps it to 0..w.
    bool clipSpaceGL = false;
    std::array<uint32_t, 2> instanceStepRate{};  // VS: VGT_INSTANCE_STEP_RATE_0/1
    // CS
    std::array<uint32_t, 3> workgroupSize{1, 1, 1};  // COMPUTE_NUM_THREAD_X/Y/Z
    std::array<bool, 3> tgidEnable{true, true, true};
    bool tgSizeEnable = false;             // COMPUTE_PGM_RSRC2.TG_SIZE_EN
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
    ScissorRegs scissorRegs;       // when .present, scissor is recomputed from these
    Raster raster;
    // CB_COLOR_CONTROL: ROP3 code (0xCC copy) and whether the mode runs it.
    uint8_t rop3 = 0xCC;
    bool degamma = false;          // CB_COLOR_CONTROL.DEGAMMA_ENABLE
    // DB_ALPHA_TO_MASK.ALPHA_TO_MASK_ENABLE: the pixel shader's alpha (export 0) becomes the sample coverage
    // mask (alpha to coverage on the host with several samples; a hardware-like alpha test at 0.5 with one).
    bool alphaToMask = false;
    uint8_t alphaToMaskOffsets = 0;
    bool earlyZ = true;            // Z_ORDER is EARLY_Z_THEN_LATE or EARLY_Z
    bool depthBeforeShader = false;
    bool psKill = false, psExportsZ = false, psExportsStencil = false, psExportsMask = false;
    PrimitiveType primitive = PrimitiveType::TriangleList;
    // Indices: 0 address = not indexed.
    uint64_t indexAddress = 0;
    bool index32 = false;
    uint8_t indexSwap = 0;         // VGT_DMA_INDEX_TYPE.SWAP_MODE: 0 none, 1 bytes in 16, 2 bytes in 32, 3 halves in 32
    uint32_t count = 0;            // vertices or indices
    uint32_t instances = 1;
    uint32_t firstVertex = 0;      // VGT_INDX_OFFSET / DrawIndexOffset
    // Indirect draw: the arguments are read by the GPU from here (draw: vertex count, instances,
    // first vertex, first instance; indexed: index count, instances, first index, base vertex,
    // first instance). count is then the index buffer's size in indices (indexed) or any non-zero.
    uint64_t indirectAddress = 0;
    Stage vs, ps;
    Stage es, gs;                  // geometry pipeline (geom.enabled): vs is then the copy shader
    GeometryState geom;
};

struct DispatchState {
    Stage cs;
    std::array<uint32_t, 3> groups{1, 1, 1};
    uint64_t indirectAddress = 0;  // set: the group counts (x, y, z) are read by the GPU from here
};

}  // namespace vpps4
