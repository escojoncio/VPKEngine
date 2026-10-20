// The geometry pipeline of Liverpool (ES -> GS -> copy shader) as VPKEngine runs it on a GPU
// without geometry shaders, and the other fixed-function primitive work done in front of Metal.
//
// Hardware model (gfx7): the ES (a vertex shader in front of the GS) runs once per vertex and
// writes its outputs to the ES->GS ring; the GS runs once per primitive (and GS instance), reads
// its input vertices' data back from the ring and emits vertices (s_sendmsg GS EMIT / CUT) whose
// data it wrote to the GS->VS ring; the copy shader (a VS) runs once per emitted vertex, reads the
// data back and exports it. Rings are accessed with glc+slc buffer instructions whose V# the
// hardware supplies: ES->GS stores use a swizzled layout, loads are linear.
//
//   ES store    bucket = (imm + voffset) / 4 (attribute * 4 + component);
//               address = es2gs_offset(SGPR) + bucket * 256 + lane * 4        (ADD_TID swizzle)
//   GS load     address = voffset + soffset + imm  with voffset = vertex offset(VGPR) * 4 and
//               soffset = bucket * 256
//   GS store    bucket = (comp * max_vert_out + emitted vertex)
//               address = gs2vs_offset(SGPR) + bucket * 256 + lane * 4
//   copy load   address = vertex_id * 4 + comp * max_vert_out * 256
//
// The emulation gives each 64-invocation compute group one wave of the ES->GS ring
// (esgsItemDwords * 256 bytes) and of the GS->VS ring (gsvsVertDwords * maxVertOut * 256 bytes),
// supplies the hardware inputs the shaders expect (es2gs_offset, the GS vertex offsets in VGPRs,
// gs2vs_offset, the copy shader's vertex id) and expands the GS output into an index list.
#pragma once

#include <cstdint>
#include <vector>

namespace vpgcn {

// DescriptorSource::Ring ids: buffers the client supplies for the geometry pipeline.
enum class GeomRing : uint16_t {
    Params,      // constant buffer of GeomParam words
    EsGs,        // ES -> GS ring (read/write)
    GsVs,        // GS -> VS ring (read/write)
    GsIndex,     // GS output: 32-bit indices into the GS->VS ring vertices, per invocation
    InputIndex,  // the draw's index buffer (16 or 32 bit), from its first index
    SlotTable,   // CPU-built primitive -> ES slot table (primitives without a closed form)
    RectTable,   // RectList: the source vertex ids of every rectangle's three vertices
};

// Words of the Params constant buffer.
namespace geomparam {
enum : uint32_t {
    EsCount,      // ES invocations: instances * entries
    Entries,      // index entries (or vertices) per instance
    FirstVertex,  // VGT_INDX_OFFSET / first vertex, added to every vertex id
    Indexed,      // 0 not indexed, 1 16-bit indices, 2 32-bit indices
    Prims,        // GS input primitives per instance
    GsCount,      // GS invocations: instances * prims * gsInstances
    GsInstances,  // VGT_GS_INSTANCE_CNT (1 if disabled)
    EsWaveBytes,  // bytes of ES->GS ring per 64 ES invocations (esgsItemDwords * 256)
    GsWaveBytes,  // bytes of GS->VS ring per 64 GS invocations (gsvsVertDwords * maxVertOut * 256)
    Instances,    // draw instances
    Words = 16
};
}

// VGT_PRIMITIVE_TYPE values a GS takes as input (the numbering of DI_PT_*).
namespace inprim {
enum : uint32_t {
    PointList = 1, LineList = 2, LineStrip = 3, TriList = 4, TriFan = 5, TriStrip = 6,
    LineListAdj = 10, LineStripAdj = 11, TriListAdj = 12, TriStripAdj = 13, Polygon = 21,
};
}

// VGT_GS_OUT_PRIM_TYPE.
namespace outprim {
enum : uint32_t { PointList = 0, LineStrip = 1, TriStrip = 2 };
}

// Vertices a GS invocation sees per input primitive.
inline unsigned inputVertices(uint32_t prim) {
    switch (prim) {
        case inprim::PointList: return 1;
        case inprim::LineList:
        case inprim::LineStrip: return 2;
        case inprim::LineListAdj:
        case inprim::LineStripAdj: return 4;
        case inprim::TriListAdj:
        case inprim::TriStripAdj: return 6;
        default: return 3;
    }
}

// Input primitives made of `entries` vertices (or indices).
inline uint32_t inputPrimitives(uint32_t prim, uint32_t n) {
    switch (prim) {
        case inprim::PointList: return n;
        case inprim::LineList: return n / 2;
        case inprim::LineStrip: return n >= 2 ? n - 1 : 0;
        case inprim::TriList: return n / 3;
        case inprim::TriFan:
        case inprim::Polygon:
        case inprim::TriStrip: return n >= 3 ? n - 2 : 0;
        case inprim::LineListAdj: return n / 4;
        case inprim::LineStripAdj: return n >= 4 ? n - 3 : 0;
        case inprim::TriListAdj: return n / 6;
        case inprim::TriStripAdj: return n >= 6 ? (n - 4) / 2 : 0;
        default: return 0;
    }
}

// Whether the primitive's vertex slots have a closed form in the kernel (else SlotTable).
inline bool slotsInKernel(uint32_t prim) { return prim != inprim::TriStripAdj; }

// Entry `slot` (position in the draw's vertex/index list) of vertex j of primitive p. For the
// strip kinds this is the order the hardware hands to the GS: odd triangles of a strip are
// (1, 0, 2), strips with adjacency follow the GL/D3D rules.
inline uint32_t primitiveSlot(uint32_t prim, uint32_t n, uint32_t p, uint32_t j) {
    switch (prim) {
        case inprim::PointList: return p;
        case inprim::LineList: return 2 * p + j;
        case inprim::LineStrip: return p + j;
        case inprim::TriList: return 3 * p + j;
        case inprim::TriStrip: return j == 0 ? p + (p & 1) : j == 1 ? p + 1 - (p & 1) : p + 2;
        case inprim::TriFan:
        case inprim::Polygon: return j == 0 ? 0 : p + j;
        case inprim::LineListAdj: return 4 * p + j;
        case inprim::LineStripAdj: return p + j;
        case inprim::TriListAdj: return 6 * p + j;
        case inprim::TriStripAdj: {
            // GL 4.x "triangle strips with adjacency": [v0 adj01 v1 adj12 v2 adj20] of triangle p.
            const uint32_t last = (n - 4) / 2 - 1;  // index of the last primitive
            uint32_t t[6];
            if (p == 0) {
                const uint32_t a[6] = {0, 1, 2, last == 0 ? 3u : 6u, 4, last == 0 ? 5u : 3u};
                for (int i = 0; i < 6; ++i) t[i] = a[i];
            } else if (p & 1) {
                const uint32_t a[6] = {2 * p + 2, 2 * p - 2, 2 * p, 2 * p + 3, 2 * p + 4, p == last ? 2 * p + 5 : 2 * p + 6};
                for (int i = 0; i < 6; ++i) t[i] = a[i];
            } else {
                const uint32_t a[6] = {2 * p, 2 * p - 2, 2 * p + 2, p == last ? 2 * p + 5 : 2 * p + 6, 2 * p + 4, 2 * p + 3};
                for (int i = 0; i < 6; ++i) t[i] = a[i];
            }
            return t[j];
        }
        default: return 3 * p + j;
    }
}

// Index slots a GS invocation reserves in its output list, for its output topology.
inline uint32_t outputIndexCapacity(uint32_t outPrim, uint32_t maxVert) {
    switch (outPrim) {
        case outprim::PointList: return maxVert;
        case outprim::LineStrip: return maxVert >= 2 ? (maxVert - 1) * 2 : 0;
        default: return maxVert >= 3 ? (maxVert - 2) * 3 : 0;
    }
}

}  // namespace vpgcn
