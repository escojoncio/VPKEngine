// Transform feedback, the part that needs no GPU: assembling the vertices of a draw into the primitives GL writes, and copying
// captured vertex records into the bound buffers. (The capture itself is the vertex shader variant vpglsl links.)
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vpgles {

// The vertices of the primitives a draw of `drawMode` (GL_POINTS ... GL_TRIANGLE_FAN) makes from the vertex sequence v[0..n), in the
// order GL writes them: points, lines (strips and loops as separate segments) or triangles (strip triangles keep their winding:
// odd ones swap the first two; a fan's triangle is (v[0], v[i], v[i+1]), the centre first, the provoking vertex last). `restart`: `restartValue` ends the primitive being assembled and starts a new sequence.
void xfbAssemble(uint32_t drawMode, const uint32_t* v, size_t n, bool restart, uint32_t restartValue, std::vector<uint32_t>& out);

// A buffer binding that captures (part of) every vertex record: `room` bytes at `dst`, `used` already taken, one vertex taking
// `stride` bytes made of `parts` (word offset in the record, words) in order.
struct XfbOut {
    uint8_t* dst = nullptr;
    uint64_t room = 0, used = 0;
    uint32_t stride = 0;
    std::vector<std::pair<uint32_t, uint32_t>> parts;
};

// Writes the primitives of `perPrim` vertices that `order` lists (relative vertex indices, a multiple of perPrim of them) taking
// each vertex's record (`W` words) from records + index * W. A primitive is written whole into every output or not at all, and
// the first one that does not fit ends the run. Returns the primitives written.
uint64_t xfbWrite(const std::vector<uint32_t>& order, uint32_t perPrim, const uint32_t* records, uint32_t W, std::vector<XfbOut>& outs);

}  // namespace vpgles
