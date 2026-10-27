// In-house CPU decoder of ASTC LDR 2D blocks (Khronos Data Format Specification, ASTC chapter): the compressed upload of a 3D image.
// Metal has no 3D ASTC / ETC2 / EAC textures, so such an image is created as RGBA8 (Unorm / Srgb) and its buffer->image copies are
// decoded slice by slice into RGBA8 at submit (commands.cpp). HDR endpoint modes (LDR profile) decode to the error colour (magenta).
#pragma once

#include <cstdint>

#include "vpgpu/core/resources.h"

namespace vpvk {

// Is `f` an ASTC format (any block size, UNORM or sRGB)?
bool astcFormat(vpgpu::Format f);
// The RGBA8 format an ASTC image of format `f` is created as when it is 3D (Unorm, or Srgb for the sRGB formats).
vpgpu::Format astcDecodedFormat(vpgpu::Format f);

// Decodes one 16-byte block of footprint bw x bh into bw*bh RGBA8 texels (row-major). srgb: the RGB channels are the sRGB-encoded
// 8-bit values (the texture is RGBA8_SRGB), alpha linear. Returns false for an invalid encoding or an HDR block; `rgba` then
// holds the error colour.
bool astcDecodeBlock(const uint8_t block[16], int bw, int bh, bool srgb, uint8_t* rgba);

// Test hooks: the ISE trit / quint groups (value of each digit) and the unquantised endpoint / weight for quantisation `levels`
// (2..256) and ISE value `v` (digit << bits | low bits). Weights are 0..64.
void astcTritDecode(unsigned t, int out[5]);
void astcQuintDecode(unsigned q, int out[3]);
int astcUnquantColor(int levels, int v);
int astcUnquantWeight(int levels, int v);

}  // namespace vpvk
