// Sampler Y'CbCr conversion (VK_KHR_sampler_ycbcr_conversion): the parameters a shader binding needs
// and the arithmetic that turns the sampled planes into RGBA. Header only and free of Vulkan types
// (the numeric values equal the Vulkan enums) so the SPIR-V reader emits exactly the constants the
// tests check, and the tests use the same reference function as the specification's formulas.
//
// Planes: plane 0 holds Y (G), plane 1 holds Cb (B) alone (3 planes: plane 2 holds Cr) or Cb in
// its first channel and Cr in its second (2 planes). The raw RGBA of a texel is (Cr, Y, Cb, 1).
#pragma once

#include <cstdint>

namespace spirv {

// VkSamplerYcbcrModelConversion
enum class YcbcrModel : uint8_t { RgbIdentity = 0, YcbcrIdentity = 1, Ycbcr709 = 2, Ycbcr601 = 3, Ycbcr2020 = 4 };
// VkSamplerYcbcrRange
enum class YcbcrRange : uint8_t { Full = 0, Narrow = 1 };

// Everything a shader needs about the immutable sampler with a conversion bound at a binding.
struct YcbcrBinding {
  uint8_t planes = 0;  // 0 = no conversion; 2 or 3
  YcbcrModel model = YcbcrModel::Ycbcr709;
  YcbcrRange range = YcbcrRange::Full;
  // Chroma sample positions of the subsampled axes (VkChromaLocation): cosited = at the luma
  // sample (COSITED_EVEN), else between two luma samples (MIDPOINT).
  bool cositedX = false, cositedY = false;
  bool subX = false, subY = false;  // chroma is half the luma size in x / y
  // VkComponentSwizzle of the conversion's `components` (IDENTITY 0, ZERO 1, ONE 2, R 3, G 4, B 5, A 6),
  // applied to the raw (Cr, Y, Cb, 1) before range expansion and the model matrix.
  uint8_t swizzle[4] = {0, 0, 0, 0};
  uint16_t planeSlot[2] = {0xFFFF, 0xFFFF};  // Texture slots of planes 1 and 2 (element e: slot + e)
  friend bool operator==(const YcbcrBinding&, const YcbcrBinding&) = default;
};

// Constants of the conversion for 8-bit planes.
//   Y'  = y  * yScale + yBias      (range expansion; narrow range: (255 y - 16) / 219)
//   Cx' = c  * cScale + cBias      (narrow: (255 c - 128) / 224; full: c - 128 / 255), in [-0.5, 0.5]
//   R = Y' + rCr Cr'        B = Y' + bCb Cb'        G = Y' + gCb Cb' + gCr Cr'
// `matrix` false (identity models): no R/G/B from Y'/Cb'/Cr'.
struct YcbcrCoeffs {
  float yScale = 1, yBias = 0, cScale = 1, cBias = 0;
  float rCr = 0, gCb = 0, gCr = 0, bCb = 0;
  bool matrix = false;
};

inline YcbcrCoeffs ycbcrCoeffs(YcbcrModel model, YcbcrRange range) {
  YcbcrCoeffs c;
  if (range == YcbcrRange::Narrow) {
    c.yScale = 255.0f / 219.0f, c.yBias = -16.0f / 219.0f;
    c.cScale = 255.0f / 224.0f, c.cBias = -128.0f / 224.0f;
  } else {
    c.cBias = -128.0f / 255.0f;
  }
  double kr = 0, kb = 0;
  switch (model) {
    case YcbcrModel::Ycbcr709: kr = 0.2126, kb = 0.0722; break;
    case YcbcrModel::Ycbcr601: kr = 0.299, kb = 0.114; break;
    case YcbcrModel::Ycbcr2020: kr = 0.2627, kb = 0.0593; break;
    default: return c;
  }
  const double kg = 1.0 - kr - kb;
  c.matrix = true;
  c.rCr = float(2.0 * (1.0 - kr));
  c.bCb = float(2.0 * (1.0 - kb));
  c.gCb = float(-2.0 * kb * (1.0 - kb) / kg);
  c.gCr = float(-2.0 * kr * (1.0 - kr) / kg);
  return c;
}

// RGB identity: the three channels are RGB already; narrow range expands each like luma.
inline float ycbcrClamp01(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

// Reference conversion of one reconstructed texel: raw = (Cr, Y, Cb, 1) as sampled (0..1). The
// shader lowering emits the same operations.
inline void ycbcrConvert(const YcbcrBinding& b, const float raw[4], float out[4]) {
  float s[4];
  for (int i = 0; i < 4; ++i) {
    switch (b.swizzle[i]) {
      case 1: s[i] = 0; break;
      case 2: s[i] = 1; break;
      case 3: s[i] = raw[0]; break;
      case 4: s[i] = raw[1]; break;
      case 5: s[i] = raw[2]; break;
      case 6: s[i] = raw[3]; break;
      default: s[i] = raw[i];
    }
  }
  const YcbcrCoeffs c = ycbcrCoeffs(b.model, b.range);
  if (b.model == YcbcrModel::RgbIdentity) {
    for (int i = 0; i < 3; ++i) out[i] = s[i] * c.yScale + c.yBias;
    out[3] = s[3];
    return;
  }
  const float y = s[1] * c.yScale + c.yBias;
  const float cb = s[2] * c.cScale + c.cBias, cr = s[0] * c.cScale + c.cBias;
  if (!c.matrix) {
    out[0] = cr, out[1] = y, out[2] = cb, out[3] = s[3];
    return;
  }
  out[0] = ycbcrClamp01(y + c.rCr * cr);
  out[1] = ycbcrClamp01(y + c.gCb * cb + c.gCr * cr);
  out[2] = ycbcrClamp01(y + c.bCb * cb);
  out[3] = s[3];
}

// Shift (in chroma texels, added to the coordinate) that makes a chroma plane sampled at the luma
// coordinate land on the chroma sample positions. Cosited chroma j lies on the centre of luma texel
// 2j, i.e. at (j + 1/4) chroma texels where the plane's own texel centre is j + 1/2: the plane is
// sampled 1/4 chroma texel further (2:1 subsampling). Midpoint chroma needs no shift.
inline float ycbcrChromaShift(bool subsampled, bool cosited) { return subsampled && cosited ? 0.25f : 0.0f; }

}  // namespace spirv
