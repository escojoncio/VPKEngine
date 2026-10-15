#include "vpgpu/core/resources.h"

#include <algorithm>
#include <bit>

namespace vpgpu {

const FormatInfo& formatInfo(Format f) {
  using enum FormatInfo::Flag;
  static constexpr FormatInfo table[] = {
#define VPGPU_X(name, bytes, bw, bh, flags) {#name, bytes, bw, bh, flags},
      VPGPU_FORMATS(VPGPU_X)
#undef VPGPU_X
  };
  static_assert(std::size(table) == static_cast<size_t>(Format::Count));
  const auto i = static_cast<size_t>(f);
  return table[i < std::size(table) ? i : 0];
}

bool validate(const TextureDesc& d) {
  const FormatInfo& fi = formatInfo(d.format);
  if (d.format == Format::Undefined || d.format >= Format::Count) return false;
  if (!d.width || !d.height || !d.depth || !d.mips || !d.layers || !d.samples) return false;
  if (!std::has_single_bit(d.samples)) return false;
  if (d.type == TextureType::Cube && (d.layers % 6 || d.width != d.height)) return false;
  if (d.type != TextureType::Tex3D && d.depth != 1) return false;
  if (d.type == TextureType::Tex1D && d.height != 1) return false;
  const uint32_t largest = std::max({d.width, d.height, d.type == TextureType::Tex3D ? d.depth : 1u});
  if (d.mips > std::bit_width(largest)) return false;
  const bool isDepth = fi.depth() || fi.stencil();
  if ((d.usage & usage::depthTarget) && !isDepth) return false;
  if ((d.usage & usage::colorTarget) && (isDepth || fi.blockSize > 1)) return false;
  if ((d.usage & usage::storage) && (isDepth || fi.blockSize > 1)) return false;
  return true;
}

}  // namespace vpgpu
