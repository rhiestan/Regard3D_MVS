// Compiled once per SIMD target via Highway's foreach_target mechanism.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "r3d/core/color_convert.cpp"
#include <hwy/foreach_target.h>  // IWYU pragma: keep

#include <hwy/highway.h>

#include "r3d/core/color_convert.hpp"

HWY_BEFORE_NAMESPACE();
namespace r3d {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Weighted sum of three u8 planes, computed in u16 lanes. Max intermediate:
// 255 * 256 + 128 < 65536, so no overflow.
template <class D16, class V8>
HWY_INLINE hn::Vec<hn::Rebind<std::uint8_t, D16>> Luma(D16 d16, V8 r, V8 g, V8 b) {
  const hn::Rebind<std::uint8_t, D16> d8;
  auto sum = hn::Mul(hn::PromoteTo(d16, r), hn::Set(d16, static_cast<std::uint16_t>(kLumaR)));
  sum = hn::MulAdd(hn::PromoteTo(d16, g), hn::Set(d16, static_cast<std::uint16_t>(kLumaG)), sum);
  sum = hn::MulAdd(hn::PromoteTo(d16, b), hn::Set(d16, static_cast<std::uint16_t>(kLumaB)), sum);
  sum = hn::Add(sum, hn::Set(d16, std::uint16_t{128}));
  return hn::DemoteTo(d8, hn::ShiftRight<8>(sum));
}

void RgbToGrayRow(const std::uint8_t* HWY_RESTRICT rgb, std::uint8_t* HWY_RESTRICT gray, std::size_t n) {
  const hn::ScalableTag<std::uint16_t> d16;
  const hn::Rebind<std::uint8_t, decltype(d16)> d8;
  const std::size_t lanes = hn::Lanes(d16);
  std::size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    hn::Vec<decltype(d8)> r, g, b;
    hn::LoadInterleaved3(d8, rgb + 3 * i, r, g, b);
    hn::StoreU(Luma(d16, r, g, b), d8, gray + i);
  }
  scalar::rgb_to_gray_row(rgb + 3 * i, gray + i, n - i);
}

void RgbaToGrayRow(const std::uint8_t* HWY_RESTRICT rgba, std::uint8_t* HWY_RESTRICT gray, std::size_t n) {
  const hn::ScalableTag<std::uint16_t> d16;
  const hn::Rebind<std::uint8_t, decltype(d16)> d8;
  const std::size_t lanes = hn::Lanes(d16);
  std::size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    hn::Vec<decltype(d8)> r, g, b, a;
    hn::LoadInterleaved4(d8, rgba + 4 * i, r, g, b, a);
    hn::StoreU(Luma(d16, r, g, b), d8, gray + i);
  }
  scalar::rgba_to_gray_row(rgba + 4 * i, gray + i, n - i);
}

}  // namespace HWY_NAMESPACE
}  // namespace r3d
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

#include <cstring>

#include "r3d/core/thread_pool.hpp"

namespace r3d {

HWY_EXPORT(RgbToGrayRow);
HWY_EXPORT(RgbaToGrayRow);

namespace scalar {

void rgb_to_gray_row(const std::uint8_t* rgb, std::uint8_t* gray, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t sum = kLumaR * rgb[3 * i] + kLumaG * rgb[3 * i + 1] + kLumaB * rgb[3 * i + 2] + 128;
    gray[i] = static_cast<std::uint8_t>(sum >> 8);
  }
}

void rgba_to_gray_row(const std::uint8_t* rgba, std::uint8_t* gray, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t sum = kLumaR * rgba[4 * i] + kLumaG * rgba[4 * i + 1] + kLumaB * rgba[4 * i + 2] + 128;
    gray[i] = static_cast<std::uint8_t>(sum >> 8);
  }
}

}  // namespace scalar

void rgb_to_gray_row(const std::uint8_t* rgb, std::uint8_t* gray, std::size_t n) {
  HWY_DYNAMIC_DISPATCH(RgbToGrayRow)(rgb, gray, n);
}

void rgba_to_gray_row(const std::uint8_t* rgba, std::uint8_t* gray, std::size_t n) {
  HWY_DYNAMIC_DISPATCH(RgbaToGrayRow)(rgba, gray, n);
}

ImageU8 to_gray(ImageView<const std::uint8_t> src) {
  const int channels = src.channels();
  if (channels != 1 && channels != 3 && channels != 4) {
    throw_error("to_gray: unsupported channel count {}", channels);
  }
  ImageU8 dst = ImageU8::uninitialized(src.width(), src.height(), 1);
  const auto width = static_cast<std::size_t>(src.width());
  parallel_for(0, static_cast<std::size_t>(src.height()), 16, [&](std::size_t y0, std::size_t y1) {
    for (std::size_t y = y0; y < y1; ++y) {
      const std::uint8_t* in = src.row(static_cast<int>(y));
      std::uint8_t* out = dst.row(static_cast<int>(y));
      if (channels == 1) {
        std::memcpy(out, in, width);
      } else if (channels == 3) {
        rgb_to_gray_row(in, out, width);
      } else {
        rgba_to_gray_row(in, out, width);
      }
    }
  });
  return dst;
}

}  // namespace r3d

#endif  // HWY_ONCE
