#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "r3d/core/image.hpp"

namespace r3d {

// Image sampling at real-valued positions. Pixel convention: integer
// coordinates are pixel centres; the image covers [-0.5, w - 0.5] x [-0.5, h - 0.5].
// Taps outside the image are clamped to the border pixel.

[[nodiscard]] inline bool inside_image(int width, int height, double x, double y) noexcept {
  return x >= -0.5 && y >= -0.5 && x <= width - 0.5 && y <= height - 0.5;
}

// Catmull-Rom (Keys, a = -0.5) weights for the taps at -1, 0, 1, 2 relative
// to floor(x), for the fractional part t in [0, 1).
[[nodiscard]] inline std::array<float, 4> cubic_weights(float t) noexcept {
  const float t2 = t * t;
  const float t3 = t2 * t;
  return {-0.5F * t3 + t2 - 0.5F * t, 1.5F * t3 - 2.5F * t2 + 1.0F, -1.5F * t3 + 2.0F * t2 + 0.5F * t,
          0.5F * t3 - 0.5F * t2};
}

// Samples all channels at (x, y) into out[0 .. channels).
template <typename T>
void sample_bicubic(const ImageView<const T>& img, double x, double y, float* out) noexcept {
  const double fx = std::floor(x);
  const double fy = std::floor(y);
  const std::array<float, 4> wx = cubic_weights(static_cast<float>(x - fx));
  const std::array<float, 4> wy = cubic_weights(static_cast<float>(y - fy));
  const int x0 = static_cast<int>(fx) - 1;
  const int y0 = static_cast<int>(fy) - 1;
  const int channels = img.channels();
  std::array<int, 4> xs{};
  for (int i = 0; i < 4; ++i) {
    xs[static_cast<std::size_t>(i)] = std::clamp(x0 + i, 0, img.width() - 1) * channels;
  }
  for (int c = 0; c < channels; ++c) {
    out[c] = 0.0F;
  }
  for (int j = 0; j < 4; ++j) {
    const T* row = img.row(std::clamp(y0 + j, 0, img.height() - 1));
    const float wyj = wy[static_cast<std::size_t>(j)];
    for (int c = 0; c < channels; ++c) {
      float acc = 0.0F;
      for (int i = 0; i < 4; ++i) {
        acc += wx[static_cast<std::size_t>(i)] * static_cast<float>(row[xs[static_cast<std::size_t>(i)] + c]);
      }
      out[c] += wyj * acc;
    }
  }
}

template <typename T>
void sample_bilinear(const ImageView<const T>& img, double x, double y, float* out) noexcept {
  const double fx = std::floor(x);
  const double fy = std::floor(y);
  const auto tx = static_cast<float>(x - fx);
  const auto ty = static_cast<float>(y - fy);
  const int channels = img.channels();
  const int xa = std::clamp(static_cast<int>(fx), 0, img.width() - 1) * channels;
  const int xb = std::clamp(static_cast<int>(fx) + 1, 0, img.width() - 1) * channels;
  const T* ra = img.row(std::clamp(static_cast<int>(fy), 0, img.height() - 1));
  const T* rb = img.row(std::clamp(static_cast<int>(fy) + 1, 0, img.height() - 1));
  for (int c = 0; c < channels; ++c) {
    const float top = static_cast<float>(ra[xa + c]) + tx * (static_cast<float>(ra[xb + c]) - static_cast<float>(ra[xa + c]));
    const float bottom =
        static_cast<float>(rb[xa + c]) + tx * (static_cast<float>(rb[xb + c]) - static_cast<float>(rb[xa + c]));
    out[c] = top + ty * (bottom - top);
  }
}

}  // namespace r3d
