#include "r3d/stereo/disparity_eval.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "r3d/core/error.hpp"

namespace r3d {

DisparityErrorStats evaluate_disparity(const ImageF32& disparity, const ImageF32& ground_truth,
                                       ImageView<const std::uint8_t> mask, int mask_min, float error_scale) {
  if (disparity.width() != ground_truth.width() || disparity.height() != ground_truth.height()) {
    throw_error("evaluate_disparity: estimate is {}x{}, ground truth {}x{}", disparity.width(), disparity.height(),
                ground_truth.width(), ground_truth.height());
  }
  if (!mask.empty() && (mask.width() != disparity.width() || mask.height() != disparity.height())) {
    throw_error("evaluate_disparity: mask size differs from the estimate");
  }
  DisparityErrorStats s;
  std::size_t estimated = 0;
  std::size_t bad[4] = {};
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  for (int y = 0; y < disparity.height(); ++y) {
    for (int x = 0; x < disparity.width(); ++x) {
      const float gt = ground_truth(x, y);
      if (!std::isfinite(gt) || (!mask.empty() && mask(x, y) < mask_min)) {
        continue;
      }
      ++s.num_evaluated;
      const float d = disparity(x, y);
      if (!std::isfinite(d)) {
        continue;
      }
      ++estimated;
      const double err = std::abs(static_cast<double>(d) - gt) * error_scale;
      sum_abs += err;
      sum_sq += err * err;
      const double limits[4] = {0.5, 1.0, 2.0, 4.0};
      for (int k = 0; k < 4; ++k) {
        if (err > limits[k]) {
          ++bad[k];
        }
      }
      const double frac = d - std::floor(d);
      s.frac_histogram[std::min(9, static_cast<int>(frac * 10.0))] += 1.0;
    }
  }
  if (s.num_evaluated > 0) {
    s.density = static_cast<double>(estimated) / static_cast<double>(s.num_evaluated);
  }
  if (estimated > 0) {
    const auto n = static_cast<double>(estimated);
    s.bad05 = static_cast<double>(bad[0]) / n;
    s.bad1 = static_cast<double>(bad[1]) / n;
    s.bad2 = static_cast<double>(bad[2]) / n;
    s.bad4 = static_cast<double>(bad[3]) / n;
    s.avg_error = sum_abs / n;
    s.rms = std::sqrt(sum_sq / n);
    for (double& h : s.frac_histogram) {
      h /= n;
    }
  }
  return s;
}

ImageF32 fill_disparity_holes(const ImageF32& disparity) {
  ImageF32 out = disparity;
  const int w = disparity.width();
  for (int y = 0; y < disparity.height(); ++y) {
    const float* in = disparity.row(y);
    float* o = out.row(y);
    int x = 0;
    while (x < w) {
      if (std::isfinite(in[x])) {
        ++x;
        continue;
      }
      int end = x;
      while (end < w && !std::isfinite(in[end])) {
        ++end;
      }
      const float left = x > 0 ? in[x - 1] : std::numeric_limits<float>::infinity();
      const float right = end < w ? in[end] : std::numeric_limits<float>::infinity();
      const float fill = std::min(left, right);
      if (std::isfinite(fill)) {
        std::fill(o + x, o + end, fill);
      }
      x = end;
    }
  }
  return out;
}

ImageU8 colorize_disparity(const ImageF32& disparity, float d_min, float d_max) {
  ImageU8 out(disparity.width(), disparity.height(), 3);
  const float range = d_max > d_min ? d_max - d_min : 1.0F;
  for (int y = 0; y < disparity.height(); ++y) {
    for (int x = 0; x < disparity.width(); ++x) {
      const float d = disparity(x, y);
      if (!std::isfinite(d)) {
        continue;
      }
      // Piecewise linear "jet": blue -> cyan -> green -> yellow -> red.
      const float t = std::clamp((d - d_min) / range, 0.0F, 1.0F);
      const float r = std::clamp(1.5F - std::abs(4.0F * t - 3.0F), 0.0F, 1.0F);
      const float g = std::clamp(1.5F - std::abs(4.0F * t - 2.0F), 0.0F, 1.0F);
      const float b = std::clamp(1.5F - std::abs(4.0F * t - 1.0F), 0.0F, 1.0F);
      out(x, y, 0) = static_cast<std::uint8_t>(std::lround(255.0F * r));
      out(x, y, 1) = static_cast<std::uint8_t>(std::lround(255.0F * g));
      out(x, y, 2) = static_cast<std::uint8_t>(std::lround(255.0F * b));
    }
  }
  return out;
}

}  // namespace r3d
