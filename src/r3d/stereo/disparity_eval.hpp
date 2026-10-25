#pragma once

#include <cstddef>
#include <cstdint>

#include "r3d/core/image.hpp"

namespace r3d {

// Disparity error statistics against ground truth, with the Middlebury 2014
// conventions: ground truth is infinite where unknown, the mask selects the
// evaluated pixels (mask0nocc.png: 255 non-occluded, 128 occluded).
struct DisparityErrorStats {
  std::size_t num_evaluated = 0;  // pixels with ground truth inside the mask
  double density = 0.0;           // fraction of those with an estimate
  // Fractions of the estimated pixels with |error| > 0.5, 1, 2, 4 px.
  double bad05 = 0.0;
  double bad1 = 0.0;
  double bad2 = 0.0;
  double bad4 = 0.0;
  double avg_error = 0.0;  // mean |error| of the estimated pixels
  double rms = 0.0;
  // Fractional part of the estimates (pixel-locking check): histogram of
  // frac(d) in 10 bins over the estimated pixels.
  double frac_histogram[10] = {};
};

// `mask_min`: pixels with mask >= mask_min are evaluated (255: non-occluded
// only, 128: all). An empty mask evaluates all pixels with ground truth.
// Errors are multiplied by `error_scale` before thresholding, e.g. 4 to score
// a quarter-resolution result in full-resolution pixels as MiddEval3 does.
[[nodiscard]] DisparityErrorStats evaluate_disparity(const ImageF32& disparity, const ImageF32& ground_truth,
                                                     ImageView<const std::uint8_t> mask, int mask_min = 255,
                                                     float error_scale = 1.0F);

// Fills invalid pixels along each row with the smaller (more distant) of the
// nearest valid disparities to the left and right, like the Middlebury SDK
// ("ii" interpolation), so that sparse results can be scored densely.
[[nodiscard]] ImageF32 fill_disparity_holes(const ImageF32& disparity);

// False-colour rendering (blue = far / small disparity, red = near) of the
// range [d_min, d_max]; invalid pixels are black.
[[nodiscard]] ImageU8 colorize_disparity(const ImageF32& disparity, float d_min, float d_max);

}  // namespace r3d
