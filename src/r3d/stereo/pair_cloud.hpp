#pragma once

#include <cstdint>

#include "r3d/core/image.hpp"
#include "r3d/core/point_cloud.hpp"
#include "r3d/rectify/rectification.hpp"
#include "r3d/stereo/stereo_matcher.hpp"

namespace r3d {

// Triangulates every valid disparity of a rectified pair into a world point
// (positions relative to the left camera centre). `color` (optional, RGB of
// the rectified left image) colours the points; the "confidence" scalar holds
// the matcher confidence. Points nearer than `min_depth` or with disparities
// at or below the disparity offset (infinitely far) are skipped.
[[nodiscard]] PointCloud disparity_to_point_cloud(const RectifiedPair& pair, const DisparityMap& disparity,
                                                  ImageView<const std::uint8_t> color = {}, double min_depth = 0.0);

}  // namespace r3d
