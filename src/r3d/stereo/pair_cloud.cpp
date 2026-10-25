#include "r3d/stereo/pair_cloud.hpp"

#include <cmath>
#include <utility>
#include <vector>

#include "r3d/core/error.hpp"

namespace r3d {

PointCloud disparity_to_point_cloud(const RectifiedPair& pair, const DisparityMap& disparity,
                                    ImageView<const std::uint8_t> color, double min_depth) {
  const ImageF32& d = disparity.disparity;
  if (d.width() != pair.left.intrinsics.width || d.height() != pair.left.intrinsics.height) {
    throw_error("disparity_to_point_cloud: disparity map is {}x{}, rectified left image {}x{}", d.width(), d.height(),
                pair.left.intrinsics.width, pair.left.intrinsics.height);
  }
  const bool with_color = !color.empty();
  if (with_color && (color.width() != d.width() || color.height() != d.height() || color.channels() < 3)) {
    throw_error("disparity_to_point_cloud: colour image does not match the disparity map");
  }
  PointCloud cloud;
  cloud.origin = pair.left.pose.center;
  std::vector<float> confidence;
  for (int y = 0; y < d.height(); ++y) {
    for (int x = 0; x < d.width(); ++x) {
      const float v = d(x, y);
      if (!std::isfinite(v) || v <= pair.disparity_offset()) {
        continue;
      }
      const double z = pair.depth_from_disparity(v);
      if (z < min_depth) {
        continue;
      }
      const Eigen::Vector3d p = pair.left.unproject(Eigen::Vector2d(x, y), z) - cloud.origin;
      cloud.positions.push_back(p.cast<float>());
      if (with_color) {
        cloud.colors.push_back({color(x, y, 0), color(x, y, 1), color(x, y, 2)});
      }
      confidence.push_back(disparity.confidence.empty() ? 0.0F : disparity.confidence(x, y));
    }
  }
  cloud.scalars.emplace_back("confidence", std::move(confidence));
  return cloud;
}

}  // namespace r3d
