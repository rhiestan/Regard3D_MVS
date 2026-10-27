#pragma once

#include <bit>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <Eigen/Core>

#include "r3d/core/camera.hpp"
#include "r3d/core/image.hpp"

namespace r3d {

// Depth map of one view on an undistorted pinhole grid (the view's
// undistorted camera, scaled by the working resolution). Depth is measured
// along the optical axis; 0 marks pixels without a depth.
struct DepthMap {
  std::uint32_t view = 0;   // index into Scene::views
  PinholeCamera camera;     // pose of the view, undistorted (scaled) intrinsics
  ImageF32 depth;           // 0 = invalid
  ImageF32 confidence;      // matcher confidence, mean over the consistent pairs, [0, 1]
  Image<std::uint16_t> pair_mask;  // bit i: the pair with partners[i] contributed to the depth
  std::vector<std::uint32_t> partners;  // scene view index of each matched pair (at most 16)
  ImageU8 support;          // number of other views that agree (geometric consistency), 0 before filtering
  ImageU8 color;            // RGB of the undistorted view

  [[nodiscard]] int width() const noexcept { return depth.width(); }
  [[nodiscard]] int height() const noexcept { return depth.height(); }
  [[nodiscard]] bool valid(int x, int y) const noexcept { return depth(x, y) > 0.0F; }
  [[nodiscard]] int pairs(int x, int y) const noexcept { return std::popcount(pair_mask(x, y)); }
  // Bit of the pair with scene view `partner_view`, 0 if there is none.
  [[nodiscard]] std::uint16_t pair_bit(std::uint32_t partner_view) const noexcept {
    for (std::size_t i = 0; i < partners.size(); ++i) {
      if (partners[i] == partner_view) {
        return static_cast<std::uint16_t>(1U << i);
      }
    }
    return 0;
  }
  [[nodiscard]] Eigen::Vector3d point(int x, int y) const {
    return camera.unproject(Eigen::Vector2d(x, y), static_cast<double>(depth(x, y)));
  }
  // Allocates all images for a w x h grid (depth 0 everywhere).
  void allocate(int w, int h);
  [[nodiscard]] std::size_t num_valid() const;
};

// Intrinsics for an image resampled by `scale` (pixel centres at integer
// coordinates: c' = (c + 0.5) * scale - 0.5).
[[nodiscard]] PinholeIntrinsics scale_intrinsics(const PinholeIntrinsics& k, double scale);

// Unit normal at a valid pixel from its neighbours' 3D points (central
// differences, one-sided where a neighbour is missing or lies across a depth
// jump larger than `max_relative_jump`), oriented towards the camera. Returns
// the zero vector when no tangent can be formed.
[[nodiscard]] Eigen::Vector3f depth_map_normal(const DepthMap& map, int x, int y, float max_relative_jump = 0.05F);

// Binary file (.r3dd): camera, partners, depth, confidence, pair mask, support and colour.
void write_depth_map(const std::filesystem::path& path, const DepthMap& map);
[[nodiscard]] DepthMap read_depth_map(const std::filesystem::path& path);

// False-colour image of the depth (near: warm, far: cool, invalid: black)
// between the 2nd and 98th percentile of the valid depths.
[[nodiscard]] ImageU8 colorize_depth(const DepthMap& map);

}  // namespace r3d
