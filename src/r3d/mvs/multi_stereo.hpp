#pragma once

#include <cstddef>
#include <span>

#include "r3d/core/camera.hpp"
#include "r3d/mvs/depth_map.hpp"
#include "r3d/rectify/rectification.hpp"
#include "r3d/scene/scene.hpp"
#include "r3d/select/view_selection.hpp"
#include "r3d/stereo/stereo_matcher.hpp"

namespace r3d {

// Multi-stereo triangulation per base view (plan.md section 3.5).
//
// All pairs of a base view share its pixel rays. In a rectified pair the
// disparity is linear in the base view's inverse depth w = 1 / z:
//   d = k * w + disparity_offset, with k = f * baseline / c,
// where c is the rectified depth of the ray point at z = 1. A pixel's pairs
// therefore vote on w: the hypothesis with the most pairs within `inlier_px`
// wins, its pairs are combined by the weighted mean w = sum(k^2 w_i) /
// sum(k^2), which minimises the sum of squared disparity (reprojection)
// errors, and pairs with a residual above `max_reprojection_px` are dropped.
struct MultiStereoOptions {
  double inlier_px = 2.0;            // disparity distance for two pairs to agree
  double max_reprojection_px = 1.0;  // final disparity residual of every used pair
  double min_angle_deg = 3.0;        // largest triangulation angle of the used pairs
  int min_pairs = 1;                 // used pairs per pixel
};

// A matched pair whose left (base) camera has the grid's centre.
struct PairDisparity {
  const RectifiedPair* pair = nullptr;
  const DisparityMap* disparity = nullptr;
  std::uint32_t partner_view = 0;  // scene view index of the right camera (recorded in DepthMap::partners)
};

// Depth map on `grid` (depth, confidence, pair mask, partners; colour and support are left
// empty / zero). A pixel needs at least min_pairs agreeing pairs, more of
// them than disagreeing ones (or two agreeing pairs on a tie), and a
// triangulation angle of at least min_angle_deg.
[[nodiscard]] DepthMap combine_pairs(const PinholeCamera& grid, std::span<const PairDisparity> pairs,
                                     const MultiStereoOptions& options = {});

// Disparity of a rectified map at a sub-pixel position: bilinear when the
// four neighbours are valid and within 1 pixel of each other, otherwise the
// nearest pixel; kInvalidDisparity outside or where unknown.
[[nodiscard]] float sample_disparity(const ImageF32& disparity, double u, double v);

struct DepthMapOptions {
  double resolution = 1.0;      // depth map and rectified images, relative to the view's undistorted camera
  int max_image_size = 0;       // > 0: the depth map's longer side is at most this (pixels), see depth_map_resolution
  int max_pairs = 4;            // stereo partners used per view (best first)
  StereoOptions stereo;
  RectificationOptions rectification;  // resolution is taken from `resolution`
  MultiStereoOptions multi_stereo;
};

struct DepthMapStats {
  int pairs_matched = 0;
  int pairs_skipped = 0;
  double load_seconds = 0.0;      // image decoding, rectification, colour
  double match_seconds = 0.0;
  double combine_seconds = 0.0;
  std::uint64_t cells = 0;        // aggregated SGM cells
};

// Rectifies and matches `view` with its best partners (ViewSelection) and
// combines the pairs into a coloured depth map.
[[nodiscard]] DepthMap compute_depth_map(const Scene& scene, const ViewSelection& selection, std::size_t view,
                                         const DepthMapOptions& options = {}, DepthMapStats* stats = nullptr);

// Undistorted camera of a view scaled by `resolution`, with the view's pose.
[[nodiscard]] PinholeCamera depth_map_camera(const Scene& scene, std::size_t view, double resolution);

// Resolution of a view's depth map: options.resolution, reduced so that the
// longer side of the undistorted image stays within options.max_image_size.
[[nodiscard]] double depth_map_resolution(const Scene& scene, std::size_t view, const DepthMapOptions& options);

}  // namespace r3d
