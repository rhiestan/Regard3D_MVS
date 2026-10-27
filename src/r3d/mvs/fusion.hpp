#pragma once

#include <cstddef>
#include <vector>

#include "r3d/core/point_cloud.hpp"
#include "r3d/mvs/depth_map.hpp"

namespace r3d {

// Simple voxel-grid fusion of filtered depth maps (Phase 4; the octree with
// best-resolution-wins follows in Phase 6). Every valid pixel becomes a
// point; the points of one voxel are averaged with weights 1 / GSD^2, so
// finer views dominate. Output attributes: normals (from the depth maps,
// oriented towards the cameras), colours, "value" (mean GSD, the point scale
// used by FSSR) and "confidence" (number of fused pixels).
struct FusionOptions {
  double voxel_size = 0.0;      // world units; 0: voxel_factor x median GSD of all pixels
  double voxel_factor = 1.0;
  float normal_jump = 0.05F;    // relative depth jump that separates surfaces for normals
};

struct FusionStats {
  std::size_t input_points = 0;
  std::size_t output_points = 0;
  double voxel_size = 0.0;
  double median_gsd = 0.0;
};

// Positions are relative to `origin` (see PointCloud).
[[nodiscard]] PointCloud fuse_depth_maps(const std::vector<DepthMap>& maps, const Eigen::Vector3d& origin,
                                         const FusionOptions& options = {}, FusionStats* stats = nullptr);

// Median ground sample distance (depth / focal length) over a regular sample of the valid pixels.
[[nodiscard]] double median_gsd(const std::vector<DepthMap>& maps);

}  // namespace r3d
