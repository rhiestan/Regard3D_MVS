#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "r3d/core/point_cloud.hpp"
#include "r3d/mvs/depth_map.hpp"

namespace r3d {

// Octree fusion of filtered depth maps (plan.md section 3.8, Wenzel 2014).
//
// Every valid pixel becomes a point with a leaf size of `voxel_factor` x its
// ground sample distance (GSD = depth / focal length), rounded to a power of
// two times the finest leaf size. The octree is linear: points are sorted by
// the Morton code of their finest cell, so every octree cell is a contiguous
// range.
//  * Best resolution wins: a point is dropped when its own leaf holds at
//    least two points whose GSD is `resolution_ratio` times finer, so coarse
//    views do not add noise where finer views see the surface.
//  * Fusion: the remaining points of a leaf become one point, averaged with
//    weights 1 / GSD^2.
//  * Fold validation: a fused point needs points from `min_sources` views,
//    in its leaf or (with `fold_neighbourhood`) in the 26 neighbouring
//    leaves of the same size, since agreeing views often fill adjacent leaves.
//
// Space is split into chunks of 2^k finest cells per axis. Chunks are fused
// independently and in batches whose points fit into `memory_budget`, each
// batch with one pass over the depth maps, so the memory does not grow with
// the scene (out-of-core-ready layout).
//
// Output attributes: normals (from the depth maps, oriented towards the
// cameras), colours, "value" (mean GSD, the point scale used by FSSR) and
// "confidence" (number of fused pixels).
struct FusionOptions {
  double voxel_size = 0.0;          // > 0: one fixed leaf size for all points (no resolution levels)
  double voxel_factor = 1.0;        // leaf size relative to the GSD of a point (1, 2, 4: output density)
  double resolution_ratio = 2.0;    // drop points whose leaf holds points this much finer; 0: keep all
  int min_sources = 1;              // distinct views per fused point
  bool fold_neighbourhood = false;  // count the views of the neighbouring leaves as well
  float normal_jump = 0.05F;        // relative depth jump that separates surfaces for normals
  bool with_views = false;          // fill PointCloud::visibility with the depth map indices of each point
  std::size_t memory_budget = std::size_t{2} << 30;  // bytes of point data per pass
};

struct FusionStats {
  std::size_t input_points = 0;
  std::size_t coarser_points = 0;     // dropped: a finer view sees the leaf
  std::size_t unsupported_points = 0; // fused points dropped for too few views
  std::size_t outside_points = 0;     // beyond the addressable grid
  std::size_t output_points = 0;
  double voxel_size = 0.0;            // finest leaf size
  int levels = 0;                     // leaf sizes voxel_size * 2^l, l < levels
  std::size_t chunks = 0;
  int passes = 0;
  double median_gsd = 0.0;
};

// Positions are relative to `origin` (see PointCloud).
[[nodiscard]] PointCloud fuse_depth_maps(const std::vector<DepthMap>& maps, const Eigen::Vector3d& origin,
                                         const FusionOptions& options = {}, FusionStats* stats = nullptr);

// The same with the maps loaded on demand: source(i) for i < count, read once
// per pass and in order. `gsd` is a GSD sample of all maps (sample_gsd).
using DepthMapSource = std::function<std::shared_ptr<const DepthMap>(std::size_t)>;
[[nodiscard]] PointCloud fuse_depth_maps(std::size_t count, const DepthMapSource& source, std::vector<float> gsd,
                                         const Eigen::Vector3d& origin, const FusionOptions& options = {},
                                         FusionStats* stats = nullptr);

// The same with the fused points handed to `sink` in parts (one per batch of
// chunks, in the same order as above) instead of being collected.
// `progress(done, total)` is called on the calling thread after every map
// read; `total` (maps to read over all passes) is exact once the number of
// passes is known, after the first pass.
using PointSink = std::function<void(PointCloud&&)>;
using FusionProgress = std::function<void(std::size_t done, std::size_t total)>;
void fuse_depth_maps(std::size_t count, const DepthMapSource& source, std::vector<float> gsd,
                     const Eigen::Vector3d& origin, const FusionOptions& options, const PointSink& sink,
                     FusionStats* stats = nullptr, const FusionProgress& progress = {});

// Appends the GSD of every 4th pixel in both directions.
void sample_gsd(const DepthMap& map, std::vector<float>& out);

// Median ground sample distance (depth / focal length) over a regular sample of the valid pixels.
[[nodiscard]] double median_gsd(const std::vector<DepthMap>& maps);

}  // namespace r3d
