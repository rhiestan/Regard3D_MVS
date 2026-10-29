#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string_view>

#include <Eigen/Core>

#include "r3d/core/cancel.hpp"
#include "r3d/io/mesher_export.hpp"
#include "r3d/mvs/consistency.hpp"
#include "r3d/mvs/fusion.hpp"
#include "r3d/mvs/multi_stereo.hpp"
#include "r3d/mvs/refinement.hpp"
#include "r3d/mvs/visibility.hpp"
#include "r3d/scene/scene.hpp"

namespace r3d {

// The dense pipeline: depth maps of all views (multi-stereo), refinement,
// geometric consistency, visibility filter and octree fusion into one
// oriented, coloured point cloud (plan.md section 3).
struct DenseOptions {
  DepthMapOptions depth;
  bool refine = true;
  RefinementOptions refinement;
  ConsistencyOptions consistency;
  bool visibility_filter = true;
  VisibilityOptions visibility;
  FusionOptions fusion;  // memory_budget: point data per fusion pass

  // Depth maps and refinement images held in memory (bytes); the rest waits
  // in the work directory, or in a temporary directory without one.
  std::size_t memory_budget = std::size_t{8} << 30;

  // Optional directory for the depth maps of every view (and debug images).
  std::filesystem::path work_dir;
  bool reuse = false;          // load the depth maps cached in work_dir instead of matching again
  bool reuse_refined = false;  // load the refined depth maps cached in work_dir (skips matching and refinement)
  bool debug_images = false;   // depth PNGs before and after filtering, in work_dir
};

// Parameter presets, from fast previews to the highest detail; each one is a
// complete DenseOptions that single options can then be changed in.
enum class DensePreset { fast, normal, high, ultra };

[[nodiscard]] DenseOptions dense_preset(DensePreset preset);
[[nodiscard]] std::optional<DensePreset> parse_dense_preset(std::string_view name);
[[nodiscard]] std::string_view to_string(DensePreset preset);

enum class DenseStage { depth_maps, refinement, consistency, visibility, fusion };
[[nodiscard]] std::string_view to_string(DenseStage stage);

struct DenseProgress {
  DenseStage stage = DenseStage::depth_maps;
  std::size_t done = 0;   // views (fusion: depth maps read) finished in this stage
  std::size_t total = 0;
  double fraction = 0.0;  // of the whole computation, estimated from typical stage run times; ends at 1
};

// Called on the thread that runs the pipeline, after each view of a stage.
using DenseProgressCallback = std::function<void(const DenseProgress&)>;

struct DenseCallbacks {
  DenseProgressCallback progress;
  // Polled before every view and every chunk of parallel work; once
  // requested, the pipeline stops with r3d::Cancelled. Not owned.
  const CancellationToken* cancel = nullptr;
};

struct DenseReport {
  std::size_t views = 0;
  std::size_t pairs_matched = 0;
  std::size_t pairs_skipped = 0;
  std::size_t cached_views = 0;  // depth maps loaded from the work directory
  RefinementStats refinement;
  ConsistencyStats consistency;
  VisibilityStats visibility;
  FusionStats fusion;
  Eigen::Vector3d origin = Eigen::Vector3d::Zero();  // of the point positions
  double depth_map_seconds = 0.0;
  double refinement_seconds = 0.0;
  double consistency_seconds = 0.0;
  double visibility_seconds = 0.0;
  double fusion_seconds = 0.0;
  double total_seconds = 0.0;
};

// Origin of the dense cloud's positions: zero, or the rounded mean camera
// centre for georeferenced scenes (coordinates above 1e4), whose positions
// are kept relative to it in float.
[[nodiscard]] Eigen::Vector3d dense_cloud_origin(const Scene& scene);

// Runs the pipeline and hands the fused points to `sink` in parts (positions
// relative to dense_cloud_origin(scene), PointCloud::origin set). With
// options.fusion.with_views, the visibility lists hold the views' source ids.
// The memory does not grow with the cloud. Throws r3d::Cancelled when
// cancelled, r3d::Error on invalid options or I/O errors. Uses the default
// thread pool; progress and stage summaries go to the log (info level).
void densify(const Scene& scene, const DenseOptions& options, const PointSink& sink,
             const DenseCallbacks& callbacks = {}, DenseReport* report = nullptr);

// densify() into a PLY file in the given layout (positions in double when the
// origin is not zero). A partly written file is removed on failure or cancellation.
void densify_to_ply(const Scene& scene, const DenseOptions& options, const std::filesystem::path& path,
                    const CloudLayoutOptions& layout = {}, const DenseCallbacks& callbacks = {},
                    DenseReport* report = nullptr);

}  // namespace r3d
