#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "r3d/scene/scene.hpp"

namespace r3d {

// Stereo partner selection from the sparse SfM structure (plan.md section 3.2),
// combining Goesele et al. 2007 (per-point angle and scale weights, spread of
// baseline directions) and Wenzel et al. 2014 (Gaussian angle weight around
// preferred intersection angles).
struct ViewSelectionOptions {
  int max_partners = 4;            // per base view
  int min_shared_points = 15;      // pairs sharing fewer points are not considered
  double min_angle_deg = 3.0;      // per-point triangulation angle limits
  double max_angle_deg = 45.0;
  std::array<double, 2> preferred_angles_deg{10.0, 20.0};
  double angle_sigma_deg = 10.0;
  double max_scale_ratio = 2.0;    // GSD ratio limit (geometric mean over shared points)
  double min_relative_score = 0.1; // partners scoring below this fraction of the best one are dropped
  int coverage_grid = 8;           // coverage cells per image side (at most 8: 64-bit mask)
  double depth_percentile = 0.01;  // robust depth range: [p, 1 - p] percentiles ...
  double depth_margin = 0.1;       // ... widened by this relative margin
};

struct DepthRange {
  double min = 0.0;
  double max = 0.0;
  [[nodiscard]] bool valid() const noexcept { return max > min && min > 0.0; }
};

struct StereoPartner {
  std::uint32_t view = 0;
  double score = 0.0;
  std::uint32_t shared_points = 0;
  double mean_angle_deg = 0.0;     // mean triangulation angle of the shared points
  double scale_ratio = 1.0;        // geometric mean of GSD(base) / GSD(partner)
  double coverage = 0.0;           // fraction of the base view's occupied grid cells
  bool epipole_in_image = false;   // forward motion: rectification degenerate
  DepthRange depth;                // of the shared points, in the base camera
};

struct ViewSelection {
  std::vector<std::vector<StereoPartner>> partners;  // per base view, best first
  std::vector<DepthRange> view_depth;                // of all points seen by a view

  // Unordered pairs (a < b) appearing in any partner list, sorted.
  [[nodiscard]] std::vector<std::pair<std::uint32_t, std::uint32_t>> stereo_pairs() const;
};

[[nodiscard]] ViewSelection select_views(const Scene& scene, const ViewSelectionOptions& options = {});

}  // namespace r3d
