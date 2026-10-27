#pragma once

#include <cstddef>
#include <vector>

#include "r3d/mvs/depth_map.hpp"
#include "r3d/select/view_selection.hpp"

namespace r3d {

// Multi-view geometric consistency (plan.md section 3.7).
//
// A depth of view v agrees with a neighbouring view n when its point,
// projected into n, meets a depth of n within `max_relative_depth_error`,
// and that depth, projected back into v, lands within `max_reprojection_px`
// of the original pixel. Pixels with fewer than `min_views` agreeing
// neighbours are removed, and pixels whose agreeing pairs (multi-stereo) plus
// agreeing views are fewer than `min_evidence`. The defaults keep a depth
// that two pairs or one other view confirm. Before the check, pixels next to a depth
// discontinuity (relative jump above `discontinuity` to a 4-neighbour) are
// removed together with their surroundings up to `edge_erosion - 1` pixels
// (flying pixels between foreground and background).
//
// The two depth maps of an image pair (matched as (v, n) and as (n, v)) tend
// to repeat each other's errors. With `independent_support`, a neighbour n
// does not count when the depth of v comes only from the pair with n and
// the depth of n only from the pair with v.
//
// With `average_depths`, every kept depth is replaced by the mean of its own
// depth and the depths of the agreeing neighbours along the same ray.
struct ConsistencyOptions {
  double max_relative_depth_error = 0.01;
  double max_reprojection_px = 2.0;
  int min_views = 0;
  int min_evidence = 2;           // pairs (multi-stereo) + agreeing views, at least
  float min_confidence = 0.0F;    // matcher confidence; lower pixels are removed before the check
  int edge_erosion = 1;           // 0: no erosion
  double discontinuity = 0.05;
  int max_neighbours = 8;
  bool independent_support = false;
  bool average_depths = true;
};

struct ConsistencyStats {
  std::size_t input = 0;      // valid pixels before filtering
  std::size_t unconfident = 0; // removed for low matcher confidence
  std::size_t eroded = 0;     // removed at depth discontinuities
  std::size_t rejected = 0;   // removed for too few agreeing views
  std::size_t output = 0;
};

// Neighbours of each view for the consistency check: its stereo partners
// and the views that chose it as a partner, best score first, at most
// max_neighbours. Indices are scene view indices.
[[nodiscard]] std::vector<std::vector<std::size_t>> consistency_neighbours(const ViewSelection& selection,
                                                                         int max_neighbours);

// Filters `maps` in place and fills DepthMap::support. `neighbours[i]` lists
// indices into `maps` (not scene views). The result does not depend on the
// order of the maps: all checks use the eroded, unfiltered maps.
void filter_depth_maps(std::vector<DepthMap>& maps, const std::vector<std::vector<std::size_t>>& neighbours,
                       const ConsistencyOptions& options = {}, ConsistencyStats* stats = nullptr);

// Removes pixels at depth discontinuities (see ConsistencyOptions); returns the number removed.
std::size_t erode_depth_edges(DepthMap& map, int radius, double discontinuity);

}  // namespace r3d
