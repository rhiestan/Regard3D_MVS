#pragma once

#include <cstddef>
#include <span>
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
//
// Finally, connected segments (4-neighbours, relative depth steps up to
// `speckle_step`) with fewer than `speckle_size` pixels are removed: most
// gross errors that survive the check (reflections, glass) are small
// fragments in regions that are otherwise empty.
struct ConsistencyOptions {
  double max_relative_depth_error = 0.01;
  double max_reprojection_px = 2.0;
  int min_views = 0;
  int min_evidence = 2;           // pairs (multi-stereo) + agreeing views, at least
  float min_confidence = 0.0F;    // matcher confidence; lower pixels are removed before the check
  int edge_erosion = 1;           // 0: no erosion
  double discontinuity = 0.05;
  bool erode_holes = false;       // erode at the border of invalid regions as well
  int max_neighbours = 8;
  bool independent_support = false;
  bool average_depths = true;
  int speckle_size = 0;           // pixels; 0: off
  double speckle_step = 0.02;     // relative depth step within a segment
};

struct ConsistencyStats {
  std::size_t input = 0;      // valid pixels before filtering
  std::size_t unconfident = 0; // removed for low matcher confidence
  std::size_t eroded = 0;     // removed at depth discontinuities
  std::size_t rejected = 0;   // removed for too few agreeing views
  std::size_t speckles = 0;   // removed as small segments
  std::size_t output = 0;
};

// Neighbours of each view for the consistency check: its stereo partners
// and the views that chose it as a partner, best score first, at most
// max_neighbours. Indices are scene view indices.
[[nodiscard]] std::vector<std::vector<std::size_t>> consistency_neighbours(const ViewSelection& selection,
                                                                         int max_neighbours);

// The check in two steps, one view at a time (statistics are added up):
// prepare_depth_map removes unconfident depths and depth edges;
// check_consistency checks a prepared map against its prepared neighbours,
// fills DepthMap::support, removes unconfirmed depths and speckles.
void prepare_depth_map(DepthMap& map, const ConsistencyOptions& options = {}, ConsistencyStats* stats = nullptr);
void check_consistency(DepthMap& map, std::span<const DepthMap* const> neighbours,
                       const ConsistencyOptions& options = {}, ConsistencyStats* stats = nullptr);

// Both steps for all maps in place. `neighbours[i]` lists indices into `maps`
// (not scene views). The result does not depend on the order of the maps:
// all checks use the prepared, unfiltered maps.
void filter_depth_maps(std::vector<DepthMap>& maps, const std::vector<std::vector<std::size_t>>& neighbours,
                       const ConsistencyOptions& options = {}, ConsistencyStats* stats = nullptr);

// Removes pixels at depth discontinuities (see ConsistencyOptions), and with
// `holes` also at the border of invalid regions; returns the number removed.
std::size_t erode_depth_edges(DepthMap& map, int radius, double discontinuity, bool holes = false);

// Removes connected segments of fewer than `min_size` pixels (4-neighbours
// whose depths differ by at most `max_relative_step`); returns the number removed.
std::size_t remove_depth_speckles(DepthMap& map, int min_size, double max_relative_step);

}  // namespace r3d
