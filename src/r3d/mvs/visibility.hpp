#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "r3d/mvs/depth_map.hpp"

namespace r3d {

// Visibility filter of the consistent depth maps (Phase 6).
//
// A depth of view v conflicts with a neighbouring view n when
//  * it lies in n's free space: projected into n, it is in front of the
//    surface n sees there (a floating point), or
//  * n sees a surface in front of it along v's ray: one of n's points,
//    projected into v, lies in front of the depth (a point behind a wall or
//    under a floor, typically a reflection).
// A conflict needs a margin of `relative_margin` in depth and must hold for
// the whole 3x3 neighbourhood of the pixel it is tested at, so depth edges
// that are misaligned by a pixel do not count. A depth is removed when at
// least `min_conflicts` neighbours conflict with it and more neighbours
// conflict than agree (DepthMap::support from the consistency check). The
// two kinds of conflict are the same configuration seen from either view,
// so the vote decides which of the two surfaces is wrong.
struct VisibilityOptions {
  double relative_margin = 0.02;  // conflicting surface at least this much (relative depth) in front
  int min_conflicts = 2;
  int support_margin = 0;         // removed when conflicts > support + support_margin
  int max_neighbours = 8;
};

struct VisibilityStats {
  std::size_t input = 0;
  std::size_t free_space = 0;   // conflicts of the first kind (pixel-neighbour pairs)
  std::size_t occluded = 0;     // conflicts of the second kind
  std::size_t removed = 0;
};

// Filters one map against its neighbours (best first, the first
// max_neighbours are used); statistics are added up.
void filter_visibility(DepthMap& map, std::span<const DepthMap* const> neighbours,
                       const VisibilityOptions& options = {}, VisibilityStats* stats = nullptr);

// Filters all `maps` in place; `neighbours[i]` lists indices into `maps`. The
// result does not depend on the order of the maps.
void filter_visibility(std::vector<DepthMap>& maps, const std::vector<std::vector<std::size_t>>& neighbours,
                       const VisibilityOptions& options = {}, VisibilityStats* stats = nullptr);

}  // namespace r3d
