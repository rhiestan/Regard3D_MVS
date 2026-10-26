#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "r3d/core/image.hpp"

namespace r3d {

// Census-SGM matcher for rectified pairs (plan section 3.4a): 9x7 Census,
// Hamming costs, 8-path semi-global aggregation, winner-takes-all with
// sub-pixel interpolation, left-right check, speckle and median filters.
// Large images are processed in independent overlapping tiles that are
// blended linearly (Hirschmüller 2008, section 2.6); the result does not
// depend on the number of threads.
//
// Hierarchical matching (DECISIONS.md D24): the coarsest pyramid level is
// matched over the full label range; every finer level searches a per-pixel
// range around the disparities of the coarser level (Rothermel et al. 2012).

enum class SubpixelMethod {
  none,
  parabola,     // fit through the aggregated costs at d - 1, d, d + 1
  equiangular,  // symmetric V fit (Shimizu & Okutomi 2001)
};

// Defaults tuned on the Middlebury 2014 training set (DECISIONS.md D19, D24).
struct StereoOptions {
  int p1 = 28;               // penalty for a disparity change of one (Census costs are 0..62)
  int p2 = 200;              // penalty for larger changes (at most 8000: 16-bit sums)
  int p2_adapt = 4;          // > 0: P2 * a / (a + |dI|) with the 8-bit intensity step dI (at least P1 + 1)
  int invalid_cost = 31;     // cost of correspondences outside the right image
  SubpixelMethod subpixel = SubpixelMethod::equiangular;
  float uniqueness = 0.0F;   // reject if best > (1 - uniqueness) * second best (outside d +- 1)
  int lr_max_diff = 1;       // left-right check tolerance in labels; < 0 disables the check
  bool lr_full_match = true;  // check against a full right-to-left match (better, twice the work)
                              // instead of the right disparities of the same cost volume
  int speckle_size = 100;    // remove segments with fewer pixels (0: off)
  float speckle_diff = 1.0F; // disparity step that separates segments
  bool median = true;        // 3x3 median of the valid disparities
  // Hierarchy: number of coarser pyramid levels (-1: automatic, 0: single
  // full-range level). A finer pixel searches the disparities (x2) of the
  // coarse pixels within range_radius, extended by range_margin labels on
  // each side; coarse pixels without disparity take the range of valid
  // pixels up to range_fill coarse pixels away, and the full range beyond.
  int levels = -1;
  int range_radius = 3;
  int range_margin = 4;
  int range_fill = 64;
  int tile_size = 512;       // core size of the tiles
  int tile_margin = 32;      // discarded border of a tile (each side with a neighbour)
  int tile_blend = 32;       // blending zone between neighbouring tiles
  std::size_t memory_budget = std::size_t{2} << 30;  // bytes for the tiles processed concurrently
  bool scalar_kernels = false;  // scalar reference kernels (tests, benchmarks)
};

inline constexpr float kInvalidDisparity = std::numeric_limits<float>::infinity();

struct DisparityMap {
  ImageF32 disparity;   // per left pixel: x_left - x_right, kInvalidDisparity where unknown
  ImageF32 confidence;  // 1 - best / second best aggregated cost, in [0, 1] (blended over tiles)
  int disparity_min = 0;
  int num_disparities = 0;
  int levels = 0;                // coarser pyramid levels used
  std::uint64_t cells = 0;       // aggregated (pixel, label) cells over all levels
};

// Per-pixel label ranges of a rectified pair: pixel (x, y) searches the
// labels lo .. lo + num - 1 (disparities disparity_min + label); num is a
// positive multiple of 16 and the range lies within the pair's labels.
struct LabelRanges {
  Image<std::uint16_t> lo;
  Image<std::uint16_t> num;
};

// Matches a rectified pair of 16-bit grayscale images (equal height). The
// label range is disparity_min .. disparity_min + num_disparities - 1;
// num_disparities is rounded up to a multiple of 16. Masks (255 = valid)
// may be empty views (all pixels valid).
[[nodiscard]] DisparityMap match_stereo(ImageView<const std::uint16_t> left, ImageView<const std::uint16_t> right,
                                        ImageView<const std::uint8_t> left_valid,
                                        ImageView<const std::uint8_t> right_valid, int disparity_min,
                                        int num_disparities, const StereoOptions& options = {});

// Post-processing filters (also used by match_stereo).
// Invalidates connected segments (4-neighbours, steps <= max_diff) with fewer than min_size pixels.
void remove_speckles(ImageF32& disparity, int min_size, float max_diff);
// 3x3 median over the valid neighbours of every valid pixel.
[[nodiscard]] ImageF32 median_filter_3x3(const ImageF32& disparity);

namespace detail {

// Aggregated cost volume of the whole image as one tile (no tiling,
// post-processing or WTA), with the optimised or scalar kernels: the labels
// of every pixel in row-major pixel order (full range: S[y][x][d]).
[[nodiscard]] std::vector<std::uint16_t> sgm_cost_volume(ImageView<const std::uint16_t> left,
                                                         ImageView<const std::uint16_t> right, int disparity_min,
                                                         int num_disparities, const StereoOptions& options,
                                                         const LabelRanges* ranges = nullptr);

// Textbook implementation of the same volume (each of the 8 paths computed
// separately in plain integer arithmetic), for testing the optimised code.
[[nodiscard]] std::vector<std::uint16_t> sgm_cost_volume_reference(ImageView<const std::uint16_t> left,
                                                                   ImageView<const std::uint16_t> right,
                                                                   int disparity_min, int num_disparities,
                                                                   const StereoOptions& options,
                                                                   const LabelRanges* ranges = nullptr);

// Full label range for every pixel.
[[nodiscard]] LabelRanges full_ranges(int width, int height, int num_disparities);

// Ranges of a level from the disparities of the next coarser pyramid level
// (pyramid.hpp geometry: fine disparity = 2 x coarse disparity), clipped to
// the labels whose match lies within the valid columns of the partner image
// (column x - d for a left base image, partner_sign = -1; x + d for a right
// base image, +1). Pixels outside fine_valid (empty: all valid) or without a
// possible match get the smallest range. With `mirrored` the ranges are
// stored for the horizontally mirrored image.
[[nodiscard]] LabelRanges ranges_from_coarse(const ImageF32& coarse, int width, int height,
                                             ImageView<const std::uint8_t> fine_valid,
                                             ImageView<const std::uint8_t> partner_valid, int partner_width,
                                             int partner_sign, int disparity_min, int num_disparities,
                                             const StereoOptions& options, bool mirrored = false);

// Number of coarser levels chosen by levels = -1.
[[nodiscard]] int automatic_levels(int width, int height, int num_disparities);

}  // namespace detail

}  // namespace r3d
