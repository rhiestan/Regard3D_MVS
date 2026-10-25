#pragma once

// Inner loops of the Census-SGM matcher (internal header). Every kernel has a
// scalar reference in namespace scalar; the dispatched versions must give
// bit-identical results (DECISIONS.md D11).

#include <cstddef>
#include <cstdint>

namespace r3d::sgm {

// Path cost buffers hold one pixel per `stride` elements with label 0 at
// offset kLabelOffset; the elements just before label 0 and after the last
// label are kSentinel, so that the d - 1 and d + 1 neighbours can be loaded
// without bound checks.
inline constexpr std::size_t kLabelOffset = 8;
inline constexpr std::uint16_t kSentinel = 0xFFFF;
// Label counts are multiples of this (every SIMD target processes whole vectors).
inline constexpr std::size_t kLabelAlign = 16;

[[nodiscard]] constexpr std::size_t path_stride(std::size_t num_disp) noexcept { return num_disp + 2 * kLabelOffset; }

// Matching costs of one row: cost[x * num_disp + d] = popcount(left[x] ^ rev[width - 1 - x + d]),
// where rev holds the right census values in reverse column order (see stereo_matcher.cpp).
void census_costs(const std::uint64_t* left, const std::uint64_t* rev, std::size_t width, std::size_t num_disp,
                  std::uint16_t* cost);

// One row of one SGM pass with four paths (Hirschmüller 2008, eq. 13):
//   L(p, d) = C(p, d) + min(L(q, d), L(q, d -+ 1) + P1, min_k L(q, k) + P2) - min_k L(q, k)
// for the predecessor q of p on each path. Pass direction +1 processes the
// row left to right with predecessors (x-1, y), (x, y-1), (x-1, y-1), (x+1, y-1);
// direction -1 right to left with (x+1, y), (x, y+1), (x+1, y+1), (x-1, y+1).
// Predecessors outside the row use the zero start buffer (L = C at the first pixel of a path).
struct SgmRowArgs {
  const std::uint16_t* cost = nullptr;      // [width][num_disp]
  const std::uint16_t* p2 = nullptr;        // [4][width] P2 per path and pixel
  const std::uint16_t* prev = nullptr;      // [3][width][stride] paths 1..3 of the preceding row
  const std::uint16_t* prev_min = nullptr;  // [3][width]
  std::uint16_t* cur = nullptr;             // [3][width][stride] paths 1..3 of this row
  std::uint16_t* cur_min = nullptr;         // [3][width]
  std::uint16_t* scratch = nullptr;         // [3][stride]: zero start buffer, two buffers for path 0
  std::uint16_t* sum = nullptr;             // [width][num_disp] aggregated costs
  std::size_t width = 0;
  std::size_t num_disp = 0;                 // multiple of kLabelAlign
  std::uint16_t p1 = 0;
  int direction = 1;
  bool accumulate = false;                  // sum += (true) or sum = (false) the four paths
};

void sgm_row(const SgmRowArgs& args);

// Winner-takes-all over one row of aggregated costs sum[width][num_disp]:
// best[x] is the first label with the lowest cost best_cost[x], second[x] the
// lowest cost of the labels outside best +- 1 (0xFFFF if there is none).
// If right_cost is not null, right_cost[i] / right_best[i] receive the lowest
// cost and its first label k over all (x, k) with x - k + num_disp - 1 = i,
// i.e. the WTA of the right image column x - k (arrays of width + num_disp - 1
// entries, initialised by the caller to 0xFFFF).
void wta_row(const std::uint16_t* sum, std::size_t width, std::size_t num_disp, std::uint16_t* best,
             std::uint16_t* best_cost, std::uint16_t* second, std::uint16_t* right_cost, std::uint16_t* right_best);

namespace scalar {
void census_costs(const std::uint64_t* left, const std::uint64_t* rev, std::size_t width, std::size_t num_disp,
                  std::uint16_t* cost);
void sgm_row(const SgmRowArgs& args);
void wta_row(const std::uint16_t* sum, std::size_t width, std::size_t num_disp, std::uint16_t* best,
             std::uint16_t* best_cost, std::uint16_t* second, std::uint16_t* right_cost, std::uint16_t* right_best);
}  // namespace scalar

}  // namespace r3d::sgm
