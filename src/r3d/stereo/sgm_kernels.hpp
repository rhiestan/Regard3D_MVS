#pragma once

// Inner loops of the Census-SGM matcher (internal header). Every kernel has a
// scalar reference in namespace scalar; the dispatched versions must give
// bit-identical results (DECISIONS.md D11).

#include <cstddef>
#include <cstdint>

namespace r3d::sgm {

// Path cost buffers hold each pixel's labels with kLabelOffset elements of
// padding on both sides; the elements just before label 0 and after the last
// label are kSentinel, so that the d - 1 and d + 1 neighbours can be loaded
// without bound checks.
inline constexpr std::size_t kLabelOffset = 8;
inline constexpr std::uint16_t kSentinel = 0xFFFF;
// Label counts are multiples of this (every SIMD target processes whole vectors).
inline constexpr std::size_t kLabelAlign = 16;

[[nodiscard]] constexpr std::size_t path_stride(std::size_t num_disp) noexcept { return num_disp + 2 * kLabelOffset; }

// Labels of one row: pixel x has the num[x] labels lo[x] .. lo[x] + num[x] - 1
// (num[x] a positive multiple of kLabelAlign). Per-pixel data of the row
// (costs, aggregated costs) is packed: pixel x starts at offset[x], with
// offset[0] = 0 and offset[x + 1] = offset[x] + num[x]. Path buffers of the
// row add the padding: label 0 of pixel x is at path_offset(x).
struct RowLabels {
  const std::int32_t* lo = nullptr;
  const std::uint16_t* num = nullptr;
  const std::size_t* offset = nullptr;  // [width + 1]

  [[nodiscard]] std::size_t path_offset(std::size_t x) const noexcept {
    return offset[x] + x * 2 * kLabelOffset + kLabelOffset;
  }
  [[nodiscard]] std::size_t path_size(std::size_t width) const noexcept {
    return offset[width] + width * 2 * kLabelOffset;
  }
};

// Matching costs of one row: cost[offset[x] + j] = popcount(left[x] ^ rev[width - 1 - x + lo[x] + j - rev_origin]),
// where rev holds the right census values in reverse column order (see stereo_matcher.cpp).
void census_costs(const std::uint64_t* left, const std::uint64_t* rev, std::size_t width, const RowLabels& labels,
                  std::int32_t rev_origin, std::uint16_t* cost);

// One row of one SGM pass with four paths (Hirschmüller 2008, eq. 13):
//   L(p, d) = C(p, d) + min(L(q, d), L(q, d -+ 1) + P1, min_k L(q, k) + P2) - min_k L(q, k)
// for the predecessor q of p on each path, where L(q, k) is infinite for
// labels k outside q's range. Pass direction +1 processes the row left to
// right with predecessors (x-1, y), (x, y-1), (x-1, y-1), (x+1, y-1);
// direction -1 right to left with (x+1, y), (x, y+1), (x+1, y+1), (x-1, y+1).
// Predecessors outside the row use the zero start buffer (L = C at the first pixel of a path).
struct SgmRowArgs {
  const std::uint16_t* cost = nullptr;      // packed per `labels`
  const std::uint16_t* p2 = nullptr;        // [4][width] P2 per path and pixel
  const std::uint16_t* prev = nullptr;      // [3][prev_labels.path_size] paths 1..3 of the preceding row
  const std::uint16_t* prev_min = nullptr;  // [3][width]
  std::uint16_t* cur = nullptr;             // [3][labels.path_size] paths 1..3 of this row
  std::uint16_t* cur_min = nullptr;         // [3][width]
  std::uint16_t* scratch = nullptr;         // [7][path_stride(max_num)]: start buffer, two buffers for
                                            // path 0, four buffers for re-aligned predecessors
  std::uint16_t* sum = nullptr;             // packed per `labels`: aggregated costs
  RowLabels labels;                         // this row
  RowLabels prev_labels;                    // preceding row (ignored if has_prev is false)
  std::size_t width = 0;
  std::size_t max_num = 0;                  // >= every num[x] of both rows
  std::uint16_t p1 = 0;
  int direction = 1;
  bool has_prev = false;                    // false for the first row of a pass
  bool accumulate = false;                  // sum += (true) or sum = (false) the four paths
};

// Number of scratch buffers of SgmRowArgs::scratch.
inline constexpr std::size_t kScratchBuffers = 7;

void sgm_row(const SgmRowArgs& args);

// Winner-takes-all over one row of aggregated costs (packed per `labels`):
// best[x] is the first label with the lowest cost best_cost[x], second[x] the
// lowest cost of the labels outside best +- 1 (0xFFFF if there is none);
// labels are absolute (lo[x] + j).
// If right_cost is not null, right_cost[i] / right_best[i] receive the lowest
// cost and its first label k over all (x, k) with x - k + right_shift = i,
// i.e. the WTA of the right image column x - k (arrays initialised by the
// caller to 0xFFFF and large enough for every i).
void wta_row(const std::uint16_t* sum, std::size_t width, const RowLabels& labels, std::uint16_t* best,
             std::uint16_t* best_cost, std::uint16_t* second, std::uint16_t* right_cost, std::uint16_t* right_best,
             std::ptrdiff_t right_shift);

namespace scalar {
void census_costs(const std::uint64_t* left, const std::uint64_t* rev, std::size_t width, const RowLabels& labels,
                  std::int32_t rev_origin, std::uint16_t* cost);
void sgm_row(const SgmRowArgs& args);
void wta_row(const std::uint16_t* sum, std::size_t width, const RowLabels& labels, std::uint16_t* best,
             std::uint16_t* best_cost, std::uint16_t* second, std::uint16_t* right_cost, std::uint16_t* right_best,
             std::ptrdiff_t right_shift);
}  // namespace scalar

}  // namespace r3d::sgm
