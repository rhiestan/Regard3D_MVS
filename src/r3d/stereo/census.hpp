#pragma once

#include <cstddef>
#include <cstdint>

#include "r3d/core/image.hpp"

namespace r3d {

// Census transform (Zabih & Woodfill 1994) over a 9x7 window: one bit per
// neighbour, set when the neighbour is darker than the centre pixel. The 62
// neighbours are visited in raster order and the first one ends up in the
// most significant bit (bit 61). Image borders are replicated.
inline constexpr int kCensusWidth = 9;
inline constexpr int kCensusHeight = 7;
inline constexpr int kCensusBits = kCensusWidth * kCensusHeight - 1;

// Census image of a 16-bit grayscale image. Rows run in parallel; with
// `scalar_reference` the scalar kernel is used (tests, benchmarks).
[[nodiscard]] Image<std::uint64_t> census_transform(ImageView<const std::uint16_t> img, bool scalar_reference = false);

// Row kernel (runtime SIMD dispatch). rows[j] points to row y - 3 + j of an
// image padded by 4 replicated pixels on both sides, at the padded start, so
// that pixel x of the window row j is rows[j][x + 4]. Writes out[0 .. n).
void census_row(const std::uint16_t* const rows[kCensusHeight], std::uint64_t* out, std::size_t n);

namespace scalar {
void census_row(const std::uint16_t* const rows[kCensusHeight], std::uint64_t* out, std::size_t n);
}  // namespace scalar

}  // namespace r3d
