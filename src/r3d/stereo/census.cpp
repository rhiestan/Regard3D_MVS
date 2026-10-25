// Compiled once per SIMD target via Highway's foreach_target mechanism.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "r3d/stereo/census.cpp"
#include <hwy/foreach_target.h>  // IWYU pragma: keep

#include <hwy/highway.h>

#include "r3d/stereo/census.hpp"

#ifndef R3D_STEREO_CENSUS_OFFSETS
#define R3D_STEREO_CENSUS_OFFSETS
namespace r3d::detail {

struct CensusOffset {
  int row;  // window row 0..6
  int col;  // window column 0..8
};

// Neighbours in raster order, without the centre.
constexpr auto make_census_offsets() {
  struct Table {
    CensusOffset v[kCensusBits];
  } table{};
  int k = 0;
  for (int j = 0; j < kCensusHeight; ++j) {
    for (int i = 0; i < kCensusWidth; ++i) {
      if (j == kCensusHeight / 2 && i == kCensusWidth / 2) {
        continue;
      }
      table.v[k++] = CensusOffset{j, i};
    }
  }
  return table;
}
inline constexpr auto kCensusOffsets = make_census_offsets();

}  // namespace r3d::detail
#endif

HWY_BEFORE_NAMESPACE();
namespace r3d {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Bits of neighbours [k0, k1) of a vector of centres, the first one most
// significant.
template <class D, class V>
HWY_INLINE V CensusBits(D d, const std::uint16_t* const* rows, std::size_t x, V center, int k0, int k1) {
  const V one = hn::Set(d, std::uint16_t{1});
  V acc = hn::Zero(d);
  for (int k = k0; k < k1; ++k) {
    const detail::CensusOffset o = detail::kCensusOffsets.v[k];
    const V nb = hn::LoadU(d, rows[o.row] + x + o.col);
    acc = hn::Or(hn::ShiftLeft<1>(acc), hn::IfThenElseZero(hn::Lt(nb, center), one));
  }
  return acc;
}

void CensusRow(const std::uint16_t* const* HWY_RESTRICT rows, std::uint64_t* HWY_RESTRICT out, std::size_t n) {
  const hn::ScalableTag<std::uint16_t> d;
  const std::size_t lanes = hn::Lanes(d);
  // The 62 bits in four 16-bit words: bits 61..48 (14 neighbours), 47..32,
  // 31..16 and 15..0 (16 neighbours each).
  HWY_ALIGN std::uint16_t words[4][HWY_MAX_BYTES / sizeof(std::uint16_t)];
  std::size_t x = 0;
  for (; x + lanes <= n; x += lanes) {
    const auto center = hn::LoadU(d, rows[kCensusHeight / 2] + x + kCensusWidth / 2);
    hn::Store(CensusBits(d, rows, x, center, 0, 14), d, words[3]);
    hn::Store(CensusBits(d, rows, x, center, 14, 30), d, words[2]);
    hn::Store(CensusBits(d, rows, x, center, 30, 46), d, words[1]);
    hn::Store(CensusBits(d, rows, x, center, 46, 62), d, words[0]);
    for (std::size_t i = 0; i < lanes; ++i) {
      out[x + i] = (std::uint64_t{words[3][i]} << 48) | (std::uint64_t{words[2][i]} << 32) |
                   (std::uint64_t{words[1][i]} << 16) | std::uint64_t{words[0][i]};
    }
  }
  if (x < n) {
    const std::uint16_t* shifted[kCensusHeight];
    for (int j = 0; j < kCensusHeight; ++j) {
      shifted[j] = rows[j] + x;
    }
    scalar::census_row(shifted, out + x, n - x);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace r3d
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

#include <algorithm>
#include <cstring>
#include <vector>

#include "r3d/core/thread_pool.hpp"

namespace r3d {

HWY_EXPORT(CensusRow);

namespace scalar {

void census_row(const std::uint16_t* const rows[kCensusHeight], std::uint64_t* out, std::size_t n) {
  for (std::size_t x = 0; x < n; ++x) {
    const std::uint16_t center = rows[kCensusHeight / 2][x + kCensusWidth / 2];
    std::uint64_t bits = 0;
    for (const detail::CensusOffset& o : detail::kCensusOffsets.v) {
      bits = (bits << 1) | (rows[o.row][x + static_cast<std::size_t>(o.col)] < center ? 1U : 0U);
    }
    out[x] = bits;
  }
}

}  // namespace scalar

void census_row(const std::uint16_t* const rows[kCensusHeight], std::uint64_t* out, std::size_t n) {
  HWY_DYNAMIC_DISPATCH(CensusRow)(rows, out, n);
}

Image<std::uint64_t> census_transform(ImageView<const std::uint16_t> img, bool scalar_reference) {
  if (img.channels() != 1) {
    throw_error("census_transform: expected a single-channel image, got {} channels", img.channels());
  }
  const int w = img.width();
  const int h = img.height();
  Image<std::uint64_t> out = Image<std::uint64_t>::uninitialized(w, h, 1);
  if (img.empty()) {
    return out;
  }
  constexpr int kPad = kCensusWidth / 2;
  const auto padded_width = static_cast<std::size_t>(w + 2 * kPad);
  parallel_for(0, static_cast<std::size_t>(h), 16, [&](std::size_t begin, std::size_t end) {
    // Horizontally padded copies of the rows of the window, reused as a ring.
    std::vector<std::uint16_t> buffer(padded_width * kCensusHeight);
    std::vector<int> cached(kCensusHeight, -1);
    auto padded_row = [&](int src_y) {
      const int slot = ((src_y % kCensusHeight) + kCensusHeight) % kCensusHeight;
      std::uint16_t* p = buffer.data() + static_cast<std::size_t>(slot) * padded_width;
      if (cached[static_cast<std::size_t>(slot)] != src_y) {
        const std::uint16_t* s = img.row(std::clamp(src_y, 0, h - 1));
        std::fill(p, p + kPad, s[0]);
        std::memcpy(p + kPad, s, static_cast<std::size_t>(w) * sizeof(std::uint16_t));
        std::fill(p + kPad + w, p + padded_width, s[w - 1]);
        cached[static_cast<std::size_t>(slot)] = src_y;
      }
      return static_cast<const std::uint16_t*>(p);
    };
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      const std::uint16_t* rows[kCensusHeight];
      for (int j = 0; j < kCensusHeight; ++j) {
        // Clamped rows map to the same source row; the ring slot is keyed by
        // the clamped index so a row is never evicted while in use.
        rows[j] = padded_row(std::clamp(y - kCensusHeight / 2 + j, 0, h - 1));
      }
      if (scalar_reference) {
        scalar::census_row(rows, out.row(y), static_cast<std::size_t>(w));
      } else {
        census_row(rows, out.row(y), static_cast<std::size_t>(w));
      }
    }
  });
  return out;
}

}  // namespace r3d

#endif  // HWY_ONCE
