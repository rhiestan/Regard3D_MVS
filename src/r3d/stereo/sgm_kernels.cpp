// Compiled once per SIMD target via Highway's foreach_target mechanism.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "r3d/stereo/sgm_kernels.cpp"
#include <hwy/foreach_target.h>  // IWYU pragma: keep

#include <hwy/highway.h>

#include "r3d/stereo/sgm_kernels.hpp"

#ifndef R3D_STEREO_SGM_PIXEL
#define R3D_STEREO_SGM_PIXEL
#include <algorithm>

namespace r3d::sgm::detail {

// Inputs and outputs of the four paths at one pixel.
struct PixelPaths {
  const std::uint16_t* prev[4];  // label 0 of this pixel's range in the predecessor's costs
  std::uint16_t* cur[4];         // label 0 of this pixel
  std::uint16_t min_prev[4];
  std::uint16_t penalty[4];      // min_prev + P2, saturated
  std::uint16_t min_cur[4];
};

// Moves path costs for align_predecessor (Mover::fill: kSentinel at
// g[-1 .. num], g has kLabelOffset elements of padding on both sides;
// Mover::copy: n elements).
struct ScalarMover {
  static void fill(std::uint16_t* g, std::size_t num) { std::fill(g - 1, g + num + 1, kSentinel); }
  static void copy(const std::uint16_t* src, std::uint16_t* dst, std::size_t n) { std::copy(src, src + n, dst); }
};

// Predecessor path costs aligned to the labels [lo, lo + num) of the current
// pixel, including the d - 1 / d + 1 neighbours at both ends. Points into q's
// buffer if q's labels cover the range; otherwise the overlap is copied into
// `buffer` (labels missing in q read as kSentinel).
template <class Mover>
inline const std::uint16_t* align_predecessor(const std::uint16_t* q, std::int32_t q_lo, std::size_t q_num,
                                              std::int32_t lo, std::size_t num, std::uint16_t* buffer) {
  const auto q_hi = q_lo + static_cast<std::int32_t>(q_num);  // exclusive
  const auto hi = lo + static_cast<std::int32_t>(num);
  if (lo >= q_lo && hi <= q_hi) {
    return q + (lo - q_lo);  // the sentinels of q cover the ends
  }
  std::uint16_t* g = buffer + kLabelOffset;
  Mover::fill(g, num);
  const std::int32_t first = std::max(lo - 1, q_lo);
  const std::int32_t last = std::min(hi + 1, q_hi);  // exclusive
  if (first < last) {
    Mover::copy(q + (first - q_lo), g + (first - lo), static_cast<std::size_t>(last - first));
  }
  return g;
}

// Pointers for pixel x (i-th pixel of the pass).
template <class Mover>
inline void setup_pixel(const SgmRowArgs& a, std::size_t x, std::size_t i, const std::uint16_t* horiz_prev,
                        std::uint16_t horiz_min, std::uint16_t* horiz_cur, PixelPaths& p) {
  const std::size_t w = a.width;
  const std::size_t stride = path_stride(a.max_num);
  const std::int32_t lo = a.labels.lo[x];
  const std::size_t num = a.labels.num[x];
  const std::uint16_t* start = a.scratch + kLabelOffset;
  std::uint16_t* align_buffers = a.scratch + 3 * stride;
  if (i == 0) {
    p.prev[0] = start;
    p.min_prev[0] = 0;
  } else {
    const std::size_t xq = a.direction > 0 ? x - 1 : x + 1;
    p.prev[0] = align_predecessor<Mover>(horiz_prev + kLabelOffset, a.labels.lo[xq], a.labels.num[xq], lo, num,
                                         align_buffers);
    p.min_prev[0] = horiz_min;
  }
  if (a.has_prev) {
    const std::size_t prev_size = a.prev_labels.path_size(w);
    const auto xs = static_cast<std::ptrdiff_t>(x);
    const std::ptrdiff_t xps[3] = {xs, xs - a.direction, xs + a.direction};
    for (std::size_t path = 0; path < 3; ++path) {
      const std::ptrdiff_t xp = xps[path];
      if (xp < 0 || xp >= static_cast<std::ptrdiff_t>(w)) {
        p.prev[path + 1] = start;
        p.min_prev[path + 1] = 0;
        continue;
      }
      const auto xq = static_cast<std::size_t>(xp);
      const std::uint16_t* q = a.prev + path * prev_size + a.prev_labels.path_offset(xq);
      p.prev[path + 1] = align_predecessor<Mover>(q, a.prev_labels.lo[xq], a.prev_labels.num[xq], lo, num,
                                                  align_buffers + (path + 1) * stride);
      p.min_prev[path + 1] = a.prev_min[path * w + xq];
    }
  } else {
    for (std::size_t k = 1; k < 4; ++k) {
      p.prev[k] = start;
      p.min_prev[k] = 0;
    }
  }
  p.cur[0] = horiz_cur + kLabelOffset;
  const std::size_t cur_size = a.labels.path_size(w);
  const std::size_t cur_offset = a.labels.path_offset(x);
  for (std::size_t k = 1; k < 4; ++k) {
    p.cur[k] = a.cur + (k - 1) * cur_size + cur_offset;
  }
  // Sentinels of the vertical paths; those of path 0 are set by run_row
  // (writing them here would stall the next pixel's loads).
  for (std::size_t k = 1; k < 4; ++k) {
    p.cur[k][-1] = kSentinel;
    p.cur[k][num] = kSentinel;
  }
  for (std::size_t k = 0; k < 4; ++k) {
    const unsigned pen = unsigned{p.min_prev[k]} + a.p2[k * w + x];
    p.penalty[k] = static_cast<std::uint16_t>(pen > 0xFFFFU ? 0xFFFFU : pen);
    p.min_cur[k] = 0xFFFF;
  }
}

// Runs the row with `pixel(cost, sum, num, paths)` doing the label loop.
template <class Mover, class PixelFn>
inline void run_row(const SgmRowArgs& a, PixelFn&& pixel) {
  const std::size_t w = a.width;
  const std::size_t stride = path_stride(a.max_num);
  std::uint16_t* horiz[2] = {a.scratch + stride, a.scratch + 2 * stride};
  // Path 0 alternates between two buffers. The sentinel before label 0 never
  // moves; the one after the last label moves only when the label count
  // differs from the buffer's previous pixel.
  std::size_t horiz_num[2] = {0, 0};
  for (std::uint16_t* h : horiz) {
    h[kLabelOffset - 1] = kSentinel;
  }
  std::uint16_t horiz_min = 0;
  for (std::size_t i = 0; i < w; ++i) {
    const std::size_t x = a.direction > 0 ? i : w - 1 - i;
    const std::size_t num = a.labels.num[x];
    if (horiz_num[i & 1] != num) {
      horiz[i & 1][kLabelOffset + num] = kSentinel;
      horiz_num[i & 1] = num;
    }
    PixelPaths p;
    setup_pixel<Mover>(a, x, i, horiz[(i + 1) & 1], horiz_min, horiz[i & 1], p);
    const std::size_t off = a.labels.offset[x];
    pixel(a.cost + off, a.sum + off, std::size_t{a.labels.num[x]}, p);
    horiz_min = p.min_cur[0];
    for (std::size_t k = 1; k < 4; ++k) {
      a.cur_min[(k - 1) * w + x] = p.min_cur[k];
    }
  }
}

}  // namespace r3d::sgm::detail
#endif

HWY_BEFORE_NAMESPACE();
namespace r3d {
namespace sgm {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void CensusCosts(const std::uint64_t* HWY_RESTRICT left, const std::uint64_t* HWY_RESTRICT rev, std::size_t width,
                 const RowLabels& labels, std::int32_t rev_origin, std::uint16_t* HWY_RESTRICT cost) {
  // At most 16 lanes, so that every label count (a multiple of 16) is a
  // whole number of vectors.
  const hn::CappedTag<std::uint64_t, kLabelAlign> d64;
  const hn::Rebind<std::uint16_t, decltype(d64)> d16;
  const std::size_t lanes = hn::Lanes(d64);
  for (std::size_t x = 0; x < width; ++x) {
    const auto l = hn::Set(d64, left[x]);
    const std::uint64_t* r = rev + (static_cast<std::ptrdiff_t>(width - 1 - x) + labels.lo[x] - rev_origin);
    std::uint16_t* c = cost + labels.offset[x];
    const std::size_t num = labels.num[x];
    for (std::size_t d = 0; d < num; d += lanes) {
      const auto bits = hn::PopulationCount(hn::Xor(l, hn::LoadU(d64, r + d)));
      hn::StoreU(hn::TruncateTo(d16, bits), d16, c + d);
    }
  }
}

template <class DT>
HWY_INLINE hn::Vec<DT> PathStep(DT dt, const std::uint16_t* HWY_RESTRICT prev, std::uint16_t* HWY_RESTRICT cur,
                                hn::Vec<DT> cost, hn::Vec<DT> p1, hn::Vec<DT> penalty, hn::Vec<DT> min_prev,
                                hn::Vec<DT>& min_cur) {
  const auto same = hn::LoadU(dt, prev);
  const auto neighbours = hn::Min(hn::LoadU(dt, prev - 1), hn::LoadU(dt, prev + 1));
  const auto best = hn::Min(hn::Min(same, hn::SaturatedAdd(neighbours, p1)), penalty);
  const auto l = hn::Add(hn::Sub(best, min_prev), cost);
  hn::StoreU(l, dt, cur);
  min_cur = hn::Min(min_cur, l);
  return l;
}

// Labels [d0, d1) of one pixel with vectors of tag DT.
template <class DT>
HWY_INLINE void PixelLabels(DT dt, std::size_t d0, std::size_t d1, const std::uint16_t* HWY_RESTRICT cost,
                            std::uint16_t* HWY_RESTRICT sum, std::uint16_t p1_value, bool accumulate,
                            detail::PixelPaths& p) {
  if (d0 >= d1) {
    return;
  }
  const std::size_t lanes = hn::Lanes(dt);
  const auto p1 = hn::Set(dt, p1_value);
  const auto pen0 = hn::Set(dt, p.penalty[0]);
  const auto pen1 = hn::Set(dt, p.penalty[1]);
  const auto pen2 = hn::Set(dt, p.penalty[2]);
  const auto pen3 = hn::Set(dt, p.penalty[3]);
  const auto mp0 = hn::Set(dt, p.min_prev[0]);
  const auto mp1 = hn::Set(dt, p.min_prev[1]);
  const auto mp2 = hn::Set(dt, p.min_prev[2]);
  const auto mp3 = hn::Set(dt, p.min_prev[3]);
  auto m0 = hn::Set(dt, std::uint16_t{0xFFFF});
  auto m1 = m0;
  auto m2 = m0;
  auto m3 = m0;
  for (std::size_t d = d0; d < d1; d += lanes) {
    const auto c = hn::LoadU(dt, cost + d);
    const auto l0 = PathStep(dt, p.prev[0] + d, p.cur[0] + d, c, p1, pen0, mp0, m0);
    const auto l1 = PathStep(dt, p.prev[1] + d, p.cur[1] + d, c, p1, pen1, mp1, m1);
    const auto l2 = PathStep(dt, p.prev[2] + d, p.cur[2] + d, c, p1, pen2, mp2, m2);
    const auto l3 = PathStep(dt, p.prev[3] + d, p.cur[3] + d, c, p1, pen3, mp3, m3);
    auto total = hn::Add(hn::Add(l0, l1), hn::Add(l2, l3));
    if (accumulate) {
      total = hn::Add(total, hn::LoadU(dt, sum + d));
    }
    hn::StoreU(total, dt, sum + d);
  }
  const std::uint16_t mins[4] = {hn::ReduceMin(dt, m0), hn::ReduceMin(dt, m1), hn::ReduceMin(dt, m2),
                                 hn::ReduceMin(dt, m3)};
  for (std::size_t k = 0; k < 4; ++k) {
    p.min_cur[k] = mins[k] < p.min_cur[k] ? mins[k] : p.min_cur[k];
  }
}

// Vector moves for align_predecessor (see detail::ScalarMover).
struct VectorMover {
  static void fill(std::uint16_t* g, std::size_t num) {
    // [g - kLabelOffset, g + num + kLabelOffset) in whole vectors of kLabelAlign lanes.
    const hn::CappedTag<std::uint16_t, kLabelAlign> d;
    const auto s = hn::Set(d, kSentinel);
    std::uint16_t* p = g - kLabelOffset;
    for (std::size_t k = 0; k < num + 2 * kLabelOffset; k += hn::Lanes(d)) {
      hn::StoreU(s, d, p + k);
    }
  }
  static void copy(const std::uint16_t* HWY_RESTRICT src, std::uint16_t* HWY_RESTRICT dst, std::size_t n) {
    const hn::CappedTag<std::uint16_t, kLabelAlign> d;
    const std::size_t lanes = hn::Lanes(d);
    std::size_t k = 0;
    for (; k + lanes <= n; k += lanes) {
      hn::StoreU(hn::LoadU(d, src + k), d, dst + k);
    }
    if (k < n) {
      hn::StoreN(hn::LoadN(d, src + k, n - k), d, dst + k, n - k);
    }
  }
};

void SgmRow(const SgmRowArgs& a) {
  const hn::ScalableTag<std::uint16_t> d;
  const hn::CappedTag<std::uint16_t, kLabelAlign> d_tail;
  const std::size_t lanes = hn::Lanes(d);
  detail::run_row<VectorMover>(a, [&](const std::uint16_t* cost, std::uint16_t* sum, std::size_t n, detail::PixelPaths& p) {
    const std::size_t main_end = n - n % lanes;
    PixelLabels(d, 0, main_end, cost, sum, a.p1, a.accumulate, p);
    PixelLabels(d_tail, main_end, n, cost, sum, a.p1, a.accumulate, p);
  });
}

// Lowest cost of labels [d0, d1).
template <class DT>
HWY_INLINE std::uint16_t RangeMin(DT dt, const std::uint16_t* HWY_RESTRICT s, std::size_t d0, std::size_t d1,
                                  std::uint16_t init) {
  if (d0 >= d1) {
    return init;
  }
  auto acc = hn::Set(dt, init);
  for (std::size_t d = d0; d < d1; d += hn::Lanes(dt)) {
    acc = hn::Min(acc, hn::LoadU(dt, s + d));
  }
  return hn::ReduceMin(dt, acc);
}

// First label of [d0, d1) with cost `value`, or d1.
template <class DT>
HWY_INLINE std::size_t FindValue(DT dt, const std::uint16_t* HWY_RESTRICT s, std::size_t d0, std::size_t d1,
                                 std::uint16_t value) {
  const auto v = hn::Set(dt, value);
  for (std::size_t d = d0; d < d1; d += hn::Lanes(dt)) {
    const auto eq = hn::Eq(hn::LoadU(dt, s + d), v);
    if (!hn::AllFalse(dt, eq)) {
      return d + hn::FindKnownFirstTrue(dt, eq);
    }
  }
  return d1;
}

// Lowest cost of labels [d0, d1) outside [lo, hi].
template <class DT>
HWY_INLINE std::uint16_t RangeMinExcept(DT dt, const std::uint16_t* HWY_RESTRICT s, std::size_t d0, std::size_t d1,
                                        std::uint16_t lo, std::uint16_t hi, std::uint16_t init) {
  if (d0 >= d1) {
    return init;
  }
  const auto vlo = hn::Set(dt, lo);
  const auto vhi = hn::Set(dt, hi);
  const auto inf = hn::Set(dt, std::uint16_t{0xFFFF});
  auto acc = hn::Set(dt, init);
  for (std::size_t d = d0; d < d1; d += hn::Lanes(dt)) {
    const auto k = hn::Iota(dt, static_cast<std::uint16_t>(d));
    const auto excluded = hn::AndNot(hn::Lt(k, vlo), hn::Not(hn::Lt(vhi, k)));
    acc = hn::Min(acc, hn::IfThenElse(excluded, inf, hn::LoadU(dt, s + d)));
  }
  return hn::ReduceMin(dt, acc);
}

// Right-image WTA update of labels [d0, d1) of one left pixel; label j of
// the pixel (absolute label first + j) belongs to right index top - j.
template <class DT>
HWY_INLINE void RightUpdate(DT dt, const std::uint16_t* HWY_RESTRICT s, std::size_t d0, std::size_t d1,
                            std::size_t top, std::uint16_t first, std::uint16_t* HWY_RESTRICT right_cost,
                            std::uint16_t* HWY_RESTRICT right_best) {
  const std::size_t lanes = hn::Lanes(dt);
  for (std::size_t d = d0; d < d1; d += lanes) {
    // Labels d .. d + lanes - 1 belong to right indices top - d down to top - d - lanes + 1.
    const std::size_t base = top - d - (lanes - 1);
    const auto v = hn::LoadU(dt, s + d);
    const auto rc = hn::Reverse(dt, hn::LoadU(dt, right_cost + base));
    const auto rb = hn::Reverse(dt, hn::LoadU(dt, right_best + base));
    const auto better = hn::Lt(v, rc);
    hn::StoreU(hn::Reverse(dt, hn::IfThenElse(better, v, rc)), dt, right_cost + base);
    hn::StoreU(hn::Reverse(dt, hn::IfThenElse(better, hn::Iota(dt, static_cast<std::uint16_t>(first + d)), rb)), dt,
               right_best + base);
  }
}

void WtaRow(const std::uint16_t* HWY_RESTRICT sum, std::size_t width, const RowLabels& labels,
            std::uint16_t* HWY_RESTRICT best, std::uint16_t* HWY_RESTRICT best_cost, std::uint16_t* HWY_RESTRICT second,
            std::uint16_t* HWY_RESTRICT right_cost, std::uint16_t* HWY_RESTRICT right_best, std::ptrdiff_t right_shift) {
  const hn::ScalableTag<std::uint16_t> d;
  const hn::CappedTag<std::uint16_t, kLabelAlign> d_tail;
  for (std::size_t x = 0; x < width; ++x) {
    const std::size_t n = labels.num[x];
    const std::size_t main_end = n - n % hn::Lanes(d);
    const std::uint16_t* s = sum + labels.offset[x];
    const std::uint16_t m = RangeMin(d_tail, s, main_end, n, RangeMin(d, s, 0, main_end, 0xFFFF));
    std::size_t k = FindValue(d, s, 0, main_end, m);
    if (k == main_end) {
      k = FindValue(d_tail, s, main_end, n, m);
    }
    const auto first = static_cast<std::uint16_t>(labels.lo[x]);
    best[x] = static_cast<std::uint16_t>(first + k);
    best_cost[x] = m;
    const auto lo = static_cast<std::uint16_t>(k > 0 ? k - 1 : 0);
    const auto hi = static_cast<std::uint16_t>(k + 1);
    second[x] = RangeMinExcept(d_tail, s, main_end, n, lo, hi, RangeMinExcept(d, s, 0, main_end, lo, hi, 0xFFFF));
    if (right_cost != nullptr) {
      const auto top = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(x) - labels.lo[x] + right_shift);
      RightUpdate(d, s, 0, main_end, top, first, right_cost, right_best);
      RightUpdate(d_tail, s, main_end, n, top, first, right_cost, right_best);
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace sgm
}  // namespace r3d
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

#include <bit>

namespace r3d::sgm {

HWY_EXPORT(CensusCosts);
HWY_EXPORT(SgmRow);
HWY_EXPORT(WtaRow);

namespace scalar {

void census_costs(const std::uint64_t* left, const std::uint64_t* rev, std::size_t width, const RowLabels& labels,
                  std::int32_t rev_origin, std::uint16_t* cost) {
  for (std::size_t x = 0; x < width; ++x) {
    const std::uint64_t* r = rev + (static_cast<std::ptrdiff_t>(width - 1 - x) + labels.lo[x] - rev_origin);
    for (std::size_t d = 0; d < labels.num[x]; ++d) {
      cost[labels.offset[x] + d] = static_cast<std::uint16_t>(std::popcount(left[x] ^ r[d]));
    }
  }
}

void sgm_row(const SgmRowArgs& a) {
  detail::run_row<detail::ScalarMover>(a, [&](const std::uint16_t* cost, std::uint16_t* sum, std::size_t n, detail::PixelPaths& p) {
    for (std::size_t d = 0; d < n; ++d) {
      unsigned total = 0;
      for (std::size_t k = 0; k < 4; ++k) {
        const std::uint16_t* q = p.prev[k] + d;
        const unsigned neighbours = std::min<unsigned>(q[-1], q[1]) + a.p1;
        const unsigned best = std::min({unsigned{q[0]}, std::min(neighbours, 0xFFFFU), unsigned{p.penalty[k]}});
        const auto l = static_cast<std::uint16_t>(best - p.min_prev[k] + cost[d]);
        p.cur[k][d] = l;
        p.min_cur[k] = std::min(p.min_cur[k], l);
        total += l;
      }
      sum[d] = static_cast<std::uint16_t>(a.accumulate ? total + sum[d] : total);
    }
  });
}

void wta_row(const std::uint16_t* sum, std::size_t width, const RowLabels& labels, std::uint16_t* best,
             std::uint16_t* best_cost, std::uint16_t* second, std::uint16_t* right_cost, std::uint16_t* right_best,
             std::ptrdiff_t right_shift) {
  for (std::size_t x = 0; x < width; ++x) {
    const std::uint16_t* s = sum + labels.offset[x];
    const std::size_t n = labels.num[x];
    std::size_t k_best = 0;
    for (std::size_t k = 1; k < n; ++k) {
      if (s[k] < s[k_best]) {
        k_best = k;
      }
    }
    std::uint16_t sec = 0xFFFF;
    for (std::size_t k = 0; k < n; ++k) {
      if (k + 1 < k_best || k > k_best + 1) {
        sec = std::min(sec, s[k]);
      }
    }
    const std::int32_t lo = labels.lo[x];
    best[x] = static_cast<std::uint16_t>(lo + static_cast<std::int32_t>(k_best));
    best_cost[x] = s[k_best];
    second[x] = sec;
    if (right_cost != nullptr) {
      for (std::size_t k = 0; k < n; ++k) {
        const auto i = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(x) - lo - static_cast<std::ptrdiff_t>(k) +
                                                right_shift);
        if (s[k] < right_cost[i]) {
          right_cost[i] = s[k];
          right_best[i] = static_cast<std::uint16_t>(lo + static_cast<std::int32_t>(k));
        }
      }
    }
  }
}

}  // namespace scalar

void wta_row(const std::uint16_t* sum, std::size_t width, const RowLabels& labels, std::uint16_t* best,
             std::uint16_t* best_cost, std::uint16_t* second, std::uint16_t* right_cost, std::uint16_t* right_best,
             std::ptrdiff_t right_shift) {
  HWY_DYNAMIC_DISPATCH(WtaRow)(sum, width, labels, best, best_cost, second, right_cost, right_best, right_shift);
}

void census_costs(const std::uint64_t* left, const std::uint64_t* rev, std::size_t width, const RowLabels& labels,
                  std::int32_t rev_origin, std::uint16_t* cost) {
  HWY_DYNAMIC_DISPATCH(CensusCosts)(left, rev, width, labels, rev_origin, cost);
}

void sgm_row(const SgmRowArgs& args) { HWY_DYNAMIC_DISPATCH(SgmRow)(args); }

}  // namespace r3d::sgm

#endif  // HWY_ONCE
