#include "r3d/stereo/stereo_matcher.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <semaphore>
#include <utility>

#include "r3d/core/aligned_buffer.hpp"
#include "r3d/core/error.hpp"
#include "r3d/core/log.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/stereo/census.hpp"
#include "r3d/stereo/sgm_kernels.hpp"

namespace r3d {
namespace {

using sgm::kLabelAlign;
using sgm::kLabelOffset;
using sgm::kSentinel;
using sgm::path_stride;

// Valid columns [first, last] of a row (empty if last < first).
struct RowInterval {
  int first = 0;
  int last = -1;
};

std::vector<RowInterval> valid_intervals(ImageView<const std::uint8_t> mask, int width, int height) {
  std::vector<RowInterval> rows(static_cast<std::size_t>(height), RowInterval{0, width - 1});
  if (mask.empty()) {
    return rows;
  }
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* m = mask.row(y);
    RowInterval& r = rows[static_cast<std::size_t>(y)];
    r = RowInterval{};
    int x = 0;
    while (x < width && m[x] < 128) {
      ++x;
    }
    if (x == width) {
      continue;
    }
    r.first = x;
    x = width - 1;
    while (m[x] < 128) {
      --x;
    }
    r.last = x;
  }
  return rows;
}

std::array<std::uint16_t, 256> p2_table(const StereoOptions& o) {
  std::array<std::uint16_t, 256> lut{};
  for (int di = 0; di < 256; ++di) {
    int p2 = o.p2;
    if (o.p2_adapt > 0) {
      p2 = std::max(o.p1 + 1, o.p2 * o.p2_adapt / (o.p2_adapt + di));
    }
    lut[static_cast<std::size_t>(di)] = static_cast<std::uint16_t>(p2);
  }
  return lut;
}

void check_options(const StereoOptions& o) {
  if (o.p1 < 0 || o.p2 < o.p1 || o.p2 > 8000) {
    throw_error("stereo: invalid penalties P1 = {}, P2 = {} (0 <= P1 <= P2 <= 8000)", o.p1, o.p2);
  }
  if (o.invalid_cost < 0 || o.invalid_cost > 255) {
    throw_error("stereo: invalid_cost {} outside [0, 255]", o.invalid_cost);
  }
  if (o.tile_size < 16 || o.tile_margin < 0 || o.tile_blend < 0) {
    throw_error("stereo: invalid tiling {} / {} / {}", o.tile_size, o.tile_margin, o.tile_blend);
  }
}

// Inputs shared by all tiles of a pair.
struct MatchContext {
  ImageView<const std::uint16_t> left;
  ImageView<const std::uint16_t> right;
  Image<std::uint64_t> census_left;
  Image<std::uint64_t> census_right;
  std::vector<RowInterval> left_rows;
  std::vector<RowInterval> right_rows;
  int disparity_min = 0;
  std::size_t num_disp = 0;
  std::array<std::uint16_t, 256> p2_lut{};
  const StereoOptions* options = nullptr;
  bool diagonal_lr = false;  // left-right check against the right disparities of the same volume
};

MatchContext make_context(ImageView<const std::uint16_t> left, ImageView<const std::uint16_t> right,
                          ImageView<const std::uint8_t> left_valid, ImageView<const std::uint8_t> right_valid,
                          int disparity_min, int num_disparities, const StereoOptions& options) {
  check_options(options);
  if (left.channels() != 1 || right.channels() != 1) {
    throw_error("stereo: expected single-channel images");
  }
  if (left.height() != right.height() || left.empty() || right.empty()) {
    throw_error("stereo: rectified images must be non-empty with equal height ({}x{}, {}x{})", left.width(),
                left.height(), right.width(), right.height());
  }
  if (!left_valid.empty() && (left_valid.width() != left.width() || left_valid.height() != left.height())) {
    throw_error("stereo: left mask size differs from the image");
  }
  if (!right_valid.empty() && (right_valid.width() != right.width() || right_valid.height() != right.height())) {
    throw_error("stereo: right mask size differs from the image");
  }
  if (num_disparities <= 0) {
    throw_error("stereo: no disparities to search");
  }
  MatchContext ctx;
  ctx.left = left;
  ctx.right = right;
  ctx.census_left = census_transform(left, options.scalar_kernels);
  ctx.census_right = census_transform(right, options.scalar_kernels);
  ctx.left_rows = valid_intervals(left_valid, left.width(), left.height());
  ctx.right_rows = valid_intervals(right_valid, right.width(), right.height());
  ctx.disparity_min = disparity_min;
  ctx.num_disp = (static_cast<std::size_t>(num_disparities) + kLabelAlign - 1) / kLabelAlign * kLabelAlign;
  ctx.p2_lut = p2_table(options);
  ctx.options = &options;
  return ctx;
}

struct Tile {
  int x0 = 0;
  int y0 = 0;
  int w = 0;
  int h = 0;
};

// Aggregation of one tile into volume[y][x][d] (tile coordinates).
class TileAggregator {
public:
  TileAggregator(const MatchContext& ctx, const Tile& tile)
      : ctx_(ctx),
        tile_(tile),
        w_(static_cast<std::size_t>(tile.w)),
        d_(ctx.num_disp),
        stride_(path_stride(ctx.num_disp)),
        rev_(w_ + d_ - 1),
        cost_(w_ * d_),
        p2_(4 * w_),
        prev_(3 * w_ * stride_),
        cur_(3 * w_ * stride_),
        prev_min_(3 * w_),
        cur_min_(3 * w_),
        scratch_(3 * stride_) {}

  void run(std::uint16_t* volume) {
    pass(1, volume);
    pass(-1, volume);
  }

  // Matching costs of tile row y (also used by the reference implementation's tests).
  const std::uint16_t* cost_row(int y) {
    const int big_y = tile_.y0 + y;
    const auto d = static_cast<int>(d_);
    const auto w = static_cast<int>(w_);
    const std::uint64_t* cl = ctx_.census_left.row(big_y) + tile_.x0;
    const std::uint64_t* cr = ctx_.census_right.row(big_y);
    const int wr = ctx_.right.width();
    // rev[k] is the right census at column base - k, so that the labels of
    // left column x are rev[w - 1 - x + d] (contiguous in d).
    const int base = tile_.x0 + w - 1 - ctx_.disparity_min;
    for (std::size_t k = 0; k < rev_.size(); ++k) {
      const int xr = base - static_cast<int>(k);
      rev_[k] = xr >= 0 && xr < wr ? cr[xr] : 0;
    }
    if (ctx_.options->scalar_kernels) {
      sgm::scalar::census_costs(cl, rev_.data(), w_, d_, cost_.data());
    } else {
      sgm::census_costs(cl, rev_.data(), w_, d_, cost_.data());
    }

    const RowInterval& lv = ctx_.left_rows[static_cast<std::size_t>(big_y)];
    const RowInterval& rv = ctx_.right_rows[static_cast<std::size_t>(big_y)];
    const auto invalid = static_cast<std::uint16_t>(ctx_.options->invalid_cost);
    for (int x = 0; x < w; ++x) {
      std::uint16_t* c = cost_.data() + static_cast<std::size_t>(x) * d_;
      const int big_x = tile_.x0 + x;
      if (big_x < lv.first || big_x > lv.last) {
        std::fill(c, c + d, std::uint16_t{0});  // no information: paths pass through
        continue;
      }
      // Label k is valid if the right column big_x - dmin - k is.
      int lo = big_x - ctx_.disparity_min - rv.last;
      int hi = big_x - ctx_.disparity_min - rv.first + 1;
      if (rv.last < rv.first) {
        lo = hi = 0;
      }
      lo = std::clamp(lo, 0, d);
      hi = std::clamp(hi, lo, d);
      std::fill(c, c + lo, invalid);
      std::fill(c + hi, c + d, invalid);
    }
    return cost_.data();
  }

private:
  void penalty_row(int y, int direction) {
    const int big_y = tile_.y0 + y;
    const std::uint16_t* img = ctx_.left.row(big_y) + tile_.x0;
    const int yp = y - direction;
    const bool has_prev_row = yp >= 0 && yp < tile_.h;
    const std::uint16_t* img_prev = has_prev_row ? ctx_.left.row(tile_.y0 + yp) + tile_.x0 : nullptr;
    const auto w = static_cast<int>(w_);
    const std::uint16_t p2 = static_cast<std::uint16_t>(ctx_.options->p2);
    auto pen = [&](std::uint16_t a, std::uint16_t b) {
      return ctx_.p2_lut[static_cast<std::size_t>(std::abs(int{a} - int{b}) >> 8)];
    };
    for (int x = 0; x < w; ++x) {
      const int xa = x - direction;
      const int xb = x + direction;
      const std::uint16_t v = img[x];
      p2_[0 * w_ + static_cast<std::size_t>(x)] = xa >= 0 && xa < w ? pen(v, img[xa]) : p2;
      p2_[1 * w_ + static_cast<std::size_t>(x)] = has_prev_row ? pen(v, img_prev[x]) : p2;
      p2_[2 * w_ + static_cast<std::size_t>(x)] = has_prev_row && xa >= 0 && xa < w ? pen(v, img_prev[xa]) : p2;
      p2_[3 * w_ + static_cast<std::size_t>(x)] = has_prev_row && xb >= 0 && xb < w ? pen(v, img_prev[xb]) : p2;
    }
  }

  // Zero path costs with sentinels around every pixel's labels.
  void reset(AlignedBuffer<std::uint16_t>& buffer) const {
    buffer.fill(0);
    for (std::size_t i = 0; i < buffer.size() / stride_; ++i) {
      std::uint16_t* slot = buffer.data() + i * stride_;
      slot[kLabelOffset - 1] = kSentinel;
      slot[kLabelOffset + d_] = kSentinel;
    }
  }

  void pass(int direction, std::uint16_t* volume) {
    reset(prev_);
    reset(cur_);
    reset(scratch_);
    prev_min_.fill(0);
    cur_min_.fill(0);
    for (int i = 0; i < tile_.h; ++i) {
      const int y = direction > 0 ? i : tile_.h - 1 - i;
      sgm::SgmRowArgs args;
      args.cost = cost_row(y);
      penalty_row(y, direction);
      args.p2 = p2_.data();
      args.prev = prev_.data();
      args.prev_min = prev_min_.data();
      args.cur = cur_.data();
      args.cur_min = cur_min_.data();
      args.scratch = scratch_.data();
      args.sum = volume + static_cast<std::size_t>(y) * w_ * d_;
      args.width = w_;
      args.num_disp = d_;
      args.p1 = static_cast<std::uint16_t>(ctx_.options->p1);
      args.direction = direction;
      args.accumulate = direction < 0;
      if (ctx_.options->scalar_kernels) {
        sgm::scalar::sgm_row(args);
      } else {
        sgm::sgm_row(args);
      }
      prev_.swap(cur_);
      prev_min_.swap(cur_min_);
    }
  }

  const MatchContext& ctx_;
  Tile tile_;
  std::size_t w_;
  std::size_t d_;
  std::size_t stride_;
  AlignedBuffer<std::uint64_t> rev_;
  AlignedBuffer<std::uint16_t> cost_;
  AlignedBuffer<std::uint16_t> p2_;
  AlignedBuffer<std::uint16_t> prev_;
  AlignedBuffer<std::uint16_t> cur_;
  AlignedBuffer<std::uint16_t> prev_min_;
  AlignedBuffer<std::uint16_t> cur_min_;
  AlignedBuffer<std::uint16_t> scratch_;
};

float subpixel_offset(SubpixelMethod method, float sm, float s0, float sp) {
  float delta = 0.0F;
  if (method == SubpixelMethod::parabola) {
    const float denom = sm - 2.0F * s0 + sp;
    delta = denom > 0.0F ? (sm - sp) / (2.0F * denom) : 0.0F;
  } else if (method == SubpixelMethod::equiangular) {
    const float slope = std::max(sm - s0, sp - s0);
    delta = slope > 0.0F ? (sm - sp) / (2.0F * slope) : 0.0F;
  }
  return std::clamp(delta, -0.5F, 0.5F);
}

struct TileResult {
  Tile tile;
  std::vector<float> disparity;
  std::vector<float> confidence;
};

// Winner-takes-all with sub-pixel interpolation, uniqueness and left-right check.
void select_disparities(const MatchContext& ctx, const std::uint16_t* volume, TileResult& out) {
  const Tile& t = out.tile;
  const StereoOptions& o = *ctx.options;
  const std::size_t w = static_cast<std::size_t>(t.w);
  const std::size_t nd = ctx.num_disp;
  const auto d = static_cast<int>(nd);
  out.disparity.assign(w * static_cast<std::size_t>(t.h), kInvalidDisparity);
  out.confidence.assign(w * static_cast<std::size_t>(t.h), 0.0F);
  // Best label of every right column reachable from the tile: index x - k + d - 1.
  std::vector<std::uint16_t> right_cost(w + nd - 1);
  std::vector<std::uint16_t> right_best(w + nd - 1);
  std::vector<std::uint16_t> best_label(w), best_value(w), second_value(w);
  for (int y = 0; y < t.h; ++y) {
    const std::uint16_t* s_row = volume + static_cast<std::size_t>(y) * w * nd;
    const int big_y = t.y0 + y;
    if (ctx.diagonal_lr) {
      std::fill(right_cost.begin(), right_cost.end(), std::uint16_t{0xFFFF});
      std::fill(right_best.begin(), right_best.end(), std::uint16_t{0xFFFF});
    }
    // Only valid left pixels: masked ones have zero costs, which would win
    // the right-image WTA of the left-right check.
    const RowInterval& lv = ctx.left_rows[static_cast<std::size_t>(big_y)];
    const RowInterval& rv = ctx.right_rows[static_cast<std::size_t>(big_y)];
    const auto xb = static_cast<std::size_t>(std::clamp(lv.first - t.x0, 0, t.w));
    const auto xe = static_cast<std::size_t>(std::clamp(lv.last + 1 - t.x0, static_cast<int>(xb), t.w));
    if (xb < xe) {
      std::uint16_t* rc = ctx.diagonal_lr ? right_cost.data() + xb : nullptr;
      const auto wta = o.scalar_kernels ? sgm::scalar::wta_row : sgm::wta_row;
      wta(s_row + xb * nd, xe - xb, nd, best_label.data() + xb, best_value.data() + xb, second_value.data() + xb, rc,
          right_best.data() + xb);
    }
    for (std::size_t x = 0; x < w; ++x) {
      const int big_x = t.x0 + static_cast<int>(x);
      if (big_x < lv.first || big_x > lv.last) {
        continue;
      }
      const std::uint16_t* s = s_row + x * nd;
      const int best = best_label[x];
      const unsigned second = second_value[x];
      const int xr = big_x - ctx.disparity_min - best;
      if (xr < rv.first || xr > rv.last) {
        continue;
      }
      if (ctx.diagonal_lr &&
          std::abs(int{right_best[x + nd - 1 - static_cast<std::size_t>(best)]} - best) > o.lr_max_diff) {
        continue;
      }
      const float best_cost = best_value[x];
      if (o.uniqueness > 0.0F && second != 0xFFFFU &&
          best_cost > (1.0F - o.uniqueness) * static_cast<float>(second)) {
        continue;
      }
      float delta = 0.0F;
      if (best > 0 && best < d - 1) {
        delta = subpixel_offset(o.subpixel, s[best - 1], best_cost, s[best + 1]);
      }
      const std::size_t i = static_cast<std::size_t>(y) * w + x;
      out.disparity[i] = static_cast<float>(ctx.disparity_min + best) + delta;
      out.confidence[i] = second != 0xFFFFU && second > 0 ? 1.0F - best_cost / static_cast<float>(second) : 0.0F;
    }
  }
}

// Weight of a tile at image coordinate v along one axis (core [c0, c1) of an
// image of `size`): linear ramps of width `blend` centred on interior core
// boundaries, so the weights of neighbouring tiles sum to one.
float axis_weight(int v, int c0, int c1, int size, int blend_px) {
  const auto blend = static_cast<float>(blend_px);
  float w = 1.0F;
  const float pos = static_cast<float>(v) + 0.5F;
  if (c0 > 0) {
    w = std::min(w, blend_px > 0 ? std::clamp((pos - static_cast<float>(c0) + 0.5F * blend) / blend, 0.0F, 1.0F)
                                 : (v >= c0 ? 1.0F : 0.0F));
  }
  if (c1 < size) {
    w = std::min(w, blend_px > 0 ? std::clamp((static_cast<float>(c1) + 0.5F * blend - pos) / blend, 0.0F, 1.0F)
                                 : (v < c1 ? 1.0F : 0.0F));
  }
  return w;
}

std::vector<int> split(int size, int tile) {
  const int n = std::max(1, (size + tile - 1) / tile);
  std::vector<int> bounds(static_cast<std::size_t>(n) + 1);
  for (int i = 0; i <= n; ++i) {
    bounds[static_cast<std::size_t>(i)] = static_cast<int>(static_cast<long long>(size) * i / n);
  }
  return bounds;
}

std::size_t tile_bytes(int w, int h, std::size_t num_disp) {
  const auto pixels = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  // Volume + path buffers (6 rows) + results.
  return pixels * num_disp * 2 + 6 * static_cast<std::size_t>(w) * path_stride(num_disp) * 2 + pixels * 8;
}

template <typename T>
Image<T> mirror(ImageView<const T> img) {
  Image<T> out = Image<T>::uninitialized(img.width(), img.height(), 1);
  for (int y = 0; y < img.height(); ++y) {
    std::reverse_copy(img.row(y), img.row(y) + img.width(), out.row(y));
  }
  return out;
}

DisparityMap match_tiles(const MatchContext& ctx);

}  // namespace

void remove_speckles(ImageF32& disparity, int min_size, float max_diff) {
  if (min_size <= 1 || disparity.empty()) {
    return;
  }
  const int w = disparity.width();
  const int h = disparity.height();
  Image<std::int32_t> label(w, h, 1);  // 0: unvisited
  std::vector<std::pair<int, int>> stack;
  std::vector<std::pair<int, int>> segment;
  std::int32_t next = 1;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (label(x, y) != 0 || !std::isfinite(disparity(x, y))) {
        continue;
      }
      segment.clear();
      stack.assign(1, {x, y});
      label(x, y) = next;
      while (!stack.empty()) {
        const auto [px, py] = stack.back();
        stack.pop_back();
        segment.emplace_back(px, py);
        const float v = disparity(px, py);
        const std::pair<int, int> nbs[4] = {{px - 1, py}, {px + 1, py}, {px, py - 1}, {px, py + 1}};
        for (const auto& [qx, qy] : nbs) {
          if (qx < 0 || qy < 0 || qx >= w || qy >= h || label(qx, qy) != 0) {
            continue;
          }
          const float u = disparity(qx, qy);
          if (std::isfinite(u) && std::abs(u - v) <= max_diff) {
            label(qx, qy) = next;
            stack.emplace_back(qx, qy);
          }
        }
      }
      ++next;
      if (static_cast<int>(segment.size()) < min_size) {
        for (const auto& [px, py] : segment) {
          disparity(px, py) = kInvalidDisparity;
        }
      }
    }
  }
}

ImageF32 median_filter_3x3(const ImageF32& disparity) {
  const int w = disparity.width();
  const int h = disparity.height();
  ImageF32 out = ImageF32::uninitialized(w, h, 1);
  parallel_for(0, static_cast<std::size_t>(h), 16, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      for (int x = 0; x < w; ++x) {
        const float c = disparity(x, y);
        if (!std::isfinite(c)) {
          out(x, y) = c;
          continue;
        }
        std::array<float, 9> v{};
        std::size_t n = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            const int qx = x + dx;
            const int qy = y + dy;
            if (qx >= 0 && qy >= 0 && qx < w && qy < h && std::isfinite(disparity(qx, qy))) {
              v[n++] = disparity(qx, qy);
            }
          }
        }
        const auto mid = v.begin() + static_cast<std::ptrdiff_t>(n / 2);
        std::nth_element(v.begin(), mid, v.begin() + static_cast<std::ptrdiff_t>(n));
        float m = *mid;
        if (n % 2 == 0) {  // even count: mean of the two middle values
          m = 0.5F * (m + *std::max_element(v.begin(), mid));
        }
        out(x, y) = m;
      }
    }
  });
  return out;
}

DisparityMap match_stereo(ImageView<const std::uint16_t> left, ImageView<const std::uint16_t> right,
                          ImageView<const std::uint8_t> left_valid, ImageView<const std::uint8_t> right_valid,
                          int disparity_min, int num_disparities, const StereoOptions& options) {
  MatchContext ctx = make_context(left, right, left_valid, right_valid, disparity_min, num_disparities, options);
  const bool check_lr = options.lr_max_diff >= 0;
  ctx.diagonal_lr = check_lr && !options.lr_full_match;
  DisparityMap out = match_tiles(ctx);
  out.disparity_min = disparity_min;

  if (check_lr && options.lr_full_match) {
    // Match right to left: the mirrored right image is the base of the
    // mirrored pair, whose disparities are d + (width_left - width_right).
    const int wl = left.width();
    const int wr = right.width();
    const ImageU16 mirrored_left = mirror(left);
    const ImageU16 mirrored_right = mirror(right);
    const ImageU8 mirrored_left_valid = left_valid.empty() ? ImageU8() : mirror(left_valid);
    const ImageU8 mirrored_right_valid = right_valid.empty() ? ImageU8() : mirror(right_valid);
    MatchContext rctx = make_context(mirrored_right.view(), mirrored_left.view(), mirrored_right_valid.view(),
                                     mirrored_left_valid.view(), disparity_min + wl - wr, num_disparities, options);
    rctx.diagonal_lr = false;
    const DisparityMap right_map = match_tiles(rctx);
    for (int y = 0; y < out.disparity.height(); ++y) {
      for (int x = 0; x < wl; ++x) {
        const float d = out.disparity(x, y);
        if (!std::isfinite(d)) {
          continue;
        }
        const auto xr = static_cast<int>(std::lround(static_cast<float>(x) - d));
        const float dr = xr >= 0 && xr < wr ? right_map.disparity(wr - 1 - xr, y) - static_cast<float>(wl - wr)
                                            : kInvalidDisparity;
        if (!std::isfinite(dr) || std::abs(d - dr) > static_cast<float>(options.lr_max_diff)) {
          out.disparity(x, y) = kInvalidDisparity;
        }
      }
    }
  }

  remove_speckles(out.disparity, options.speckle_size, options.speckle_diff);
  if (options.median) {
    out.disparity = median_filter_3x3(out.disparity);
  }
  return out;
}

namespace {

// Matches all tiles of a pair and blends them (no post-processing).
DisparityMap match_tiles(const MatchContext& ctx) {
  const StereoOptions& options = *ctx.options;
  const int width = ctx.left.width();
  const int height = ctx.left.height();

  // Tile layout: depends only on the image size and the options.
  int tile_size = options.tile_size;
  const int ext = options.tile_margin + (options.tile_blend + 1) / 2;
  while (tile_size > 64 && tile_bytes(std::min(width, tile_size + 2 * ext), std::min(height, tile_size + 2 * ext),
                                      ctx.num_disp) > options.memory_budget) {
    tile_size /= 2;
  }
  const std::vector<int> bx = split(width, tile_size);
  const std::vector<int> by = split(height, tile_size);
  std::vector<TileResult> tiles;
  std::vector<std::array<int, 4>> cores;  // c0x, c1x, c0y, c1y
  for (std::size_t j = 0; j + 1 < by.size(); ++j) {
    for (std::size_t i = 0; i + 1 < bx.size(); ++i) {
      const int x0 = std::max(0, bx[i] - ext);
      const int x1 = std::min(width, bx[i + 1] + ext);
      const int y0 = std::max(0, by[j] - ext);
      const int y1 = std::min(height, by[j + 1] + ext);
      tiles.push_back(TileResult{Tile{x0, y0, x1 - x0, y1 - y0}, {}, {}});
      cores.push_back({bx[i], bx[i + 1], by[j], by[j + 1]});
    }
  }
  std::size_t max_bytes = 0;
  for (const TileResult& t : tiles) {
    max_bytes = std::max(max_bytes, tile_bytes(t.tile.w, t.tile.h, ctx.num_disp));
  }
  const auto concurrent = static_cast<std::ptrdiff_t>(std::clamp<std::size_t>(
      options.memory_budget / std::max<std::size_t>(max_bytes, 1), 1, default_thread_pool().num_threads()));
  if (max_bytes > options.memory_budget) {
    log_warn("stereo: a tile needs {} MB, more than the memory budget of {} MB", max_bytes >> 20,
             options.memory_budget >> 20);
  }
  log_debug("stereo: {}x{}, {} labels, {} tiles of up to {} MB, {} concurrent", width, height, ctx.num_disp,
            tiles.size(), max_bytes >> 20, concurrent);

  std::counting_semaphore<> slots(concurrent);
  parallel_for(0, tiles.size(), 1, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      slots.acquire();
      try {
        const Tile& t = tiles[i].tile;
        AlignedBuffer<std::uint16_t> volume(static_cast<std::size_t>(t.w) * static_cast<std::size_t>(t.h) *
                                            ctx.num_disp);
        TileAggregator(ctx, t).run(volume.data());
        select_disparities(ctx, volume.data(), tiles[i]);
      } catch (...) {
        slots.release();
        throw;
      }
      slots.release();
    }
  });

  // Blend the tiles.
  const auto npix = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  std::vector<float> wsum(npix, 0.0F), wall(npix, 0.0F), dsum(npix, 0.0F), csum(npix, 0.0F), wbest(npix, 0.0F),
      dbest(npix, kInvalidDisparity);
  for (std::size_t i = 0; i < tiles.size(); ++i) {
    const TileResult& r = tiles[i];
    const Tile& t = r.tile;
    const std::array<int, 4>& c = cores[i];
    for (int y = 0; y < t.h; ++y) {
      const int big_y = t.y0 + y;
      const float wy = axis_weight(big_y, c[2], c[3], height, options.tile_blend);
      if (wy <= 0.0F) {
        continue;
      }
      for (int x = 0; x < t.w; ++x) {
        const int big_x = t.x0 + x;
        const float wgt = wy * axis_weight(big_x, c[0], c[1], width, options.tile_blend);
        if (wgt <= 0.0F) {
          continue;
        }
        const std::size_t ti = static_cast<std::size_t>(y) * static_cast<std::size_t>(t.w) + static_cast<std::size_t>(x);
        const std::size_t pi = static_cast<std::size_t>(big_y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(big_x);
        wall[pi] += wgt;
        const float dv = r.disparity[ti];
        if (std::isfinite(dv)) {
          wsum[pi] += wgt;
          dsum[pi] += wgt * dv;
          csum[pi] += wgt * r.confidence[ti];
          if (wgt > wbest[pi]) {
            wbest[pi] = wgt;
            dbest[pi] = dv;
          }
        }
      }
    }
  }
  DisparityMap out;
  out.disparity_min = ctx.disparity_min;
  out.num_disparities = static_cast<int>(ctx.num_disp);
  out.disparity = ImageF32::uninitialized(width, height, 1);
  out.confidence = ImageF32(width, height, 1);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t pi = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
      float dv = kInvalidDisparity;
      if (wsum[pi] > 0.0F && wsum[pi] >= 0.5F * wall[pi]) {
        dv = dsum[pi] / wsum[pi];
        if (std::abs(dv - dbest[pi]) > 1.0F) {  // tiles disagree: keep the dominant one
          dv = dbest[pi];
        }
        out.confidence(x, y) = csum[pi] / wsum[pi];
      }
      out.disparity(x, y) = dv;
    }
  }
  return out;
}

}  // namespace

namespace detail {

std::vector<std::uint16_t> sgm_cost_volume(ImageView<const std::uint16_t> left, ImageView<const std::uint16_t> right,
                                           int disparity_min, int num_disparities, const StereoOptions& options) {
  const MatchContext ctx = make_context(left, right, {}, {}, disparity_min, num_disparities, options);
  const Tile tile{0, 0, left.width(), left.height()};
  std::vector<std::uint16_t> volume(static_cast<std::size_t>(tile.w) * static_cast<std::size_t>(tile.h) *
                                    ctx.num_disp);
  TileAggregator(ctx, tile).run(volume.data());
  return volume;
}

std::vector<std::uint16_t> sgm_cost_volume_reference(ImageView<const std::uint16_t> left,
                                                     ImageView<const std::uint16_t> right, int disparity_min,
                                                     int num_disparities, const StereoOptions& options) {
  check_options(options);
  const int w = left.width();
  const int h = left.height();
  const int wr = right.width();
  const int nd = (num_disparities + static_cast<int>(kLabelAlign) - 1) / static_cast<int>(kLabelAlign) *
                 static_cast<int>(kLabelAlign);
  const Image<std::uint64_t> cl = census_transform(left, true);
  const Image<std::uint64_t> cr = census_transform(right, true);
  const std::array<std::uint16_t, 256> lut = p2_table(options);
  auto at = [&](int x, int y, int d) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) *
               static_cast<std::size_t>(nd) +
           static_cast<std::size_t>(d);
  };
  std::vector<int> cost(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(nd));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int d = 0; d < nd; ++d) {
        const int xr = x - disparity_min - d;
        cost[at(x, y, d)] = xr >= 0 && xr < wr ? std::popcount(cl(x, y) ^ cr(xr, y)) : options.invalid_cost;
      }
    }
  }
  std::vector<int> total(cost.size(), 0);
  std::vector<int> path(cost.size());
  const int dirs[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {-1, 1}, {1, -1}};
  for (const auto& r : dirs) {
    const int dx = r[0];
    const int dy = r[1];
    for (int iy = 0; iy < h; ++iy) {
      const int y = dy >= 0 ? iy : h - 1 - iy;
      for (int ix = 0; ix < w; ++ix) {
        const int x = dx >= 0 ? ix : w - 1 - ix;
        const int qx = x - dx;
        const int qy = y - dy;
        if (qx < 0 || qy < 0 || qx >= w || qy >= h) {
          for (int d = 0; d < nd; ++d) {
            path[at(x, y, d)] = cost[at(x, y, d)];
          }
          continue;
        }
        int min_q = path[at(qx, qy, 0)];
        for (int d = 1; d < nd; ++d) {
          min_q = std::min(min_q, path[at(qx, qy, d)]);
        }
        const int p2 = lut[static_cast<std::size_t>(std::abs(int{left(x, y)} - int{left(qx, qy)}) >> 8)];
        for (int d = 0; d < nd; ++d) {
          int best = std::min(path[at(qx, qy, d)], min_q + p2);
          if (d > 0) {
            best = std::min(best, path[at(qx, qy, d - 1)] + options.p1);
          }
          if (d + 1 < nd) {
            best = std::min(best, path[at(qx, qy, d + 1)] + options.p1);
          }
          path[at(x, y, d)] = cost[at(x, y, d)] + best - min_q;
        }
      }
    }
    for (std::size_t i = 0; i < total.size(); ++i) {
      total[i] += path[i];
    }
  }
  std::vector<std::uint16_t> out(total.size());
  for (std::size_t i = 0; i < total.size(); ++i) {
    out[i] = static_cast<std::uint16_t>(total[i]);
  }
  return out;
}

}  // namespace detail
}  // namespace r3d
