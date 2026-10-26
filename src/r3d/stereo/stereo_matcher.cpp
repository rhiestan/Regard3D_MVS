#include "r3d/stereo/stereo_matcher.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>

#include "r3d/core/aligned_buffer.hpp"
#include "r3d/core/error.hpp"
#include "r3d/core/log.hpp"
#include "r3d/core/pyramid.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/core/timer.hpp"
#include "r3d/stereo/census.hpp"
#include "r3d/stereo/sgm_kernels.hpp"

namespace r3d {
namespace {

using sgm::kLabelAlign;
using sgm::kLabelOffset;
using sgm::path_stride;

int align_labels(int n) {
  const auto a = static_cast<int>(kLabelAlign);
  return (n + a - 1) / a * a;
}

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
  if (o.levels < -1 || o.levels > 12 || o.range_radius < 0 || o.range_margin < 0 || o.range_fill < 0) {
    throw_error("stereo: invalid hierarchy levels {} / radius {} / margin {} / fill {}", o.levels, o.range_radius,
                o.range_margin, o.range_fill);
  }
}

void check_ranges(const LabelRanges& r, int width, int height, int num_disp) {
  if (r.lo.width() != width || r.lo.height() != height || r.num.width() != width || r.num.height() != height) {
    throw_error("stereo: label ranges of size {}x{} for an image of {}x{}", r.lo.width(), r.lo.height(), width,
                height);
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int lo = r.lo(x, y);
      const int n = r.num(x, y);
      if (n <= 0 || n % static_cast<int>(kLabelAlign) != 0 || lo + n > num_disp) {
        throw_error("stereo: invalid label range {} + {} at ({}, {}) for {} labels", lo, n, x, y, num_disp);
      }
    }
  }
}

// Inputs shared by all tiles of a pair.
struct MatchContext {
  ImageView<const std::uint16_t> left;
  ImageView<const std::uint16_t> right;
  std::vector<RowInterval> left_rows;
  std::vector<RowInterval> right_rows;
  int disparity_min = 0;
  std::size_t num_disp = 0;
  const LabelRanges* ranges = nullptr;
  std::array<std::uint16_t, 256> p2_lut{};
  const StereoOptions* options = nullptr;
  bool diagonal_lr = false;  // left-right check against the right disparities of the same volume
  bool confidence = true;    // compute DisparityMap::confidence
};

MatchContext make_context(ImageView<const std::uint16_t> left, ImageView<const std::uint16_t> right,
                          ImageView<const std::uint8_t> left_valid, ImageView<const std::uint8_t> right_valid,
                          int disparity_min, int num_disparities, const LabelRanges* ranges,
                          const StereoOptions& options) {
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
  ctx.left_rows = valid_intervals(left_valid, left.width(), left.height());
  ctx.right_rows = valid_intervals(right_valid, right.width(), right.height());
  ctx.disparity_min = disparity_min;
  ctx.num_disp = static_cast<std::size_t>(align_labels(num_disparities));
  if (ranges != nullptr) {
    check_ranges(*ranges, left.width(), left.height(), static_cast<int>(ctx.num_disp));
  }
  ctx.ranges = ranges;
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

// Label layout of a tile (see sgm::RowLabels).
struct TileLabels {
  std::size_t w = 0;
  std::size_t h = 0;
  std::vector<std::int32_t> lo;        // [h][w]
  std::vector<std::uint16_t> num;      // [h][w]
  std::vector<std::size_t> offset;     // [h][w + 1], within the row
  std::vector<std::size_t> row_start;  // [h + 1], start of each row in the tile's volume
  std::size_t max_num = 0;
  std::size_t max_row_cells = 0;
  std::size_t max_row_path = 0;  // largest RowLabels::path_size

  TileLabels(const MatchContext& ctx, const Tile& t)
      : w(static_cast<std::size_t>(t.w)),
        h(static_cast<std::size_t>(t.h)),
        lo(w * h),
        num(w * h),
        offset(h * (w + 1)),
        row_start(h + 1, 0) {
    for (std::size_t y = 0; y < h; ++y) {
      std::size_t* off = offset.data() + y * (w + 1);
      off[0] = 0;
      for (std::size_t x = 0; x < w; ++x) {
        const int bx = t.x0 + static_cast<int>(x);
        const int by = t.y0 + static_cast<int>(y);
        const std::size_t i = y * w + x;
        if (ctx.ranges != nullptr) {
          lo[i] = ctx.ranges->lo(bx, by);
          num[i] = ctx.ranges->num(bx, by);
        } else {
          lo[i] = 0;
          num[i] = static_cast<std::uint16_t>(ctx.num_disp);
        }
        off[x + 1] = off[x] + num[i];
        max_num = std::max<std::size_t>(max_num, num[i]);
      }
      row_start[y + 1] = row_start[y] + off[w];
      max_row_cells = std::max(max_row_cells, off[w]);
      max_row_path = std::max(max_row_path, row(y).path_size(w));
    }
  }

  [[nodiscard]] sgm::RowLabels row(std::size_t y) const {
    return sgm::RowLabels{lo.data() + y * w, num.data() + y * w, offset.data() + y * (w + 1)};
  }
  [[nodiscard]] std::size_t cells() const { return row_start[h]; }
  // Volume, cost row, path buffers (6 rows), WTA rows, results and Census
  // (tile and right rows of at most right_width columns).
  [[nodiscard]] std::size_t bytes(std::size_t right_width) const {
    return cells() * 2 + max_row_cells * 2 + 6 * max_row_path * 2 + w * h * 16 + lo.size() * 6 +
           offset.size() * sizeof(std::size_t) + (w + right_width + 16) * (h + 6) * 8;
  }
};

// Census values of the columns [first, last] of rows [y0, y1) of an image,
// computed on a region enlarged by the window radius, so that they equal
// those of the whole image.
struct CensusRegion {
  Image<std::uint64_t> census;
  int x0 = 0;  // origin of `census` in the image
  int y0 = 0;
  int first = 0;
  int last = -1;

  CensusRegion(ImageView<const std::uint16_t> img, int x_first, int x_last, int y_begin, int y_end, bool scalar) {
    first = std::max(x_first, 0);
    last = std::min(x_last, img.width() - 1);
    if (first > last) {
      return;
    }
    x0 = std::max(0, first - kCensusWidth / 2);
    y0 = std::max(0, y_begin - kCensusHeight / 2);
    const int x1 = std::min(img.width(), last + 1 + kCensusWidth / 2);
    const int y1 = std::min(img.height(), y_end + kCensusHeight / 2);
    census = census_transform(img.subview(x0, y0, x1 - x0, y1 - y0), scalar);
  }
  // Census of image column 0 in image row y (valid for columns first .. last).
  [[nodiscard]] const std::uint64_t* row(int y) const { return census.row(y - y0) - x0; }
};

// Aggregation of one tile into its volume (layout of TileLabels).
class TileAggregator {
public:
  TileAggregator(const MatchContext& ctx, const Tile& tile, const TileLabels& labels)
      : ctx_(ctx),
        tile_(tile),
        labels_(labels),
        w_(static_cast<std::size_t>(tile.w)),
        cost_(labels.max_row_cells),
        p2_(4 * w_),
        prev_(3 * labels.max_row_path),
        cur_(3 * labels.max_row_path),
        prev_min_(3 * w_),
        cur_min_(3 * w_),
        scratch_(sgm::kScratchBuffers * path_stride(labels.max_num)) {
    scratch_.fill(0);  // the first buffer is the zero start of every path
    std::size_t rev_size = 0;
    std::int32_t origin_min = INT_MAX;
    std::int32_t end_max = INT_MIN;
    for (std::size_t y = 0; y < labels.h; ++y) {
      const auto [origin, end] = rev_extent(labels.row(y));
      rev_size = std::max(rev_size, static_cast<std::size_t>(end - origin));
      origin_min = std::min(origin_min, origin);
      end_max = std::max(end_max, end);
    }
    rev_.reset_uninitialized(rev_size);
    // Right columns reached by the labels (see cost_row).
    const int base = tile.x0 + tile.w - 1 - ctx.disparity_min;
    census_left_.emplace(ctx.left, tile.x0, tile.x0 + tile.w - 1, tile.y0, tile.y0 + tile.h,
                         ctx.options->scalar_kernels);
    census_right_.emplace(ctx.right, base - end_max + 1, base - origin_min, tile.y0, tile.y0 + tile.h,
                          ctx.options->scalar_kernels);
  }

  void run(std::uint16_t* volume) {
    pass(1, volume);
    pass(-1, volume);
  }

private:
  // Range of rev indices (w - 1 - x + label) used by a row: [origin, end).
  [[nodiscard]] std::pair<std::int32_t, std::int32_t> rev_extent(const sgm::RowLabels& rl) const {
    std::int32_t origin = INT_MAX;
    std::int32_t end = INT_MIN;
    for (std::size_t x = 0; x < w_; ++x) {
      const std::int32_t s = static_cast<std::int32_t>(w_ - 1 - x) + rl.lo[x];
      origin = std::min(origin, s);
      end = std::max(end, s + static_cast<std::int32_t>(rl.num[x]));
    }
    return {origin, end};
  }

  // Matching costs of tile row y.
  const std::uint16_t* cost_row(int y) {
    const int big_y = tile_.y0 + y;
    const sgm::RowLabels rl = labels_.row(static_cast<std::size_t>(y));
    const std::uint64_t* cl = census_left_->row(big_y) + tile_.x0;
    const CensusRegion& right = *census_right_;
    const std::uint64_t* cr = right.first <= right.last ? right.row(big_y) : nullptr;
    const auto w = static_cast<int>(w_);
    // rev[k] is the right census at column base - k, so that the labels of
    // left column x are rev[w - 1 - x + label - origin] (contiguous in label).
    const auto [origin, end] = rev_extent(rl);
    const int base = tile_.x0 + w - 1 - ctx_.disparity_min - origin;
    for (std::int32_t k = 0; k < end - origin; ++k) {
      const int xr = base - k;
      rev_[static_cast<std::size_t>(k)] = xr >= right.first && xr <= right.last ? cr[xr] : 0;
    }
    if (ctx_.options->scalar_kernels) {
      sgm::scalar::census_costs(cl, rev_.data(), w_, rl, origin, cost_.data());
    } else {
      sgm::census_costs(cl, rev_.data(), w_, rl, origin, cost_.data());
    }

    const RowInterval& lv = ctx_.left_rows[static_cast<std::size_t>(big_y)];
    const RowInterval& rv = ctx_.right_rows[static_cast<std::size_t>(big_y)];
    const auto invalid = static_cast<std::uint16_t>(ctx_.options->invalid_cost);
    for (int x = 0; x < w; ++x) {
      const auto n = static_cast<int>(rl.num[x]);
      std::uint16_t* c = cost_.data() + rl.offset[x];
      const int big_x = tile_.x0 + x;
      if (big_x < lv.first || big_x > lv.last) {
        std::fill(c, c + n, std::uint16_t{0});  // no information: paths pass through
        continue;
      }
      // Label k is valid if the right column big_x - dmin - k is.
      int lo = big_x - ctx_.disparity_min - rv.last - rl.lo[x];
      int hi = big_x - ctx_.disparity_min - rv.first + 1 - rl.lo[x];
      if (rv.last < rv.first) {
        lo = hi = 0;
      }
      lo = std::clamp(lo, 0, n);
      hi = std::clamp(hi, lo, n);
      if (lo > 0) {
        std::fill(c, c + lo, invalid);
      }
      if (hi < n) {
        std::fill(c + hi, c + n, invalid);
      }
    }
    return cost_.data();
  }

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

  void pass(int direction, std::uint16_t* volume) {
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
      args.sum = volume + labels_.row_start[static_cast<std::size_t>(y)];
      args.labels = labels_.row(static_cast<std::size_t>(y));
      args.has_prev = i > 0;
      if (args.has_prev) {
        args.prev_labels = labels_.row(static_cast<std::size_t>(y - direction));
      }
      args.width = w_;
      args.max_num = labels_.max_num;
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
  const TileLabels& labels_;
  std::size_t w_;
  std::optional<CensusRegion> census_left_;
  std::optional<CensusRegion> census_right_;
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
  std::size_t cells = 0;
};

// Winner-takes-all with sub-pixel interpolation, uniqueness and left-right check.
void select_disparities(const MatchContext& ctx, const TileLabels& labels, const std::uint16_t* volume,
                        TileResult& out) {
  const Tile& t = out.tile;
  const StereoOptions& o = *ctx.options;
  const std::size_t w = static_cast<std::size_t>(t.w);
  out.disparity.assign(w * static_cast<std::size_t>(t.h), kInvalidDisparity);
  if (ctx.confidence) {
    out.confidence.assign(w * static_cast<std::size_t>(t.h), 0.0F);
  }
  // Best label of every right column reachable from the row: index x - k + shift.
  std::vector<std::uint16_t> right_cost;
  std::vector<std::uint16_t> right_best;
  std::vector<std::uint16_t> best_label(w), best_value(w), second_value(w);
  for (int y = 0; y < t.h; ++y) {
    const sgm::RowLabels rl = labels.row(static_cast<std::size_t>(y));
    const std::uint16_t* s_row = volume + labels.row_start[static_cast<std::size_t>(y)];
    const int big_y = t.y0 + y;
    std::ptrdiff_t shift = 0;
    if (ctx.diagonal_lr) {
      std::ptrdiff_t top = 0;
      shift = PTRDIFF_MIN;
      for (std::size_t x = 0; x < w; ++x) {
        const auto xs = static_cast<std::ptrdiff_t>(x);
        shift = std::max<std::ptrdiff_t>(shift, rl.lo[x] + rl.num[x] - 1 - xs);
        top = std::max<std::ptrdiff_t>(top, xs - rl.lo[x]);
      }
      right_cost.assign(static_cast<std::size_t>(top + shift + 1), 0xFFFF);
      right_best.assign(right_cost.size(), 0xFFFF);
    }
    // Only valid left pixels: masked ones have zero costs, which would win
    // the right-image WTA of the left-right check.
    const RowInterval& lv = ctx.left_rows[static_cast<std::size_t>(big_y)];
    const RowInterval& rv = ctx.right_rows[static_cast<std::size_t>(big_y)];
    const auto xb = static_cast<std::size_t>(std::clamp(lv.first - t.x0, 0, t.w));
    const auto xe = static_cast<std::size_t>(std::clamp(lv.last + 1 - t.x0, static_cast<int>(xb), t.w));
    if (xb < xe) {
      const sgm::RowLabels sub{rl.lo + xb, rl.num + xb, rl.offset + xb};
      std::uint16_t* rc = ctx.diagonal_lr ? right_cost.data() + xb : nullptr;
      std::uint16_t* rb = ctx.diagonal_lr ? right_best.data() + xb : nullptr;
      const auto wta = o.scalar_kernels ? sgm::scalar::wta_row : sgm::wta_row;
      wta(s_row, xe - xb, sub, best_label.data() + xb, best_value.data() + xb, second_value.data() + xb, rc, rb,
          shift);
    }
    for (std::size_t x = xb; x < xe; ++x) {
      const int big_x = t.x0 + static_cast<int>(x);
      const std::uint16_t* s = s_row + rl.offset[x];
      const int best = best_label[x];
      const int j = best - rl.lo[x];
      const auto n = static_cast<int>(rl.num[x]);
      const unsigned second = second_value[x];
      const int xr = big_x - ctx.disparity_min - best;
      if (xr < rv.first || xr > rv.last) {
        continue;
      }
      // A minimum on the edge of a reduced range probably lies outside it.
      if ((j == 0 && rl.lo[x] > 0) || (j == n - 1 && rl.lo[x] + n < static_cast<int>(ctx.num_disp))) {
        continue;
      }
      if (ctx.diagonal_lr &&
          std::abs(int{right_best[static_cast<std::size_t>(static_cast<std::ptrdiff_t>(x) - best + shift)]} - best) >
              o.lr_max_diff) {
        continue;
      }
      const float best_cost = best_value[x];
      if (o.uniqueness > 0.0F && second != 0xFFFFU &&
          best_cost > (1.0F - o.uniqueness) * static_cast<float>(second)) {
        continue;
      }
      float delta = 0.0F;
      if (j > 0 && j < n - 1) {
        delta = subpixel_offset(o.subpixel, s[j - 1], best_cost, s[j + 1]);
      }
      const std::size_t i = static_cast<std::size_t>(y) * w + x;
      out.disparity[i] = static_cast<float>(ctx.disparity_min + best) + delta;
      if (ctx.confidence) {
        out.confidence[i] = second != 0xFFFFU && second > 0 ? 1.0F - best_cost / static_cast<float>(second) : 0.0F;
      }
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

// Number of labels of the pixels of [x0, x1) x [y0, y1).
std::uint64_t count_cells(const MatchContext& ctx, int x0, int y0, int x1, int y1) {
  if (ctx.ranges == nullptr) {
    return static_cast<std::uint64_t>(x1 - x0) * static_cast<std::uint64_t>(y1 - y0) * ctx.num_disp;
  }
  std::uint64_t cells = 0;
  for (int y = y0; y < y1; ++y) {
    const std::uint16_t* num = ctx.ranges->num.row(y);
    for (int x = x0; x < x1; ++x) {
      cells += num[x];
    }
  }
  return cells;
}

// Estimated bytes of a tile (TileLabels::bytes with average rows).
std::size_t estimated_tile_bytes(const MatchContext& ctx, int x0, int y0, int x1, int y1, std::size_t right_width) {
  const auto cells = static_cast<std::size_t>(count_cells(ctx, x0, y0, x1, y1));
  const auto w = static_cast<std::size_t>(x1 - x0);
  const auto h = static_cast<std::size_t>(y1 - y0);
  const std::size_t row_cells = cells / std::max<std::size_t>(h, 1);
  return cells * 2 + row_cells * 2 + 6 * (row_cells + w * 2 * kLabelOffset) * 2 + w * h * 32 +
         (w + right_width + 16) * (h + 6) * 8;
}

// Lets tiles run while their memory fits into the budget (always at least one).
class MemoryGate {
public:
  explicit MemoryGate(std::size_t budget) : budget_(budget) {}
  void acquire(std::size_t bytes) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return used_ == 0 || used_ + bytes <= budget_; });
    used_ += bytes;
  }
  void release(std::size_t bytes) {
    {
      const std::lock_guard lock(mutex_);
      used_ -= bytes;
    }
    cv_.notify_all();
  }

private:
  std::size_t budget_;
  std::size_t used_ = 0;
  std::mutex mutex_;
  std::condition_variable cv_;
};

template <typename T>
Image<T> mirror(ImageView<const T> img) {
  Image<T> out = Image<T>::uninitialized(img.width(), img.height(), 1);
  for (int y = 0; y < img.height(); ++y) {
    std::reverse_copy(img.row(y), img.row(y) + img.width(), out.row(y));
  }
  return out;
}

// Matches all tiles of a pair and blends them (no post-processing).
DisparityMap match_tiles(const MatchContext& ctx) {
  const StereoOptions& options = *ctx.options;
  const int width = ctx.left.width();
  const int height = ctx.left.height();

  // Tile layout: depends only on the image size, the ranges and the options.
  const int ext = options.tile_margin + (options.tile_blend + 1) / 2;
  auto layout = [&](int tile_size) {
    std::vector<std::pair<Tile, std::array<int, 4>>> tiles;  // tile, core c0x, c1x, c0y, c1y
    const std::vector<int> bx = split(width, tile_size);
    const std::vector<int> by = split(height, tile_size);
    for (std::size_t j = 0; j + 1 < by.size(); ++j) {
      for (std::size_t i = 0; i + 1 < bx.size(); ++i) {
        const int x0 = std::max(0, bx[i] - ext);
        const int x1 = std::min(width, bx[i + 1] + ext);
        const int y0 = std::max(0, by[j] - ext);
        const int y1 = std::min(height, by[j + 1] + ext);
        tiles.push_back({Tile{x0, y0, x1 - x0, y1 - y0}, {bx[i], bx[i + 1], by[j], by[j + 1]}});
      }
    }
    return tiles;
  };
  auto max_bytes = [&](const auto& tiles) {
    std::size_t m = 0;
    for (const auto& [t, core] : tiles) {
      m = std::max(m, estimated_tile_bytes(ctx, t.x0, t.y0, t.x0 + t.w, t.y0 + t.h,
                                           static_cast<std::size_t>(ctx.right.width())));
    }
    return m;
  };
  int tile_size = options.tile_size;
  auto layout_tiles = layout(tile_size);
  while (tile_size > 64 && max_bytes(layout_tiles) > options.memory_budget) {
    tile_size /= 2;
    layout_tiles = layout(tile_size);
  }
  const std::size_t largest = max_bytes(layout_tiles);
  if (largest > options.memory_budget) {
    log_warn("stereo: a tile needs about {} MB, more than the memory budget of {} MB", largest >> 20,
             options.memory_budget >> 20);
  }
  std::vector<TileResult> tiles;
  for (const auto& [t, core] : layout_tiles) {
    tiles.push_back(TileResult{t, {}, {}, 0});
  }
  log_debug("stereo: {}x{}, {} labels, {} tiles of up to {} MB", width, height, ctx.num_disp, tiles.size(),
            largest >> 20);

  MemoryGate gate(options.memory_budget);
  const Stopwatch watch;
  std::atomic<std::int64_t> aggregate_us{0}, select_us{0};
  parallel_for(0, tiles.size(), 1, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      const Tile& t = tiles[i].tile;
      const TileLabels labels(ctx, t);
      const std::size_t bytes = labels.bytes(static_cast<std::size_t>(ctx.right.width()));
      gate.acquire(bytes);
      try {
        AlignedBuffer<std::uint16_t> volume(labels.cells());
        const Stopwatch tile_watch;
        TileAggregator(ctx, t, labels).run(volume.data());
        const double t_aggregate = tile_watch.elapsed_seconds();
        select_disparities(ctx, labels, volume.data(), tiles[i]);
        aggregate_us += static_cast<std::int64_t>(t_aggregate * 1e6);
        select_us += static_cast<std::int64_t>((tile_watch.elapsed_seconds() - t_aggregate) * 1e6);
        tiles[i].cells = labels.cells();
      } catch (...) {
        gate.release(bytes);
        throw;
      }
      gate.release(bytes);
    }
  });

  const double t_tiles = watch.elapsed_seconds();
  // Blend the tiles, in parallel over the output rows.
  std::uint64_t cells = 0;
  std::vector<std::vector<float>> column_weights(tiles.size());
  for (std::size_t i = 0; i < tiles.size(); ++i) {
    const Tile& t = tiles[i].tile;
    const std::array<int, 4>& c = layout_tiles[i].second;
    cells += tiles[i].cells;
    column_weights[i].resize(static_cast<std::size_t>(t.w));
    for (int x = 0; x < t.w; ++x) {
      column_weights[i][static_cast<std::size_t>(x)] = axis_weight(t.x0 + x, c[0], c[1], width, options.tile_blend);
    }
  }
  DisparityMap out;
  out.disparity_min = ctx.disparity_min;
  out.num_disparities = static_cast<int>(ctx.num_disp);
  out.cells = cells;
  out.disparity = ImageF32::uninitialized(width, height, 1);
  if (ctx.confidence) {
    out.confidence = ImageF32(width, height, 1);
  }
  parallel_for(0, static_cast<std::size_t>(height), 8, [&](std::size_t begin, std::size_t end) {
    const auto w = static_cast<std::size_t>(width);
    std::vector<float> wsum(w), wall(w), dsum(w), csum(w), wbest(w), dbest(w);
    for (auto big_y = static_cast<int>(begin); big_y < static_cast<int>(end); ++big_y) {
      std::fill(wsum.begin(), wsum.end(), 0.0F);
      std::fill(wall.begin(), wall.end(), 0.0F);
      std::fill(dsum.begin(), dsum.end(), 0.0F);
      std::fill(csum.begin(), csum.end(), 0.0F);
      std::fill(wbest.begin(), wbest.end(), 0.0F);
      std::fill(dbest.begin(), dbest.end(), kInvalidDisparity);
      for (std::size_t i = 0; i < tiles.size(); ++i) {
        const TileResult& r = tiles[i];
        const Tile& t = r.tile;
        if (big_y < t.y0 || big_y >= t.y0 + t.h) {
          continue;
        }
        const std::array<int, 4>& c = layout_tiles[i].second;
        const float wy = axis_weight(big_y, c[2], c[3], height, options.tile_blend);
        if (wy <= 0.0F) {
          continue;
        }
        const std::size_t row = static_cast<std::size_t>(big_y - t.y0) * static_cast<std::size_t>(t.w);
        for (int x = 0; x < t.w; ++x) {
          const float wgt = wy * column_weights[i][static_cast<std::size_t>(x)];
          if (wgt <= 0.0F) {
            continue;
          }
          const std::size_t ti = row + static_cast<std::size_t>(x);
          const auto pi = static_cast<std::size_t>(t.x0 + x);
          wall[pi] += wgt;
          const float dv = r.disparity[ti];
          if (std::isfinite(dv)) {
            wsum[pi] += wgt;
            dsum[pi] += wgt * dv;
            if (ctx.confidence) {
              csum[pi] += wgt * r.confidence[ti];
            }
            if (wgt > wbest[pi]) {
              wbest[pi] = wgt;
              dbest[pi] = dv;
            }
          }
        }
      }
      for (std::size_t x = 0; x < w; ++x) {
        float dv = kInvalidDisparity;
        if (wsum[x] > 0.0F && wsum[x] >= 0.5F * wall[x]) {
          dv = dsum[x] / wsum[x];
          if (std::abs(dv - dbest[x]) > 1.0F) {  // tiles disagree: keep the dominant one
            dv = dbest[x];
          }
          if (ctx.confidence) {
            out.confidence(static_cast<int>(x), big_y) = csum[x] / wsum[x];
          }
        }
        out.disparity(static_cast<int>(x), big_y) = dv;
      }
    }
  });
  log_debug("stereo: tiles {:.3f} s (aggregation {:.3f} s, selection {:.3f} s summed over threads), blending {:.3f} s",
            t_tiles, static_cast<double>(aggregate_us.load()) * 1e-6, static_cast<double>(select_us.load()) * 1e-6,
            watch.elapsed_seconds() - t_tiles);
  return out;
}

// Invalidates the disparities of `map` (base image of the pair) whose match
// in `other` (disparities x_l - x_r of the other image, indexed by its own
// pixels) disagrees by more than max_diff. sign = -1 for a left base image
// (other column x - d), +1 for a right base image (other column x + d).
bool left_right_differ(const ImageF32& map, const ImageF32& other, int x, int y, int sign, int max_diff) {
  const float d = map(x, y);
  const auto xo = static_cast<int>(std::lround(static_cast<float>(x) + static_cast<float>(sign) * d));
  const float dother = xo >= 0 && xo < other.width() ? other(xo, y) : kInvalidDisparity;
  return !std::isfinite(dother) || std::abs(d - dother) > static_cast<float>(max_diff);
}

void check_left_right(ImageF32& map, const ImageF32& other, int sign, int max_diff) {
  for (int y = 0; y < map.height(); ++y) {
    for (int x = 0; x < map.width(); ++x) {
      if (std::isfinite(map(x, y)) && left_right_differ(map, other, x, y, sign, max_diff)) {
        map(x, y) = kInvalidDisparity;
      }
    }
  }
}

// Pixels that check_left_right would invalidate (1), without changing `map`.
std::vector<std::uint8_t> left_right_mismatch(const ImageF32& map, const ImageF32& other, int sign, int max_diff) {
  std::vector<std::uint8_t> bad(static_cast<std::size_t>(map.width()) * static_cast<std::size_t>(map.height()), 0);
  for (int y = 0; y < map.height(); ++y) {
    for (int x = 0; x < map.width(); ++x) {
      if (std::isfinite(map(x, y)) && left_right_differ(map, other, x, y, sign, max_diff)) {
        bad[static_cast<std::size_t>(y) * static_cast<std::size_t>(map.width()) + static_cast<std::size_t>(x)] = 1;
      }
    }
  }
  return bad;
}

// One pyramid level of a pair.
struct LevelInput {
  ImageView<const std::uint16_t> left;
  ImageView<const std::uint16_t> right;
  ImageView<const std::uint8_t> left_valid;
  ImageView<const std::uint8_t> right_valid;
  int disparity_min = 0;
  int num_disparities = 0;
};

struct LevelOutput {
  DisparityMap left;
  ImageF32 right;  // disparities x_l - x_r of the right pixels (only with a full right-to-left match)
};

void post_filter(ImageF32& disparity, const StereoOptions& options, int speckle_size) {
  remove_speckles(disparity, speckle_size, options.speckle_diff);
  if (options.median) {
    disparity = median_filter_3x3(disparity);
  }
}

// Matches and filters one level; right_ranges are per pixel of the mirrored
// right image (null: full range).
LevelOutput match_level(const LevelInput& in, const LabelRanges* left_ranges, const LabelRanges* right_ranges,
                        bool need_right, int speckle_size, const StereoOptions& options) {
  const Stopwatch watch;
  MatchContext ctx = make_context(in.left, in.right, in.left_valid, in.right_valid, in.disparity_min,
                                  in.num_disparities, left_ranges, options);
  const bool check_lr = options.lr_max_diff >= 0;
  ctx.diagonal_lr = check_lr && !options.lr_full_match;
  log_debug("stereo: context (Census) {:.3f} s", watch.elapsed_seconds());
  LevelOutput out;
  out.left = match_tiles(ctx);

  if (check_lr && options.lr_full_match) {
    // Match right to left: the mirrored right image is the base of the
    // mirrored pair, whose disparities are d + (width_right - width_left).
    const int wl = in.left.width();
    const int wr = in.right.width();
    const ImageU16 mirrored_left = mirror(in.left);
    const ImageU16 mirrored_right = mirror(in.right);
    const ImageU8 mirrored_left_valid = in.left_valid.empty() ? ImageU8() : mirror(in.left_valid);
    const ImageU8 mirrored_right_valid = in.right_valid.empty() ? ImageU8() : mirror(in.right_valid);
    MatchContext rctx = make_context(mirrored_right.view(), mirrored_left.view(), mirrored_right_valid.view(),
                                     mirrored_left_valid.view(), in.disparity_min + wr - wl, in.num_disparities,
                                     right_ranges, options);
    rctx.diagonal_lr = false;
    rctx.confidence = false;
    DisparityMap right_map = match_tiles(rctx);
    out.left.cells += right_map.cells;
    out.right = ImageF32::uninitialized(wr, in.right.height(), 1);
    for (int y = 0; y < in.right.height(); ++y) {
      for (int x = 0; x < wr; ++x) {
        out.right(x, y) = right_map.disparity(wr - 1 - x, y) - static_cast<float>(wr - wl);
      }
    }
    if (need_right) {
      // Both checks compare the unchecked maps.
      const std::vector<std::uint8_t> right_bad = left_right_mismatch(out.right, out.left.disparity, 1,
                                                                      options.lr_max_diff);
      check_left_right(out.left.disparity, out.right, -1, options.lr_max_diff);
      for (int y = 0; y < out.right.height(); ++y) {
        for (int x = 0; x < wr; ++x) {
          if (right_bad[static_cast<std::size_t>(y) * static_cast<std::size_t>(wr) + static_cast<std::size_t>(x)] != 0) {
            out.right(x, y) = kInvalidDisparity;
          }
        }
      }
      post_filter(out.right, options, speckle_size);
    } else {
      check_left_right(out.left.disparity, out.right, -1, options.lr_max_diff);
      out.right = ImageF32();
    }
  }
  const Stopwatch filter_watch;
  post_filter(out.left.disparity, options, speckle_size);
  log_debug("stereo: filters {:.3f} s, level total {:.3f} s", filter_watch.elapsed_seconds(), watch.elapsed_seconds());
  return out;
}

// Floor and ceiling of v / 2^k.
int floor_shift(int v, int k) { return v >= 0 ? v >> k : -((-v + (1 << k) - 1) >> k); }
int ceil_shift(int v, int k) { return -floor_shift(-v, k); }

ImageU8 reduce_mask(ImageView<const std::uint8_t> mask) {
  ImageU8 m = pyramid_down(mask);
  for (int y = 0; y < m.height(); ++y) {
    for (int x = 0; x < m.width(); ++x) {
      m(x, y) = m(x, y) >= 128 ? 255 : 0;
    }
  }
  return m;
}

// Median of 9 values with a sorting network (Paeth, Graphics Gems 1990).
float median9(std::array<float, 9>& p) {
  auto sort2 = [&](std::size_t i, std::size_t j) {
    const float a = p[i];
    const float b = p[j];
    p[i] = std::min(a, b);
    p[j] = std::max(a, b);
  };
  sort2(1, 2); sort2(4, 5); sort2(7, 8); sort2(0, 1); sort2(3, 4); sort2(6, 7);
  sort2(1, 2); sort2(4, 5); sort2(7, 8); sort2(0, 3); sort2(5, 8); sort2(4, 7);
  sort2(3, 6); sort2(1, 4); sort2(2, 5); sort2(4, 7); sort2(4, 2); sort2(6, 4);
  sort2(4, 2);
  return p[4];
}

}  // namespace

void remove_speckles(ImageF32& disparity, int min_size, float max_diff) {
  if (min_size <= 1 || disparity.empty()) {
    return;
  }
  const int w = disparity.width();
  const int h = disparity.height();
  const auto n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  // Contiguous copy; visited marks pixels that belong to a segment already.
  std::vector<float> d(n);
  for (int y = 0; y < h; ++y) {
    std::copy(disparity.row(y), disparity.row(y) + w, d.begin() + static_cast<std::ptrdiff_t>(y) * w);
  }
  std::vector<std::uint8_t> visited(n, 0);
  std::vector<std::uint32_t> segment;  // breadth-first queue, finally the whole segment
  for (std::size_t start = 0; start < n; ++start) {
    if (visited[start] != 0 || !std::isfinite(d[start])) {
      continue;
    }
    segment.clear();
    segment.push_back(static_cast<std::uint32_t>(start));
    visited[start] = 1;
    for (std::size_t head = 0; head < segment.size(); ++head) {
      const std::size_t i = segment[head];
      const float v = d[i];
      const auto x = static_cast<int>(i % static_cast<std::size_t>(w));
      auto visit = [&](std::size_t j) {
        if (visited[j] == 0 && std::abs(d[j] - v) <= max_diff) {  // false for invalid d[j]
          visited[j] = 1;
          segment.push_back(static_cast<std::uint32_t>(j));
        }
      };
      if (x > 0) {
        visit(i - 1);
      }
      if (x + 1 < w) {
        visit(i + 1);
      }
      if (i >= static_cast<std::size_t>(w)) {
        visit(i - static_cast<std::size_t>(w));
      }
      if (i + static_cast<std::size_t>(w) < n) {
        visit(i + static_cast<std::size_t>(w));
      }
    }
    if (static_cast<int>(segment.size()) < min_size) {
      for (const std::uint32_t i : segment) {
        disparity(static_cast<int>(i % static_cast<std::uint32_t>(w)), static_cast<int>(i / static_cast<std::uint32_t>(w))) =
            kInvalidDisparity;
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
        if (x > 0 && y > 0 && x + 1 < w && y + 1 < h) {
          std::array<float, 9> a{};
          bool all_valid = true;
          for (int dy = -1; dy <= 1; ++dy) {
            const float* row = disparity.row(y + dy) + x - 1;
            for (int dx = 0; dx < 3; ++dx) {
              a[static_cast<std::size_t>((dy + 1) * 3 + dx)] = row[dx];
              all_valid = all_valid && std::isfinite(row[dx]);
            }
          }
          if (all_valid) {
            out(x, y) = median9(a);
            continue;
          }
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
  check_options(options);
  if (num_disparities <= 0) {
    throw_error("stereo: no disparities to search");
  }
  const int levels = options.levels >= 0 ? options.levels
                                         : detail::automatic_levels(left.width(), left.height(), num_disparities);
  const bool lr_full = options.lr_max_diff >= 0 && options.lr_full_match;
  const int disparity_max = disparity_min + num_disparities - 1;

  // Pyramids (level 0 refers to the inputs).
  std::vector<ImageU16> left_pyr, right_pyr;
  std::vector<ImageU8> left_valid_pyr, right_valid_pyr;
  for (int l = 1; l <= levels; ++l) {
    left_pyr.push_back(pyramid_down(l == 1 ? left : left_pyr.back().view()));
    right_pyr.push_back(pyramid_down(l == 1 ? right : right_pyr.back().view()));
    if (!left_valid.empty()) {
      left_valid_pyr.push_back(reduce_mask(l == 1 ? left_valid : left_valid_pyr.back().view()));
    }
    if (!right_valid.empty()) {
      right_valid_pyr.push_back(reduce_mask(l == 1 ? right_valid : right_valid_pyr.back().view()));
    }
  }
  auto level_input = [&](int l) {
    LevelInput in;
    const auto i = static_cast<std::size_t>(l - 1);
    in.left = l == 0 ? left : left_pyr[i].view();
    in.right = l == 0 ? right : right_pyr[i].view();
    in.left_valid = l == 0 || left_valid.empty() ? left_valid : left_valid_pyr[i].view();
    in.right_valid = l == 0 || right_valid.empty() ? right_valid : right_valid_pyr[i].view();
    in.disparity_min = floor_shift(disparity_min, l);
    in.num_disparities = align_labels(ceil_shift(disparity_max, l) - in.disparity_min + 1);
    return in;
  };

  LevelOutput coarse;
  std::uint64_t cells = 0;
  for (int l = levels; l >= 0; --l) {
    const LevelInput in = level_input(l);
    const int speckle = options.speckle_size >> (2 * l);
    if (l == levels) {
      coarse = match_level(in, nullptr, nullptr, lr_full && l > 0, speckle, options);
    } else {
      const Stopwatch range_watch;
      const LabelRanges left_ranges =
          detail::ranges_from_coarse(coarse.left.disparity, in.left.width(), in.left.height(), in.left_valid,
                                     in.right_valid, in.right.width(), -1, in.disparity_min, in.num_disparities,
                                     options);
      LabelRanges right_ranges;
      if (lr_full) {
        // Mirrored, for the right-to-left match on the mirrored pair.
        right_ranges = detail::ranges_from_coarse(coarse.right, in.right.width(), in.right.height(), in.right_valid,
                                                  in.left_valid, in.left.width(), 1, in.disparity_min,
                                                  in.num_disparities, options, true);
      }
      log_debug("stereo: ranges {:.3f} s", range_watch.elapsed_seconds());
      coarse = match_level(in, &left_ranges, lr_full ? &right_ranges : nullptr, lr_full && l > 0, speckle, options);
    }
    cells += coarse.left.cells;
    log_debug("stereo: level {} ({}x{}): {} labels from {}, {:.1f} cells per pixel", l, in.left.width(),
              in.left.height(), in.num_disparities, in.disparity_min,
              static_cast<double>(coarse.left.cells) /
                  (static_cast<double>(in.left.width()) * static_cast<double>(in.left.height())));
  }
  DisparityMap out = std::move(coarse.left);
  out.disparity_min = disparity_min;
  out.num_disparities = align_labels(num_disparities);
  out.levels = levels;
  out.cells = cells;
  return out;
}

namespace detail {

LabelRanges full_ranges(int width, int height, int num_disparities) {
  LabelRanges r{Image<std::uint16_t>(width, height, 1), Image<std::uint16_t>(width, height, 1)};
  r.num.fill(static_cast<std::uint16_t>(align_labels(num_disparities)));
  return r;
}

int automatic_levels(int width, int height, int num_disparities) {
  int levels = 0;
  int w = width;
  int h = height;
  int n = num_disparities;
  while (levels < 8) {
    const int nw = (w + 1) / 2;
    const int nh = (h + 1) / 2;
    if (std::min(nw, nh) < 96 || n <= 48) {
      break;
    }
    w = nw;
    h = nh;
    n = (n + 1) / 2;
    ++levels;
  }
  return levels;
}

LabelRanges ranges_from_coarse(const ImageF32& coarse, int width, int height, ImageView<const std::uint8_t> fine_valid,
                               ImageView<const std::uint8_t> partner_valid, int partner_width, int partner_sign,
                               int disparity_min, int num_disparities, const StereoOptions& options, bool mirrored) {
  const int wc = coarse.width();
  const int hc = coarse.height();
  if (wc != (width + 1) / 2 || hc != (height + 1) / 2) {
    throw_error("stereo: coarse map of {}x{} for a level of {}x{}", wc, hc, width, height);
  }
  constexpr float inf = std::numeric_limits<float>::infinity();
  // Min / max of the valid disparities within range_radius (separable).
  const int r = options.range_radius;
  ImageF32 hmin = ImageF32::uninitialized(wc, hc, 1);
  ImageF32 hmax = ImageF32::uninitialized(wc, hc, 1);
  parallel_for(0, static_cast<std::size_t>(hc), 32, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      for (int x = 0; x < wc; ++x) {
        float mn = inf;
        float mx = -inf;
        for (int k = std::max(0, x - r); k <= std::min(wc - 1, x + r); ++k) {
          const float v = coarse(k, y);
          if (std::isfinite(v)) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
          }
        }
        hmin(x, y) = mn;
        hmax(x, y) = mx;
      }
    }
  });
  ImageF32 cmin = ImageF32::uninitialized(wc, hc, 1);
  ImageF32 cmax = ImageF32::uninitialized(wc, hc, 1);
  parallel_for(0, static_cast<std::size_t>(hc), 32, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      const int k0 = std::max(0, y - r);
      const int k1 = std::min(hc - 1, y + r);
      for (int x = 0; x < wc; ++x) {
        float mn = inf;
        float mx = -inf;
        for (int k = k0; k <= k1; ++k) {
          mn = std::min(mn, hmin(x, k));
          mx = std::max(mx, hmax(x, k));
        }
        cmin(x, y) = mn;
        cmax(x, y) = mx;
      }
    }
  });
  // Pixels without a valid disparity take the range of the nearest valid
  // pixels, looked up in a min / max pyramid (3x3 blocks of 2^k coarse
  // pixels, k = 1, 2, ...) up to about range_fill coarse pixels away.
  std::vector<ImageF32> pyr_min{cmin};
  std::vector<ImageF32> pyr_max{cmax};
  while ((2 << pyr_min.size()) <= 2 * options.range_fill && pyr_min.back().width() > 1) {
    const ImageF32& a = pyr_min.back();
    const ImageF32& b = pyr_max.back();
    const int nw = (a.width() + 1) / 2;
    const int nh = (a.height() + 1) / 2;
    ImageF32 mn = ImageF32::uninitialized(nw, nh, 1);
    ImageF32 mx = ImageF32::uninitialized(nw, nh, 1);
    for (int y = 0; y < nh; ++y) {
      for (int x = 0; x < nw; ++x) {
        const int x1 = std::min(2 * x + 1, a.width() - 1);
        const int y1 = std::min(2 * y + 1, a.height() - 1);
        mn(x, y) = std::min(std::min(a(2 * x, 2 * y), a(x1, 2 * y)), std::min(a(2 * x, y1), a(x1, y1)));
        mx(x, y) = std::max(std::max(b(2 * x, 2 * y), b(x1, 2 * y)), std::max(b(2 * x, y1), b(x1, y1)));
      }
    }
    pyr_min.push_back(std::move(mn));
    pyr_max.push_back(std::move(mx));
  }
  parallel_for(0, static_cast<std::size_t>(hc), 32, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      for (int x = 0; x < wc; ++x) {
        if (pyr_min[0](x, y) != inf) {
          continue;
        }
        for (std::size_t k = 1; k < pyr_min.size(); ++k) {
          const ImageF32& a = pyr_min[k];
          const ImageF32& b = pyr_max[k];
          const int bx = x >> k;
          const int by = y >> k;
          float mn = inf;
          float mx = -inf;
          for (int qy = std::max(0, by - 1); qy <= std::min(a.height() - 1, by + 1); ++qy) {
            for (int qx = std::max(0, bx - 1); qx <= std::min(a.width() - 1, bx + 1); ++qx) {
              mn = std::min(mn, a(qx, qy));
              mx = std::max(mx, b(qx, qy));
            }
          }
          if (mn != inf) {
            cmin(x, y) = mn;  // pyr_min[0] is a copy: no race with the lookups
            cmax(x, y) = mx;
            break;
          }
        }
      }
    }
  });

  const int n_all = align_labels(num_disparities);
  const auto align = static_cast<int>(kLabelAlign);
  LabelRanges out{Image<std::uint16_t>::uninitialized(width, height, 1),
                  Image<std::uint16_t>::uninitialized(width, height, 1)};
  const std::vector<RowInterval> partner_rows = valid_intervals(partner_valid, partner_width, height);
  // The 2x2 fine pixels of a block share their range (the coarse pixels k
  // and k + 1 cover fine pixels 2k and 2k + 1), so that most path
  // predecessors have the same labels as their successors.
  parallel_for(0, static_cast<std::size_t>(height), 32, [&](std::size_t begin, std::size_t end) {
  for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
    const RowInterval& pr = partner_rows[static_cast<std::size_t>(y)];
    const int cy0 = y / 2;
    const int cy1 = std::min(cy0 + 1, hc - 1);
    for (int x = 0; x < width; ++x) {
      const int cx0 = x / 2;
      const int cx1 = std::min(cx0 + 1, wc - 1);
      const float mn = std::min(std::min(cmin(cx0, cy0), cmin(cx1, cy0)), std::min(cmin(cx0, cy1), cmin(cx1, cy1)));
      const float mx = std::max(std::max(cmax(cx0, cy0), cmax(cx1, cy0)), std::max(cmax(cx0, cy1), cmax(cx1, cy1)));
      int lo = 0;
      int hi = n_all - 1;
      if (mn != inf) {
        lo = static_cast<int>(std::floor(2.0F * mn)) - disparity_min - options.range_margin;
        hi = static_cast<int>(std::ceil(2.0F * mx)) - disparity_min + options.range_margin;
      }
      // Labels whose match lies within the partner's valid columns.
      const int match_lo = (partner_sign < 0 ? x - pr.last : pr.first - x) - disparity_min;
      const int match_hi = (partner_sign < 0 ? x - pr.first : pr.last - x) - disparity_min;
      lo = std::max(lo, match_lo);
      hi = std::min(hi, match_hi);
      if (lo > hi || pr.last < pr.first || (!fine_valid.empty() && fine_valid(x, y) < 128)) {
        hi = lo;  // no matching: the smallest range
      }
      lo = std::clamp(lo, 0, n_all - 1);
      hi = std::clamp(hi, lo, n_all - 1);
      const int n = std::min(n_all, (hi - lo + align) / align * align);
      lo = std::clamp(lo - (n - (hi - lo + 1)) / 2, 0, n_all - n);
      const int xo = mirrored ? width - 1 - x : x;
      out.lo(xo, y) = static_cast<std::uint16_t>(lo);
      out.num(xo, y) = static_cast<std::uint16_t>(n);
    }
  }
  });
  return out;
}

std::vector<std::uint16_t> sgm_cost_volume(ImageView<const std::uint16_t> left, ImageView<const std::uint16_t> right,
                                           int disparity_min, int num_disparities, const StereoOptions& options,
                                           const LabelRanges* ranges) {
  const MatchContext ctx = make_context(left, right, {}, {}, disparity_min, num_disparities, ranges, options);
  const Tile tile{0, 0, left.width(), left.height()};
  const TileLabels labels(ctx, tile);
  std::vector<std::uint16_t> volume(labels.cells());
  TileAggregator(ctx, tile, labels).run(volume.data());
  return volume;
}

std::vector<std::uint16_t> sgm_cost_volume_reference(ImageView<const std::uint16_t> left,
                                                     ImageView<const std::uint16_t> right, int disparity_min,
                                                     int num_disparities, const StereoOptions& options,
                                                     const LabelRanges* ranges) {
  check_options(options);
  const int w = left.width();
  const int h = left.height();
  const int wr = right.width();
  const int nd = align_labels(num_disparities);
  if (ranges != nullptr) {
    check_ranges(*ranges, w, h, nd);
  }
  auto first = [&](int x, int y) { return ranges != nullptr ? int{ranges->lo(x, y)} : 0; };
  auto count = [&](int x, int y) { return ranges != nullptr ? int{ranges->num(x, y)} : nd; };
  auto in_range = [&](int x, int y, int d) { return d >= first(x, y) && d < first(x, y) + count(x, y); };
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
  constexpr int kInf = 1 << 28;  // L of labels outside a pixel's range
  std::vector<int> total(cost.size(), 0);
  std::vector<int> path(cost.size(), kInf);
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
        const int d0 = first(x, y);
        const int d1 = d0 + count(x, y);
        if (qx < 0 || qy < 0 || qx >= w || qy >= h) {
          for (int d = d0; d < d1; ++d) {
            path[at(x, y, d)] = cost[at(x, y, d)];
          }
          continue;
        }
        int min_q = kInf;
        for (int d = first(qx, qy); d < first(qx, qy) + count(qx, qy); ++d) {
          min_q = std::min(min_q, path[at(qx, qy, d)]);
        }
        auto lq = [&](int d) { return d >= 0 && d < nd && in_range(qx, qy, d) ? path[at(qx, qy, d)] : kInf; };
        const int p2 = lut[static_cast<std::size_t>(std::abs(int{left(x, y)} - int{left(qx, qy)}) >> 8)];
        for (int d = d0; d < d1; ++d) {
          const int best = std::min({lq(d), lq(d - 1) + options.p1, lq(d + 1) + options.p1, min_q + p2});
          path[at(x, y, d)] = cost[at(x, y, d)] + best - min_q;
        }
      }
    }
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        for (int d = first(x, y); d < first(x, y) + count(x, y); ++d) {
          total[at(x, y, d)] += path[at(x, y, d)];
        }
      }
    }
  }
  std::vector<std::uint16_t> out;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int d = first(x, y); d < first(x, y) + count(x, y); ++d) {
        out.push_back(static_cast<std::uint16_t>(total[at(x, y, d)]));
      }
    }
  }
  return out;
}

}  // namespace detail
}  // namespace r3d
