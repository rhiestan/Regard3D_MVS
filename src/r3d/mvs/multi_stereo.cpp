#include "r3d/mvs/multi_stereo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/LU>

#include "r3d/core/error.hpp"
#include "r3d/core/log.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/core/timer.hpp"
#include "r3d/io/image_io.hpp"
#include "r3d/scene/undistort.hpp"

namespace r3d {

namespace {

constexpr int kMaxPairs = 16;

// Per pair: grid pixel -> rectified pixel homography and triangulation constants.
struct PairGeometry {
  Eigen::Matrix3d grid_to_rect;   // K_r R_r R_g^T K_g^-1; third row gives c
  double fb = 0.0;                // f * baseline
  double offset = 0.0;            // disparity offset
  Eigen::Vector3d partner_center;
  const ImageF32* disparity = nullptr;
  const ImageF32* confidence = nullptr;
};

struct Candidate {
  double w = 0.0;     // inverse depth
  double k = 0.0;     // d = k * w + offset
  float confidence = 0.0F;
  int pair = 0;
};

}  // namespace

float sample_disparity(const ImageF32& disparity, double u, double v) {
  const int w = disparity.width();
  const int h = disparity.height();
  if (!(u > -0.5 && v > -0.5 && u < w - 0.5 && v < h - 0.5)) {
    return kInvalidDisparity;
  }
  const int x0 = std::clamp(static_cast<int>(std::floor(u)), 0, w - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(v)), 0, h - 1);
  const int x1 = std::min(x0 + 1, w - 1);
  const int y1 = std::min(y0 + 1, h - 1);
  const float d00 = disparity(x0, y0);
  const float d10 = disparity(x1, y0);
  const float d01 = disparity(x0, y1);
  const float d11 = disparity(x1, y1);
  if (std::isfinite(d00) && std::isfinite(d10) && std::isfinite(d01) && std::isfinite(d11)) {
    const float lo = std::min({d00, d10, d01, d11});
    const float hi = std::max({d00, d10, d01, d11});
    if (hi - lo <= 1.0F) {
      const auto fx = static_cast<float>(std::clamp(u - x0, 0.0, 1.0));
      const auto fy = static_cast<float>(std::clamp(v - y0, 0.0, 1.0));
      const float top = d00 + fx * (d10 - d00);
      const float bottom = d01 + fx * (d11 - d01);
      return top + fy * (bottom - top);
    }
  }
  const int xn = std::clamp(static_cast<int>(std::lround(u)), 0, w - 1);
  const int yn = std::clamp(static_cast<int>(std::lround(v)), 0, h - 1);
  return disparity(xn, yn);
}

DepthMap combine_pairs(const PinholeCamera& grid, std::span<const PairDisparity> pairs,
                       const MultiStereoOptions& options) {
  DepthMap map;
  map.camera = grid;
  map.allocate(grid.intrinsics.width, grid.intrinsics.height);
  if (pairs.size() > static_cast<std::size_t>(kMaxPairs)) {
    throw_error("combine_pairs: at most {} pairs per view ({} given)", kMaxPairs, pairs.size());
  }
  const Eigen::Matrix3d grid_inv = grid.intrinsics.K().inverse();
  const Eigen::Matrix3d grid_rot_t = grid.pose.rotation.transpose();
  std::vector<PairGeometry> geo;
  for (const PairDisparity& pd : pairs) {
    map.partners.push_back(pd.partner_view);
    const RectifiedPair& rp = *pd.pair;
    if ((rp.left.pose.center - grid.pose.center).norm() > 1e-9 * (1.0 + grid.pose.center.norm())) {
      throw_error("combine_pairs: pair does not share the grid's camera centre");
    }
    PairGeometry g;
    g.grid_to_rect = rp.left.intrinsics.K() * rp.left.pose.rotation * grid_rot_t * grid_inv;
    g.fb = rp.focal() * rp.baseline;
    g.offset = rp.disparity_offset();
    g.partner_center = rp.right.pose.center;
    g.disparity = &pd.disparity->disparity;
    g.confidence = pd.disparity->confidence.empty() ? nullptr : &pd.disparity->confidence;
    geo.push_back(g);
  }
  const double cos_min_angle = std::cos(options.min_angle_deg * std::numbers::pi / 180.0);
  const int min_pairs = std::max(1, options.min_pairs);
  const int w = grid.intrinsics.width;
  const int h = grid.intrinsics.height;

  parallel_for(0, static_cast<std::size_t>(h), 8, [&](std::size_t begin, std::size_t end) {
    std::array<Candidate, kMaxPairs> cand{};
    std::array<int, kMaxPairs> used{};
    for (auto yy = static_cast<int>(begin); yy < static_cast<int>(end); ++yy) {
      for (int x = 0; x < w; ++x) {
        const Eigen::Vector3d pix(x, yy, 1.0);
        int n = 0;
        for (std::size_t i = 0; i < geo.size(); ++i) {
          const PairGeometry& g = geo[i];
          const Eigen::Vector3d r = g.grid_to_rect * pix;
          const double c = r.z();
          if (!(c > 1e-9)) {
            continue;
          }
          const double u = r.x() / c;
          const double v = r.y() / c;
          const float d = sample_disparity(*g.disparity, u, v);
          if (!std::isfinite(d) || d <= g.offset) {
            continue;
          }
          Candidate& cd = cand[static_cast<std::size_t>(n++)];
          cd.k = g.fb / c;
          cd.w = (d - g.offset) / cd.k;
          cd.pair = static_cast<int>(i);
          cd.confidence = 0.0F;
          if (g.confidence != nullptr) {
            const int xn = std::clamp(static_cast<int>(std::lround(u)), 0, g.confidence->width() - 1);
            const int yn = std::clamp(static_cast<int>(std::lround(v)), 0, g.confidence->height() - 1);
            cd.confidence = (*g.confidence)(xn, yn);
          }
        }
        if (n < min_pairs) {
          continue;
        }
        // Hypothesis with the most agreeing pairs (ties: higher summed confidence).
        int best = -1, best_support = 0;
        float best_conf = -1.0F;
        for (int i = 0; i < n; ++i) {
          int support = 0;
          float conf = 0.0F;
          for (int j = 0; j < n; ++j) {
            if (std::abs(cand[j].k * (cand[j].w - cand[i].w)) <= options.inlier_px) {
              ++support;
              conf += cand[j].confidence;
            }
          }
          if (support > best_support || (support == best_support && conf > best_conf)) {
            best = i;
            best_support = support;
            best_conf = conf;
          }
        }
        int m = 0;
        for (int j = 0; j < n; ++j) {
          if (std::abs(cand[j].k * (cand[j].w - cand[best].w)) <= options.inlier_px) {
            used[m++] = j;
          }
        }
        // Weighted least squares on w; drop the worst pair while a residual is too large.
        double w_opt = 0.0;
        while (m > 0) {
          double sw = 0.0, swx = 0.0;
          for (int t = 0; t < m; ++t) {
            const Candidate& cd = cand[used[t]];
            sw += cd.k * cd.k;
            swx += cd.k * cd.k * cd.w;
          }
          w_opt = swx / sw;
          int worst = -1;
          double worst_res = options.max_reprojection_px;
          for (int t = 0; t < m; ++t) {
            const Candidate& cd = cand[used[t]];
            const double res = std::abs(cd.k * (cd.w - w_opt));
            if (res > worst_res) {
              worst_res = res;
              worst = t;
            }
          }
          if (worst < 0) {
            break;
          }
          used[worst] = used[--m];
        }
        const bool majority = m > n - m || (m >= 2 && m >= n - m);
        if (m < min_pairs || !majority || !(w_opt > 0.0)) {
          continue;
        }
        const double z = 1.0 / w_opt;
        const Eigen::Vector3d ray = grid_rot_t * (grid_inv * pix);  // world direction with z = 1 in the grid camera
        const Eigen::Vector3d x_world = grid.pose.center + z * ray;
        const Eigen::Vector3d to_base = (grid.pose.center - x_world).normalized();
        bool wide = false;
        float conf = 0.0F;
        std::uint16_t mask = 0;
        for (int t = 0; t < m; ++t) {
          const Candidate& cd = cand[used[t]];
          mask = static_cast<std::uint16_t>(mask | (1U << cd.pair));
          const Eigen::Vector3d to_partner = (geo[static_cast<std::size_t>(cd.pair)].partner_center - x_world).normalized();
          wide = wide || to_base.dot(to_partner) <= cos_min_angle;
          conf += cd.confidence;
        }
        if (!wide) {
          continue;
        }
        map.depth(x, yy) = static_cast<float>(z);
        map.confidence(x, yy) = conf / static_cast<float>(m);
        map.pair_mask(x, yy) = mask;
      }
    }
  });
  return map;
}

PinholeCamera depth_map_camera(const Scene& scene, std::size_t view, double resolution) {
  PinholeCamera cam;
  cam.intrinsics = scale_intrinsics(undistorted_camera(scene.camera_of(view)), resolution);
  cam.pose = scene.views[view].pose;
  return cam;
}

double depth_map_resolution(const Scene& scene, std::size_t view, const DepthMapOptions& options) {
  if (options.max_image_size <= 0) {
    return options.resolution;
  }
  const PinholeIntrinsics cam = undistorted_camera(scene.camera_of(view));
  const int longer = std::max(cam.width, cam.height);
  return std::min(options.resolution, static_cast<double>(options.max_image_size) / std::max(longer, 1));
}

DepthMap compute_depth_map(const Scene& scene, const ViewSelection& selection, std::size_t view,
                           const DepthMapOptions& options, DepthMapStats* stats) {
  DepthMapStats local;
  DepthMapStats& st = stats != nullptr ? *stats : local;
  st = DepthMapStats{};
  const SceneView& vb = scene.views[view];
  const double resolution = depth_map_resolution(scene, view, options);
  const PinholeCamera grid = depth_map_camera(scene, view, resolution);
  RectificationOptions rect_options = options.rectification;
  rect_options.resolution = resolution;

  Stopwatch watch;
  const ImageU16 base_gray = read_image_u16(vb.image_path, 1);
  st.load_seconds += watch.elapsed_seconds();

  std::vector<RectifiedPair> rect_pairs;
  std::vector<DisparityMap> disparities;
  std::vector<std::uint32_t> partner_views;
  rect_pairs.reserve(static_cast<std::size_t>(options.max_pairs));
  disparities.reserve(static_cast<std::size_t>(options.max_pairs));
  int used = 0;
  for (const StereoPartner& sp : selection.partners[view]) {
    if (used >= options.max_pairs) {
      break;
    }
    ++used;
    const SceneView& vp = scene.views[sp.view];
    const DepthRange depth = sp.depth.valid() ? sp.depth : selection.view_depth[view];
    RectifiedPair rp =
        compute_rectification(scene.camera_of(view), vb.pose, scene.camera_of(sp.view), vp.pose, depth, rect_options);
    if (!rp.valid) {
      log_debug("view {}: pair with {} skipped: {}", vb.source_id, vp.source_id, rp.reason);
      ++st.pairs_skipped;
      continue;
    }
    watch.restart();
    ImageU8 mask_left, mask_right;
    const ImageU16 left = rectify_image(base_gray, scene.camera_of(view), vb.pose, rp.left, &mask_left);
    const ImageU16 right =
        rectify_image(read_image_u16(vp.image_path, 1), scene.camera_of(sp.view), vp.pose, rp.right, &mask_right);
    st.load_seconds += watch.elapsed_seconds();
    watch.restart();
    DisparityMap disp = match_stereo(left.view(), right.view(), mask_left.view(), mask_right.view(), rp.disparity_min,
                                     rp.num_disparities, options.stereo);
    st.match_seconds += watch.elapsed_seconds();
    st.cells += disp.cells;
    ++st.pairs_matched;
    rect_pairs.push_back(std::move(rp));
    partner_views.push_back(sp.view);
    disparities.push_back(std::move(disp));
  }

  watch.restart();
  std::vector<PairDisparity> inputs;
  for (std::size_t i = 0; i < rect_pairs.size(); ++i) {
    inputs.push_back({&rect_pairs[i], &disparities[i], partner_views[i]});
  }
  DepthMap map = combine_pairs(grid, inputs, options.multi_stereo);
  map.view = static_cast<std::uint32_t>(view);
  st.combine_seconds += watch.elapsed_seconds();

  watch.restart();
  map.color = rectify_image(read_image_u8(vb.image_path, 3), scene.camera_of(view), vb.pose, grid);
  st.load_seconds += watch.elapsed_seconds();
  return map;
}

}  // namespace r3d
