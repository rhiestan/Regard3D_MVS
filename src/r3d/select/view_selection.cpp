#include "r3d/select/view_selection.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <unordered_map>

#include "r3d/core/error.hpp"

namespace r3d {
namespace {

constexpr double kRadToDeg = 180.0 / std::numbers::pi;

struct PairAccum {
  std::uint32_t shared = 0;
  double weight = 0.0;         // sum of per-point angle x scale weights
  double angle_sum = 0.0;      // degrees
  double log_scale_sum = 0.0;  // log(GSD(base) / GSD(partner))
  std::uint64_t cells = 0;     // base-image grid cells with a usable shared point
};

struct ObservationGeometry {
  std::uint32_t view = 0;
  Eigen::Vector3d ray = Eigen::Vector3d::Zero();  // unit, world coordinates, camera to point
  double depth = 0.0;
  double gsd = 0.0;  // ground sample distance: depth / focal length
  int cell = 0;
};

double angle_weight(double angle_deg, const ViewSelectionOptions& options) {
  if (angle_deg < options.min_angle_deg || angle_deg > options.max_angle_deg) {
    return 0.0;
  }
  double w = 0.0;
  for (const double preferred : options.preferred_angles_deg) {
    const double t = (angle_deg - preferred) / options.angle_sigma_deg;
    w = std::max(w, std::exp(-0.5 * t * t));
  }
  return w;
}

// Goesele et al. 2007: full weight up to a resolution ratio of 2, then 2/q.
double scale_weight(double ratio) {
  const double q = std::max(ratio, 1.0 / ratio);
  return q <= 2.0 ? 1.0 : 2.0 / q;
}

DepthRange robust_range(std::vector<double>& depths, const ViewSelectionOptions& options) {
  DepthRange range;
  if (depths.empty()) {
    return range;
  }
  std::sort(depths.begin(), depths.end());
  const double last = static_cast<double>(depths.size() - 1);
  const double p = std::clamp(options.depth_percentile, 0.0, 0.5);
  range.min = depths[static_cast<std::size_t>(std::floor(p * last))] / (1.0 + options.depth_margin);
  range.max = depths[static_cast<std::size_t>(std::ceil((1.0 - p) * last))] * (1.0 + options.depth_margin);
  return range;
}

double wrapped_angle_difference(double a, double b) {
  const double d = std::fmod(std::abs(a - b), 2.0 * std::numbers::pi);
  return d > std::numbers::pi ? 2.0 * std::numbers::pi - d : d;
}

}  // namespace

std::vector<std::pair<std::uint32_t, std::uint32_t>> ViewSelection::stereo_pairs() const {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
  for (std::size_t base = 0; base < partners.size(); ++base) {
    for (const StereoPartner& p : partners[base]) {
      const auto a = static_cast<std::uint32_t>(base);
      pairs.emplace_back(std::min(a, p.view), std::max(a, p.view));
    }
  }
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  return pairs;
}

ViewSelection select_views(const Scene& scene, const ViewSelectionOptions& options) {
  scene.validate();
  if (options.coverage_grid < 1 || options.coverage_grid > 8) {
    throw_error("select_views: coverage_grid must be in [1, 8]");
  }
  const std::size_t num_views = scene.views.size();
  const int grid = options.coverage_grid;

  std::vector<std::uint32_t> view_points(num_views, 0);
  std::vector<std::uint64_t> view_cells(num_views, 0);
  std::vector<std::vector<double>> view_depths(num_views);
  std::vector<std::unordered_map<std::uint32_t, PairAccum>> accum(num_views);

  // Pass 1: per-point geometry and pairwise accumulation over each track.
  std::vector<ObservationGeometry> geo;
  for (std::size_t i = 0; i < scene.num_points(); ++i) {
    const Eigen::Vector3d& x = scene.points[i];
    geo.clear();
    for (const TrackObservation& obs : scene.track(i)) {
      const SceneView& view = scene.views[obs.view];
      const double depth = view.pose.world_to_camera(x).z();
      if (!(depth > 0.0)) {
        continue;
      }
      ObservationGeometry g;
      g.view = obs.view;
      g.ray = (x - view.pose.center).normalized();
      g.depth = depth;
      g.gsd = depth / scene.camera_of(obs.view).fx;
      const int cx = std::clamp(static_cast<int>((obs.x + 0.5F) * static_cast<float>(grid) / static_cast<float>(view.width)), 0, grid - 1);
      const int cy = std::clamp(static_cast<int>((obs.y + 0.5F) * static_cast<float>(grid) / static_cast<float>(view.height)), 0, grid - 1);
      g.cell = cy * grid + cx;
      geo.push_back(g);
      ++view_points[obs.view];
      view_cells[obs.view] |= std::uint64_t{1} << g.cell;
      view_depths[obs.view].push_back(depth);
    }
    for (std::size_t a = 0; a < geo.size(); ++a) {
      for (std::size_t b = a + 1; b < geo.size(); ++b) {
        const ObservationGeometry& ga = geo[a];
        const ObservationGeometry& gb = geo[b];
        if (ga.view == gb.view) {
          continue;
        }
        const double angle = std::acos(std::clamp(ga.ray.dot(gb.ray), -1.0, 1.0)) * kRadToDeg;
        const double wa = angle_weight(angle, options);
        const double ratio = ga.gsd / gb.gsd;
        const double w = wa * scale_weight(ratio);
        const double log_ratio = std::log(ratio);
        PairAccum& ab = accum[ga.view][gb.view];
        PairAccum& ba = accum[gb.view][ga.view];
        for (PairAccum* acc : {&ab, &ba}) {
          ++acc->shared;
          acc->weight += w;
          acc->angle_sum += angle;
        }
        ab.log_scale_sum += log_ratio;
        ba.log_scale_sum -= log_ratio;
        if (w > 0.0) {
          ab.cells |= std::uint64_t{1} << ga.cell;
          ba.cells |= std::uint64_t{1} << gb.cell;
        }
      }
    }
  }

  ViewSelection result;
  result.partners.resize(num_views);
  result.view_depth.resize(num_views);
  for (std::size_t v = 0; v < num_views; ++v) {
    result.view_depth[v] = robust_range(view_depths[v], options);
  }

  // Scoring and greedy selection with a spread of baseline directions.
  const double log_max_ratio = std::log(std::max(options.max_scale_ratio, 1.0));
  for (std::size_t base = 0; base < num_views; ++base) {
    const SceneView& bv = scene.views[base];
    const PinholeIntrinsics k = scene.camera_of(base).pinhole();
    const int base_cells = std::popcount(view_cells[base]);
    struct Candidate {
      StereoPartner partner;
      double direction = 0.0;  // of the baseline in the base image plane
    };
    std::vector<Candidate> candidates;
    for (const auto& [other, acc] : accum[base]) {
      if (acc.shared < static_cast<std::uint32_t>(options.min_shared_points) || acc.weight <= 0.0 || base_cells == 0) {
        continue;
      }
      const double mean_log_ratio = acc.log_scale_sum / acc.shared;
      if (std::abs(mean_log_ratio) > log_max_ratio) {
        continue;
      }
      Candidate c;
      c.partner.view = other;
      c.partner.shared_points = acc.shared;
      c.partner.mean_angle_deg = acc.angle_sum / acc.shared;
      c.partner.scale_ratio = std::exp(mean_log_ratio);
      c.partner.coverage = static_cast<double>(std::popcount(acc.cells)) / base_cells;
      c.partner.score = acc.weight / view_points[base] * c.partner.coverage;
      const Eigen::Vector3d b = bv.pose.rotation * (scene.views[other].pose.center - bv.pose.center);
      c.direction = std::atan2(b.y(), b.x());
      if (std::abs(b.z()) > 1e-12 * b.norm()) {
        c.partner.epipole_in_image = k.contains(Eigen::Vector2d(k.fx * b.x() / b.z() + k.cx, k.fy * b.y() / b.z() + k.cy));
      }
      candidates.push_back(c);
    }
    // Deterministic order regardless of hash-map iteration.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.partner.view < b.partner.view; });
    double best = 0.0;
    for (const Candidate& c : candidates) {
      best = std::max(best, c.partner.score);
    }
    std::erase_if(candidates, [&](const Candidate& c) { return c.partner.score < options.min_relative_score * best; });

    std::vector<double> chosen_directions;
    while (static_cast<int>(result.partners[base].size()) < options.max_partners && !candidates.empty()) {
      std::size_t pick = 0;
      double pick_value = -1.0;
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        double spread = 1.0;
        for (const double d : chosen_directions) {
          spread = std::min(spread, wrapped_angle_difference(candidates[i].direction, d) / (0.5 * std::numbers::pi));
        }
        const double value = candidates[i].partner.score * (0.25 + 0.75 * spread);
        if (value > pick_value) {
          pick_value = value;
          pick = i;
        }
      }
      chosen_directions.push_back(candidates[pick].direction);
      result.partners[base].push_back(candidates[pick].partner);
      candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(pick));
    }
  }

  // Pass 2: depth ranges of the points shared with the selected partners.
  std::vector<std::vector<std::vector<double>>> pair_depths(num_views);
  for (std::size_t v = 0; v < num_views; ++v) {
    pair_depths[v].resize(result.partners[v].size());
  }
  for (std::size_t i = 0; i < scene.num_points(); ++i) {
    const auto track = scene.track(i);
    for (const TrackObservation& a : track) {
      const std::vector<StereoPartner>& partners = result.partners[a.view];
      if (partners.empty()) {
        continue;
      }
      double depth = -1.0;
      for (const TrackObservation& b : track) {
        for (std::size_t p = 0; p < partners.size(); ++p) {
          if (partners[p].view != b.view) {
            continue;
          }
          if (depth < 0.0) {
            depth = scene.views[a.view].pose.world_to_camera(scene.points[i]).z();
          }
          if (depth > 0.0) {
            pair_depths[a.view][p].push_back(depth);
          }
        }
      }
    }
  }
  for (std::size_t v = 0; v < num_views; ++v) {
    for (std::size_t p = 0; p < result.partners[v].size(); ++p) {
      result.partners[v][p].depth = robust_range(pair_depths[v][p], options);
    }
  }
  return result;
}

}  // namespace r3d
