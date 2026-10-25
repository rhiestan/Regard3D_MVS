#include "r3d/scene/scene.hpp"

#include <algorithm>
#include <cmath>

#include "r3d/core/error.hpp"

namespace r3d {

void Scene::add_point(const Eigen::Vector3d& x, std::span<const TrackObservation> track) {
  points.push_back(x);
  observations.insert(observations.end(), track.begin(), track.end());
  track_offsets.push_back(static_cast<std::uint32_t>(observations.size()));
}

void Scene::validate() const {
  for (std::size_t v = 0; v < views.size(); ++v) {
    const SceneView& view = views[v];
    if (view.camera >= cameras.size()) {
      throw_error("scene: view {} refers to camera {} of {}", v, view.camera, cameras.size());
    }
    if (view.width <= 0 || view.height <= 0) {
      throw_error("scene: view {} has invalid size {}x{}", v, view.width, view.height);
    }
  }
  if (track_offsets.size() != points.size() + 1 || track_offsets.front() != 0 ||
      track_offsets.back() != observations.size()) {
    throw_error("scene: track offsets are inconsistent with {} points / {} observations", points.size(),
                observations.size());
  }
  if (!std::is_sorted(track_offsets.begin(), track_offsets.end())) {
    throw_error("scene: track offsets are not monotonic");
  }
  for (const TrackObservation& obs : observations) {
    if (obs.view >= views.size()) {
      throw_error("scene: observation refers to view {} of {}", obs.view, views.size());
    }
  }
}

ReprojectionStats compute_reprojection_stats(const Scene& scene) {
  ReprojectionStats stats;
  std::vector<double> errors;
  errors.reserve(scene.observations.size());
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < scene.num_points(); ++i) {
    for (const TrackObservation& obs : scene.track(i)) {
      const SceneView& view = scene.views[obs.view];
      const Eigen::Vector3d x_cam = view.pose.world_to_camera(scene.points[i]);
      ++stats.num_observations;
      if (x_cam.z() <= 0.0) {
        ++stats.num_behind_camera;
        continue;
      }
      const Eigen::Vector2d projected = scene.cameras[view.camera].project(x_cam);
      const double err = (projected - Eigen::Vector2d(obs.x, obs.y)).norm();
      errors.push_back(err);
      sum_sq += err * err;
      stats.max = std::max(stats.max, err);
    }
  }
  if (!errors.empty()) {
    stats.rms = std::sqrt(sum_sq / static_cast<double>(errors.size()));
    const auto mid = errors.begin() + static_cast<std::ptrdiff_t>(errors.size() / 2);
    std::nth_element(errors.begin(), mid, errors.end());
    stats.median = *mid;
  }
  return stats;
}

}  // namespace r3d
