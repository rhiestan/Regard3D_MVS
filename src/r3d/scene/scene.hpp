#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "r3d/core/camera.hpp"
#include "r3d/core/camera_model.hpp"

namespace r3d {

// One calibrated image of the scene.
struct SceneView {
  std::uint32_t source_id = 0;        // id in the source (OpenMVG id_view, COLMAP image_id)
  std::filesystem::path image_path;   // resolved path of the original (distorted) image
  int width = 0;
  int height = 0;
  std::uint32_t camera = 0;           // index into Scene::cameras
  Pose pose;
};

// Observation of a sparse point in a view, in distorted pixel coordinates of
// the original image (as stored by SfM).
struct TrackObservation {
  std::uint32_t view = 0;  // index into Scene::views
  float x = 0.0F;
  float y = 0.0F;
};

// Calibrated scene: cameras, views and sparse SfM structure. Only views with
// both a pose and a supported intrinsic are kept, so every view is usable.
// Views are sorted by source id and tracks by source landmark id, which makes
// all results independent of the hash-map order in the input file.
struct Scene {
  std::vector<CameraIntrinsics> cameras;
  std::vector<SceneView> views;

  // Sparse points with tracks in CSR layout: the observations of point i are
  // observations[track_offsets[i] .. track_offsets[i + 1]).
  std::vector<Eigen::Vector3d> points;
  std::vector<std::uint32_t> track_offsets{0};
  std::vector<TrackObservation> observations;

  [[nodiscard]] std::size_t num_points() const noexcept { return points.size(); }
  [[nodiscard]] std::span<const TrackObservation> track(std::size_t point) const {
    return std::span<const TrackObservation>(observations)
        .subspan(track_offsets[point], track_offsets[point + 1] - track_offsets[point]);
  }
  [[nodiscard]] const CameraIntrinsics& camera_of(std::size_t view) const { return cameras[views[view].camera]; }

  // Appends a point with its track (observations must refer to valid views).
  void add_point(const Eigen::Vector3d& x, std::span<const TrackObservation> track);

  // Throws r3d::Error if indices or array sizes are inconsistent.
  void validate() const;
};

// Reprojection error statistics of the sparse structure (a sanity check of a
// scene import: SfM results have a sub-pixel RMS error).
struct ReprojectionStats {
  std::size_t num_observations = 0;
  std::size_t num_behind_camera = 0;
  double rms = 0.0;
  double median = 0.0;
  double max = 0.0;
};

[[nodiscard]] ReprojectionStats compute_reprojection_stats(const Scene& scene);

}  // namespace r3d
