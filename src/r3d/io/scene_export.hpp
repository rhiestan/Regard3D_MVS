#pragma once

#include <filesystem>

#include "r3d/core/point_cloud.hpp"
#include "r3d/scene/scene.hpp"

namespace r3d {

// Sparse SfM structure as a point cloud: positions (relative to the scene
// centroid), the view ids of each track as visibility and a "track_length"
// scalar. With `with_colors`, colours are averaged over all observations,
// which reads every image of the scene once.
[[nodiscard]] PointCloud sparse_point_cloud(const Scene& scene, bool with_colors);

// Camera frustums as a triangle mesh PLY (binary): apex at the camera centre,
// the image rectangle at distance `size` along the viewing direction, one
// colour per camera. For visual checks in MeshLab / CloudCompare.
void write_camera_frustums_ply(const std::filesystem::path& path, const Scene& scene, double size);

}  // namespace r3d
