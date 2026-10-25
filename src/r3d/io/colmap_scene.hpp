#pragma once

#include <cstddef>
#include <filesystem>

#include "r3d/scene/scene.hpp"

namespace r3d {

struct ColmapReadOptions {
  // Directory the image names are relative to. If empty: <model>/../../images
  // (COLMAP project layout sparse/0), <model>/../images, <model>/images, <model>.
  std::filesystem::path image_dir;
  bool load_structure = true;
  bool require_images = true;
};

struct ColmapImportReport {
  std::size_t num_cameras = 0;
  std::size_t num_images = 0;
  std::size_t num_images_unsupported = 0;  // camera model not supported
  std::size_t num_points = 0;
  std::size_t num_points_dropped = 0;
};

// Reads a COLMAP text model (cameras.txt, images.txt, points3D.txt) from
// `model_dir`, for benchmark datasets (ETH3D, Tanks and Temples).
// Supported camera models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL,
// OPENCV, OPENCV_FISHEYE, and FULL_OPENCV without rational terms.
// COLMAP pixel coordinates (origin at the corner of the top-left pixel) are
// shifted by -0.5 to the r3d pixel-centre convention (DECISIONS.md D10).
[[nodiscard]] Scene read_colmap_scene(const std::filesystem::path& model_dir, const ColmapReadOptions& options = {},
                                      ColmapImportReport* report = nullptr);

}  // namespace r3d
