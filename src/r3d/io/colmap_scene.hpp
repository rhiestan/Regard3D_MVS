#pragma once

#include <cstddef>
#include <filesystem>

#include "r3d/scene/scene.hpp"

namespace r3d {

struct ColmapReadOptions {
  // Directory the image names are relative to. If empty: <model>/../../images
  // (COLMAP project layout sparse/0), <model>/../images, <model>/images,
  // <model>/.. and <model>.
  std::filesystem::path image_dir;
  bool load_structure = true;
  bool require_images = true;
};

struct ColmapImportReport {
  std::filesystem::path model_dir;  // the directory the model was read from
  bool binary = false;              // .bin files (otherwise .txt)
  std::size_t num_cameras = 0;
  std::size_t num_images = 0;
  std::size_t num_images_unsupported = 0;  // camera model not supported
  std::size_t num_points = 0;
  std::size_t num_points_dropped = 0;
};

// Directory of the COLMAP sparse model for `path`: `path` itself if it holds
// cameras.bin or cameras.txt, else <path>/sparse/0 or <path>/sparse.
// Throws r3d::IoError if none of them holds a model.
[[nodiscard]] std::filesystem::path find_colmap_model(const std::filesystem::path& path);

// Reads a COLMAP sparse model: binary (cameras.bin, images.bin, points3D.bin)
// if cameras.bin exists, otherwise text (cameras.txt, images.txt, points3D.txt).
// `path` is the model directory or a project directory (see find_colmap_model).
// Supported camera models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL,
// OPENCV, OPENCV_FISHEYE, and FULL_OPENCV without rational terms; images of
// other cameras are skipped.
// COLMAP pixel coordinates (origin at the corner of the top-left pixel) are
// shifted by -0.5 to the r3d pixel-centre convention (DECISIONS.md D10).
[[nodiscard]] Scene read_colmap_scene(const std::filesystem::path& path, const ColmapReadOptions& options = {},
                                      ColmapImportReport* report = nullptr);

}  // namespace r3d
