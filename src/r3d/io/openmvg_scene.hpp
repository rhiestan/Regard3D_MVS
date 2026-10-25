#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "r3d/scene/scene.hpp"

namespace r3d {

struct OpenMvgReadOptions {
  // Directory searched first for the images. If empty, images are looked up
  // relative to root_path, then next to the sfm_data file.
  std::filesystem::path image_dir;
  // Read the sparse structure (landmarks with observations).
  bool load_structure = true;
  // Throw if an image cannot be found; otherwise keep the first candidate path.
  bool require_images = true;
};

// What the import kept and dropped, for logging and the CLI.
struct OpenMvgImportReport {
  std::string version;                    // sfm_data_version, e.g. "0.3"
  std::string root_path;                  // as stored in the file
  std::size_t num_views = 0;              // views in the file
  std::size_t num_views_without_pose = 0;
  std::size_t num_views_unsupported = 0;  // missing or non-perspective intrinsic, size mismatch
  std::size_t num_points = 0;             // landmarks in the file
  std::size_t num_points_dropped = 0;     // all observations in dropped views
  std::size_t num_observations_dropped = 0;
};

// Reads an OpenMVG scene: sfm_data.json (cereal JSON) or sfm_data.bin
// (cereal portable binary), selected by the file extension. Supports
// sfm_data versions 0.1-0.3, the camera models of r3d::CameraModel and
// view_priors. Throws r3d::IoError on unreadable or malformed files.
[[nodiscard]] Scene read_openmvg_scene(const std::filesystem::path& path, const OpenMvgReadOptions& options = {},
                                       OpenMvgImportReport* report = nullptr);

}  // namespace r3d
