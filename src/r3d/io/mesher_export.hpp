#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string_view>

#include "r3d/core/camera.hpp"
#include "r3d/core/point_cloud.hpp"
#include "r3d/scene/scene.hpp"

namespace r3d {

// Point attributes for an external surface reconstruction tool.
enum class CloudLayout {
  full,     // everything the cloud has (positions, normals, colours, "value", "confidence", view ids)
  poisson,  // PoissonRecon: positions, normals, colours; normals optionally scaled by the confidence
  fssr,     // MVE FSSR: positions, normals, colours, "value" (sample scale) and "confidence" in (0, 1]
};

[[nodiscard]] std::optional<CloudLayout> parse_cloud_layout(std::string_view name);

struct CloudLayoutOptions {
  CloudLayout layout = CloudLayout::full;
  // fssr: "value" = scale_factor x the point's footprint (its GSD, stored in "value").
  double fssr_scale = 8.0;
  // fssr: "confidence" = min(1, confidence / fssr_full_confidence), the
  // confidence being the number of fused depths.
  double fssr_full_confidence = 4.0;
  // poisson: normal length = min(1, confidence / poisson_full_confidence), for
  // PoissonRecon --confidence; 0 writes unit normals.
  double poisson_full_confidence = 4.0;
};

// The cloud with the attributes of `options.layout`. Throws if the layout
// needs an attribute the cloud lacks (normals; "value" for fssr).
[[nodiscard]] PointCloud cloud_for_mesher(const PointCloud& cloud, const CloudLayoutOptions& options);

// texrecon (mvs-texturing) camera file: "tx ty tz R00 .. R22" and
// "f d0 d1 paspect ppx ppy", f normalised by the larger image side, the
// principal point in units of the image size (pixel corner convention).
void write_texrecon_camera(const std::filesystem::path& path, const PinholeIntrinsics& k, const Pose& pose);

struct TexreconExportOptions {
  int jpeg_quality = 95;
};

struct TexreconExportReport {
  std::size_t images = 0;
  std::size_t copied = 0;  // images without distortion, copied unchanged
};

// texrecon scene folder: one undistorted image (<name>.jpg) and one
// <name>.cam per view, so that "texrecon <dir> mesh.ply out" runs directly.
// Images of cameras without distortion that are already JPEG are copied.
void write_texrecon_scene(const Scene& scene, const std::filesystem::path& dir,
                          const TexreconExportOptions& options = {}, TexreconExportReport* report = nullptr);

}  // namespace r3d
