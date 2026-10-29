#include "r3d/io/mesher_export.hpp"

#include <algorithm>
#include <cctype>
#include <format>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "r3d/core/error.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/io/image_io.hpp"
#include "r3d/scene/undistort.hpp"

namespace r3d {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

const std::vector<float>& required_scalar(const PointCloud& cloud, std::string_view name, std::string_view layout) {
  const std::vector<float>* values = cloud.find_scalar(name);
  if (values == nullptr) {
    throw_error("the {} layout needs the point attribute '{}'", layout, name);
  }
  return *values;
}

}  // namespace

std::optional<CloudLayout> parse_cloud_layout(std::string_view name) {
  if (name == "full") {
    return CloudLayout::full;
  }
  if (name == "poisson") {
    return CloudLayout::poisson;
  }
  if (name == "fssr") {
    return CloudLayout::fssr;
  }
  return std::nullopt;
}

std::string_view to_string(CloudLayout layout) {
  switch (layout) {
    case CloudLayout::full: return "full";
    case CloudLayout::poisson: return "poisson";
    case CloudLayout::fssr: return "fssr";
  }
  return "unknown";
}

namespace {

PlyWriteOptions with_layout_comment(PlyWriteOptions ply, CloudLayout layout) {
  if (layout != CloudLayout::full) {
    ply.comments.push_back(std::format("r3dmvs layout {}", to_string(layout)));
  }
  return ply;
}

}  // namespace

CloudWriter::CloudWriter(const std::filesystem::path& path, const CloudLayoutOptions& layout, PlyWriteOptions ply)
    : layout_(layout), writer_(path, with_layout_comment(std::move(ply), layout.layout)) {}

void CloudWriter::append(const PointCloud& part) {
  if (layout_.layout == CloudLayout::full) {
    writer_.append(part);
  } else {
    writer_.append(cloud_for_mesher(part, layout_));
  }
}

PointCloud cloud_for_mesher(const PointCloud& cloud, const CloudLayoutOptions& options) {
  cloud.validate();
  if (options.layout == CloudLayout::full) {
    return cloud;
  }
  const std::string_view name = options.layout == CloudLayout::poisson ? "poisson" : "fssr";
  if (!cloud.has_normals() && cloud.size() > 0) {
    throw_error("the {} layout needs normals", name);
  }
  PointCloud out;
  out.origin = cloud.origin;
  out.positions = cloud.positions;
  out.normals = cloud.normals;
  out.colors = cloud.colors;
  const std::vector<float>* confidence = cloud.find_scalar("confidence");
  if (options.layout == CloudLayout::poisson) {
    if (options.poisson_full_confidence > 0.0 && confidence != nullptr) {
      const auto inv = static_cast<float>(1.0 / options.poisson_full_confidence);
      for (std::size_t i = 0; i < out.size(); ++i) {
        out.normals[i] *= std::clamp((*confidence)[i] * inv, 0.0F, 1.0F);
      }
    }
    return out;
  }
  const std::vector<float>& footprint = required_scalar(cloud, "value", name);
  std::vector<float> value(footprint.size());
  const auto scale = static_cast<float>(options.fssr_scale);
  std::transform(footprint.begin(), footprint.end(), value.begin(), [&](float f) { return scale * f; });
  out.scalars.emplace_back("value", std::move(value));
  std::vector<float> conf(out.size(), 1.0F);
  if (confidence != nullptr && options.fssr_full_confidence > 0.0) {
    const auto inv = static_cast<float>(1.0 / options.fssr_full_confidence);
    for (std::size_t i = 0; i < out.size(); ++i) {
      conf[i] = std::clamp((*confidence)[i] * inv, 0.0F, 1.0F);
    }
  }
  out.scalars.emplace_back("confidence", std::move(conf));
  return out;
}

void write_texrecon_camera(const std::filesystem::path& path, const PinholeIntrinsics& k, const Pose& pose) {
  const Eigen::Vector3d t = pose.translation();
  const Eigen::Matrix3d& r = pose.rotation;
  // MVE's focal length is relative to the larger side, the principal point to the size, pixels span [i, i + 1].
  const double flen = k.width >= k.height ? k.fx / k.width : k.fy / k.height;
  std::string text = std::format("{:.17g} {:.17g} {:.17g}", t.x(), t.y(), t.z());
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      text += std::format(" {:.17g}", r(i, j));
    }
  }
  text += std::format("\n{:.17g} 0 0 {:.17g} {:.17g} {:.17g}\n", flen, k.fy / k.fx, (k.cx + 0.5) / k.width,
                      (k.cy + 0.5) / k.height);
  write_file(path, std::vector<std::uint8_t>(text.begin(), text.end()));
}

void write_texrecon_scene(const Scene& scene, const std::filesystem::path& dir, const TexreconExportOptions& options,
                          TexreconExportReport* report) {
  std::filesystem::create_directories(dir);
  // Unique names: the image stem, with the view id appended where stems repeat.
  // Stems compare case-insensitively (Windows file systems).
  std::vector<std::string> names(scene.views.size());
  std::map<std::string, int> stems;
  for (std::size_t v = 0; v < scene.views.size(); ++v) {
    names[v] = scene.views[v].image_path.stem().string();
    ++stems[lower(names[v])];
  }
  for (std::size_t v = 0; v < scene.views.size(); ++v) {
    if (stems[lower(names[v])] > 1) {
      names[v] += std::format("_{}", scene.views[v].source_id);
    }
  }
  std::vector<std::uint8_t> copied(scene.views.size(), 0);
  // One view per task: the JPEG encoder is single-threaded.
  parallel_for(0, scene.views.size(), 1, [&](std::size_t begin, std::size_t end) {
    for (std::size_t v = begin; v < end; ++v) {
      const SceneView& view = scene.views[v];
      const CameraIntrinsics& cam = scene.camera_of(v);
      UndistortOptions undistort;
      undistort.fit = UndistortFit::same_size;
      const PinholeIntrinsics target = undistorted_camera(cam, undistort);
      const std::filesystem::path image = dir / (names[v] + ".jpg");
      const std::string ext = lower(view.image_path.extension().string());
      bool copy = !cam.has_distortion() && (ext == ".jpg" || ext == ".jpeg");
      if (copy) {
        const ImageFileInfo info = read_image_info(view.image_path);
        copy = info.width == cam.width && info.height == cam.height;
      }
      if (copy) {
        std::filesystem::copy_file(view.image_path, image, std::filesystem::copy_options::overwrite_existing);
        copied[v] = 1;
      } else {
        const ImageU8 img = undistort_image(read_image_u8(view.image_path, 3), cam, target);
        write_jpeg(image, img.view(), options.jpeg_quality);
      }
      write_texrecon_camera(dir / (names[v] + ".cam"), target, view.pose);
    }
  });
  if (report != nullptr) {
    report->images = scene.views.size();
    report->copied = static_cast<std::size_t>(std::count(copied.begin(), copied.end(), 1));
  }
}

}  // namespace r3d
