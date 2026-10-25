#include "r3d/io/scene_export.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <string>
#include <vector>

#include "r3d/core/error.hpp"
#include "r3d/core/sampling.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/io/image_io.hpp"

namespace r3d {
namespace {

std::array<std::uint8_t, 3> camera_color(std::size_t index, std::size_t count) {
  // Hue wheel, full saturation and value.
  const double h = 6.0 * static_cast<double>(index) / static_cast<double>(std::max<std::size_t>(count, 1));
  const double f = h - std::floor(h);
  const auto to_u8 = [](double v) { return static_cast<std::uint8_t>(std::lround(255.0 * v)); };
  switch (static_cast<int>(h) % 6) {
    case 0:
      return {255, to_u8(f), 0};
    case 1:
      return {to_u8(1 - f), 255, 0};
    case 2:
      return {0, 255, to_u8(f)};
    case 3:
      return {0, to_u8(1 - f), 255};
    case 4:
      return {to_u8(f), 0, 255};
    default:
      return {255, 0, to_u8(1 - f)};
  }
}

template <typename T>
void append(std::string& out, T value) {
  char bytes[sizeof(T)];
  std::memcpy(bytes, &value, sizeof(T));
  if constexpr (std::endian::native == std::endian::big) {
    std::reverse(bytes, bytes + sizeof(T));
  }
  out.append(bytes, sizeof(T));
}

}  // namespace

PointCloud sparse_point_cloud(const Scene& scene, bool with_colors) {
  scene.validate();
  PointCloud cloud;
  const std::size_t n = scene.num_points();
  if (n == 0) {
    return cloud;
  }
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : scene.points) {
    centroid += p;
  }
  cloud.origin = (centroid / static_cast<double>(n)).array().round().matrix();
  cloud.positions.reserve(n);
  cloud.visibility.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    cloud.positions.push_back((scene.points[i] - cloud.origin).cast<float>());
    for (const TrackObservation& obs : scene.track(i)) {
      cloud.visibility[i].push_back(scene.views[obs.view].source_id);
    }
  }
  std::vector<float>& track_length = cloud.scalar("track_length");
  for (std::size_t i = 0; i < n; ++i) {
    track_length[i] = static_cast<float>(cloud.visibility[i].size());
  }

  if (with_colors) {
    // Colour of every observation, sampled from its (distorted) image; each
    // view writes only its own observations, so views run in parallel.
    std::vector<std::vector<std::uint32_t>> obs_of_view(scene.views.size());
    for (std::size_t k = 0; k < scene.observations.size(); ++k) {
      obs_of_view[scene.observations[k].view].push_back(static_cast<std::uint32_t>(k));
    }
    std::vector<std::array<float, 3>> obs_color(scene.observations.size());
    parallel_for(0, scene.views.size(), 1, [&](std::size_t begin, std::size_t end) {
      for (std::size_t v = begin; v < end; ++v) {
        const ImageU8 img = read_image_u8(scene.views[v].image_path, 3);
        for (const std::uint32_t k : obs_of_view[v]) {
          const TrackObservation& obs = scene.observations[k];
          sample_bilinear(img.view(), obs.x, obs.y, obs_color[k].data());
        }
      }
    });
    cloud.colors.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      std::array<float, 3> sum{};
      for (std::uint32_t k = scene.track_offsets[i]; k < scene.track_offsets[i + 1]; ++k) {
        for (int c = 0; c < 3; ++c) {
          sum[static_cast<std::size_t>(c)] += obs_color[k][static_cast<std::size_t>(c)];
        }
      }
      const float count = static_cast<float>(scene.track_offsets[i + 1] - scene.track_offsets[i]);
      for (int c = 0; c < 3; ++c) {
        cloud.colors[i][static_cast<std::size_t>(c)] =
            static_cast<std::uint8_t>(std::clamp(std::lround(sum[static_cast<std::size_t>(c)] / count), 0L, 255L));
      }
    }
  }
  cloud.validate();
  return cloud;
}

void write_camera_frustums_ply(const std::filesystem::path& path, const Scene& scene, double size) {
  if (!(size > 0.0)) {
    throw_error("write_camera_frustums_ply: size must be positive");
  }
  const std::size_t num_views = scene.views.size();
  std::string data = std::format(
      "ply\nformat binary_little_endian 1.0\ncomment r3dmvs camera frustums\n"
      "element vertex {}\nproperty double x\nproperty double y\nproperty double z\n"
      "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      "element face {}\nproperty list uchar int vertex_indices\nend_header\n",
      5 * num_views, 6 * num_views);
  for (std::size_t v = 0; v < num_views; ++v) {
    const SceneView& view = scene.views[v];
    const PinholeIntrinsics k = scene.camera_of(v).pinhole();
    const std::array<Eigen::Vector2d, 4> corners = {Eigen::Vector2d(-0.5, -0.5), Eigen::Vector2d(k.width - 0.5, -0.5),
                                                    Eigen::Vector2d(k.width - 0.5, k.height - 0.5),
                                                    Eigen::Vector2d(-0.5, k.height - 0.5)};
    // Image rectangle at `size` along the optical axis, scaled so that its
    // larger side is `size` long whatever the focal length.
    const double depth = size * std::max(k.fx, k.fy) / std::max(k.width, k.height);
    std::array<Eigen::Vector3d, 5> vertices;
    vertices[0] = view.pose.center;
    for (std::size_t c = 0; c < 4; ++c) {
      vertices[c + 1] = view.pose.camera_to_world(k.unproject(corners[c], depth));
    }
    const std::array<std::uint8_t, 3> color = camera_color(v, num_views);
    for (const Eigen::Vector3d& p : vertices) {
      append(data, p.x());
      append(data, p.y());
      append(data, p.z());
      data.append(reinterpret_cast<const char*>(color.data()), 3);
    }
  }
  for (std::size_t v = 0; v < num_views; ++v) {
    const auto base = static_cast<std::int32_t>(5 * v);
    const std::array<std::array<std::int32_t, 3>, 6> faces = {{{base, base + 1, base + 2},
                                                               {base, base + 2, base + 3},
                                                               {base, base + 3, base + 4},
                                                               {base, base + 4, base + 1},
                                                               {base + 1, base + 3, base + 2},
                                                               {base + 1, base + 4, base + 3}}};
    for (const auto& f : faces) {
      data.push_back(3);
      for (const std::int32_t index : f) {
        append(data, index);
      }
    }
  }
  write_file(path, std::vector<std::uint8_t>(data.begin(), data.end()));
}

}  // namespace r3d
