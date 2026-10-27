#include "r3d/mvs/depth_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/LU>

#include "r3d/core/error.hpp"
#include "r3d/stereo/disparity_eval.hpp"

namespace r3d {

void DepthMap::allocate(int w, int h) {
  depth = ImageF32(w, h, 1);
  confidence = ImageF32(w, h, 1);
  pair_mask = Image<std::uint16_t>(w, h, 1);
  support = ImageU8(w, h, 1);
  color = ImageU8(w, h, 3);
}

std::size_t DepthMap::num_valid() const {
  std::size_t n = 0;
  for (int y = 0; y < height(); ++y) {
    const float* row = depth.row(y);
    for (int x = 0; x < width(); ++x) {
      n += row[x] > 0.0F ? 1 : 0;
    }
  }
  return n;
}

PinholeIntrinsics scale_intrinsics(const PinholeIntrinsics& k, double scale) {
  PinholeIntrinsics s;
  s.width = std::max(1, static_cast<int>(std::lround(k.width * scale)));
  s.height = std::max(1, static_cast<int>(std::lround(k.height * scale)));
  s.fx = k.fx * scale;
  s.fy = k.fy * scale;
  s.cx = (k.cx + 0.5) * scale - 0.5;
  s.cy = (k.cy + 0.5) * scale - 0.5;
  return s;
}

Eigen::Vector3f depth_map_normal(const DepthMap& map, int x, int y, float max_relative_jump) {
  const float z = map.depth(x, y);
  if (z <= 0.0F) {
    return Eigen::Vector3f::Zero();
  }
  // Neighbour usable: valid and on the same surface.
  auto usable = [&](int nx, int ny) {
    if (nx < 0 || ny < 0 || nx >= map.width() || ny >= map.height()) {
      return false;
    }
    const float zn = map.depth(nx, ny);
    return zn > 0.0F && std::abs(zn - z) <= max_relative_jump * z;
  };
  const Eigen::Vector3d c = map.point(x, y);
  // Tangent along one axis: central difference, else one-sided.
  auto tangent = [&](int dx, int dy, Eigen::Vector3d& t) {
    const bool plus = usable(x + dx, y + dy);
    const bool minus = usable(x - dx, y - dy);
    if (plus && minus) {
      t = map.point(x + dx, y + dy) - map.point(x - dx, y - dy);
    } else if (plus) {
      t = map.point(x + dx, y + dy) - c;
    } else if (minus) {
      t = c - map.point(x - dx, y - dy);
    } else {
      return false;
    }
    return true;
  };
  Eigen::Vector3d tx, ty;
  if (!tangent(1, 0, tx) || !tangent(0, 1, ty)) {
    return Eigen::Vector3f::Zero();
  }
  Eigen::Vector3d n = ty.cross(tx);
  const double len = n.norm();
  if (!(len > 0.0)) {
    return Eigen::Vector3f::Zero();
  }
  n /= len;
  if (n.dot(map.camera.pose.center - c) < 0.0) {
    n = -n;
  }
  return n.cast<float>();
}

namespace {

constexpr std::array<char, 4> kMagic{'R', '3', 'D', 'D'};
constexpr std::uint32_t kVersion = 2;

template <typename T>
void write_pod(std::ofstream& f, const T& v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
void read_pod(std::ifstream& f, T& v) {
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
}

template <typename T>
void write_image(std::ofstream& f, const Image<T>& img) {
  const std::size_t row_bytes = sizeof(T) * static_cast<std::size_t>(img.width()) * static_cast<std::size_t>(img.channels());
  for (int y = 0; y < img.height(); ++y) {
    f.write(reinterpret_cast<const char*>(img.row(y)), static_cast<std::streamsize>(row_bytes));
  }
}

template <typename T>
void read_image(std::ifstream& f, Image<T>& img) {
  const std::size_t row_bytes = sizeof(T) * static_cast<std::size_t>(img.width()) * static_cast<std::size_t>(img.channels());
  for (int y = 0; y < img.height(); ++y) {
    f.read(reinterpret_cast<char*>(img.row(y)), static_cast<std::streamsize>(row_bytes));
  }
}

}  // namespace

void write_depth_map(const std::filesystem::path& path, const DepthMap& map) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    throw_error("cannot write depth map {}", path.string());
  }
  f.write(kMagic.data(), kMagic.size());
  write_pod(f, kVersion);
  write_pod(f, map.view);
  const PinholeIntrinsics& k = map.camera.intrinsics;
  write_pod(f, static_cast<std::int32_t>(k.width));
  write_pod(f, static_cast<std::int32_t>(k.height));
  for (const double v : {k.fx, k.fy, k.cx, k.cy}) {
    write_pod(f, v);
  }
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      write_pod(f, map.camera.pose.rotation(r, c));
    }
  }
  for (int i = 0; i < 3; ++i) {
    write_pod(f, map.camera.pose.center(i));
  }
  write_pod(f, static_cast<std::uint32_t>(map.partners.size()));
  for (const std::uint32_t p : map.partners) {
    write_pod(f, p);
  }
  write_pod(f, static_cast<std::int32_t>(map.width()));
  write_pod(f, static_cast<std::int32_t>(map.height()));
  write_image(f, map.depth);
  write_image(f, map.confidence);
  write_image(f, map.pair_mask);
  write_image(f, map.support);
  write_image(f, map.color);
  if (!f) {
    throw_error("error writing depth map {}", path.string());
  }
}

DepthMap read_depth_map(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw_error("cannot open depth map {}", path.string());
  }
  std::array<char, 4> magic{};
  f.read(magic.data(), magic.size());
  std::uint32_t version = 0;
  read_pod(f, version);
  if (magic != kMagic || version != kVersion) {
    throw_error("{} is not an r3d depth map (version {})", path.string(), kVersion);
  }
  DepthMap map;
  read_pod(f, map.view);
  PinholeIntrinsics& k = map.camera.intrinsics;
  std::int32_t w = 0, h = 0;
  read_pod(f, w);
  read_pod(f, h);
  k.width = w;
  k.height = h;
  read_pod(f, k.fx);
  read_pod(f, k.fy);
  read_pod(f, k.cx);
  read_pod(f, k.cy);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      read_pod(f, map.camera.pose.rotation(r, c));
    }
  }
  for (int i = 0; i < 3; ++i) {
    read_pod(f, map.camera.pose.center(i));
  }
  std::uint32_t num_partners = 0;
  read_pod(f, num_partners);
  if (!f || num_partners > 16) {
    throw_error("corrupt depth map {}", path.string());
  }
  map.partners.resize(num_partners);
  for (std::uint32_t& p : map.partners) {
    read_pod(f, p);
  }
  read_pod(f, w);
  read_pod(f, h);
  if (!f || w <= 0 || h <= 0 || w > (1 << 16) || h > (1 << 16)) {
    throw_error("corrupt depth map {}", path.string());
  }
  map.allocate(w, h);
  read_image(f, map.depth);
  read_image(f, map.confidence);
  read_image(f, map.pair_mask);
  read_image(f, map.support);
  read_image(f, map.color);
  if (!f) {
    throw_error("truncated depth map {}", path.string());
  }
  return map;
}

ImageU8 colorize_depth(const DepthMap& map) {
  // Inverse depth, so that the colour scale follows disparity (near: warm).
  ImageF32 inv(map.width(), map.height(), 1);
  std::vector<float> values;
  for (int y = 0; y < map.height(); ++y) {
    for (int x = 0; x < map.width(); ++x) {
      const float z = map.depth(x, y);
      inv(x, y) = z > 0.0F ? 1.0F / z : std::numeric_limits<float>::infinity();
      if (z > 0.0F) {
        values.push_back(1.0F / z);
      }
    }
  }
  if (values.empty()) {
    return colorize_disparity(inv, 0.0F, 1.0F);
  }
  auto percentile = [&](double p) {
    const auto k = static_cast<std::ptrdiff_t>(p * static_cast<double>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + k, values.end());
    return values[static_cast<std::size_t>(k)];
  };
  const float lo = percentile(0.02);
  const float hi = percentile(0.98);
  return colorize_disparity(inv, lo, hi);
}

}  // namespace r3d
