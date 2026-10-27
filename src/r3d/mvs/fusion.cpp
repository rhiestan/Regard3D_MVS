#include "r3d/mvs/fusion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

#include "r3d/core/error.hpp"
#include "r3d/core/thread_pool.hpp"

namespace r3d {

namespace {

constexpr int kAxisBits = 21;
constexpr std::int64_t kAxisCells = std::int64_t{1} << kAxisBits;

struct Entry {
  std::uint64_t key;
  std::uint32_t map;
  std::uint32_t pixel;
};

std::array<std::int64_t, 3> voxel_of(const Eigen::Vector3d& p, const Eigen::Vector3d& origin, double inv_voxel) {
  return {static_cast<std::int64_t>(std::floor((p.x() - origin.x()) * inv_voxel)),
          static_cast<std::int64_t>(std::floor((p.y() - origin.y()) * inv_voxel)),
          static_cast<std::int64_t>(std::floor((p.z() - origin.z()) * inv_voxel))};
}

// Sorts in parallel: chunks sorted independently, then merged pairwise.
void parallel_sort(std::vector<Entry>& entries) {
  auto less = [](const Entry& a, const Entry& b) {
    return a.key != b.key ? a.key < b.key : (a.map != b.map ? a.map < b.map : a.pixel < b.pixel);
  };
  const std::size_t n = entries.size();
  const std::size_t chunks = std::max<std::size_t>(1, std::min<std::size_t>(default_thread_pool().num_threads(), n / 65536));
  std::vector<std::size_t> bounds(chunks + 1);
  for (std::size_t c = 0; c <= chunks; ++c) {
    bounds[c] = n * c / chunks;
  }
  parallel_for(0, chunks, 1, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
      std::sort(entries.begin() + static_cast<std::ptrdiff_t>(bounds[c]),
                entries.begin() + static_cast<std::ptrdiff_t>(bounds[c + 1]), less);
    }
  });
  for (std::size_t width = 1; width < chunks; width *= 2) {
    const std::size_t merges = (chunks + 2 * width - 1) / (2 * width);
    parallel_for(0, merges, 1, [&](std::size_t begin, std::size_t end) {
      for (std::size_t m = begin; m < end; ++m) {
        const std::size_t lo = 2 * width * m;
        const std::size_t mid = std::min(lo + width, chunks);
        const std::size_t hi = std::min(lo + 2 * width, chunks);
        if (mid < hi) {
          std::inplace_merge(entries.begin() + static_cast<std::ptrdiff_t>(bounds[lo]),
                             entries.begin() + static_cast<std::ptrdiff_t>(bounds[mid]),
                             entries.begin() + static_cast<std::ptrdiff_t>(bounds[hi]), less);
        }
      }
    });
  }
}

}  // namespace

double median_gsd(const std::vector<DepthMap>& maps) {
  std::vector<float> gsd;
  for (const DepthMap& m : maps) {
    const auto inv_f = static_cast<float>(1.0 / m.camera.intrinsics.fx);
    for (int y = 0; y < m.height(); y += 4) {
      for (int x = 0; x < m.width(); x += 4) {
        const float z = m.depth(x, y);
        if (z > 0.0F) {
          gsd.push_back(z * inv_f);
        }
      }
    }
  }
  if (gsd.empty()) {
    return 0.0;
  }
  const auto mid = gsd.begin() + static_cast<std::ptrdiff_t>(gsd.size() / 2);
  std::nth_element(gsd.begin(), mid, gsd.end());
  return *mid;
}

PointCloud fuse_depth_maps(const std::vector<DepthMap>& maps, const Eigen::Vector3d& origin,
                           const FusionOptions& options, FusionStats* stats) {
  FusionStats st;
  st.median_gsd = median_gsd(maps);
  st.voxel_size = options.voxel_size > 0.0 ? options.voxel_size : options.voxel_factor * st.median_gsd;
  PointCloud cloud;
  cloud.origin = origin;
  if (!(st.voxel_size > 0.0)) {
    if (stats != nullptr) {
      *stats = st;
    }
    return cloud;
  }
  const double inv_voxel = 1.0 / st.voxel_size;

  // Pass 1: voxel bounding box and point count per row.
  std::array<std::int64_t, 3> lo{std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max(),
                                 std::numeric_limits<std::int64_t>::max()};
  std::array<std::int64_t, 3> hi{std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::min(),
                                 std::numeric_limits<std::int64_t>::min()};
  std::vector<std::vector<std::size_t>> row_counts(maps.size());
  std::mutex box_mutex;
  for (std::size_t mi = 0; mi < maps.size(); ++mi) {
    const DepthMap& m = maps[mi];
    row_counts[mi].assign(static_cast<std::size_t>(m.height()), 0);
    parallel_for(0, static_cast<std::size_t>(m.height()), 16, [&](std::size_t begin, std::size_t end) {
      std::array<std::int64_t, 3> l{}, h{};
      l.fill(std::numeric_limits<std::int64_t>::max());
      h.fill(std::numeric_limits<std::int64_t>::min());
      for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
        std::size_t count = 0;
        for (int x = 0; x < m.width(); ++x) {
          if (!m.valid(x, y)) {
            continue;
          }
          const auto v = voxel_of(m.point(x, y), origin, inv_voxel);
          for (int a = 0; a < 3; ++a) {
            l[a] = std::min(l[a], v[a]);
            h[a] = std::max(h[a], v[a]);
          }
          ++count;
        }
        row_counts[mi][static_cast<std::size_t>(y)] = count;
      }
      const std::lock_guard lock(box_mutex);
      for (int a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], l[a]);
        hi[a] = std::max(hi[a], h[a]);
      }
    });
  }
  std::size_t total = 0;
  for (const auto& rc : row_counts) {
    for (const std::size_t c : rc) {
      total += c;
    }
  }
  st.input_points = total;
  if (total == 0) {
    if (stats != nullptr) {
      *stats = st;
    }
    return cloud;
  }
  for (int a = 0; a < 3; ++a) {
    if (hi[a] - lo[a] >= kAxisCells) {
      throw_error("fusion: the scene spans {} voxels along axis {} (at most {}); use a larger voxel size",
                  hi[a] - lo[a] + 1, a, kAxisCells);
    }
  }

  // Pass 2: (voxel key, map, pixel) entries.
  std::vector<Entry> entries(total);
  std::size_t offset = 0;
  for (std::size_t mi = 0; mi < maps.size(); ++mi) {
    const DepthMap& m = maps[mi];
    std::vector<std::size_t> row_start(static_cast<std::size_t>(m.height()));
    for (int y = 0; y < m.height(); ++y) {
      row_start[static_cast<std::size_t>(y)] = offset;
      offset += row_counts[mi][static_cast<std::size_t>(y)];
    }
    parallel_for(0, static_cast<std::size_t>(m.height()), 16, [&](std::size_t begin, std::size_t end) {
      for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
        std::size_t i = row_start[static_cast<std::size_t>(y)];
        for (int x = 0; x < m.width(); ++x) {
          if (!m.valid(x, y)) {
            continue;
          }
          const auto v = voxel_of(m.point(x, y), origin, inv_voxel);
          const auto key = (static_cast<std::uint64_t>(v[0] - lo[0]) << (2 * kAxisBits)) |
                           (static_cast<std::uint64_t>(v[1] - lo[1]) << kAxisBits) |
                           static_cast<std::uint64_t>(v[2] - lo[2]);
          entries[i++] = {key, static_cast<std::uint32_t>(mi),
                          static_cast<std::uint32_t>(y) * static_cast<std::uint32_t>(m.width()) +
                              static_cast<std::uint32_t>(x)};
        }
      }
    });
  }
  parallel_sort(entries);

  // Pass 3: one point per voxel. Chunks start at voxel boundaries.
  const std::size_t chunks = std::max<std::size_t>(1, std::min<std::size_t>(4 * default_thread_pool().num_threads(),
                                                                            total / 16384));
  std::vector<std::size_t> starts;
  for (std::size_t c = 0; c < chunks; ++c) {
    std::size_t s = total * c / chunks;
    while (s > 0 && s < total && entries[s].key == entries[s - 1].key) {
      ++s;
    }
    if (starts.empty() || s > starts.back()) {
      starts.push_back(s);
    }
  }
  starts.push_back(total);
  const std::size_t num_chunks = starts.size() - 1;
  std::vector<std::size_t> chunk_voxels(num_chunks + 1, 0);
  parallel_for(0, num_chunks, 1, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
      std::size_t count = 0;
      for (std::size_t i = starts[c]; i < starts[c + 1]; ++i) {
        count += (i == starts[c] || entries[i].key != entries[i - 1].key) ? 1 : 0;
      }
      chunk_voxels[c + 1] = count;
    }
  });
  for (std::size_t c = 0; c < num_chunks; ++c) {
    chunk_voxels[c + 1] += chunk_voxels[c];
  }
  const std::size_t num_points = chunk_voxels[num_chunks];
  cloud.positions.resize(num_points);
  cloud.normals.resize(num_points);
  cloud.colors.resize(num_points);
  std::vector<float> value(num_points);
  std::vector<float> confidence(num_points);
  parallel_for(0, num_chunks, 1, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
      std::size_t out = chunk_voxels[c];
      std::size_t i = starts[c];
      while (i < starts[c + 1]) {
        std::size_t j = i;
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        Eigen::Vector3d color = Eigen::Vector3d::Zero();
        Eigen::Vector3d toward_camera = Eigen::Vector3d::Zero();
        double weight = 0.0;
        double gsd_sum = 0.0;
        for (; j < starts[c + 1] && entries[j].key == entries[i].key; ++j) {
          const DepthMap& m = maps[entries[j].map];
          const auto x = static_cast<int>(entries[j].pixel % static_cast<std::uint32_t>(m.width()));
          const auto y = static_cast<int>(entries[j].pixel / static_cast<std::uint32_t>(m.width()));
          const Eigen::Vector3d p = m.point(x, y);
          const double gsd = m.depth(x, y) / m.camera.intrinsics.fx;
          const double wt = 1.0 / (gsd * gsd);
          pos += wt * p;
          normal += wt * depth_map_normal(m, x, y, options.normal_jump).cast<double>();
          color += wt * Eigen::Vector3d(m.color(x, y, 0), m.color(x, y, 1), m.color(x, y, 2));
          toward_camera += wt * (m.camera.pose.center - p).normalized();
          gsd_sum += wt * gsd;
          weight += wt;
        }
        pos /= weight;
        color /= weight;
        if (normal.squaredNorm() < 1e-12) {
          normal = toward_camera;
        }
        normal.normalize();
        cloud.positions[out] = (pos - origin).cast<float>();
        cloud.normals[out] = normal.cast<float>();
        cloud.colors[out] = {static_cast<std::uint8_t>(std::lround(std::clamp(color.x(), 0.0, 255.0))),
                             static_cast<std::uint8_t>(std::lround(std::clamp(color.y(), 0.0, 255.0))),
                             static_cast<std::uint8_t>(std::lround(std::clamp(color.z(), 0.0, 255.0)))};
        value[out] = static_cast<float>(gsd_sum / weight);
        confidence[out] = static_cast<float>(j - i);
        ++out;
        i = j;
      }
    }
  });
  cloud.scalars.emplace_back("value", std::move(value));
  cloud.scalars.emplace_back("confidence", std::move(confidence));
  st.output_points = num_points;
  if (stats != nullptr) {
    *stats = st;
  }
  return cloud;
}

}  // namespace r3d
