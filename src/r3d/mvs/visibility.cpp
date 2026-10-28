#include "r3d/mvs/visibility.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#include <Eigen/LU>

#include "r3d/core/error.hpp"
#include "r3d/core/thread_pool.hpp"

namespace r3d {

namespace {

// Maps pixel (x, y) at depth z of `from` to h = H (x, y, 1) z + t in `to`:
// h.z() is the depth in `to`, h.xy() / h.z() the pixel.
struct Transfer {
  Eigen::Matrix3d h;
  Eigen::Vector3d t;
  Transfer(const PinholeCamera& from, const PinholeCamera& to) {
    const Eigen::Matrix3d k_to = to.intrinsics.K();
    h = k_to * to.pose.rotation * from.pose.rotation.transpose() * from.intrinsics.K().inverse();
    t = k_to * to.pose.rotation * (from.pose.center - to.pose.center);
  }
  [[nodiscard]] Eigen::Vector3d operator()(int x, int y, float z) const {
    return h * Eigen::Vector3d(x, y, 1.0) * static_cast<double>(z) + t;
  }
};

constexpr float kNone = std::numeric_limits<float>::infinity();

// Renders the points of `src` into `dst`'s grid, nearest depth per pixel.
void render_depths(const DepthMap& src, const PinholeCamera& dst, ImageF32& out) {
  for (int y = 0; y < out.height(); ++y) {
    std::fill(out.row(y), out.row(y) + out.width(), kNone);
  }
  const Transfer to_dst(src.camera, dst);
  const int w = out.width();
  const int h = out.height();
  parallel_for(0, static_cast<std::size_t>(src.height()), 16, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      for (int x = 0; x < src.width(); ++x) {
        const float z = src.depth(x, y);
        if (z <= 0.0F) {
          continue;
        }
        const Eigen::Vector3d p = to_dst(x, y, z);
        if (!(p.z() > 0.0)) {
          continue;
        }
        const auto qx = static_cast<int>(std::lround(p.x() / p.z()));
        const auto qy = static_cast<int>(std::lround(p.y() / p.z()));
        if (qx < 0 || qy < 0 || qx >= w || qy >= h) {
          continue;
        }
        // Positive floats order like their bit patterns.
        std::atomic_ref<std::uint32_t> cell(*reinterpret_cast<std::uint32_t*>(&out(qx, qy)));
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(static_cast<float>(p.z()));
        std::uint32_t old = cell.load(std::memory_order_relaxed);
        while (bits < old && !cell.compare_exchange_weak(old, bits, std::memory_order_relaxed)) {
        }
      }
    }
  });
}

}  // namespace

void filter_visibility(DepthMap& m, std::span<const DepthMap* const> neighbours, const VisibilityOptions& options,
                       VisibilityStats* stats) {
  VisibilityStats st;
  st.input = m.num_valid();
  if (m.width() == 0) {
    if (stats != nullptr) {
      stats->input += st.input;
    }
    return;
  }
  const auto front = static_cast<float>(1.0 - options.relative_margin);
  const int w = m.width();
  const int h = m.height();
  ImageU8 conflicts(w, h, 1);
  ImageF32 rendered(w, h, 1);
  std::atomic<std::size_t> free_space{0};
  std::atomic<std::size_t> occluded{0};
  const std::size_t count = std::min(neighbours.size(), static_cast<std::size_t>(std::max(0, options.max_neighbours)));
  for (std::size_t k = 0; k < count; ++k) {
    const DepthMap& n = *neighbours[k];
    if (n.width() == 0) {
      continue;
    }
    render_depths(n, m.camera, rendered);
    const Transfer to_n(m.camera, n.camera);
    parallel_for(0, static_cast<std::size_t>(h), 8, [&](std::size_t begin, std::size_t end) {
      std::size_t local_free = 0;
      std::size_t local_occluded = 0;
      for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
        for (int x = 0; x < w; ++x) {
          const float z = m.depth(x, y);
          if (z <= 0.0F) {
            continue;
          }
          // n sees a surface in front of the depth along this view's ray.
          bool occ = rendered(x, y) < kNone;
          for (int dy = -1; dy <= 1 && occ; ++dy) {
            for (int dx = -1; dx <= 1 && occ; ++dx) {
              const float r = rendered(std::clamp(x + dx, 0, w - 1), std::clamp(y + dy, 0, h - 1));
              occ = r == kNone || r < front * z;
            }
          }
          // The depth lies in n's free space.
          bool free = false;
          if (!occ) {
            const Eigen::Vector3d p = to_n(x, y, z);
            if (p.z() > 0.0) {
              const auto qx = static_cast<int>(std::lround(p.x() / p.z()));
              const auto qy = static_cast<int>(std::lround(p.y() / p.z()));
              if (qx >= 0 && qy >= 0 && qx < n.width() && qy < n.height() && n.depth(qx, qy) > 0.0F) {
                free = true;
                const auto pz = static_cast<float>(p.z());
                for (int dy = -1; dy <= 1 && free; ++dy) {
                  for (int dx = -1; dx <= 1 && free; ++dx) {
                    const float zn =
                        n.depth(std::clamp(qx + dx, 0, n.width() - 1), std::clamp(qy + dy, 0, n.height() - 1));
                    free = zn <= 0.0F || pz < front * zn;
                  }
                }
              }
            }
          }
          if (occ || free) {
            local_occluded += occ ? 1 : 0;
            local_free += free ? 1 : 0;
            conflicts(x, y) = static_cast<std::uint8_t>(conflicts(x, y) + 1);
          }
        }
      }
      free_space += local_free;
      occluded += local_occluded;
    });
  }
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (m.depth(x, y) <= 0.0F) {
        continue;
      }
      const int c = conflicts(x, y);
      const int s = m.support.width() > 0 ? m.support(x, y) : 0;
      if (c >= options.min_conflicts && c > s + options.support_margin) {
        m.depth(x, y) = 0.0F;
        ++st.removed;
      }
    }
  }
  if (stats != nullptr) {
    stats->input += st.input;
    stats->free_space += free_space.load();
    stats->occluded += occluded.load();
    stats->removed += st.removed;
  }
}

void filter_visibility(std::vector<DepthMap>& maps, const std::vector<std::vector<std::size_t>>& neighbours,
                       const VisibilityOptions& options, VisibilityStats* stats) {
  if (neighbours.size() != maps.size()) {
    throw_error("filter_visibility: {} neighbour lists for {} depth maps", neighbours.size(), maps.size());
  }
  VisibilityStats st;
  std::vector<DepthMap> filtered(maps.size());
  std::vector<const DepthMap*> nb;
  for (std::size_t i = 0; i < maps.size(); ++i) {
    nb.clear();
    for (const std::size_t j : neighbours[i]) {
      nb.push_back(&maps[j]);
    }
    filtered[i] = maps[i];
    filter_visibility(filtered[i], nb, options, &st);
  }
  maps = std::move(filtered);
  if (stats != nullptr) {
    *stats = st;
  }
}

}  // namespace r3d
