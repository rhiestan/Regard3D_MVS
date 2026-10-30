#include "r3d/mvs/consistency.hpp"

#include <algorithm>
#include <atomic>
#include <span>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/LU>

#include "r3d/core/error.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/stereo/stereo_matcher.hpp"

namespace r3d {

std::vector<std::vector<std::size_t>> consistency_neighbours(const ViewSelection& selection, int max_neighbours) {
  const std::size_t n = selection.partners.size();
  std::vector<std::vector<std::pair<double, std::size_t>>> scored(n);
  for (std::size_t v = 0; v < n; ++v) {
    for (const StereoPartner& p : selection.partners[v]) {
      scored[v].emplace_back(p.score, p.view);
      scored[p.view].emplace_back(p.score, v);
    }
  }
  std::vector<std::vector<std::size_t>> result(n);
  for (std::size_t v = 0; v < n; ++v) {
    auto& s = scored[v];
    // Best score per view, then best first (ties by index for determinism).
    std::sort(s.begin(), s.end(), [](const auto& a, const auto& b) {
      return a.second != b.second ? a.second < b.second : a.first > b.first;
    });
    s.erase(std::unique(s.begin(), s.end(), [](const auto& a, const auto& b) { return a.second == b.second; }),
            s.end());
    std::sort(s.begin(), s.end(), [](const auto& a, const auto& b) {
      return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    for (const auto& [score, view] : s) {
      if (static_cast<int>(result[v].size()) >= max_neighbours) {
        break;
      }
      result[v].push_back(view);
    }
  }
  return result;
}

std::size_t erode_depth_edges(DepthMap& map, int radius, double discontinuity, bool holes) {
  if (radius <= 0) {
    return 0;
  }
  const int w = map.width();
  const int h = map.height();
  const auto jump = static_cast<float>(discontinuity);
  ImageU8 edge(w, h, 1);
  parallel_for(0, static_cast<std::size_t>(h), 16, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      for (int x = 0; x < w; ++x) {
        const float z = map.depth(x, y);
        if (z <= 0.0F) {
          continue;
        }
        auto differs = [&](int nx, int ny) {
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
            return false;
          }
          const float zn = map.depth(nx, ny);
          return zn > 0.0F ? std::abs(zn - z) > jump * std::min(z, zn) : holes;
        };
        if (differs(x - 1, y) || differs(x + 1, y) || differs(x, y - 1) || differs(x, y + 1)) {
          edge(x, y) = 1;
        }
      }
    }
  });
  const int r = radius - 1;
  std::atomic<std::size_t> removed{0};
  ImageF32 out = map.depth;
  parallel_for(0, static_cast<std::size_t>(h), 16, [&](std::size_t begin, std::size_t end) {
    std::size_t local = 0;
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      for (int x = 0; x < w; ++x) {
        if (map.depth(x, y) <= 0.0F) {
          continue;
        }
        bool near_edge = false;
        for (int dy = -r; dy <= r && !near_edge; ++dy) {
          for (int dx = -r; dx <= r && !near_edge; ++dx) {
            const int nx = x + dx;
            const int ny = y + dy;
            near_edge = nx >= 0 && ny >= 0 && nx < w && ny < h && edge(nx, ny) != 0;
          }
        }
        if (near_edge) {
          out(x, y) = 0.0F;
          ++local;
        }
      }
    }
    removed += local;
  });
  map.depth = std::move(out);
  return removed.load();
}

std::size_t remove_depth_speckles(DepthMap& map, int min_size, double max_relative_step) {
  if (min_size <= 1 || map.width() == 0) {
    return 0;
  }
  // Segments of log depth with absolute steps.
  ImageF32 log_depth(map.width(), map.height(), 1);
  for (int y = 0; y < map.height(); ++y) {
    for (int x = 0; x < map.width(); ++x) {
      const float z = map.depth(x, y);
      log_depth(x, y) = z > 0.0F ? std::log(z) : kInvalidDisparity;
    }
  }
  remove_speckles(log_depth, min_size, static_cast<float>(std::log1p(max_relative_step)));
  std::size_t removed = 0;
  for (int y = 0; y < map.height(); ++y) {
    for (int x = 0; x < map.width(); ++x) {
      if (map.depth(x, y) > 0.0F && !std::isfinite(log_depth(x, y))) {
        map.depth(x, y) = 0.0F;
        ++removed;
      }
    }
  }
  return removed;
}

std::size_t remove_weak_depths(DepthMap& map, const ConsistencyOptions& options) {
  const int w = map.width();
  const int h = map.height();
  const bool segments = options.weak_segment > 0.0;
  if (options.weak_texture <= 0.0 || (!segments && !options.reject_weak) || map.color.empty() || w == 0) {
    return 0;
  }
  // Weak pixels: no texture in the 7 x 7 window.
  constexpr int r = 3;
  const double max_var = options.weak_texture * options.weak_texture;
  const auto n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  const auto index = [w](int x, int y) { return static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x); };
  std::vector<std::uint8_t> weak(n, 0);
  parallel_for(0, static_cast<std::size_t>(h), 16, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      for (int x = 0; x < w; ++x) {
        if (map.depth(x, y) <= 0.0F) {
          continue;
        }
        double sum = 0.0;
        double sum2 = 0.0;
        int count = 0;
        for (int v = std::max(y - r, 0); v <= std::min(y + r, h - 1); ++v) {
          for (int u = std::max(x - r, 0); u <= std::min(x + r, w - 1); ++u) {
            const double g = 0.299 * map.color(u, v, 0) + 0.587 * map.color(u, v, 1) + 0.114 * map.color(u, v, 2);
            sum += g;
            sum2 += g * g;
            ++count;
          }
        }
        const double mean = sum / count;
        if (sum2 / count - mean * mean < max_var) {
          weak[index(x, y)] = 1;
        }
      }
    }
  });
  std::size_t removed = 0;
  if (options.reject_weak) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (weak[index(x, y)] != 0) {
          map.depth(x, y) = 0.0F;
          ++removed;
        }
      }
    }
    return removed;
  }
  // Segments of log depth; small ones that are mostly weak are removed.
  const auto max_size = static_cast<std::size_t>(options.weak_segment * static_cast<double>(n));
  const auto max_step = static_cast<float>(std::log1p(options.speckle_step));
  std::vector<float> d(n);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float z = map.depth(x, y);
      d[index(x, y)] = z > 0.0F ? std::log(z) : kInvalidDisparity;
    }
  }
  std::vector<std::uint8_t> visited(n, 0);
  std::vector<std::uint32_t> segment;  // breadth-first queue, finally the whole segment
  for (std::size_t start = 0; start < n; ++start) {
    if (visited[start] != 0 || !std::isfinite(d[start])) {
      continue;
    }
    segment.clear();
    segment.push_back(static_cast<std::uint32_t>(start));
    visited[start] = 1;
    std::size_t weak_count = 0;
    for (std::size_t head = 0; head < segment.size(); ++head) {
      const std::size_t i = segment[head];
      weak_count += weak[i];
      const float v = d[i];
      const auto x = static_cast<int>(i % static_cast<std::size_t>(w));
      auto visit = [&](std::size_t j) {
        if (visited[j] == 0 && std::abs(d[j] - v) <= max_step) {  // false for invalid d[j]
          visited[j] = 1;
          segment.push_back(static_cast<std::uint32_t>(j));
        }
      };
      if (x > 0) {
        visit(i - 1);
      }
      if (x + 1 < w) {
        visit(i + 1);
      }
      if (i >= static_cast<std::size_t>(w)) {
        visit(i - static_cast<std::size_t>(w));
      }
      if (i + static_cast<std::size_t>(w) < n) {
        visit(i + static_cast<std::size_t>(w));
      }
    }
    if (segment.size() < max_size && 2 * weak_count >= segment.size()) {
      for (const std::uint32_t i : segment) {
        map.depth(static_cast<int>(i % static_cast<std::size_t>(w)), static_cast<int>(i / static_cast<std::size_t>(w))) =
            0.0F;
      }
      removed += segment.size();
    }
  }
  return removed;
}

void prepare_depth_map(DepthMap& map, const ConsistencyOptions& options, ConsistencyStats* stats) {
  ConsistencyStats st;
  st.input = map.num_valid();
  if (options.min_confidence > 0.0F) {
    for (int y = 0; y < map.height(); ++y) {
      for (int x = 0; x < map.width(); ++x) {
        if (map.depth(x, y) > 0.0F && map.confidence(x, y) < options.min_confidence) {
          map.depth(x, y) = 0.0F;
          ++st.unconfident;
        }
      }
    }
  }
  st.eroded = erode_depth_edges(map, options.edge_erosion, options.discontinuity, options.erode_holes);
  if (stats != nullptr) {
    stats->input += st.input;
    stats->unconfident += st.unconfident;
    stats->eroded += st.eroded;
  }
}

void check_consistency(DepthMap& m, std::span<const DepthMap* const> neighbours, const ConsistencyOptions& options,
                       ConsistencyStats* stats) {
  ConsistencyStats st;
  ImageU8 support(m.width(), m.height(), 1);
  ImageF32 averaged;
  if (options.average_depths) {
    averaged = ImageF32(m.width(), m.height(), 1);
  }
  const auto rel_tol = static_cast<float>(options.max_relative_depth_error);
  const double reproj_tol2 = options.max_reprojection_px * options.max_reprojection_px;
  const Eigen::Matrix3d k_inv = m.camera.intrinsics.K().inverse();
  const Eigen::Matrix3d rt = m.camera.pose.rotation.transpose();
  parallel_for(0, static_cast<std::size_t>(m.height()), 8, [&](std::size_t begin, std::size_t end) {
    for (auto y = static_cast<int>(begin); y < static_cast<int>(end); ++y) {
      for (int x = 0; x < m.width(); ++x) {
        const float z = m.depth(x, y);
        if (z <= 0.0F) {
          continue;
        }
        const Eigen::Vector3d xw = m.camera.pose.center + rt * (k_inv * Eigen::Vector3d(x, y, 1.0)) * z;
        int agree = 0;
        double depth_sum = z;
        const std::uint16_t mask = m.pair_mask(x, y);
        for (const DepthMap* np : neighbours) {
          const DepthMap& n = *np;
          if (n.width() == 0) {
            continue;
          }
          const std::uint16_t bit_m = m.pair_bit(n.view);
          const Eigen::Vector3d xc = n.camera.pose.world_to_camera(xw);
          if (!(xc.z() > 0.0)) {
            continue;
          }
          const Eigen::Vector2d q = n.camera.intrinsics.project(xc);
          const auto qx = static_cast<int>(std::lround(q.x()));
          const auto qy = static_cast<int>(std::lround(q.y()));
          if (qx < 0 || qy < 0 || qx >= n.width() || qy >= n.height()) {
            continue;
          }
          const float zn = n.depth(qx, qy);
          if (zn <= 0.0F || std::abs(zn - static_cast<float>(xc.z())) > rel_tol * zn) {
            continue;
          }
          const Eigen::Vector3d back = n.point(qx, qy);
          const Eigen::Vector3d bc = m.camera.pose.world_to_camera(back);
          if (!(bc.z() > 0.0)) {
            continue;
          }
          const Eigen::Vector2d p = m.camera.intrinsics.project(bc);
          if ((p - Eigen::Vector2d(x, y)).squaredNorm() > reproj_tol2) {
            continue;
          }
          if (options.independent_support && bit_m != 0 && mask == bit_m) {
            const std::uint16_t bit_n = n.pair_bit(m.view);
            if (bit_n != 0 && n.pair_mask(qx, qy) == bit_n) {
              continue;  // only the same image pair, matched in both directions
            }
          }
          ++agree;
          depth_sum += bc.z();
        }
        if (options.average_depths) {
          averaged(x, y) = static_cast<float>(depth_sum / (agree + 1));
        }
        support(x, y) = static_cast<std::uint8_t>(std::min(agree, 255));
      }
    }
  });
  for (int y = 0; y < m.height(); ++y) {
    for (int x = 0; x < m.width(); ++x) {
      if (m.depth(x, y) <= 0.0F) {
        continue;
      }
      if (support(x, y) < options.min_views || support(x, y) + m.pairs(x, y) < options.min_evidence) {
        m.depth(x, y) = 0.0F;
        ++st.rejected;
      } else if (options.average_depths) {
        m.depth(x, y) = averaged(x, y);
      }
    }
  }
  m.support = std::move(support);
  st.weak = remove_weak_depths(m, options);
  st.speckles = remove_depth_speckles(m, options.speckle_size, options.speckle_step);
  st.output = m.num_valid();
  if (stats != nullptr) {
    stats->rejected += st.rejected;
    stats->weak += st.weak;
    stats->speckles += st.speckles;
    stats->output += st.output;
  }
}

void filter_depth_maps(std::vector<DepthMap>& maps, const std::vector<std::vector<std::size_t>>& neighbours,
                       const ConsistencyOptions& options, ConsistencyStats* stats) {
  if (neighbours.size() != maps.size()) {
    throw_error("filter_depth_maps: {} neighbour lists for {} depth maps", neighbours.size(), maps.size());
  }
  ConsistencyStats st;
  for (DepthMap& m : maps) {
    prepare_depth_map(m, options, &st);
  }
  // All checks use the prepared, unfiltered maps.
  std::vector<DepthMap> filtered(maps.size());
  std::vector<const DepthMap*> nb;
  for (std::size_t i = 0; i < maps.size(); ++i) {
    nb.clear();
    for (const std::size_t j : neighbours[i]) {
      nb.push_back(&maps[j]);
    }
    filtered[i] = maps[i];
    check_consistency(filtered[i], nb, options, &st);
  }
  maps = std::move(filtered);
  if (stats != nullptr) {
    *stats = st;
  }
}

}  // namespace r3d
