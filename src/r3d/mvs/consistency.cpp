#include "r3d/mvs/consistency.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <utility>

#include <Eigen/Geometry>
#include <Eigen/LU>

#include "r3d/core/error.hpp"
#include "r3d/core/thread_pool.hpp"

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

void filter_depth_maps(std::vector<DepthMap>& maps, const std::vector<std::vector<std::size_t>>& neighbours,
                       const ConsistencyOptions& options, ConsistencyStats* stats) {
  if (neighbours.size() != maps.size()) {
    throw_error("filter_depth_maps: {} neighbour lists for {} depth maps", neighbours.size(), maps.size());
  }
  ConsistencyStats st;
  for (DepthMap& m : maps) {
    st.input += m.num_valid();
    if (options.min_confidence > 0.0F) {
      for (int y = 0; y < m.height(); ++y) {
        for (int x = 0; x < m.width(); ++x) {
          if (m.depth(x, y) > 0.0F && m.confidence(x, y) < options.min_confidence) {
            m.depth(x, y) = 0.0F;
            ++st.unconfident;
          }
        }
      }
    }
    st.eroded += erode_depth_edges(m, options.edge_erosion, options.discontinuity, options.erode_holes);
  }
  // Support of every pixel against the (eroded, unfiltered) neighbours.
  std::vector<ImageU8> support(maps.size());
  std::vector<ImageF32> averaged(maps.size());
  const auto rel_tol = static_cast<float>(options.max_relative_depth_error);
  const double reproj_tol2 = options.max_reprojection_px * options.max_reprojection_px;
  for (std::size_t i = 0; i < maps.size(); ++i) {
    const DepthMap& m = maps[i];
    support[i] = ImageU8(m.width(), m.height(), 1);
    if (options.average_depths) {
      averaged[i] = ImageF32(m.width(), m.height(), 1);
    }
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
          for (const std::size_t j : neighbours[i]) {
            const DepthMap& n = maps[j];
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
            averaged[i](x, y) = static_cast<float>(depth_sum / (agree + 1));
          }
          support[i](x, y) = static_cast<std::uint8_t>(std::min(agree, 255));
        }
      }
    });
  }
  for (std::size_t i = 0; i < maps.size(); ++i) {
    DepthMap& m = maps[i];
    for (int y = 0; y < m.height(); ++y) {
      for (int x = 0; x < m.width(); ++x) {
        if (m.depth(x, y) <= 0.0F) {
          continue;
        }
        if (support[i](x, y) < options.min_views || support[i](x, y) + m.pairs(x, y) < options.min_evidence) {
          m.depth(x, y) = 0.0F;
          ++st.rejected;
        } else {
          ++st.output;
          if (options.average_depths) {
            m.depth(x, y) = averaged[i](x, y);
          }
        }
      }
    }
    m.support = std::move(support[i]);
  }
  if (stats != nullptr) {
    *stats = st;
  }
}

}  // namespace r3d
