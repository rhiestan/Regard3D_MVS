#include "r3d/mvs/fusion.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "r3d/core/thread_pool.hpp"

namespace r3d {

namespace {

constexpr int kChunkBits = 21;                       // chunk coordinates per axis (signed, offset)
constexpr std::int64_t kChunkOffset = std::int64_t{1} << (kChunkBits - 1);
constexpr int kMaxLevels = 12;

// Spreads the low 21 bits of v to every third bit.
std::uint64_t spread3(std::uint64_t v) {
  v &= 0x1fffffULL;
  v = (v | v << 32) & 0x1f00000000ffffULL;
  v = (v | v << 16) & 0x1f0000ff0000ffULL;
  v = (v | v << 8) & 0x100f00f00f00f00fULL;
  v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
  v = (v | v << 2) & 0x1249249249249249ULL;
  return v;
}

std::uint64_t morton(std::uint64_t x, std::uint64_t y, std::uint64_t z) {
  return spread3(x) | (spread3(y) << 1) | (spread3(z) << 2);
}

// Inverse of spread3.
std::uint64_t compact3(std::uint64_t v) {
  v &= 0x1249249249249249ULL;
  v = (v | v >> 2) & 0x10c30c30c30c30c3ULL;
  v = (v | v >> 4) & 0x100f00f00f00f00fULL;
  v = (v | v >> 8) & 0x1f0000ff0000ffULL;
  v = (v | v >> 16) & 0x1f00000000ffffULL;
  v = (v | v >> 32) & 0x1fffffULL;
  return v;
}

std::array<std::int64_t, 3> demorton(std::uint64_t key) {
  return {static_cast<std::int64_t>(compact3(key)), static_cast<std::int64_t>(compact3(key >> 1)),
          static_cast<std::int64_t>(compact3(key >> 2))};
}

// One depth pixel, 40 bytes.
struct Entry {
  std::uint64_t key;                  // Morton code of the finest cell within the chunk
  std::array<float, 3> position;      // relative to the fusion origin
  float gsd;
  std::uint32_t view;                 // index into the depth maps
  std::array<std::int16_t, 3> normal; // unit normal x 32767, zero if unknown
  std::array<std::uint8_t, 3> color;
  std::uint8_t level;
};
static_assert(sizeof(Entry) == 40);

bool entry_less(const Entry& a, const Entry& b) {
  if (a.key != b.key) {
    return a.key < b.key;
  }
  if (a.view != b.view) {
    return a.view < b.view;
  }
  for (int i = 0; i < 3; ++i) {
    const auto ba = std::bit_cast<std::uint32_t>(a.position[i]);
    const auto bb = std::bit_cast<std::uint32_t>(b.position[i]);
    if (ba != bb) {
      return ba < bb;
    }
  }
  return false;
}

// Grid of the finest cells, its chunks and the level of each point.
struct Grid {
  Eigen::Vector3d origin;
  double cell = 0.0;          // finest leaf size
  double inv_cell = 0.0;
  double factor = 1.0;
  int levels = 1;
  int chunk_bits = 6;         // finest cells per chunk axis: 2^chunk_bits

  // Chunk id (packed, offset coordinates) and Morton key within the chunk; false outside the grid.
  [[nodiscard]] bool locate(const Eigen::Vector3d& p, std::uint64_t& chunk, std::uint64_t& key) const {
    std::array<std::int64_t, 3> c{};
    for (int a = 0; a < 3; ++a) {
      const double f = std::floor((p[a] - origin[a]) * inv_cell);
      if (!(std::abs(f) < 1e15)) {
        return false;
      }
      c[a] = static_cast<std::int64_t>(f);
    }
    chunk = 0;
    std::array<std::uint64_t, 3> local{};
    for (int a = 0; a < 3; ++a) {
      const std::int64_t ch = (c[a] >> chunk_bits) + kChunkOffset;
      if (ch < 0 || ch >= (std::int64_t{1} << kChunkBits)) {
        return false;
      }
      chunk = (chunk << kChunkBits) | static_cast<std::uint64_t>(ch);
      local[a] = static_cast<std::uint64_t>(c[a] & ((std::int64_t{1} << chunk_bits) - 1));
    }
    key = morton(local[0], local[1], local[2]);
    return true;
  }
  [[nodiscard]] int level(double gsd) const {
    if (levels <= 1) {
      return 0;
    }
    const auto l = static_cast<int>(std::lround(std::log2(factor * gsd * inv_cell)));
    return std::clamp(l, 0, levels - 1);
  }
};

// Spatially coherent order of chunks: Morton code of the chunk coordinates.
std::uint64_t chunk_order(std::uint64_t chunk) {
  const std::uint64_t mask = (std::uint64_t{1} << kChunkBits) - 1;
  return morton((chunk >> (2 * kChunkBits)) & mask, (chunk >> kChunkBits) & mask, chunk & mask);
}

// Visits the rows of every map in parallel: body(map, map index, first row, end row).
template <typename Body>
void for_each_point(std::size_t count, const DepthMapSource& source, Body&& body) {
  for (std::size_t mi = 0; mi < count; ++mi) {
    const std::shared_ptr<const DepthMap> m = source(mi);
    parallel_for(0, static_cast<std::size_t>(m->height()), 16, [&](std::size_t begin, std::size_t end) {
      body(*m, mi, static_cast<int>(begin), static_cast<int>(end));
    });
  }
}

float percentile(std::vector<float>& v, double q) {
  const auto k = static_cast<std::ptrdiff_t>(std::clamp(q, 0.0, 1.0) * static_cast<double>(v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[static_cast<std::size_t>(k)];
}

struct ChunkInfo {
  std::uint64_t id = 0;
  std::size_t count = 0;
  int batch = 0;
  std::size_t offset = 0;     // within the batch buffer
};

struct Output {
  std::vector<Eigen::Vector3f> positions;
  std::vector<Eigen::Vector3f> normals;
  std::vector<std::array<std::uint8_t, 3>> colors;
  std::vector<float> value;
  std::vector<float> confidence;
  std::size_t coarser = 0;
  std::size_t unsupported = 0;
};

// Fuses the entries of one chunk.
void fuse_chunk(std::span<Entry> entries, const Grid& grid, const FusionOptions& options,
                const std::vector<Eigen::Vector3d>& centres, const Eigen::Vector3d& origin, Output& out) {
  std::sort(entries.begin(), entries.end(), entry_less);
  const std::size_t n = entries.size();
  std::vector<std::uint8_t> keep(n, 1);
  // Best resolution wins: the second-finest GSD of each cell at level L.
  if (options.resolution_ratio > 0.0) {
    const auto ratio = static_cast<float>(options.resolution_ratio);
    for (int level = 1; level < grid.levels; ++level) {
      const int shift = 3 * level;
      std::size_t i = 0;
      while (i < n) {
        const std::uint64_t cell = entries[i].key >> shift;
        std::size_t j = i;
        float best = std::numeric_limits<float>::infinity();
        float second = best;
        for (; j < n && (entries[j].key >> shift) == cell; ++j) {
          const float g = entries[j].gsd;
          if (g < best) {
            second = best;
            best = g;
          } else if (g < second) {
            second = g;
          }
        }
        for (std::size_t k = i; k < j; ++k) {
          if (entries[k].level == level && keep[k] != 0 && entries[k].gsd > ratio * second) {
            keep[k] = 0;
            ++out.coarser;
          }
        }
        i = j;
      }
    }
  }
  // Leaves: the kept points of each level, grouped by their cell at that level.
  struct Leaf {
    std::uint64_t cell = 0;
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    Eigen::Vector3d color = Eigen::Vector3d::Zero();
    Eigen::Vector3d toward_camera = Eigen::Vector3d::Zero();
    double weight = 0.0;
    double gsd_sum = 0.0;
    std::size_t count = 0;
    std::size_t views_begin = 0;  // its distinct views in `views`
    std::size_t views_end = 0;
  };
  std::vector<std::uint32_t> members;
  std::vector<Leaf> leaves;
  std::vector<std::uint32_t> views;
  std::vector<std::uint32_t> pool;
  for (int level = 0; level < grid.levels; ++level) {
    members.clear();
    for (std::size_t k = 0; k < n; ++k) {
      if (keep[k] != 0 && entries[k].level == level) {
        members.push_back(static_cast<std::uint32_t>(k));
      }
    }
    const int shift = 3 * level;
    leaves.clear();
    views.clear();
    for (std::size_t i = 0; i < members.size();) {
      Leaf leaf;
      leaf.cell = entries[members[i]].key >> shift;
      leaf.views_begin = views.size();
      std::size_t j = i;
      for (; j < members.size() && (entries[members[j]].key >> shift) == leaf.cell; ++j) {
        const Entry& e = entries[members[j]];
        const Eigen::Vector3d p(e.position[0], e.position[1], e.position[2]);
        const double wt = 1.0 / (static_cast<double>(e.gsd) * e.gsd);
        leaf.pos += wt * p;
        leaf.normal += wt * Eigen::Vector3d(e.normal[0], e.normal[1], e.normal[2]);
        leaf.color += wt * Eigen::Vector3d(e.color[0], e.color[1], e.color[2]);
        leaf.toward_camera += wt * (centres[e.view] - origin - p).normalized();
        leaf.gsd_sum += wt * e.gsd;
        leaf.weight += wt;
        views.push_back(e.view);
      }
      leaf.count = j - i;
      i = j;
      const auto first = views.begin() + static_cast<std::ptrdiff_t>(leaf.views_begin);
      std::sort(first, views.end());
      views.erase(std::unique(first, views.end()), views.end());
      leaf.views_end = views.size();
      leaves.push_back(leaf);
    }
    // Fold validation, optionally with the 26 neighbouring leaves of this level within the chunk.
    const std::int64_t cells_per_axis = std::int64_t{1} << (grid.chunk_bits - level);
    auto find_leaf = [&](std::uint64_t cell) -> const Leaf* {
      const auto it = std::lower_bound(leaves.begin(), leaves.end(), cell,
                                       [](const Leaf& l, std::uint64_t c) { return l.cell < c; });
      return it != leaves.end() && it->cell == cell ? &*it : nullptr;
    };
    const auto needed = static_cast<std::size_t>(std::max(options.min_sources, 1));
    for (const Leaf& leaf : leaves) {
      if (leaf.views_end - leaf.views_begin < needed) {
        bool supported = false;
        if (options.fold_neighbourhood) {
          pool.assign(views.begin() + static_cast<std::ptrdiff_t>(leaf.views_begin),
                      views.begin() + static_cast<std::ptrdiff_t>(leaf.views_end));
          const std::array<std::int64_t, 3> c = demorton(leaf.cell);
          for (int d = 0; d < 27 && !supported; ++d) {
            const std::int64_t x = c[0] + d % 3 - 1;
            const std::int64_t y = c[1] + d / 3 % 3 - 1;
            const std::int64_t z = c[2] + d / 9 - 1;
            if (d == 13 || x < 0 || y < 0 || z < 0 || x >= cells_per_axis || y >= cells_per_axis ||
                z >= cells_per_axis) {
              continue;
            }
            const Leaf* nb = find_leaf(
                morton(static_cast<std::uint64_t>(x), static_cast<std::uint64_t>(y), static_cast<std::uint64_t>(z)));
            if (nb == nullptr) {
              continue;
            }
            for (std::size_t k = nb->views_begin; k < nb->views_end; ++k) {
              if (std::find(pool.begin(), pool.end(), views[k]) == pool.end()) {
                pool.push_back(views[k]);
              }
            }
            supported = pool.size() >= needed;
          }
        }
        if (!supported) {
          ++out.unsupported;
          continue;
        }
      }
      const Eigen::Vector3d pos = leaf.pos / leaf.weight;
      const Eigen::Vector3d color = leaf.color / leaf.weight;
      Eigen::Vector3d normal = leaf.normal.squaredNorm() < 1e-12 ? leaf.toward_camera : leaf.normal;
      normal.normalize();
      out.positions.push_back(pos.cast<float>());
      out.normals.push_back(normal.cast<float>());
      out.colors.push_back({static_cast<std::uint8_t>(std::lround(std::clamp(color.x(), 0.0, 255.0))),
                            static_cast<std::uint8_t>(std::lround(std::clamp(color.y(), 0.0, 255.0))),
                            static_cast<std::uint8_t>(std::lround(std::clamp(color.z(), 0.0, 255.0)))});
      out.value.push_back(static_cast<float>(leaf.gsd_sum / leaf.weight));
      out.confidence.push_back(static_cast<float>(leaf.count));
    }
  }
}

}  // namespace

void sample_gsd(const DepthMap& m, std::vector<float>& out) {
  const auto inv_f = static_cast<float>(1.0 / m.camera.intrinsics.fx);
  for (int y = 0; y < m.height(); y += 4) {
    for (int x = 0; x < m.width(); x += 4) {
      const float z = m.depth(x, y);
      if (z > 0.0F) {
        out.push_back(z * inv_f);
      }
    }
  }
}

double median_gsd(const std::vector<DepthMap>& maps) {
  std::vector<float> gsd;
  for (const DepthMap& m : maps) {
    sample_gsd(m, gsd);
  }
  return gsd.empty() ? 0.0 : percentile(gsd, 0.5);
}

PointCloud fuse_depth_maps(const std::vector<DepthMap>& maps, const Eigen::Vector3d& origin,
                           const FusionOptions& options, FusionStats* stats) {
  std::vector<float> gsd;
  for (const DepthMap& m : maps) {
    sample_gsd(m, gsd);
  }
  return fuse_depth_maps(
      maps.size(),
      [&](std::size_t i) { return std::shared_ptr<const DepthMap>(std::shared_ptr<const DepthMap>{}, &maps[i]); },
      std::move(gsd), origin, options, stats);
}

PointCloud fuse_depth_maps(std::size_t count, const DepthMapSource& source, std::vector<float> gsd,
                           const Eigen::Vector3d& origin, const FusionOptions& options, FusionStats* stats) {
  FusionStats st;
  PointCloud cloud;
  cloud.origin = origin;
  auto finish = [&]() {
    if (stats != nullptr) {
      *stats = st;
    }
    return cloud;
  };
  if (gsd.empty()) {
    return finish();
  }
  st.median_gsd = percentile(gsd, 0.5);

  // Leaf sizes: the finest from the 1st percentile of the GSD, levels up to the 99.9th.
  Grid grid;
  grid.origin = origin;
  grid.factor = options.voxel_factor;
  if (options.voxel_size > 0.0) {
    grid.cell = options.voxel_size;
    grid.levels = 1;
  } else {
    grid.cell = options.voxel_factor * percentile(gsd, 0.01);
    const double coarsest = options.voxel_factor * percentile(gsd, 0.999);
    grid.levels = std::clamp(static_cast<int>(std::lround(std::log2(coarsest / grid.cell))) + 1, 1, kMaxLevels);
  }
  if (!(grid.cell > 0.0)) {
    return finish();
  }
  grid.inv_cell = 1.0 / grid.cell;
  grid.chunk_bits = std::min(grid.levels - 1 + 6, 21);
  st.voxel_size = grid.cell;
  st.levels = grid.levels;

  // Pass 1: points per chunk.
  std::unordered_map<std::uint64_t, std::size_t> counts;
  std::mutex counts_mutex;
  std::atomic<std::size_t> outside{0};
  for_each_point(count, source, [&](const DepthMap& m, std::size_t, int y0, int y1) {
    std::unordered_map<std::uint64_t, std::size_t> local;
    std::size_t local_outside = 0;
    for (int y = y0; y < y1; ++y) {
      for (int x = 0; x < m.width(); ++x) {
        if (!m.valid(x, y)) {
          continue;
        }
        std::uint64_t chunk = 0;
        std::uint64_t key = 0;
        if (grid.locate(m.point(x, y), chunk, key)) {
          ++local[chunk];
        } else {
          ++local_outside;
        }
      }
    }
    outside += local_outside;
    const std::lock_guard lock(counts_mutex);
    for (const auto& [chunk, c] : local) {
      counts[chunk] += c;
    }
  });
  st.outside_points = outside.load();

  // Chunks in spatial order, grouped into batches within the memory budget.
  std::vector<ChunkInfo> chunks;
  chunks.reserve(counts.size());
  for (const auto& [chunk, c] : counts) {
    chunks.push_back({chunk, c, 0, 0});
    st.input_points += c;
  }
  st.input_points += st.outside_points;
  std::sort(chunks.begin(), chunks.end(),
            [](const ChunkInfo& a, const ChunkInfo& b) { return chunk_order(a.id) < chunk_order(b.id); });
  const std::size_t budget = std::max<std::size_t>(options.memory_budget / sizeof(Entry), 1);
  std::vector<std::size_t> batch_size;
  for (ChunkInfo& c : chunks) {
    if (batch_size.empty() || (batch_size.back() > 0 && batch_size.back() + c.count > budget)) {
      batch_size.push_back(0);
    }
    c.batch = static_cast<int>(batch_size.size()) - 1;
    c.offset = batch_size.back();
    batch_size.back() += c.count;
  }
  std::unordered_map<std::uint64_t, std::uint32_t> chunk_index;
  chunk_index.reserve(chunks.size());
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    chunk_index.emplace(chunks[i].id, static_cast<std::uint32_t>(i));
  }
  st.chunks = chunks.size();
  st.passes = static_cast<int>(batch_size.size());

  std::vector<Eigen::Vector3d> centres(count);
  std::vector<float> value;
  std::vector<float> confidence;
  std::vector<Entry> buffer;
  std::size_t first_chunk = 0;
  for (std::size_t b = 0; b < batch_size.size(); ++b) {
    std::size_t end_chunk = first_chunk;
    while (end_chunk < chunks.size() && chunks[end_chunk].batch == static_cast<int>(b)) {
      ++end_chunk;
    }
    // Pass 2: the entries of this batch's chunks.
    buffer.resize(batch_size[b]);
    std::vector<std::atomic<std::size_t>> cursor(end_chunk - first_chunk);
    for (std::size_t c = first_chunk; c < end_chunk; ++c) {
      cursor[c - first_chunk].store(chunks[c].offset, std::memory_order_relaxed);
    }
    for_each_point(count, source, [&](const DepthMap& m, std::size_t mi, int y0, int y1) {
      if (y0 == 0) {
        centres[mi] = m.camera.pose.center;
      }
      const double inv_f = 1.0 / m.camera.intrinsics.fx;
      for (int y = y0; y < y1; ++y) {
        for (int x = 0; x < m.width(); ++x) {
          if (!m.valid(x, y)) {
            continue;
          }
          const Eigen::Vector3d p = m.point(x, y);
          std::uint64_t chunk = 0;
          std::uint64_t key = 0;
          if (!grid.locate(p, chunk, key)) {
            continue;
          }
          const std::uint32_t ci = chunk_index.find(chunk)->second;
          if (ci < first_chunk || ci >= end_chunk) {
            continue;
          }
          Entry& e = buffer[cursor[ci - first_chunk].fetch_add(1, std::memory_order_relaxed)];
          const Eigen::Vector3f rel = (p - origin).cast<float>();
          const Eigen::Vector3f nrm = depth_map_normal(m, x, y, options.normal_jump);
          e.key = key;
          e.position = {rel.x(), rel.y(), rel.z()};
          e.gsd = static_cast<float>(m.depth(x, y) * inv_f);
          e.view = static_cast<std::uint32_t>(mi);
          for (int a = 0; a < 3; ++a) {
            e.normal[a] = static_cast<std::int16_t>(std::lround(std::clamp(nrm[a], -1.0F, 1.0F) * 32767.0F));
            e.color[a] = m.color(x, y, a);
          }
          e.level = static_cast<std::uint8_t>(grid.level(e.gsd));
        }
      }
    });
    // Fusion per chunk, appended in chunk order.
    std::vector<Output> outputs(end_chunk - first_chunk);
    parallel_for(first_chunk, end_chunk, 1, [&](std::size_t begin, std::size_t end) {
      for (std::size_t c = begin; c < end; ++c) {
        fuse_chunk(std::span<Entry>(buffer.data() + chunks[c].offset, chunks[c].count), grid, options, centres, origin,
                   outputs[c - first_chunk]);
      }
    });
    for (Output& o : outputs) {
      cloud.positions.insert(cloud.positions.end(), o.positions.begin(), o.positions.end());
      cloud.normals.insert(cloud.normals.end(), o.normals.begin(), o.normals.end());
      cloud.colors.insert(cloud.colors.end(), o.colors.begin(), o.colors.end());
      value.insert(value.end(), o.value.begin(), o.value.end());
      confidence.insert(confidence.end(), o.confidence.begin(), o.confidence.end());
      st.coarser_points += o.coarser;
      st.unsupported_points += o.unsupported;
      o = Output{};
    }
    first_chunk = end_chunk;
  }
  buffer = {};
  cloud.scalars.emplace_back("value", std::move(value));
  cloud.scalars.emplace_back("confidence", std::move(confidence));
  st.output_points = cloud.size();
  return finish();
}

}  // namespace r3d
