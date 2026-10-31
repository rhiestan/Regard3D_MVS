#include "r3d/pipeline/dense.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <format>
#include <list>
#include <memory>
#include <optional>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "r3d/core/error.hpp"
#include "r3d/core/log.hpp"
#include "r3d/core/memory.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/core/timer.hpp"
#include "r3d/io/image_io.hpp"
#include "r3d/mvs/depth_map_store.hpp"
#include "r3d/rectify/rectification.hpp"
#include "r3d/select/view_selection.hpp"

namespace r3d {

namespace {

// Grey image of a view resampled onto its depth map grid, in 8-bit grey levels.
ImageF32 grey_on_grid(const Scene& scene, std::size_t view, const PinholeCamera& grid) {
  const SceneView& sv = scene.views[view];
  const ImageU16 img = rectify_image(read_image_u16(sv.image_path, 1), scene.camera_of(view), sv.pose, grid);
  ImageF32 out(img.width(), img.height(), 1);
  for (int y = 0; y < img.height(); ++y) {
    for (int x = 0; x < img.width(); ++x) {
      out(x, y) = static_cast<float>(img(x, y)) * (1.0F / 257.0F);
    }
  }
  return out;
}

// A new directory below the system's temporary directory, removed with the object; inactive if !create.
class TempDirectory {
 public:
  explicit TempDirectory(bool create) {
    if (!create) {
      return;
    }
    static std::atomic<unsigned> counter{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() / std::format("r3dmvs-{:x}-{}", stamp, counter.fetch_add(1));
    std::filesystem::create_directories(path_);
  }
  ~TempDirectory() {
    if (!path_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(path_, ec);
    }
  }
  TempDirectory(const TempDirectory&) = delete;
  TempDirectory& operator=(const TempDirectory&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

// Refinement images of the views within a memory budget (least recently used leave first).
class RefinementImageCache {
 public:
  using Loader = std::function<RefinementImage(std::size_t)>;
  RefinementImageCache(std::size_t budget, Loader loader) : budget_(budget), loader_(std::move(loader)) {}

  // The images of `views`, missing ones loaded in parallel.
  std::vector<std::shared_ptr<const RefinementImage>> get(const std::vector<std::size_t>& views) {
    std::vector<std::size_t> missing;
    for (const std::size_t v : views) {
      if (!images_.contains(v) && std::find(missing.begin(), missing.end(), v) == missing.end()) {
        missing.push_back(v);
      }
    }
    const Stopwatch watch;
    std::vector<std::shared_ptr<const RefinementImage>> loaded(missing.size());
    parallel_for(0, missing.size(), 1, [&](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i) {
        loaded[i] = std::make_shared<const RefinementImage>(loader_(missing[i]));
      }
    });
    load_seconds_ += watch.elapsed_seconds();
    for (std::size_t i = 0; i < missing.size(); ++i) {
      images_[missing[i]] = loaded[i];
      bytes_ += bytes_of(*loaded[i]);
    }
    std::vector<std::shared_ptr<const RefinementImage>> result;
    for (const std::size_t v : views) {
      result.push_back(images_[v]);
      lru_.remove(v);
      lru_.push_front(v);
    }
    // Keeps the images of this call.
    while (bytes_ > budget_ && lru_.size() > views.size()) {
      const std::size_t v = lru_.back();
      lru_.pop_back();
      bytes_ -= bytes_of(*images_[v]);
      images_.erase(v);
    }
    return result;
  }
  [[nodiscard]] double load_seconds() const noexcept { return load_seconds_; }

 private:
  static std::size_t bytes_of(const RefinementImage& img) {
    return static_cast<std::size_t>(img.data.width()) * static_cast<std::size_t>(img.data.height()) *
           static_cast<std::size_t>(img.data.channels()) * sizeof(float);
  }
  std::size_t budget_;
  Loader loader_;
  std::unordered_map<std::size_t, std::shared_ptr<const RefinementImage>> images_;
  std::list<std::size_t> lru_;
  std::size_t bytes_ = 0;
  double load_seconds_ = 0.0;
};

// Overall progress from the stages' typical shares of the run time (ETH3D,
// half resolution): matching dominates, refinement next.
class ProgressTracker {
 public:
  ProgressTracker(const DenseProgressCallback& callback, const DenseOptions& options) : callback_(callback) {
    weight_ = {0.62, options.refine ? 0.20 : 0.0, 0.05, options.visibility_filter ? 0.04 : 0.0, 0.09};
    double sum = 0.0;
    for (const double w : weight_) {
      sum += w;
    }
    double start = 0.0;
    for (std::size_t s = 0; s < weight_.size(); ++s) {
      weight_[s] /= sum;
      start_[s] = start;
      start += weight_[s];
    }
  }

  void report(DenseStage stage, std::size_t done, std::size_t total) const {
    if (!callback_) {
      return;
    }
    const auto s = static_cast<std::size_t>(stage);
    const double part = total > 0 ? static_cast<double>(std::min(done, total)) / static_cast<double>(total) : 1.0;
    const double fraction = start_[s] + weight_[s] * part;
    // The end is exactly 1, without the rounding of the sums.
    callback_(DenseProgress{.stage = stage, .done = done, .total = total, .fraction = fraction > 1.0 - 1e-9 ? 1.0 : fraction});
  }

 private:
  const DenseProgressCallback& callback_;
  std::array<double, 5> weight_{};
  std::array<double, 5> start_{};
};

double percent(std::size_t part, std::size_t whole) {
  return 100.0 * static_cast<double>(part) / std::max(1.0, static_cast<double>(whole));
}

}  // namespace

DenseOptions dense_preset(DensePreset preset) {
  DenseOptions o;
  switch (preset) {
    case DensePreset::fast:
      o.depth.max_image_size = 1600;
      o.refine = false;
      break;
    case DensePreset::normal:
      o.depth.max_image_size = 3200;
      break;
    case DensePreset::high:
      o.depth.max_image_size = 6400;
      o.refine = false;
      break;
    case DensePreset::ultra:
      o.depth.max_image_size = 0;
      o.refine = false;
      break;
  }
  return o;
}

std::optional<DensePreset> parse_dense_preset(std::string_view name) {
  for (const DensePreset p : {DensePreset::fast, DensePreset::normal, DensePreset::high, DensePreset::ultra}) {
    if (name == to_string(p)) {
      return p;
    }
  }
  return std::nullopt;
}

std::string_view to_string(DensePreset preset) {
  switch (preset) {
    case DensePreset::fast: return "fast";
    case DensePreset::normal: return "normal";
    case DensePreset::high: return "high";
    case DensePreset::ultra: return "ultra";
  }
  return "unknown";
}

std::string_view to_string(DenseStage stage) {
  switch (stage) {
    case DenseStage::depth_maps: return "depth maps";
    case DenseStage::refinement: return "refinement";
    case DenseStage::consistency: return "consistency";
    case DenseStage::visibility: return "visibility";
    case DenseStage::fusion: return "fusion";
  }
  return "unknown";
}

Eigen::Vector3d dense_cloud_origin(const Scene& scene) {
  Eigen::Vector3d centre = Eigen::Vector3d::Zero();
  for (const SceneView& view : scene.views) {
    centre += view.pose.center / static_cast<double>(scene.views.size());
  }
  const bool georeferenced = centre.cwiseAbs().maxCoeff() > 1e4;
  return georeferenced ? Eigen::Vector3d(centre.array().round()) : Eigen::Vector3d::Zero();
}

void densify(const Scene& scene, const DenseOptions& options, const PointSink& sink, const DenseCallbacks& callbacks,
             DenseReport* report) {
  const CancellationScope cancel_scope(callbacks.cancel);
  const ProgressTracker progress(callbacks.progress, options);
  DenseReport rep;
  const Stopwatch total_watch;
  const std::size_t num_views = scene.views.size();
  rep.views = num_views;
  ViewSelectionOptions select_options;
  select_options.max_partners = options.depth.max_pairs;
  const ViewSelection selection = select_views(scene, select_options);

  const std::filesystem::path& work_dir = options.work_dir;
  if (!work_dir.empty()) {
    std::filesystem::create_directories(work_dir);
  } else if (options.reuse || options.reuse_refined || options.debug_images) {
    throw_error("reusing depth maps and debug images need a work directory");
  }
  // Depth maps that do not fit into memory go to the work directory, or to a temporary one.
  const TempDirectory temp(work_dir.empty());
  const std::filesystem::path spill_dir = work_dir.empty() ? temp.path() : work_dir;
  auto cache_path = [&](std::size_t v, const char* suffix) {
    return work_dir / std::format("depth_{}{}", scene.views[v].source_id, suffix);
  };
  // Two stages hold depth maps at a time, the refinement images take the rest.
  const std::size_t budget = options.memory_budget / 3;
  auto make_store = [&](const char* persistent_suffix, const char* stage) {
    if (!work_dir.empty() && persistent_suffix != nullptr) {
      return std::make_unique<DepthMapStore>(
          num_views, [&, persistent_suffix](std::size_t v) { return cache_path(v, persistent_suffix); }, budget, true);
    }
    return std::make_unique<DepthMapStore>(
        num_views, [&, stage](std::size_t v) { return spill_dir / std::format("{}_{}.r3dd", stage, scene.views[v].source_id); },
        budget, false);
  };

  // Stage 1: depth maps.
  auto raw = make_store(".r3dd", "raw");
  auto refined = make_store("_refined.r3dd", "refined");
  progress.report(DenseStage::depth_maps, 0, num_views);
  for (std::size_t v = 0; v < num_views; ++v) {
    throw_if_cancelled();
    const Stopwatch watch;
    const std::uint32_t id = scene.views[v].source_id;
    if (options.reuse_refined && std::filesystem::exists(cache_path(v, "_refined.r3dd"))) {
      refined->adopt_file(v);
      ++rep.cached_views;
      log_info("view {}: cached, refined", id);
    } else if ((options.reuse || options.reuse_refined) && std::filesystem::exists(cache_path(v, ".r3dd"))) {
      raw->adopt_file(v);
      ++rep.cached_views;
      log_info("view {}: cached", id);
    } else {
      DepthMapStats st;
      DepthMap map = compute_depth_map(scene, selection, v, options.depth, &st);
      rep.pairs_matched += static_cast<std::size_t>(st.pairs_matched);
      rep.pairs_skipped += static_cast<std::size_t>(st.pairs_skipped);
      log_info("view {}: {} pairs, {} skipped, {:.1f}% valid, load {:.2f} s, match {:.2f} s, combine {:.2f} s", id,
               st.pairs_matched, st.pairs_skipped,
               percent(map.num_valid(), static_cast<std::size_t>(map.width()) * static_cast<std::size_t>(map.height())),
               st.load_seconds, st.match_seconds, st.combine_seconds);
      if (options.debug_images) {
        write_png(cache_path(v, ".png"), colorize_depth(map).view(), 2);
      }
      raw->put(v, std::move(map));
      rep.depth_map_seconds += watch.elapsed_seconds();
    }
    progress.report(DenseStage::depth_maps, v + 1, num_views);
  }

  // Stage 2: refinement against the neighbouring images (consistency neighbours, best first).
  Stopwatch watch;
  {
    const auto neighbours = consistency_neighbours(selection, options.refinement.max_views);
    RefinementImageCache images(budget, [&](std::size_t v) {
      const PinholeCamera grid = depth_map_camera(scene, v, depth_map_resolution(scene, v, options.depth));
      return make_refinement_image(grid, grey_on_grid(scene, v, grid), options.refinement.blur_sigma);
    });
    std::size_t count = 0;
    for (std::size_t v = 0; v < num_views; ++v) {
      throw_if_cancelled();
      if (!refined->contains(v)) {
        DepthMap map = *raw->get(v);
        if (options.refine && map.num_valid() > 0) {
          std::vector<std::size_t> wanted{v};
          wanted.insert(wanted.end(), neighbours[v].begin(), neighbours[v].end());
          const auto loaded = images.get(wanted);
          std::vector<const RefinementImage*> views;
          for (std::size_t k = 1; k < loaded.size(); ++k) {
            views.push_back(loaded[k].get());
          }
          RefinementStats st;
          refine_depth_map(map, *loaded[0], views, options.refinement, &st);
          rep.refinement.input += st.input;
          rep.refinement.refined += st.refined;
          rep.refinement.kept += st.kept;
          rep.refinement.unconfirmed += st.unconfirmed;
          ++count;
          if (options.debug_images) {
            write_png(cache_path(v, "_refined.png"), colorize_depth(map).view(), 2);
          }
        }
        refined->put(v, std::move(map));
      }
      if (options.refine) {
        progress.report(DenseStage::refinement, v + 1, num_views);
      }
    }
    rep.refinement_seconds = watch.elapsed_seconds();
    if (count > 0) {
      log_info("refinement: {} depths, {} refined ({:.1f}%), {} unconfirmed, images {:.1f} s, {:.1f} s",
               rep.refinement.input, rep.refinement.refined, percent(rep.refinement.refined, rep.refinement.input),
               rep.refinement.unconfirmed, images.load_seconds(), rep.refinement_seconds);
    }
  }
  raw.reset();

  // Stage 3: geometric consistency, against the prepared (eroded) neighbours.
  watch.restart();
  auto consistent = make_store(nullptr, "consistent");
  {
    auto prepared = make_store(nullptr, "prepared");
    for (std::size_t v = 0; v < num_views; ++v) {
      throw_if_cancelled();
      DepthMap map = *refined->get(v);
      prepare_depth_map(map, options.consistency, &rep.consistency);
      prepared->put(v, std::move(map));
      progress.report(DenseStage::consistency, v + 1, 2 * num_views);
    }
    refined.reset();
    const auto neighbours = consistency_neighbours(selection, options.consistency.max_neighbours);
    for (std::size_t v = 0; v < num_views; ++v) {
      throw_if_cancelled();
      DepthMap map = *prepared->get(v);
      std::vector<std::shared_ptr<const DepthMap>> held;
      std::vector<const DepthMap*> nb;
      for (const std::size_t n : neighbours[v]) {
        held.push_back(prepared->get(n));
        nb.push_back(held.back().get());
      }
      check_consistency(map, nb, options.consistency, &rep.consistency);
      consistent->put(v, std::move(map));
      progress.report(DenseStage::consistency, num_views + v + 1, 2 * num_views);
    }
  }
  rep.consistency_seconds = watch.elapsed_seconds();
  const ConsistencyStats& cst = rep.consistency;
  log_info("consistency: {} depths, {} unconfident, {} at discontinuities, {} unconfirmed, {} weak, {} halo, "
           "{} speckles, {} kept ({:.1f}%), {:.2f} s",
           cst.input, cst.unconfident, cst.eroded, cst.rejected, cst.weak, cst.halo, cst.speckles, cst.output,
           percent(cst.output, cst.input), rep.consistency_seconds);

  // Stage 4: visibility.
  watch.restart();
  FusionOptions fusion_options = options.fusion;
  if (fusion_options.precision_px > 0.0) {
    fusion_options.view_centres.clear();
    for (const SceneView& view : scene.views) {
      fusion_options.view_centres.push_back(view.pose.center);
    }
  }
  std::vector<float> gsd;
  auto final_maps = make_store(nullptr, "final");
  {
    const auto neighbours = consistency_neighbours(selection, options.visibility.max_neighbours);
    for (std::size_t v = 0; v < num_views; ++v) {
      throw_if_cancelled();
      DepthMap map = *consistent->get(v);
      if (options.visibility_filter) {
        std::vector<std::shared_ptr<const DepthMap>> held;
        std::vector<const DepthMap*> nb;
        for (const std::size_t n : neighbours[v]) {
          held.push_back(consistent->get(n));
          nb.push_back(held.back().get());
        }
        filter_visibility(map, nb, options.visibility, &rep.visibility);
        progress.report(DenseStage::visibility, v + 1, num_views);
      }
      sample_gsd(map, gsd, fusion_options);
      if (options.debug_images) {
        write_png(cache_path(v, "_filtered.png"), colorize_depth(map).view(), 2);
        write_depth_map(cache_path(v, "_filtered.r3dd"), map);
      }
      final_maps->put(v, std::move(map));
    }
  }
  consistent.reset();
  rep.visibility_seconds = watch.elapsed_seconds();
  if (options.visibility_filter) {
    const VisibilityStats& vst = rep.visibility;
    log_info("visibility: {} depths, conflicts {} free space / {} occluded, {} removed ({:.1f}%), {:.2f} s", vst.input,
             vst.free_space, vst.occluded, vst.removed, percent(vst.removed, vst.input), rep.visibility_seconds);
  }

  // Stage 5: fusion, streamed to the sink.
  watch.restart();
  rep.origin = dense_cloud_origin(scene);
  fuse_depth_maps(
      num_views,
      [&](std::size_t v) {
        throw_if_cancelled();
        return final_maps->get(v);
      },
      std::move(gsd), rep.origin, fusion_options,
      [&](PointCloud&& part) {
        for (std::vector<std::uint32_t>& views : part.visibility) {
          for (std::uint32_t& v : views) {
            v = scene.views[v].source_id;
          }
        }
        sink(std::move(part));
      },
      &rep.fusion,
      [&](std::size_t done, std::size_t total) { progress.report(DenseStage::fusion, done, total); });
  final_maps.reset();
  rep.fusion_seconds = watch.elapsed_seconds();
  const FusionStats& fst = rep.fusion;
  log_info("fusion: {} points, {} coarser, {} unsupported -> {} (leaf {:.4g} x 2^0..{}, median GSD {:.4g}, "
           "{} chunks, {} passes), {:.2f} s",
           fst.input_points, fst.coarser_points, fst.unsupported_points, fst.output_points, fst.voxel_size,
           fst.levels - 1, fst.median_gsd, fst.chunks, fst.passes, rep.fusion_seconds);
  progress.report(DenseStage::fusion, 1, 1);

  rep.total_seconds = total_watch.elapsed_seconds();
  log_info("depth maps {:.1f} s, refinement {:.1f} s, consistency {:.1f} s, visibility {:.1f} s, fusion {:.1f} s, "
           "total {:.1f} s, peak memory {:.0f} MB",
           rep.depth_map_seconds, rep.refinement_seconds, rep.consistency_seconds, rep.visibility_seconds,
           rep.fusion_seconds, rep.total_seconds, static_cast<double>(peak_memory_bytes()) / (1 << 20));
  if (report != nullptr) {
    *report = rep;
  }
}

void densify_to_ply(const Scene& scene, const DenseOptions& options, const std::filesystem::path& path,
                    const CloudLayoutOptions& layout, const DenseCallbacks& callbacks, DenseReport* report) {
  PlyWriteOptions ply;
  ply.double_positions = !dense_cloud_origin(scene).isZero();
  ply.comments.push_back("r3dmvs dense cloud");
  std::optional<CloudWriter> writer(std::in_place, path, layout, ply);
  try {
    densify(scene, options, [&](PointCloud&& part) { writer->append(part); }, callbacks, report);
    writer->close();
  } catch (...) {
    writer.reset();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    throw;
  }
}

}  // namespace r3d
