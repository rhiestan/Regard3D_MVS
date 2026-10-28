#include "dense_command.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <format>
#include <functional>
#include <list>
#include <memory>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "r3d/core/error.hpp"
#include "r3d/core/memory.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/core/timer.hpp"
#include "r3d/io/image_io.hpp"
#include "r3d/io/ply.hpp"
#include "r3d/mvs/depth_map_store.hpp"
#include "r3d/rectify/rectification.hpp"
#include "r3d/select/view_selection.hpp"
#include "stereo_commands.hpp"

namespace r3dmvs {

namespace {

std::filesystem::path utf8_path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

// Grey image of a view resampled onto its depth map grid, in 8-bit grey levels.
r3d::ImageF32 grey_on_grid(const r3d::Scene& scene, std::size_t view, const r3d::PinholeCamera& grid) {
  const r3d::SceneView& sv = scene.views[view];
  const r3d::ImageU16 img =
      r3d::rectify_image(r3d::read_image_u16(sv.image_path, 1), scene.camera_of(view), sv.pose, grid);
  r3d::ImageF32 out(img.width(), img.height(), 1);
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
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() / std::format("r3dmvs-{:x}", stamp);
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
  using Loader = std::function<r3d::RefinementImage(std::size_t)>;
  RefinementImageCache(std::size_t budget, Loader loader) : budget_(budget), loader_(std::move(loader)) {}

  // The images of `views`, missing ones loaded in parallel.
  std::vector<std::shared_ptr<const r3d::RefinementImage>> get(const std::vector<std::size_t>& views) {
    std::vector<std::size_t> missing;
    for (const std::size_t v : views) {
      if (!images_.contains(v) && std::find(missing.begin(), missing.end(), v) == missing.end()) {
        missing.push_back(v);
      }
    }
    const r3d::Stopwatch watch;
    std::vector<std::shared_ptr<const r3d::RefinementImage>> loaded(missing.size());
    r3d::parallel_for(0, missing.size(), 1, [&](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i) {
        loaded[i] = std::make_shared<const r3d::RefinementImage>(loader_(missing[i]));
      }
    });
    load_seconds_ += watch.elapsed_seconds();
    for (std::size_t i = 0; i < missing.size(); ++i) {
      images_[missing[i]] = loaded[i];
      bytes_ += bytes_of(*loaded[i]);
    }
    std::vector<std::shared_ptr<const r3d::RefinementImage>> result;
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
  static std::size_t bytes_of(const r3d::RefinementImage& img) {
    return static_cast<std::size_t>(img.data.width()) * static_cast<std::size_t>(img.data.height()) *
           static_cast<std::size_t>(img.data.channels()) * sizeof(float);
  }
  std::size_t budget_;
  Loader loader_;
  std::unordered_map<std::size_t, std::shared_ptr<const r3d::RefinementImage>> images_;
  std::list<std::size_t> lru_;
  std::size_t bytes_ = 0;
  double load_seconds_ = 0.0;
};

}  // namespace

void add_dense_options(CLI::App& command, DenseArgs& args) {
  command.add_option("-o,--output", args.output, "Output PLY file")->required();
  command.add_option("--work-dir", args.work_dir, "Directory for cached depth maps (and debug images)");
  command.add_flag("--reuse", args.reuse, "Load depth maps cached in --work-dir instead of matching again");
  command.add_flag("--reuse-refined", args.reuse_refined,
                   "Load the refined depth maps cached in --work-dir (skips matching and refinement)");
  command.add_flag("--debug-images", args.debug_images, "Write depth images before and after filtering to --work-dir");
  command.add_option("--resolution", args.depth.resolution, "Depth map resolution relative to the images")
      ->check(CLI::Range(0.05, 4.0))
      ->capture_default_str();
  command.add_option("--max-pairs", args.depth.max_pairs, "Stereo partners per view")
      ->check(CLI::Range(1, 16))
      ->capture_default_str();
  command.add_option("--inlier-px", args.depth.multi_stereo.inlier_px, "Disparity distance for pairs to agree")
      ->capture_default_str();
  command.add_option("--max-reproj", args.depth.multi_stereo.max_reprojection_px,
                     "Maximum disparity residual of a used pair")
      ->capture_default_str();
  command.add_option("--min-angle", args.depth.multi_stereo.min_angle_deg, "Minimum triangulation angle (degrees)")
      ->capture_default_str();
  command.add_option("--min-pairs", args.depth.multi_stereo.min_pairs, "Agreeing pairs per pixel")
      ->capture_default_str();
  command.add_flag("--refine,!--no-refine", args.refine, "Refine depths by patch matching in the images (default)");
  command.add_option("--refine-radius", args.refinement.radius, "Refinement window half size (pixels)")
      ->check(CLI::Range(1, 7))
      ->capture_default_str();
  command.add_option("--refine-step", args.refinement.step, "Spacing of the refinement window samples")
      ->check(CLI::Range(1, 7))
      ->capture_default_str();
  command.add_option("--refine-iterations", args.refinement.iterations, "Gauss-Newton steps per pixel")
      ->capture_default_str();
  command.add_option("--refine-views", args.refinement.max_views, "Neighbouring images used for refinement")
      ->check(CLI::Range(1, 8))
      ->capture_default_str();
  command.add_option("--refine-min-ncc", args.refinement.min_ncc, "NCC a view needs to take part")
      ->capture_default_str();
  command.add_option("--refine-blur", args.refinement.blur_sigma, "Gaussian pre-filter of the refinement images (pixels)")
      ->capture_default_str();
  command.add_option("--refine-min-gain", args.refinement.min_gain, "Mean NCC gain a refined depth needs")
      ->capture_default_str();
  command.add_option("--refine-max-shift", args.refinement.max_shift_px,
                     "Largest change of a refined depth (pixels in the used views)")
      ->capture_default_str();
  command.add_option("--min-views", args.consistency.min_views, "Agreeing neighbour views per pixel")
      ->capture_default_str();
  command.add_option("--min-evidence", args.consistency.min_evidence, "Agreeing pairs + agreeing views per pixel (2: two pairs or one other view)")
      ->capture_default_str();
  command.add_option("--min-confidence", args.consistency.min_confidence, "Minimum matcher confidence")
      ->capture_default_str();
  command.add_option("--depth-tol", args.consistency.max_relative_depth_error,
                     "Relative depth tolerance of the consistency check")
      ->capture_default_str();
  command.add_option("--consistency-reproj", args.consistency.max_reprojection_px,
                     "Forward-backward reprojection tolerance (pixels)")
      ->capture_default_str();
  command.add_flag("--independent-support,!--any-support", args.consistency.independent_support,
                   "Ignore agreement that rests only on the same image pair matched in both directions (default: any agreement counts)");
  command.add_flag("--average,!--no-average", args.consistency.average_depths,
                   "Average each depth with the agreeing neighbours' depths (default)");
  command.add_option("--neighbours", args.consistency.max_neighbours, "Views checked per view")->capture_default_str();
  command.add_option("--edge-erosion", args.consistency.edge_erosion, "Pixels removed at depth discontinuities")
      ->capture_default_str();
  command.add_flag("--erode-holes,!--no-erode-holes", args.consistency.erode_holes,
                   "Erode at the border of regions without depth as well");
  command.add_option("--depth-speckle-size", args.consistency.speckle_size,
                     "Smallest connected depth segment kept after the consistency check (pixels, 0: off)")
      ->capture_default_str();
  command.add_option("--depth-speckle-step", args.consistency.speckle_step, "Relative depth step within a segment")
      ->capture_default_str();
  command.add_option("--discontinuity", args.consistency.discontinuity, "Relative depth jump of a discontinuity")
      ->capture_default_str();
  command.add_flag("--visibility,!--no-visibility", args.visibility_filter,
                   "Remove depths that conflict with the surfaces neighbouring views see (default)");
  command.add_option("--visibility-margin", args.visibility.relative_margin,
                     "Relative depth by which a conflicting surface lies in front")
      ->capture_default_str();
  command.add_option("--visibility-conflicts", args.visibility.min_conflicts, "Conflicting views needed to remove a depth")
      ->capture_default_str();
  command.add_option("--visibility-margin-views", args.visibility.support_margin,
                     "Removed when conflicting views exceed agreeing views by more than this")
      ->capture_default_str();
  command.add_option("--visibility-neighbours", args.visibility.max_neighbours, "Views checked per view")
      ->capture_default_str();
  command.add_option("--memory", args.memory_mb,
                     "Memory for depth maps and refinement images (MB); the rest waits in the work directory")
      ->check(CLI::Range(std::size_t{64}, std::size_t{1} << 30))
      ->capture_default_str();
  command.add_option("--voxel", args.fusion.voxel_size, "Fixed fusion leaf size (0: adaptive, --voxel-factor x GSD)")
      ->capture_default_str();
  command.add_option("--voxel-factor", args.fusion.voxel_factor, "Leaf size relative to the GSD of a point (1, 2, 4: density)")
      ->capture_default_str();
  command.add_option("--resolution-ratio", args.fusion.resolution_ratio,
                     "Drop points whose leaf holds points this much finer (0: keep all)")
      ->capture_default_str();
  command.add_option("--min-sources", args.fusion.min_sources, "Views per fused point")->capture_default_str();
  command.add_flag("--fold-neighbourhood,!--fold-leaf", args.fusion.fold_neighbourhood,
                   "Count the views of the neighbouring leaves for --min-sources (default: the leaf only)");
  command.add_option("--fusion-memory", args.fusion_memory_mb, "Memory for the fusion's point data per pass (MB)")
      ->capture_default_str();
  add_matcher_options(command, args.depth.stereo);
}

void run_dense(const DenseArgs& args, const r3d::Scene& scene) {
  const r3d::Stopwatch total_watch;
  r3d::ViewSelectionOptions select_options;
  select_options.max_partners = args.depth.max_pairs;
  const r3d::ViewSelection selection = r3d::select_views(scene, select_options);
  const std::size_t num_views = scene.views.size();

  const std::filesystem::path work_dir = utf8_path(args.work_dir);
  if (!work_dir.empty()) {
    std::filesystem::create_directories(work_dir);
  } else if (args.reuse || args.reuse_refined || args.debug_images) {
    r3d::throw_error("--reuse, --reuse-refined and --debug-images need --work-dir");
  }
  // Depth maps that do not fit into memory go to the work directory, or to a temporary one.
  const TempDirectory temp(work_dir.empty());
  const std::filesystem::path spill_dir = work_dir.empty() ? temp.path() : work_dir;
  auto cache_path = [&](std::size_t v, const char* suffix) {
    return work_dir / std::format("depth_{}{}", scene.views[v].source_id, suffix);
  };
  // Two stages hold depth maps at a time, the refinement images take the rest.
  const std::size_t budget = (args.memory_mb << 20) / 3;
  auto make_store = [&](const char* persistent_suffix, const char* stage) {
    if (!work_dir.empty() && persistent_suffix != nullptr) {
      return std::make_unique<r3d::DepthMapStore>(
          num_views, [&, persistent_suffix](std::size_t v) { return cache_path(v, persistent_suffix); }, budget, true);
    }
    return std::make_unique<r3d::DepthMapStore>(
        num_views,
        [&, stage](std::size_t v) { return spill_dir / std::format("{}_{}.r3dd", stage, scene.views[v].source_id); },
        budget, false);
  };

  // Stage 1: depth maps.
  std::printf("%5s %6s %6s %9s %7s %7s %7s\n", "view", "pairs", "skip", "valid", "load s", "match s", "comb s");
  auto raw = make_store(".r3dd", "raw");
  auto refined = make_store("_refined.r3dd", "refined");
  double depth_seconds = 0.0;
  for (std::size_t v = 0; v < num_views; ++v) {
    const r3d::Stopwatch watch;
    if (args.reuse_refined && std::filesystem::exists(cache_path(v, "_refined.r3dd"))) {
      refined->adopt_file(v);
      std::printf("%5u  cached, refined\n", scene.views[v].source_id);
      continue;
    }
    if ((args.reuse || args.reuse_refined) && std::filesystem::exists(cache_path(v, ".r3dd"))) {
      raw->adopt_file(v);
      std::printf("%5u  cached\n", scene.views[v].source_id);
      continue;
    }
    r3d::DepthMapStats st;
    r3d::DepthMap map = r3d::compute_depth_map(scene, selection, v, args.depth, &st);
    std::printf("%5u %6d %6d %8.1f%% %7.2f %7.2f %7.2f\n", scene.views[v].source_id, st.pairs_matched,
                st.pairs_skipped,
                100.0 * static_cast<double>(map.num_valid()) /
                    std::max(1.0, static_cast<double>(map.width()) * map.height()),
                st.load_seconds, st.match_seconds, st.combine_seconds);
    std::fflush(stdout);
    if (args.debug_images) {
      r3d::write_png(cache_path(v, ".png"), r3d::colorize_depth(map).view(), 2);
    }
    raw->put(v, std::move(map));
    depth_seconds += watch.elapsed_seconds();
  }

  // Stage 2: refinement against the neighbouring images (consistency neighbours, best first).
  r3d::Stopwatch watch;
  double refine_seconds = 0.0;
  {
    const auto neighbours = r3d::consistency_neighbours(selection, args.refinement.max_views);
    RefinementImageCache images(budget, [&](std::size_t v) {
      const r3d::PinholeCamera grid = r3d::depth_map_camera(scene, v, args.depth.resolution);
      return r3d::make_refinement_image(grid, grey_on_grid(scene, v, grid), args.refinement.blur_sigma);
    });
    r3d::RefinementStats total;
    std::size_t count = 0;
    for (std::size_t v = 0; v < num_views; ++v) {
      if (refined->contains(v)) {
        continue;
      }
      r3d::DepthMap map = *raw->get(v);
      if (args.refine && map.num_valid() > 0) {
        std::vector<std::size_t> wanted{v};
        wanted.insert(wanted.end(), neighbours[v].begin(), neighbours[v].end());
        const auto loaded = images.get(wanted);
        std::vector<const r3d::RefinementImage*> views;
        for (std::size_t k = 1; k < loaded.size(); ++k) {
          views.push_back(loaded[k].get());
        }
        r3d::RefinementStats st;
        r3d::refine_depth_map(map, *loaded[0], views, args.refinement, &st);
        total.input += st.input;
        total.refined += st.refined;
        total.kept += st.kept;
        ++count;
        if (args.debug_images) {
          r3d::write_png(cache_path(v, "_refined.png"), r3d::colorize_depth(map).view(), 2);
        }
      }
      refined->put(v, std::move(map));
    }
    refine_seconds = watch.elapsed_seconds();
    if (count > 0) {
      std::printf("refinement: %zu depths, %zu refined (%.1f%%), images %.1f s, %.1f s\n", total.input, total.refined,
                  100.0 * static_cast<double>(total.refined) / std::max<double>(1.0, static_cast<double>(total.input)),
                  images.load_seconds(), refine_seconds);
      std::fflush(stdout);
    }
  }
  raw.reset();

  // Stage 3: geometric consistency, against the prepared (eroded) neighbours.
  watch.restart();
  r3d::ConsistencyStats cst;
  auto consistent = make_store(nullptr, "consistent");
  {
    auto prepared = make_store(nullptr, "prepared");
    for (std::size_t v = 0; v < num_views; ++v) {
      r3d::DepthMap map = *refined->get(v);
      r3d::prepare_depth_map(map, args.consistency, &cst);
      prepared->put(v, std::move(map));
    }
    refined.reset();
    const auto neighbours = r3d::consistency_neighbours(selection, args.consistency.max_neighbours);
    for (std::size_t v = 0; v < num_views; ++v) {
      r3d::DepthMap map = *prepared->get(v);
      std::vector<std::shared_ptr<const r3d::DepthMap>> held;
      std::vector<const r3d::DepthMap*> nb;
      for (const std::size_t n : neighbours[v]) {
        held.push_back(prepared->get(n));
        nb.push_back(held.back().get());
      }
      r3d::check_consistency(map, nb, args.consistency, &cst);
      consistent->put(v, std::move(map));
    }
  }
  const double filter_seconds = watch.elapsed_seconds();
  std::printf("consistency: %zu depths, %zu unconfident, %zu at discontinuities, %zu unconfirmed, %zu speckles, "
              "%zu kept (%.1f%%), %.2f s\n",
              cst.input, cst.unconfident, cst.eroded, cst.rejected, cst.speckles, cst.output,
              100.0 * static_cast<double>(cst.output) / std::max<double>(1.0, static_cast<double>(cst.input)),
              filter_seconds);

  // Stage 4: visibility.
  watch.restart();
  std::vector<float> gsd;
  auto final_maps = make_store(nullptr, "final");
  r3d::VisibilityStats vst;
  {
    const auto neighbours = r3d::consistency_neighbours(selection, args.visibility.max_neighbours);
    for (std::size_t v = 0; v < num_views; ++v) {
      r3d::DepthMap map = *consistent->get(v);
      if (args.visibility_filter) {
        std::vector<std::shared_ptr<const r3d::DepthMap>> held;
        std::vector<const r3d::DepthMap*> nb;
        for (const std::size_t n : neighbours[v]) {
          held.push_back(consistent->get(n));
          nb.push_back(held.back().get());
        }
        r3d::filter_visibility(map, nb, args.visibility, &vst);
      }
      r3d::sample_gsd(map, gsd);
      if (args.debug_images) {
        r3d::write_png(cache_path(v, "_filtered.png"), r3d::colorize_depth(map).view(), 2);
        r3d::write_depth_map(cache_path(v, "_filtered.r3dd"), map);
      }
      final_maps->put(v, std::move(map));
    }
  }
  consistent.reset();
  const double visibility_seconds = watch.elapsed_seconds();
  if (args.visibility_filter) {
    std::printf("visibility: %zu depths, conflicts %zu free space / %zu occluded, %zu removed (%.1f%%), %.2f s\n",
                vst.input, vst.free_space, vst.occluded, vst.removed,
                100.0 * static_cast<double>(vst.removed) / std::max<double>(1.0, static_cast<double>(vst.input)),
                visibility_seconds);
  }

  // Stage 5: fusion. Positions are written as world coordinates; a large
  // offset (georeferenced scenes) is kept in double precision.
  watch.restart();
  Eigen::Vector3d centre = Eigen::Vector3d::Zero();
  for (const r3d::SceneView& view : scene.views) {
    centre += view.pose.center / static_cast<double>(num_views);
  }
  const bool georeferenced = centre.cwiseAbs().maxCoeff() > 1e4;
  const Eigen::Vector3d origin = georeferenced ? Eigen::Vector3d(centre.array().round()) : Eigen::Vector3d::Zero();
  r3d::FusionStats fst;
  r3d::FusionOptions fusion = args.fusion;
  fusion.memory_budget = args.fusion_memory_mb << 20;
  const r3d::PointCloud cloud = r3d::fuse_depth_maps(
      num_views, [&](std::size_t v) { return final_maps->get(v); }, std::move(gsd), origin, fusion, &fst);
  final_maps.reset();
  const double fuse_seconds = watch.elapsed_seconds();
  std::printf("fusion: %zu points, %zu coarser, %zu unsupported -> %zu (leaf %.4g x 2^0..%d, median GSD %.4g, "
              "%zu chunks, %d passes), %.2f s\n",
              fst.input_points, fst.coarser_points, fst.unsupported_points, fst.output_points, fst.voxel_size,
              fst.levels - 1, fst.median_gsd, fst.chunks, fst.passes, fuse_seconds);

  r3d::PlyWriteOptions ply;
  ply.double_positions = georeferenced;
  ply.comments.push_back("r3dmvs dense cloud");
  r3d::write_ply(utf8_path(args.output), cloud, ply);
  std::printf("depth maps %.1f s, refinement %.1f s, consistency %.1f s, visibility %.1f s, fusion %.1f s, "
              "total %.1f s, peak memory %.0f MB\n",
              depth_seconds, refine_seconds, filter_seconds, visibility_seconds, fuse_seconds,
              total_watch.elapsed_seconds(), static_cast<double>(r3d::peak_memory_bytes()) / (1 << 20));
}

}  // namespace r3dmvs
