#include "dense_command.hpp"

#include <cstdio>
#include <filesystem>
#include <format>
#include <vector>

#include "r3d/core/error.hpp"
#include "r3d/core/timer.hpp"
#include "r3d/io/image_io.hpp"
#include "r3d/io/ply.hpp"
#include "r3d/select/view_selection.hpp"
#include "stereo_commands.hpp"

namespace r3dmvs {

namespace {

std::filesystem::path utf8_path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

}  // namespace

void add_dense_options(CLI::App& command, DenseArgs& args) {
  command.add_option("-o,--output", args.output, "Output PLY file")->required();
  command.add_option("--work-dir", args.work_dir, "Directory for cached depth maps (and debug images)");
  command.add_flag("--reuse", args.reuse, "Load depth maps cached in --work-dir instead of matching again");
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
  command.add_option("--discontinuity", args.consistency.discontinuity, "Relative depth jump of a discontinuity")
      ->capture_default_str();
  command.add_option("--voxel", args.fusion.voxel_size, "Fusion voxel size (0: --voxel-factor x median GSD)")
      ->capture_default_str();
  command.add_option("--voxel-factor", args.fusion.voxel_factor, "Voxel size relative to the median GSD")
      ->capture_default_str();
  add_matcher_options(command, args.depth.stereo);
}

void run_dense(const DenseArgs& args, const r3d::Scene& scene) {
  const r3d::Stopwatch total_watch;
  r3d::ViewSelectionOptions select_options;
  select_options.max_partners = args.depth.max_pairs;
  const r3d::ViewSelection selection = r3d::select_views(scene, select_options);

  const std::filesystem::path work_dir = utf8_path(args.work_dir);
  if (!work_dir.empty()) {
    std::filesystem::create_directories(work_dir);
  } else if (args.reuse || args.debug_images) {
    r3d::throw_error("--reuse and --debug-images need --work-dir");
  }
  auto cache_path = [&](std::size_t v, const char* suffix) {
    return work_dir / std::format("depth_{}{}", scene.views[v].source_id, suffix);
  };

  // Stage 1: depth maps.
  std::printf("%5s %6s %6s %9s %7s %7s %7s\n", "view", "pairs", "skip", "valid", "load s", "match s", "comb s");
  std::vector<r3d::DepthMap> maps(scene.views.size());
  double depth_seconds = 0.0;
  for (std::size_t v = 0; v < scene.views.size(); ++v) {
    const r3d::Stopwatch watch;
    if (args.reuse && std::filesystem::exists(cache_path(v, ".r3dd"))) {
      maps[v] = r3d::read_depth_map(cache_path(v, ".r3dd"));
      std::printf("%5u  cached %8.1f%%\n", scene.views[v].source_id,
                  100.0 * static_cast<double>(maps[v].num_valid()) /
                      std::max(1.0, static_cast<double>(maps[v].width()) * maps[v].height()));
      continue;
    }
    r3d::DepthMapStats st;
    maps[v] = r3d::compute_depth_map(scene, selection, v, args.depth, &st);
    std::printf("%5u %6d %6d %8.1f%% %7.2f %7.2f %7.2f\n", scene.views[v].source_id, st.pairs_matched,
                st.pairs_skipped,
                100.0 * static_cast<double>(maps[v].num_valid()) /
                    std::max(1.0, static_cast<double>(maps[v].width()) * maps[v].height()),
                st.load_seconds, st.match_seconds, st.combine_seconds);
    std::fflush(stdout);
    if (!work_dir.empty()) {
      r3d::write_depth_map(cache_path(v, ".r3dd"), maps[v]);
      if (args.debug_images) {
        r3d::write_png(cache_path(v, ".png"), r3d::colorize_depth(maps[v]).view(), 2);
      }
    }
    depth_seconds += watch.elapsed_seconds();
  }

  // Stage 2: geometric consistency.
  r3d::Stopwatch watch;
  r3d::ConsistencyStats cst;
  r3d::filter_depth_maps(maps, r3d::consistency_neighbours(selection, args.consistency.max_neighbours),
                         args.consistency, &cst);
  const double filter_seconds = watch.elapsed_seconds();
  std::printf("consistency: %zu depths, %zu unconfident, %zu at discontinuities, %zu unconfirmed, %zu kept (%.1f%%), "
              "%.2f s\n",
              cst.input, cst.unconfident, cst.eroded, cst.rejected, cst.output,
              100.0 * static_cast<double>(cst.output) / std::max<double>(1.0, static_cast<double>(cst.input)),
              filter_seconds);
  if (args.debug_images) {
    for (std::size_t v = 0; v < maps.size(); ++v) {
      r3d::write_png(cache_path(v, "_filtered.png"), r3d::colorize_depth(maps[v]).view(), 2);
    }
  }

  // Stage 3: fusion. Positions are written as world coordinates; a large
  // offset (georeferenced scenes) is kept in double precision.
  watch.restart();
  Eigen::Vector3d centre = Eigen::Vector3d::Zero();
  for (const r3d::SceneView& view : scene.views) {
    centre += view.pose.center / static_cast<double>(scene.views.size());
  }
  const bool georeferenced = centre.cwiseAbs().maxCoeff() > 1e4;
  const Eigen::Vector3d origin = georeferenced ? Eigen::Vector3d(centre.array().round()) : Eigen::Vector3d::Zero();
  r3d::FusionStats fst;
  const r3d::PointCloud cloud = r3d::fuse_depth_maps(maps, origin, args.fusion, &fst);
  const double fuse_seconds = watch.elapsed_seconds();
  std::printf("fusion: %zu points -> %zu (voxel %.4g, median GSD %.4g), %.2f s\n", fst.input_points,
              fst.output_points, fst.voxel_size, fst.median_gsd, fuse_seconds);

  r3d::PlyWriteOptions ply;
  ply.double_positions = georeferenced;
  ply.comments.push_back("r3dmvs dense cloud");
  r3d::write_ply(utf8_path(args.output), cloud, ply);
  std::printf("depth maps %.1f s, consistency %.1f s, fusion %.1f s, total %.1f s\n", depth_seconds, filter_seconds,
              fuse_seconds, total_watch.elapsed_seconds());
}

}  // namespace r3dmvs
