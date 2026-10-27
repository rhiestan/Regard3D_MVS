// r3dmvs: command line front end of the R3DMVS library.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <format>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <CLI/CLI.hpp>

#include "r3d/core/log.hpp"
#include "r3d/core/simd.hpp"
#include "r3d/core/thread_pool.hpp"
#include "r3d/core/timer.hpp"
#include "r3d/io/colmap_scene.hpp"
#include "r3d/io/image_io.hpp"
#include "r3d/io/openmvg_scene.hpp"
#include "r3d/io/ply.hpp"
#include "r3d/io/pfm.hpp"
#include "r3d/io/scene_export.hpp"
#include "r3d/rectify/rectification.hpp"
#include "r3d/scene/scene.hpp"
#include "r3d/scene/undistort.hpp"
#include "r3d/select/view_selection.hpp"
#include "r3d/stereo/disparity_eval.hpp"
#include "r3d/stereo/pair_cloud.hpp"
#include "r3d/stereo/stereo_matcher.hpp"
#include "r3d/version.hpp"
#include "dense_command.hpp"
#include "stereo_commands.hpp"

namespace {

std::string compiler_description() {
#if defined(__clang__)
  return std::format("Clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(_MSC_VER)
  return std::format("MSVC {}", _MSC_FULL_VER);
#elif defined(__GNUC__)
  return std::format("GCC {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
  return "unknown compiler";
#endif
}

void print_info() {
  std::string targets;
  for (const std::int64_t target : r3d::simd_available_targets()) {
    targets += (targets.empty() ? "" : ", ") + r3d::simd_target_name(target);
  }
  std::printf("R3DMVS %s\n", r3d::kVersionString);
  std::printf("  compiler:        %s\n", compiler_description().c_str());
  std::printf("  hardware threads: %u\n", std::thread::hardware_concurrency());
  std::printf("  worker threads:  %u\n", r3d::default_thread_pool().num_threads());
  std::printf("  SIMD active:     %s\n", r3d::simd_active_target_name().c_str());
  std::printf("  SIMD available:  %s\n", targets.c_str());
}

struct SceneArgs {
  std::string input;
  std::string image_dir;
  bool pairs = false;
  int max_partners = 4;
  std::string sparse_ply;
  bool colors = false;
  std::string cameras_ply;
  std::string undistort_dir;
};

std::filesystem::path utf8_path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

void print_pairs(const r3d::Scene& scene, const r3d::ViewSelection& selection) {
  std::printf("view selection (partner: score, shared points, mean angle, GSD ratio, coverage, depth range):\n");
  for (std::size_t v = 0; v < scene.views.size(); ++v) {
    const r3d::DepthRange& d = selection.view_depth[v];
    std::printf("  view %u (%s), depth %.3g..%.3g\n", scene.views[v].source_id,
                scene.views[v].image_path.filename().string().c_str(), d.min, d.max);
    for (const r3d::StereoPartner& p : selection.partners[v]) {
      std::printf("    -> %4u: %.3f %6u pts %5.1f deg %.2f %3.0f%% %.3g..%.3g%s\n", scene.views[p.view].source_id,
                  p.score, p.shared_points, p.mean_angle_deg, p.scale_ratio, 100.0 * p.coverage, p.depth.min,
                  p.depth.max, p.epipole_in_image ? "  [epipole in image]" : "");
    }
    if (selection.partners[v].empty()) {
      std::printf("    (no partner)\n");
    }
  }
  std::printf("  %zu stereo pairs\n", selection.stereo_pairs().size());
}

// OpenMVG sfm_data file, or a directory with a COLMAP text model.
r3d::Scene read_scene(const SceneArgs& args) {
  const std::filesystem::path input = utf8_path(args.input);
  const r3d::Stopwatch watch;
  if (std::filesystem::is_directory(input)) {
    r3d::ColmapReadOptions options;
    options.image_dir = utf8_path(args.image_dir);
    r3d::ColmapImportReport report;
    r3d::Scene scene = r3d::read_colmap_scene(input, options, &report);
    std::printf("COLMAP model, read in %.3f s\n", watch.elapsed_seconds());
    std::printf("  views:        %zu of %zu (%zu with unsupported camera)\n", scene.views.size(), report.num_images,
                report.num_images_unsupported);
    std::printf("  points:       %zu (%zu dropped), %zu observations\n", scene.num_points(), report.num_points_dropped,
                scene.observations.size());
    return scene;
  }
  r3d::OpenMvgReadOptions options;
  options.image_dir = utf8_path(args.image_dir);
  r3d::OpenMvgImportReport report;
  r3d::Scene scene = r3d::read_openmvg_scene(input, options, &report);
  std::printf("sfm_data %s, root_path '%s', read in %.3f s\n", report.version.c_str(), report.root_path.c_str(),
              watch.elapsed_seconds());
  std::printf("  views:        %zu calibrated of %zu (%zu without pose, %zu unsupported)\n", scene.views.size(),
              report.num_views, report.num_views_without_pose, report.num_views_unsupported);
  std::printf("  points:       %zu (%zu dropped), %zu observations\n", scene.num_points(), report.num_points_dropped,
              scene.observations.size());
  return scene;
}

void run_scene(const SceneArgs& args) {
  const r3d::Scene scene = read_scene(args);
  const r3d::ReprojectionStats reproj = r3d::compute_reprojection_stats(scene);
  for (std::size_t i = 0; i < scene.cameras.size(); ++i) {
    const r3d::CameraIntrinsics& cam = scene.cameras[i];
    std::printf("  camera %zu:     %s %dx%d f=%.2f c=(%.2f, %.2f)", i,
                std::string(r3d::openmvg_camera_model_name(cam.model)).c_str(), cam.width, cam.height, cam.fx, cam.cx,
                cam.cy);
    for (int k = 0; k < r3d::distortion_parameter_count(cam.model); ++k) {
      std::printf(" %.6g", cam.params[static_cast<std::size_t>(k)]);
    }
    std::printf("\n");
  }
  std::printf("  reprojection: rms %.3f px, median %.3f px, max %.3f px, %zu behind camera\n", reproj.rms,
              reproj.median, reproj.max, reproj.num_behind_camera);
  if (!scene.views.empty()) {
    std::printf("  first image:  %s\n", scene.views.front().image_path.string().c_str());
  }
  for (std::size_t i = 0; i < scene.cameras.size(); ++i) {
    if (!r3d::distortion_is_monotonic(scene.cameras[i])) {
      std::printf("  warning: distortion of camera %zu folds over inside the image\n", i);
    }
  }

  r3d::ViewSelectionOptions selection_options;
  selection_options.max_partners = args.max_partners;
  const r3d::Stopwatch selection_watch;
  const r3d::ViewSelection selection = r3d::select_views(scene, selection_options);
  const double selection_seconds = selection_watch.elapsed_seconds();
  if (args.pairs) {
    print_pairs(scene, selection);
    std::printf("  selection took %.3f s\n", selection_seconds);
  }

  if (!args.sparse_ply.empty()) {
    r3d::PlyWriteOptions ply;
    ply.comments.push_back("r3dmvs sparse SfM structure");
    r3d::write_ply(utf8_path(args.sparse_ply), r3d::sparse_point_cloud(scene, args.colors), ply);
    std::printf("wrote %s\n", args.sparse_ply.c_str());
  }
  if (!args.cameras_ply.empty()) {
    // Frustum size: 10% of the median view depth.
    std::vector<double> depths;
    for (const r3d::DepthRange& d : selection.view_depth) {
      if (d.valid()) {
        depths.push_back(0.5 * (d.min + d.max));
      }
    }
    double size = 1.0;
    if (!depths.empty()) {
      std::nth_element(depths.begin(), depths.begin() + static_cast<std::ptrdiff_t>(depths.size() / 2), depths.end());
      size = 0.1 * depths[depths.size() / 2];
    }
    r3d::write_camera_frustums_ply(utf8_path(args.cameras_ply), scene, size);
    std::printf("wrote %s\n", args.cameras_ply.c_str());
  }
  if (!args.undistort_dir.empty()) {
    const std::filesystem::path dir = utf8_path(args.undistort_dir);
    std::filesystem::create_directories(dir);
    const r3d::Stopwatch undistort_watch;
    // One view per task: PNG encoding is single-threaded, so views run in
    // parallel (the undistortion inside a task then runs serially).
    std::vector<std::string> lines(scene.views.size());
    r3d::parallel_for(0, scene.views.size(), 1, [&](std::size_t begin, std::size_t end) {
      for (std::size_t v = begin; v < end; ++v) {
        const r3d::CameraIntrinsics& cam = scene.camera_of(v);
        const r3d::PinholeIntrinsics target = r3d::undistorted_camera(cam);
        const r3d::ImageU8 img = r3d::read_image_u8(scene.views[v].image_path, 3);
        const r3d::ImageU8 out = r3d::undistort_image(img, cam, target);
        const std::filesystem::path file = dir / scene.views[v].image_path.filename().replace_extension(".png");
        r3d::write_png(file, out.view(), 2);  // fast: inspection output only
        lines[v] = std::format("  {}: {}x{} -> {}x{}, f={:.2f} c=({:.2f}, {:.2f})", file.filename().string(), cam.width,
                               cam.height, target.width, target.height, target.fx, target.cx, target.cy);
      }
    });
    for (const std::string& line : lines) {
      std::printf("%s\n", line.c_str());
    }
    std::printf("undistorted %zu images in %.2f s\n", scene.views.size(), undistort_watch.elapsed_seconds());
  }
}

struct PairArgs {
  std::string input;
  std::string image_dir;
  long long base = -1;     // source view ids
  long long partner = -1;
  bool all = false;
  std::string out_dir;
  double resolution = 1.0;
  r3d::StereoOptions options;
};

std::size_t view_index(const r3d::Scene& scene, long long source_id) {
  for (std::size_t v = 0; v < scene.views.size(); ++v) {
    if (static_cast<long long>(scene.views[v].source_id) == source_id) {
      return v;
    }
  }
  r3d::throw_error("no calibrated view with id {}", source_id);
}

// Rectifies and matches stereo pairs of a scene; writes rectified images,
// disparity maps and coloured point clouds for inspection.
void run_pair(const PairArgs& args) {
  SceneArgs scene_args;
  scene_args.input = args.input;
  scene_args.image_dir = args.image_dir;
  const r3d::Scene scene = read_scene(scene_args);
  const r3d::ViewSelection selection = r3d::select_views(scene);

  std::vector<std::pair<std::size_t, std::size_t>> jobs;
  if (args.all) {
    for (std::size_t v = 0; v < scene.views.size(); ++v) {
      if (!selection.partners[v].empty()) {
        jobs.emplace_back(v, selection.partners[v].front().view);
      }
    }
  } else {
    const std::size_t base = args.base >= 0 ? view_index(scene, args.base) : 0;
    std::size_t partner = 0;
    if (args.partner >= 0) {
      partner = view_index(scene, args.partner);
    } else if (!selection.partners[base].empty()) {
      partner = selection.partners[base].front().view;
    } else {
      r3d::throw_error("view {} has no stereo partner", scene.views[base].source_id);
    }
    jobs.emplace_back(base, partner);
  }
  const std::filesystem::path out_dir = utf8_path(args.out_dir);
  if (!out_dir.empty()) {
    std::filesystem::create_directories(out_dir);
  }
  r3d::RectificationOptions rect_options;
  rect_options.resolution = args.resolution;

  std::printf("%5s %5s %11s %11s %6s %5s %7s %7s %8s %8s\n", "base", "pair", "left", "right", "dmin", "nd", "rect s",
              "match s", "Mcell/s", "density");
  for (const auto& [b, p] : jobs) {
    const r3d::SceneView& vb = scene.views[b];
    const r3d::SceneView& vp = scene.views[p];
    r3d::DepthRange depth = selection.view_depth[b];
    for (const r3d::StereoPartner& sp : selection.partners[b]) {
      if (sp.view == p && sp.depth.valid()) {
        depth = sp.depth;
      }
    }
    const r3d::RectifiedPair rp =
        r3d::compute_rectification(scene.camera_of(b), vb.pose, scene.camera_of(p), vp.pose, depth, rect_options);
    if (!rp.valid) {
      std::printf("%5u %5u  skipped: %s\n", vb.source_id, vp.source_id, rp.reason.c_str());
      continue;
    }
    const r3d::Stopwatch rect_watch;
    r3d::ImageU8 mask_left, mask_right;
    const r3d::ImageU16 left = r3d::rectify_image(r3d::read_image_u16(vb.image_path, 1), scene.camera_of(b),
                                                  vb.pose, rp.left, &mask_left);
    const r3d::ImageU16 right = r3d::rectify_image(r3d::read_image_u16(vp.image_path, 1), scene.camera_of(p),
                                                   vp.pose, rp.right, &mask_right);
    const double rect_seconds = rect_watch.elapsed_seconds();
    const r3d::Stopwatch match_watch;
    const r3d::DisparityMap disp = r3d::match_stereo(left.view(), right.view(), mask_left.view(), mask_right.view(),
                                                     rp.disparity_min, rp.num_disparities, args.options);
    const double match_seconds = match_watch.elapsed_seconds();
    std::size_t valid_pixels = 0;
    std::size_t matched = 0;
    for (int y = 0; y < left.height(); ++y) {
      for (int x = 0; x < left.width(); ++x) {
        valid_pixels += mask_left(x, y) != 0 ? 1 : 0;
        matched += std::isfinite(disp.disparity(x, y)) ? 1 : 0;
      }
    }
    const double cells = static_cast<double>(left.width()) * left.height() * disp.num_disparities;
    std::printf("%5u %5u %5dx%-5d %5dx%-5d %6d %5d %7.2f %7.2f %8.1f %7.1f%%\n", vb.source_id, vp.source_id,
                left.width(), left.height(), right.width(), right.height(), rp.disparity_min, disp.num_disparities,
                rect_seconds, match_seconds, cells / match_seconds * 1e-6,
                100.0 * static_cast<double>(matched) / static_cast<double>(std::max<std::size_t>(valid_pixels, 1)));
    if (!out_dir.empty()) {
      const std::string name = std::format("pair_{}_{}", vb.source_id, vp.source_id);
      const r3d::ImageU8 color =
          r3d::rectify_image(r3d::read_image_u8(vb.image_path, 3), scene.camera_of(b), vb.pose, rp.left);
      r3d::PlyWriteOptions ply;
      ply.comments.push_back("r3dmvs stereo pair " + name);
      r3d::write_ply(out_dir / (name + ".ply"), r3d::disparity_to_point_cloud(rp, disp, color.view()), ply);
      const auto lo = static_cast<float>(rp.disparity_min);
      r3d::write_png(out_dir / (name + "_disparity.png"),
                     r3d::colorize_disparity(disp.disparity, lo, lo + static_cast<float>(rp.num_disparities)).view(), 2);
      r3d::write_png(out_dir / (name + "_left.png"), left.view(), 2);
      r3d::write_png(out_dir / (name + "_right.png"), right.view(), 2);
      r3d::write_pfm(out_dir / (name + "_disparity.pfm"), disp.disparity.view());
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"R3DMVS - multi-view stereo densification for OpenMVG scenes"};
  app.set_version_flag("--version", r3d::kVersionString);
  app.require_subcommand(1);

  unsigned threads = 0;
  std::string log_level = "info";
  app.add_option("-j,--threads", threads, "Number of worker threads (0 = all hardware threads)");
  app.add_option("--log-level", log_level, "trace, debug, info, warn, error or off")
      ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "off"}));

  CLI::App* info = app.add_subcommand("info", "Print build and system information");

  SceneArgs scene_args;
  CLI::App* scene = app.add_subcommand("scene", "Read an OpenMVG scene (or COLMAP text model directory), select stereo pairs, export for inspection");
  scene->add_option("input", scene_args.input, "sfm_data.json, sfm_data.bin or a COLMAP model directory")->required();
  scene->add_option("--image-dir", scene_args.image_dir, "Directory with the images (default: from the scene)");
  scene->add_flag("--pairs", scene_args.pairs, "Print the selected stereo partners of every view");
  scene->add_option("--max-partners", scene_args.max_partners, "Stereo partners per view")->check(CLI::Range(1, 32));
  scene->add_option("--sparse-ply", scene_args.sparse_ply, "Write the sparse points to a PLY file");
  scene->add_flag("--colors", scene_args.colors, "Colour the sparse points from the images");
  scene->add_option("--cameras-ply", scene_args.cameras_ply, "Write the camera frustums to a PLY file");
  scene->add_option("--undistort-dir", scene_args.undistort_dir, "Write undistorted images (PNG) to this directory");

  r3dmvs::StereoArgs stereo_args;
  CLI::App* stereo = nullptr;
  r3dmvs::add_stereo_command(app, stereo_args, stereo);

  PairArgs pair_args;
  CLI::App* pair = app.add_subcommand("pair", "Rectify and match stereo pairs of a scene, export disparity and points");
  pair->add_option("input", pair_args.input, "sfm_data.json, sfm_data.bin or a COLMAP model directory")->required();
  pair->add_option("--image-dir", pair_args.image_dir, "Directory with the images (default: from the scene)");
  pair->add_option("--base", pair_args.base, "Base view id (default: the first view)");
  pair->add_option("--partner", pair_args.partner, "Partner view id (default: the best partner of the base view)");
  pair->add_flag("--all", pair_args.all, "Every view with its best partner");
  pair->add_option("--out-dir", pair_args.out_dir, "Write rectified images, disparity maps and point clouds here");
  pair->add_option("--resolution", pair_args.resolution, "Rectified resolution relative to the base view")
      ->check(CLI::Range(0.05, 4.0));
  r3dmvs::add_matcher_options(*pair, pair_args.options);

  SceneArgs dense_scene;
  r3dmvs::DenseArgs dense_args;
  CLI::App* dense = app.add_subcommand("dense", "Compute the dense point cloud of a scene");
  dense->add_option("input", dense_scene.input, "sfm_data.json, sfm_data.bin or a COLMAP model directory")->required();
  dense->add_option("--image-dir", dense_scene.image_dir, "Directory with the images (default: from the scene)");
  r3dmvs::add_dense_options(*dense, dense_args);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  try {
    r3d::LogLevel level = r3d::LogLevel::info;
    r3d::parse_log_level(log_level, level);
    r3d::set_log_level(level);
    r3d::set_default_thread_count(threads);

    if (info->parsed()) {
      print_info();
    } else if (scene->parsed()) {
      run_scene(scene_args);
    } else if (stereo->parsed()) {
      r3dmvs::run_stereo(stereo_args);
    } else if (pair->parsed()) {
      run_pair(pair_args);
    } else if (dense->parsed()) {
      r3dmvs::run_dense(dense_args, read_scene(dense_scene));
    }
  } catch (const std::exception& e) {
    r3d::log_error("{}", e.what());
    return 1;
  }
  return 0;
}
