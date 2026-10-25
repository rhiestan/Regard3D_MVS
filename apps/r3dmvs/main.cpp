// r3dmvs: command line front end of the R3DMVS library.

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <format>
#include <string>
#include <thread>
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
#include "r3d/io/scene_export.hpp"
#include "r3d/scene/scene.hpp"
#include "r3d/scene/undistort.hpp"
#include "r3d/select/view_selection.hpp"
#include "r3d/version.hpp"

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
    }
  } catch (const std::exception& e) {
    r3d::log_error("{}", e.what());
    return 1;
  }
  return 0;
}
