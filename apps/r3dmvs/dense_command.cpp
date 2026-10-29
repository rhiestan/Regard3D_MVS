#include "dense_command.hpp"

#include <csignal>
#include <cstdio>
#include <filesystem>
#include <string_view>

#include "stereo_commands.hpp"

namespace r3dmvs {

namespace {

std::filesystem::path utf8_path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

r3d::CancellationToken g_cancel;

extern "C" void on_cancel_signal(int signal) {
  g_cancel.request();
  std::signal(signal, SIG_DFL);  // the next one terminates
}

}  // namespace

r3d::CancellationToken& cancellation_token() { return g_cancel; }

void install_cancel_handlers() {
  std::signal(SIGINT, on_cancel_signal);
  std::signal(SIGTERM, on_cancel_signal);
#ifdef SIGBREAK
  std::signal(SIGBREAK, on_cancel_signal);  // Windows: Ctrl+Break, which a parent process can send
#endif
}

r3d::DensePreset preset_from_args(int argc, const char* const* argv) {
  constexpr std::string_view kOption = "--preset";
  std::string_view name;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == kOption && i + 1 < argc) {
      name = argv[i + 1];
    } else if (arg.starts_with(kOption) && arg.size() > kOption.size() && arg[kOption.size()] == '=') {
      name = arg.substr(kOption.size() + 1);
    }
  }
  return r3d::parse_dense_preset(name).value_or(r3d::DensePreset::normal);
}

void add_dense_options(CLI::App& command, DenseArgs& args) {
  args.memory_mb = args.options.memory_budget >> 20;
  args.fusion_memory_mb = args.options.fusion.memory_budget >> 20;
  command.add_option("-o,--output", args.output, "Output PLY file")->required();
  command
      .add_option("--preset", args.preset,
                  "Parameter preset: fast, normal, high or ultra; the defaults shown below are the preset's")
      ->check(CLI::IsMember({"fast", "normal", "high", "ultra"}))
      ->capture_default_str();
  command.add_flag("--progress", args.progress, "Print 'progress <percent> <stage> <done>/<total>' lines on stdout");
  command.add_option("--work-dir", args.work_dir, "Directory for cached depth maps (and debug images)");
  command.add_flag("--reuse", args.options.reuse, "Load depth maps cached in --work-dir instead of matching again");
  command.add_flag("--reuse-refined", args.options.reuse_refined,
                   "Load the refined depth maps cached in --work-dir (skips matching and refinement)");
  command.add_flag("--debug-images", args.options.debug_images, "Write depth images before and after filtering to --work-dir");
  command.add_option("--resolution", args.options.depth.resolution, "Depth map resolution relative to the images")
      ->check(CLI::Range(0.05, 4.0))
      ->capture_default_str();
  command.add_option("--max-image-size", args.options.depth.max_image_size,
                     "Longest side of the depth maps in pixels (0: no limit; applied after --resolution)")
      ->capture_default_str();
  command.add_option("--max-disparity-ratio", args.options.depth.rectification.max_disparity_ratio,
                     "Skip pairs whose disparity range exceeds this times the rectified image width")
      ->capture_default_str();
  command.add_option("--max-pairs", args.options.depth.max_pairs, "Stereo partners per view")
      ->check(CLI::Range(1, 16))
      ->capture_default_str();
  command.add_option("--inlier-px", args.options.depth.multi_stereo.inlier_px, "Disparity distance for pairs to agree")
      ->capture_default_str();
  command.add_option("--max-reproj", args.options.depth.multi_stereo.max_reprojection_px,
                     "Maximum disparity residual of a used pair")
      ->capture_default_str();
  command.add_option("--min-angle", args.options.depth.multi_stereo.min_angle_deg, "Minimum triangulation angle (degrees)")
      ->capture_default_str();
  command.add_option("--min-pairs", args.options.depth.multi_stereo.min_pairs, "Agreeing pairs per pixel")
      ->capture_default_str();
  command.add_flag("--refine,!--no-refine", args.options.refine, "Refine depths by patch matching in the images");
  command.add_option("--refine-radius", args.options.refinement.radius, "Refinement window half size (pixels)")
      ->check(CLI::Range(1, 7))
      ->capture_default_str();
  command.add_option("--refine-step", args.options.refinement.step, "Spacing of the refinement window samples")
      ->check(CLI::Range(1, 7))
      ->capture_default_str();
  command.add_option("--refine-iterations", args.options.refinement.iterations, "Gauss-Newton steps per pixel")
      ->capture_default_str();
  command.add_option("--refine-views", args.options.refinement.max_views, "Neighbouring images used for refinement")
      ->check(CLI::Range(1, 8))
      ->capture_default_str();
  command.add_option("--refine-min-ncc", args.options.refinement.min_ncc, "NCC a view needs to take part")
      ->capture_default_str();
  command.add_option("--refine-blur", args.options.refinement.blur_sigma, "Gaussian pre-filter of the refinement images (pixels)")
      ->capture_default_str();
  command.add_option("--refine-min-gain", args.options.refinement.min_gain, "Mean NCC gain a refined depth needs")
      ->capture_default_str();
  command.add_option("--refine-max-shift", args.options.refinement.max_shift_px,
                     "Largest change of a refined depth (pixels in the used views)")
      ->capture_default_str();
  command.add_option("--min-views", args.options.consistency.min_views, "Agreeing neighbour views per pixel")
      ->capture_default_str();
  command.add_option("--min-evidence", args.options.consistency.min_evidence, "Agreeing pairs + agreeing views per pixel (2: two pairs or one other view)")
      ->capture_default_str();
  command.add_option("--min-confidence", args.options.consistency.min_confidence, "Minimum matcher confidence")
      ->capture_default_str();
  command.add_option("--depth-tol", args.options.consistency.max_relative_depth_error,
                     "Relative depth tolerance of the consistency check")
      ->capture_default_str();
  command.add_option("--consistency-reproj", args.options.consistency.max_reprojection_px,
                     "Forward-backward reprojection tolerance (pixels)")
      ->capture_default_str();
  command.add_flag("--independent-support,!--any-support", args.options.consistency.independent_support,
                   "Ignore agreement that rests only on the same image pair matched in both directions (default: any agreement counts)");
  command.add_flag("--average,!--no-average", args.options.consistency.average_depths,
                   "Average each depth with the agreeing neighbours' depths (default)");
  command.add_option("--neighbours", args.options.consistency.max_neighbours, "Views checked per view")->capture_default_str();
  command.add_option("--edge-erosion", args.options.consistency.edge_erosion, "Pixels removed at depth discontinuities")
      ->capture_default_str();
  command.add_flag("--erode-holes,!--no-erode-holes", args.options.consistency.erode_holes,
                   "Erode at the border of regions without depth as well");
  command.add_option("--depth-speckle-size", args.options.consistency.speckle_size,
                     "Smallest connected depth segment kept after the consistency check (pixels, 0: off)")
      ->capture_default_str();
  command.add_option("--depth-speckle-step", args.options.consistency.speckle_step, "Relative depth step within a segment")
      ->capture_default_str();
  command.add_option("--discontinuity", args.options.consistency.discontinuity, "Relative depth jump of a discontinuity")
      ->capture_default_str();
  command.add_flag("--visibility,!--no-visibility", args.options.visibility_filter,
                   "Remove depths that conflict with the surfaces neighbouring views see (default)");
  command.add_option("--visibility-margin", args.options.visibility.relative_margin,
                     "Relative depth by which a conflicting surface lies in front")
      ->capture_default_str();
  command.add_option("--visibility-conflicts", args.options.visibility.min_conflicts, "Conflicting views needed to remove a depth")
      ->capture_default_str();
  command.add_option("--visibility-margin-views", args.options.visibility.support_margin,
                     "Removed when conflicting views exceed agreeing views by more than this")
      ->capture_default_str();
  command.add_option("--visibility-neighbours", args.options.visibility.max_neighbours, "Views checked per view")
      ->capture_default_str();
  command.add_option("--memory", args.memory_mb,
                     "Memory for depth maps and refinement images (MB); the rest waits in the work directory")
      ->check(CLI::Range(std::size_t{64}, std::size_t{1} << 30))
      ->capture_default_str();
  command.add_option("--voxel", args.options.fusion.voxel_size, "Fixed fusion leaf size (0: adaptive, --voxel-factor x GSD)")
      ->capture_default_str();
  command.add_option("--voxel-factor", args.options.fusion.voxel_factor, "Leaf size relative to the GSD of a point (1, 2, 4: density)")
      ->capture_default_str();
  command.add_option("--resolution-ratio", args.options.fusion.resolution_ratio,
                     "Drop points whose leaf holds points this much finer (0: keep all)")
      ->capture_default_str();
  command.add_option("--min-sources", args.options.fusion.min_sources, "Views per fused point")->capture_default_str();
  command.add_flag("--fold-neighbourhood,!--fold-leaf", args.options.fusion.fold_neighbourhood,
                   "Count the views of the neighbouring leaves for --min-sources (default: the leaf only)");
  command.add_option("--fusion-memory", args.fusion_memory_mb, "Memory for the fusion's point data per pass (MB)")
      ->capture_default_str();
  command.add_flag("--with-views", args.options.fusion.with_views, "Write the ids of the views of each point (view_ids)");
  add_layout_options(command, args.layout);
  add_matcher_options(command, args.options.depth.stereo);
}

void run_dense(DenseArgs args, const r3d::Scene& scene) {
  r3d::DenseOptions& options = args.options;
  options.work_dir = utf8_path(args.work_dir);
  options.memory_budget = args.memory_mb << 20;
  options.fusion.memory_budget = args.fusion_memory_mb << 20;
  r3d::DenseCallbacks callbacks;
  callbacks.cancel = &cancellation_token();
  if (args.progress) {
    callbacks.progress = [](const r3d::DenseProgress& p) {
      std::printf("progress %.1f %s %zu/%zu\n", 100.0 * p.fraction, std::string(r3d::to_string(p.stage)).c_str(),
                  p.done, p.total);
      std::fflush(stdout);
    };
  }
  r3d::densify_to_ply(scene, options, utf8_path(args.output), layout_options(args.layout), callbacks);
}

}  // namespace r3dmvs
