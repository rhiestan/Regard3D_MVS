#include "stereo_commands.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "r3d/core/error.hpp"
#include "r3d/core/timer.hpp"
#include "r3d/io/image_io.hpp"
#include "r3d/io/pfm.hpp"
#include "r3d/stereo/disparity_eval.hpp"

namespace r3dmvs {
namespace {

namespace fs = std::filesystem;

fs::path utf8_path(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }

// key=value lines of a Middlebury calib.txt.
std::map<std::string, std::string> read_calib(const fs::path& path) {
  std::map<std::string, std::string> values;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const auto eq = line.find('=');
    if (eq != std::string::npos) {
      values[line.substr(0, eq)] = line.substr(eq + 1);
    }
  }
  return values;
}

struct SceneResult {
  std::string name;
  double seconds = 0.0;
  double mcells_per_s = 0.0;     // full-range cells per second
  double cells_per_pixel = 0.0;  // aggregated cells (all levels) per pixel
  bool has_gt = false;
  r3d::DisparityErrorStats nonocc;        // sparse estimate, non-occluded pixels
  r3d::DisparityErrorStats nonocc_dense;  // holes filled
  r3d::DisparityErrorStats all_dense;
};

SceneResult match_scene(const StereoArgs& args, const fs::path& left_path, const fs::path& right_path, int dmin,
                        int ndisp, const fs::path& gt_dir, const std::string& name) {
  const r3d::ImageU16 left = r3d::read_image_u16(left_path, 1);
  const r3d::ImageU16 right = r3d::read_image_u16(right_path, 1);
  SceneResult res;
  res.name = name;
  r3d::DisparityMap result;
  if (!args.load_pattern.empty()) {
    // Evaluate an existing result instead of matching.
    std::string file = args.load_pattern;
    if (const auto at = file.find("{scene}"); at != std::string::npos) {
      file.replace(at, 7, name);
    }
    result.disparity = r3d::read_pfm(utf8_path(file));
    result.num_disparities = ndisp;
  } else {
    const r3d::Stopwatch watch;
    for (int i = 0; i < std::max(1, args.repeat); ++i) {
      result = r3d::match_stereo(left.view(), right.view(), {}, {}, dmin, ndisp, args.options);
    }
    res.seconds = watch.elapsed_seconds() / std::max(1, args.repeat);
    const double pixels = static_cast<double>(left.width()) * left.height();
    res.mcells_per_s = pixels * result.num_disparities / res.seconds * 1e-6;
    res.cells_per_pixel = static_cast<double>(result.cells) / pixels;
  }

  const fs::path gt_path = gt_dir / "disp0GT.pfm";
  const fs::path mask_path = gt_dir / "mask0nocc.png";
  if (!gt_dir.empty() && fs::exists(gt_path)) {
    const r3d::ImageF32 gt = r3d::read_pfm(gt_path);
    r3d::ImageU8 mask;
    if (fs::exists(mask_path)) {
      mask = r3d::read_image_u8(mask_path, 1);
    }
    const r3d::ImageF32 filled = r3d::fill_disparity_holes(result.disparity);
    res.has_gt = true;
    // MiddEval3 scores every resolution in full-resolution pixels.
    float scale = args.error_scale;
    if (scale <= 0.0F) {
      const std::string set = gt_dir.parent_path().filename().string();
      const char res_code = set.empty() ? 'F' : set.back();
      scale = res_code == 'Q' ? 4.0F : res_code == 'H' ? 2.0F : 1.0F;
    }
    res.nonocc = r3d::evaluate_disparity(result.disparity, gt, mask.view(), 255, scale);
    res.nonocc_dense = r3d::evaluate_disparity(filled, gt, mask.view(), 255, scale);
    res.all_dense = r3d::evaluate_disparity(filled, gt, mask.view(), 128, scale);
  }
  if (!args.out_pfm.empty()) {
    r3d::write_pfm(utf8_path(args.out_pfm), result.disparity.view());
  }
  if (!args.out_png.empty()) {
    const float lo = static_cast<float>(dmin);
    r3d::write_png(utf8_path(args.out_png),
                   r3d::colorize_disparity(result.disparity, lo, lo + static_cast<float>(ndisp)).view());
  }
  return res;
}

// MiddEval3 training weights: these scenes count half in the averages.
double middlebury_weight(const std::string& name) {
  for (const char* half : {"PianoL", "Playroom", "Playtable", "Teddy", "Vintage"}) {
    if (name == half) {
      return 0.5;
    }
  }
  return 1.0;
}

void print_header() {
  std::printf("%-12s %7s %8s %6s | %7s %6s %6s %6s | %6s %6s %6s | %6s\n", "scene", "time s", "Mcell/s", "cells",
              "density", "bad1", "bad2", "avgerr", "bad2", "bad4", "avgerr", "bad2");
  std::printf("%-12s %7s %8s %6s | %28s | %20s | %6s\n", "", "", "", "/pixel", "non-occluded, estimates only",
              "non-occluded, filled", "all");
}

void print_row(const SceneResult& r) {
  std::printf("%-12s %7.3f %8.1f %6.1f", r.name.c_str(), r.seconds, r.mcells_per_s, r.cells_per_pixel);
  if (r.has_gt) {
    std::printf(" | %6.1f%% %5.1f%% %5.1f%% %6.2f | %5.1f%% %5.1f%% %6.2f | %5.1f%%", 100 * r.nonocc.density,
                100 * r.nonocc.bad1, 100 * r.nonocc.bad2, r.nonocc.avg_error, 100 * r.nonocc_dense.bad2,
                100 * r.nonocc_dense.bad4, r.nonocc_dense.avg_error, 100 * r.all_dense.bad2);
  }
  std::printf("\n");
}

}  // namespace

void add_matcher_options(CLI::App& command, r3d::StereoOptions& o) {
  command.add_option("--p1", o.p1, "SGM penalty for disparity steps of one")->capture_default_str();
  command.add_option("--p2", o.p2, "SGM penalty for larger steps")->capture_default_str();
  command.add_option("--p2-adapt", o.p2_adapt, "Intensity adaptive P2 (0 = constant)")->capture_default_str();
  command.add_option("--invalid-cost", o.invalid_cost, "Cost of matches outside the right image")
      ->capture_default_str();
  command.add_option("--uniqueness", o.uniqueness, "Uniqueness margin (0 = off)")->capture_default_str();
  command.add_option("--lr-diff", o.lr_max_diff, "Left-right check tolerance (< 0: off)")->capture_default_str();
  command.add_flag("--lr-full,!--lr-diagonal", o.lr_full_match,
                   "Left-right check against a full right-to-left match (default) or against the right "
                   "disparities of the same cost volume (faster)");
  command.add_option("--speckle-size", o.speckle_size, "Minimum segment size (0 = off)")->capture_default_str();
  command.add_option("--speckle-diff", o.speckle_diff, "Disparity step between segments")->capture_default_str();
  command.add_flag("!--no-median", o.median, "Disable the 3x3 median filter");
  command.add_option("--levels", o.levels, "Coarser pyramid levels (-1: automatic, 0: full range only)")
      ->capture_default_str();
  command.add_option("--range-radius", o.range_radius, "Coarse window radius of the per-pixel ranges")
      ->capture_default_str();
  command.add_option("--range-margin", o.range_margin, "Labels added on each side of the per-pixel ranges")
      ->capture_default_str();
  command.add_option("--range-fill", o.range_fill, "Coarse pixels to fill ranges of pixels without disparity")
      ->capture_default_str();
  command.add_option("--tile-size", o.tile_size, "Tile core size")->capture_default_str();
  command.add_option("--tile-margin", o.tile_margin, "Discarded tile border")->capture_default_str();
  command.add_option("--tile-blend", o.tile_blend, "Tile blending zone")->capture_default_str();
  command.add_option("--subpixel", o.subpixel, "Sub-pixel method")
      ->transform(CLI::CheckedTransformer(std::map<std::string, r3d::SubpixelMethod>{
          {"none", r3d::SubpixelMethod::none},
          {"parabola", r3d::SubpixelMethod::parabola},
          {"equiangular", r3d::SubpixelMethod::equiangular}}));
  command
      .add_option_function<std::size_t>(
          "--memory-budget", [&o](std::size_t mb) { o.memory_budget = mb << 20; },
          "Memory for the tiles matched at once, in MB (default 2048)")
      ->check(CLI::Range(std::size_t{16}, std::size_t{1} << 24));
  command.add_flag("--scalar", o.scalar_kernels, "Use the scalar reference kernels");
}

void add_stereo_command(CLI::App& app, StereoArgs& args, CLI::App*& command) {
  command = app.add_subcommand("stereo", "Match a rectified stereo pair (Middlebury 2014 layout) and evaluate it");
  command->add_option("input", args.input,
                      "Scene directory with im0.png, im1.png, calib.txt [, disp0GT.pfm, mask0nocc.png], or a "
                      "directory of such scenes");
  command->add_option("--left", args.left, "Left image (instead of a scene directory)");
  command->add_option("--right", args.right, "Right image");
  command->add_option("--dmin", args.disparity_min, "Smallest disparity")->capture_default_str();
  command->add_option("--ndisp", args.num_disparities, "Number of disparities (default: calib.txt)");
  command->add_option("--out-pfm", args.out_pfm, "Write the disparity map (PFM)");
  command->add_option("--out-png", args.out_png, "Write a colour rendering of the disparity map");
  command->add_option("--repeat", args.repeat, "Repeat the matching (timing)")->check(CLI::Range(1, 100));
  command->add_option("--load", args.load_pattern,
                      "Evaluate this PFM instead of matching ({scene} is replaced by the scene name)");
  command->add_option("--error-scale", args.error_scale,
                      "Error scale for the evaluation (0: MiddEval3 full-resolution pixels from the Q/H/F set name)");
  add_matcher_options(*command, args.options);
}

void run_stereo(const StereoArgs& args) {
  if (!args.left.empty()) {
    if (args.right.empty() || args.num_disparities <= 0) {
      r3d::throw_error("stereo: --left needs --right and --ndisp");
    }
    print_header();
    print_row(match_scene(args, utf8_path(args.left), utf8_path(args.right), args.disparity_min,
                          args.num_disparities, {}, utf8_path(args.left).stem().string()));
    return;
  }
  const fs::path input = utf8_path(args.input);
  std::vector<fs::path> scenes;
  if (fs::exists(input / "im0.png")) {
    scenes.push_back(input);
  } else if (fs::is_directory(input)) {
    for (const fs::directory_entry& e : fs::directory_iterator(input)) {
      if (e.is_directory() && fs::exists(e.path() / "im0.png")) {
        scenes.push_back(e.path());
      }
    }
    std::sort(scenes.begin(), scenes.end());
  }
  if (scenes.empty()) {
    r3d::throw_error("stereo: no Middlebury scene (im0.png) in {}", args.input);
  }
  print_header();
  SceneResult mean;
  mean.name = "weighted";
  double gt_weight = 0.0;
  for (const fs::path& scene : scenes) {
    int ndisp = args.num_disparities;
    if (ndisp <= 0) {
      const auto calib = read_calib(scene / "calib.txt");
      const auto it = calib.find("ndisp");
      if (it == calib.end()) {
        r3d::throw_error("stereo: {} has no ndisp in calib.txt (use --ndisp)", scene.string());
      }
      ndisp = std::stoi(it->second);
    }
    const SceneResult r = match_scene(args, scene / "im0.png", scene / "im1.png", args.disparity_min, ndisp, scene,
                                      scene.filename().string());
    print_row(r);
    mean.seconds += r.seconds / static_cast<double>(scenes.size());
    mean.mcells_per_s += r.mcells_per_s / static_cast<double>(scenes.size());
    mean.cells_per_pixel += r.cells_per_pixel / static_cast<double>(scenes.size());
    if (r.has_gt) {
      const double wgt = middlebury_weight(r.name);
      gt_weight += wgt;
      mean.has_gt = true;
      for (auto [dst, src] : {std::pair{&mean.nonocc, &r.nonocc}, std::pair{&mean.nonocc_dense, &r.nonocc_dense},
                              std::pair{&mean.all_dense, &r.all_dense}}) {
        dst->density += wgt * src->density;
        dst->bad1 += wgt * src->bad1;
        dst->bad2 += wgt * src->bad2;
        dst->bad4 += wgt * src->bad4;
        dst->avg_error += wgt * src->avg_error;
      }
    }
  }
  if (scenes.size() > 1) {
    if (gt_weight > 0.0) {
      for (r3d::DisparityErrorStats* s : {&mean.nonocc, &mean.nonocc_dense, &mean.all_dense}) {
        s->density /= gt_weight;
        s->bad1 /= gt_weight;
        s->bad2 /= gt_weight;
        s->bad4 /= gt_weight;
        s->avg_error /= gt_weight;
      }
    }
    print_row(mean);
  }
}

}  // namespace r3dmvs
