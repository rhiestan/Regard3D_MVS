#include "export_command.hpp"

#include <cstdio>
#include <filesystem>

#include "r3d/core/error.hpp"
#include "r3d/core/timer.hpp"

namespace r3dmvs {

namespace {

std::filesystem::path utf8_path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

}  // namespace

void add_layout_options(CLI::App& command, LayoutArgs& args) {
  command
      .add_option("--for", args.layout,
                  "Point attributes for a mesher: full (all), poisson (PoissonRecon), fssr (MVE fssrecon)")
      ->check(CLI::IsMember({"full", "poisson", "fssr"}))
      ->capture_default_str();
  command.add_option("--fssr-scale", args.options.fssr_scale, "fssr: sample scale = this x the point footprint")
      ->capture_default_str();
  command
      .add_option("--fssr-full-confidence", args.options.fssr_full_confidence,
                  "fssr: fused depths per point for confidence 1 (0: all 1)")
      ->capture_default_str();
  command
      .add_option("--poisson-full-confidence", args.options.poisson_full_confidence,
                  "poisson: fused depths per point for normal length 1, for PoissonRecon --confidence (0: unit normals)")
      ->capture_default_str();
}

r3d::CloudLayoutOptions layout_options(const LayoutArgs& args) {
  r3d::CloudLayoutOptions options = args.options;
  options.layout = r3d::parse_cloud_layout(args.layout).value_or(r3d::CloudLayout::full);
  return options;
}

void add_export_options(CLI::App& command, ExportArgs& args) {
  command.add_option("--cloud", args.cloud, "Dense cloud (PLY) written by 'dense'");
  command.add_option("-o,--output", args.output, "Cloud in the layout of --for (PLY)");
  add_layout_options(command, args.layout);
  command.add_option("--texrecon-dir", args.texrecon_dir,
                     "Write undistorted images and .cam files for texrecon (needs the scene)");
  command.add_option("--jpeg-quality", args.texrecon.jpeg_quality, "Quality of re-encoded images")
      ->check(CLI::Range(1, 100))
      ->capture_default_str();
}

void run_export(const ExportArgs& args, const r3d::Scene* scene) {
  if (args.cloud.empty() != args.output.empty()) {
    r3d::throw_error("--cloud and --output go together");
  }
  if (args.cloud.empty() && args.texrecon_dir.empty()) {
    r3d::throw_error("nothing to export: give --cloud / --output or --texrecon-dir");
  }
  if (!args.cloud.empty()) {
    const r3d::Stopwatch watch;
    const r3d::PointCloud cloud = r3d::read_ply(utf8_path(args.cloud));
    r3d::PlyWriteOptions ply;
    ply.double_positions = !cloud.origin.isZero();
    ply.comments.push_back("r3dmvs dense cloud");
    r3d::CloudWriter writer(utf8_path(args.output), layout_options(args.layout), ply);
    writer.append(cloud);
    writer.close();
    std::printf("wrote %zu points (%s) to %s in %.2f s\n", cloud.size(), args.layout.layout.c_str(),
                args.output.c_str(), watch.elapsed_seconds());
  }
  if (!args.texrecon_dir.empty()) {
    if (scene == nullptr) {
      r3d::throw_error("--texrecon-dir needs the scene");
    }
    const r3d::Stopwatch watch;
    r3d::TexreconExportReport report;
    r3d::write_texrecon_scene(*scene, utf8_path(args.texrecon_dir), args.texrecon, &report);
    std::printf("texrecon scene: %zu views (%zu images copied) in %s, %.2f s\n", report.images, report.copied,
                args.texrecon_dir.c_str(), watch.elapsed_seconds());
  }
}

}  // namespace r3dmvs
