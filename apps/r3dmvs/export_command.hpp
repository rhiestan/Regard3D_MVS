#pragma once

#include <cstddef>
#include <string>

#include <CLI/CLI.hpp>

#include "r3d/core/point_cloud.hpp"
#include "r3d/io/mesher_export.hpp"
#include "r3d/io/ply.hpp"
#include "r3d/scene/scene.hpp"

namespace r3dmvs {

// Point attributes of the written cloud (--for and its parameters).
struct LayoutArgs {
  std::string layout = "full";
  r3d::CloudLayoutOptions options;
};

void add_layout_options(CLI::App& command, LayoutArgs& args);
[[nodiscard]] r3d::CloudLayoutOptions layout_options(const LayoutArgs& args);

// `export`: an existing cloud in another layout, and / or the texrecon scene folder.
struct ExportArgs {
  std::string cloud;         // input PLY
  std::string output;        // output PLY
  LayoutArgs layout;
  std::string texrecon_dir;
  r3d::TexreconExportOptions texrecon;
};

void add_export_options(CLI::App& command, ExportArgs& args);
// `scene` may be null if no texrecon folder is requested.
void run_export(const ExportArgs& args, const r3d::Scene* scene);

}  // namespace r3dmvs
