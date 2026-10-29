#pragma once

#include <cstddef>
#include <string>

#include <CLI/CLI.hpp>

#include "export_command.hpp"
#include "r3d/pipeline/dense.hpp"
#include "r3d/scene/scene.hpp"

namespace r3dmvs {

// `dense`: the dense pipeline (r3d::densify) into a PLY file.
struct DenseArgs {
  std::string output;               // PLY file
  std::string work_dir;             // depth map cache and debug images
  std::string preset = "normal";
  r3d::DenseOptions options;        // holds the preset before the options are parsed
  std::size_t memory_mb = 0;        // options.memory_budget in MB
  std::size_t fusion_memory_mb = 0; // options.fusion.memory_budget in MB
  LayoutArgs layout;                // point attributes of the output
  bool progress = false;            // "progress <percent> ..." lines on stdout
};

// The preset that --preset in argv names (normal if none or unknown), so
// that its values become the defaults of the single options.
[[nodiscard]] r3d::DensePreset preset_from_args(int argc, const char* const* argv);

// Token requested by Ctrl+C (SIGINT), SIGTERM or (Windows) Ctrl+Break once
// install_cancel_handlers() has run; a second signal terminates the process at once.
[[nodiscard]] r3d::CancellationToken& cancellation_token();
void install_cancel_handlers();

// args.options must hold the preset's values already.
void add_dense_options(CLI::App& command, DenseArgs& args);
void run_dense(DenseArgs args, const r3d::Scene& scene);

}  // namespace r3dmvs
