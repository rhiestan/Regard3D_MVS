#pragma once

#include <string>

#include <CLI/CLI.hpp>

#include "r3d/mvs/consistency.hpp"
#include "r3d/mvs/fusion.hpp"
#include "r3d/mvs/multi_stereo.hpp"
#include "r3d/scene/scene.hpp"

namespace r3dmvs {

// `dense`: depth maps of all views (multi-stereo), geometric consistency,
// voxel fusion, coloured PLY.
struct DenseArgs {
  std::string output;         // PLY file
  std::string work_dir;       // depth map cache and debug images
  bool reuse = false;         // load cached depth maps from work_dir
  bool debug_images = false;  // depth PNGs before / after filtering
  r3d::DepthMapOptions depth;
  r3d::ConsistencyOptions consistency;
  r3d::FusionOptions fusion;
};

void add_dense_options(CLI::App& command, DenseArgs& args);
void run_dense(const DenseArgs& args, const r3d::Scene& scene);

}  // namespace r3dmvs
