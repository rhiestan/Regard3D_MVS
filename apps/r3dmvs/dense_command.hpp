#pragma once

#include <string>

#include <CLI/CLI.hpp>

#include "r3d/mvs/consistency.hpp"
#include "r3d/mvs/fusion.hpp"
#include "r3d/mvs/multi_stereo.hpp"
#include "r3d/mvs/refinement.hpp"
#include "r3d/mvs/visibility.hpp"
#include "r3d/scene/scene.hpp"

namespace r3dmvs {

// `dense`: depth maps of all views (multi-stereo), refinement, geometric consistency,
// voxel fusion, coloured PLY.
struct DenseArgs {
  std::string output;         // PLY file
  std::string work_dir;       // depth map cache and debug images
  bool reuse = false;         // load cached depth maps from work_dir
  bool reuse_refined = false; // load cached refined depth maps (skips matching and refinement)
  bool debug_images = false;  // depth PNGs before / after filtering
  r3d::DepthMapOptions depth;
  bool refine = true;
  r3d::RefinementOptions refinement;
  r3d::ConsistencyOptions consistency;
  bool visibility_filter = true;
  r3d::VisibilityOptions visibility;
  r3d::FusionOptions fusion;
  std::size_t fusion_memory_mb = 2048;
  std::size_t memory_mb = 8192;     // depth maps and refinement images in memory
};

void add_dense_options(CLI::App& command, DenseArgs& args);
void run_dense(const DenseArgs& args, const r3d::Scene& scene);

}  // namespace r3dmvs
