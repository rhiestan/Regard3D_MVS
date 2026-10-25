#pragma once

#include <string>

#include <CLI/CLI.hpp>

#include "r3d/stereo/stereo_matcher.hpp"

namespace r3dmvs {

// `stereo`: match a rectified pair (or Middlebury 2014 scene directories)
// and evaluate against ground truth when available.
struct StereoArgs {
  std::string input;      // Middlebury scene directory, or a directory of scenes
  std::string left;       // or explicit images
  std::string right;
  int disparity_min = 0;
  int num_disparities = 0;  // 0: from calib.txt
  std::string out_pfm;
  std::string out_png;
  int repeat = 1;
  std::string load_pattern;  // evaluate existing results
  float error_scale = 0.0F;  // 0: from the Middlebury set name (Q: 4, H: 2)
  r3d::StereoOptions options;
};

void add_stereo_command(CLI::App& app, StereoArgs& args, CLI::App*& command);
void run_stereo(const StereoArgs& args);

// Matcher options shared by `stereo` and `pair`.
void add_matcher_options(CLI::App& command, r3d::StereoOptions& options);

}  // namespace r3dmvs
