// r3dmvs: command line front end of the R3DMVS library.

#include <cstdio>
#include <exception>
#include <format>
#include <string>
#include <thread>

#include <CLI/CLI.hpp>

#include "r3d/core/log.hpp"
#include "r3d/core/simd.hpp"
#include "r3d/core/thread_pool.hpp"
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
    }
  } catch (const std::exception& e) {
    r3d::log_error("{}", e.what());
    return 1;
  }
  return 0;
}
