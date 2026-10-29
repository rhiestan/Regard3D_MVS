// Minimal use of the R3DMVS library: an OpenMVG scene to a dense cloud for
// PoissonRecon, with a preset, progress output and cancellation.
//
//   densify_example sfm_data.bin cloud.ply [fast|normal|high|ultra]

#include <cstddef>
#include <cstdio>
#include <exception>
#include <string>

#include "r3d/core/log.hpp"
#include "r3d/io/openmvg_scene.hpp"
#include "r3d/pipeline/dense.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: densify_example sfm_data.bin cloud.ply [preset]\n");
    return 2;
  }
  try {
    const r3d::Scene scene = r3d::read_openmvg_scene(argv[1]);

    // A preset, then single options changed as needed.
    r3d::DenseOptions options = r3d::dense_preset(r3d::parse_dense_preset(argc > 3 ? argv[3] : "normal")
                                                      .value_or(r3d::DensePreset::normal));
    options.memory_budget = std::size_t{4} << 30;

    // Library messages to a GUI log window would go through r3d::set_log_sink.
    r3d::set_log_level(r3d::LogLevel::warn);

    // A GUI keeps the token and calls request() from its cancel button.
    r3d::CancellationToken cancel;
    r3d::DenseCallbacks callbacks;
    callbacks.cancel = &cancel;
    callbacks.progress = [](const r3d::DenseProgress& p) {
      std::printf("\r%5.1f%%  %-12s %zu/%zu", 100.0 * p.fraction, std::string(r3d::to_string(p.stage)).c_str(),
                  p.done, p.total);
      std::fflush(stdout);
    };

    r3d::CloudLayoutOptions layout;
    layout.layout = r3d::CloudLayout::poisson;
    r3d::DenseReport report;
    r3d::densify_to_ply(scene, options, argv[2], layout, callbacks, &report);
    std::printf("\n%zu points in %.1f s\n", report.fusion.output_points, report.total_seconds);
  } catch (const r3d::Cancelled&) {
    std::fprintf(stderr, "\ncancelled\n");
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\nerror: %s\n", e.what());
    return 1;
  }
  return 0;
}
