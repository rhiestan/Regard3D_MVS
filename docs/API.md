# Using R3DMVS from another program

R3DMVS can be used in two ways: linked as a C++ library (`r3d`, a static library), or as the
`r3dmvs` command line program started as a child process. Both run the same pipeline
(`r3d::densify`); the library adds a progress callback and cancellation from the calling program,
the command line program reports progress on stdout and stops cleanly on Ctrl+C / SIGTERM.

## Linking the library

The library is a CMake project meant to be added to the parent project's source tree:

```cmake
add_subdirectory(path/to/Regard3D_MVS r3dmvs)    # or FetchContent_Declare / FetchContent_MakeAvailable
target_link_libraries(my_app PRIVATE R3D::r3d)
```

As a subproject it builds only the library: tests, benchmarks, examples and the command line program
are off unless `R3D_BUILD_TESTS`, `R3D_BUILD_BENCH`, `R3D_BUILD_EXAMPLES` or `R3D_BUILD_APPS` are set.
The parent needs C++20 (`<format>`, see the README for compiler versions).

**Eigen.** Eigen types appear in the public headers (poses, points). Eigen must therefore be the same
copy, with the same alignment settings, in the library and in the program: two copies in one binary
corrupt the objects passed between them. If the parent project defines the target `Eigen3::Eigen`
before `add_subdirectory`, the library uses it; otherwise `-DR3D_USE_SYSTEM_DEPS=ON` finds an installed
Eigen 3.4, and without either Eigen 3.4.0 is downloaded. Highway, nlohmann/json and stb are private
dependencies and never appear in the headers. `EIGEN_MPL2_ONLY` is set for the library's own sources
only.

`examples/densify_example.cpp` is a complete program in 50 lines.

## Reading a scene

```cpp
#include "r3d/io/openmvg_scene.hpp"   // OpenMVG sfm_data.json / .bin
#include "r3d/io/colmap_scene.hpp"    // COLMAP text model directory

r3d::OpenMvgReadOptions read;
read.image_dir = "path/to/images";    // optional: overrides the sfm_data root_path
const r3d::Scene scene = r3d::read_openmvg_scene("sfm_data.bin", read);
```

Only views with a pose and a supported camera model are kept; `OpenMvgImportReport` (third argument)
says how many were dropped. `r3d::Scene` can also be filled directly (cameras, views with image paths
and poses, sparse points with their observations, which the pair selection uses).

## Running the pipeline

```cpp
#include "r3d/pipeline/dense.hpp"

r3d::DenseOptions options = r3d::dense_preset(r3d::DensePreset::normal);
options.memory_budget = std::size_t{4} << 30;   // depth maps and images held in memory
options.work_dir = project_dir / "dense_work";  // optional, see below

r3d::CloudLayoutOptions layout;                 // full (default), poisson or fssr
layout.layout = r3d::CloudLayout::poisson;
r3d::DenseReport report;
r3d::densify_to_ply(scene, options, "dense.ply", layout, callbacks, &report);
```

`densify_to_ply` writes the cloud while it is fused, so the memory does not grow with the cloud; a
partly written file is removed when the run fails or is cancelled. `densify` hands the points to a
callback in parts instead (`PointCloud` with positions relative to `PointCloud::origin`, normals,
colours, and the scalars `value` (point footprint) and `confidence` (fused depths); with
`options.fusion.with_views` also the source ids of the views that see each point).

### Presets

| Preset | Depth maps (longer side) | Refinement | ETH3D (6048 px images): F1 at 2 cm, time |
|---|---|---|---|
| `fast` | 1600 px | off | 0.61-0.81, 35-115 s (previews) |
| `normal` | 3200 px | on | 0.64-0.84, 95-360 s (default) |
| `high` | 6400 px | off | 0.61-0.82, 460-1850 s |
| `ultra` | full image size | off | same as `high` on ETH3D |

On ETH3D, depth maps larger than `normal`'s are slower and score lower at 2 cm (D53). `high` and `ultra`
skip the refinement, which gains nothing at that size; set `options.refine = true` (`--refine`) to keep it.

Measured quality and run times on ETH3D are in `docs/DECISIONS.md` (D46, D53). Every preset is a complete
`DenseOptions`; single fields can be changed afterwards. `depth.max_image_size` limits the depth map
size whatever the image size (0: no limit), `depth.resolution` scales relative to the image.

### Memory

`memory_budget` bounds the depth maps and refinement images held at once (8 GB by default); the
others wait in `work_dir`, or in a temporary directory that is removed at the end.
`fusion.memory_budget` bounds the point data of one fusion pass (2 GB; more passes if needed). The
stereo matcher's tiles use `depth.stereo.memory_budget` (2 GB).

### Work directory and reuse

With a `work_dir`, the depth maps of every view are kept there (`depth_<id>.r3dd` and
`depth_<id>_refined.r3dd`). `reuse` loads the matched maps instead of matching again, `reuse_refined`
the refined maps (skipping matching and refinement), so a cancelled run can be resumed and the
filter and fusion options can be changed cheaply. The cache does not record the options: reuse only
maps computed with the same depth map and refinement options. `debug_images` writes depth PNGs of
every stage there.

## Progress and cancellation

```cpp
r3d::CancellationToken cancel;          // the GUI's cancel button calls cancel.request()
r3d::DenseCallbacks callbacks;
callbacks.cancel = &cancel;
callbacks.progress = [&](const r3d::DenseProgress& p) {
  // p.fraction: 0..1 of the whole run; p.stage, p.done / p.total within the stage
};
```

* The progress callback runs on the thread that called `densify`, once per view and stage (fusion:
  per depth map read in each pass). It must return quickly; a GUI posts the values to its event loop.
  `fraction` is estimated from the typical share of each stage in the run time and ends at exactly 1.
* `request()` may be called from any thread. The pipeline polls the token before every view and
  before every chunk of parallel work (image tiles, rows), so it stops within a fraction of a second
  and throws `r3d::Cancelled`. The token is not reset by the pipeline (`reset()` to reuse it).
* Errors are reported as exceptions derived from `r3d::Error` (`r3d::IoError` for files);
  `r3d::Cancelled` is one of them, so catch it first.

The same token works for every other library function that uses `parallel_for`:
`r3d::CancellationScope scope(&token);` makes it the calling thread's token for the scope.

## Logging

Stage summaries and one line per view are logged at `info` level, to stderr by default.
`r3d::set_log_sink([](r3d::LogLevel level, std::string_view message) { ... })` forwards them
(e.g. to a log window); the sink is called from the pipeline's threads but never concurrently.
`r3d::set_log_level(r3d::LogLevel::warn)` silences the per-view lines.

## Threads

The library uses one process-wide pool of worker threads (`r3d::default_thread_pool()`, all hardware
threads). `r3d::set_default_thread_count(n)` changes its size; call it only while no library function
runs. Several `densify` calls may run concurrently from different threads; they share the pool, and
each one has its own memory budgets. The pool's threads are separate from OpenMP's.

## Running `r3dmvs` as a child process

```sh
r3dmvs dense sfm_data.bin --image-dir images --preset normal --work-dir work --progress -o dense.ply
```

* `--progress` prints one line per step on stdout, flushed:
  `progress <percent> <stage> <done>/<total>`, e.g. `progress 42.7 depth maps 31/48`, where the
  stage is one of `depth maps`, `refinement`, `consistency`, `visibility`, `fusion`. Log messages
  go to stderr; the scene summary at the start goes to stdout.
* Exit codes: 0 success, 1 error (message on stderr), 130 cancelled.
* Ctrl+C, SIGTERM or, on Windows, Ctrl+Break cancels cleanly (partly written output removed,
  temporary files deleted); a second signal ends the process at once. A parent process on Windows
  starts `r3dmvs` with `CREATE_NEW_PROCESS_GROUP` and sends
  `GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid)`; on Linux and macOS it sends SIGTERM. If the process is killed instead, the output file is
  incomplete, and spilled depth maps stay in the work directory, or in a `r3dmvs-*` directory under
  the system's temporary directory without `--work-dir`; give a work directory inside the project
  for that reason.
* `--preset` sets the defaults of all other options; options given as well override them.
  `r3dmvs dense --preset high --help` lists the resulting values.

For meshing and texturing, see `r3dmvs export` (layouts for PoissonRecon / FSSR, texrecon scene folder)
and [tools/README.md](../tools/README.md).
