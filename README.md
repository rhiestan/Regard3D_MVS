# R3DMVS

Multi-view stereo densification of structure-from-motion results, developed for
[Regard3D](https://www.regard3d.org).

R3DMVS takes a calibrated scene, a [COLMAP](https://colmap.github.io/format.html) sparse model (binary or
text), and computes a dense point cloud with colours, oriented normals, per-point scale and confidence.
The cloud is meant to be meshed and textured with external tools: PoissonRecon, FSSR and texrecon.

How it works:

1. Every view is matched with up to four stereo partners by hierarchical Census + semi-global matching
   on rectified pairs.
2. The depth maps are refined patch by patch (plane fitting on the normalised cross-correlation) and
   checked for consistency and visibility against the neighbouring views.
3. Everything is fused in an octree at the resolution of the best view, and stays within a memory
   budget.

On the ETH3D training scenes (F1 at 2 cm, preset `normal`) R3DMVS scores courtyard 0.82, office 0.68,
pipes 0.64 and terrains 0.84. COLMAP 4.2 at half resolution scores 0.77, 0.37, 0.38 and 0.72.

R3DMVS is released under the [MIT license](LICENSE). All dependencies are permissively licensed.

## Documentation

- [docs/API.md](docs/API.md): using R3DMVS from another program, either as a C++ library (progress
  callbacks, cancellation) or as a child process (progress on stdout, exit codes, clean cancellation).
- [docs/DECISIONS.md](docs/DECISIONS.md): design decisions, with the benchmarks behind them.
- [tools/README.md](tools/README.md): meshing and texturing end to end (PoissonRecon or FSSR, then
  texrecon).
- [bench/README.md](bench/README.md): matcher throughput benchmark.

## Input

A scene is either:

- a COLMAP model directory with `cameras`, `images` and `points3D` as `.bin` or `.txt` (binary wins if
  both exist), or
- a project directory with the model in `sparse/0` or `sparse`.

Image names are looked up under `images/` next to the project or the model, then next to the model.
`--image-dir` overrides this.

Supported camera models: `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`,
`OPENCV_FISHEYE`, and `FULL_OPENCV` without rational terms.

COLMAP and GLOMAP write this format directly, and so do many other tools. Convert OpenMVG scenes
(`sfm_data.bin` / `.json`, e.g. from Regard3D) with OpenMVG's converter:

```sh
openMVG_main_openMVG2Colmap -i sfm_data.bin -o project/sparse -f bin
```

## Usage

```sh
# Dense point cloud. Presets: fast, normal (default), high, ultra.
r3dmvs dense path/to/project --preset normal --work-dir work -o dense.ply

# Cleaner clouds: remove small untextured islands (sky, dark background), and also require two views:
r3dmvs dense path/to/project --weak-segment 0.001 -o dense.ply
r3dmvs dense path/to/project --weak-segment 0.001 --min-sources 2 -o dense.ply

# Point attributes for a mesher (--for poisson / fssr, also on 'dense'), and the scene for texrecon:
r3dmvs export --cloud dense.ply --for poisson -o poisson_in.ply
r3dmvs export path/to/project --texrecon-dir texrecon

# Mesh and texture end to end:
python tools/r3d_mesh.py path/to/project --out-dir mesh --tools path/to/external_tools

# Inspect a scene: summary, stereo pairs, sparse cloud, camera frustums, undistorted images:
r3dmvs scene path/to/project --pairs --sparse-ply sparse.ply --cameras-ply cameras.ply
```

`r3dmvs dense --preset high --help` lists all options and their values under a preset; options you
give override the preset.

How `dense` behaves:

- Depth maps are cached in the work directory, so `--reuse` skips the matching.
- `--progress` prints machine-readable progress lines.
- Ctrl+C cancels cleanly.

## Binary releases

Prebuilt `r3dmvs` programs are attached to the
[GitHub releases](https://github.com/rhiestan/Regard3D_MVS/releases). They need no installation:

| Platform | Requirement |
|---|---|
| Windows | x64 |
| Linux | x64, glibc 2.35 or newer (e.g. Ubuntu 22.04) |
| macOS | Apple Silicon, macOS 13.3 or newer |

The macOS program is not notarized. After downloading it with a browser, remove the quarantine flag:
`xattr -d com.apple.quarantine r3dmvs`.

Releases are built by `.github/workflows/release.yml` when a tag `vX.Y.Z` matching the version in
`CMakeLists.txt` is pushed. The workflow creates a draft release with the packages.

## Building

Requirements:

- CMake ≥ 3.25 and Ninja.
- A C++20 compiler with `<format>`: MSVC 19.30+ (VS 2022), clang-cl, GCC 13+, Clang 17+, or
  AppleClang 15+ (macOS 13.3+).

Dependencies are downloaded at configure time (pinned versions, SHA256-verified):

| Dependency | License | Notes |
|---|---|---|
| Eigen | MPL-2.0 | built with `EIGEN_MPL2_ONLY` |
| Google Highway | Apache-2.0 | |
| CLI11 | BSD-3 | |
| stb_image | public domain / MIT | |
| doctest | MIT | tests only |

```sh
cmake --preset gcc-release          # or: msvc-release, clang-cl-release, clang-release, macos-release
cmake --build --preset gcc-release
ctest --preset gcc-release
```

Notes:

- On Windows, run these from a Visual Studio developer prompt.
- `-DR3D_USE_SYSTEM_DEPS=ON` prefers installed packages (e.g. from vcpkg or a Linux distribution) over
  the downloads.
- Tests on real scenes run when `R3D_TESTDATA_DIR` (default `../testdata`) contains
  `<scene>/sparse/cameras.bin`.
