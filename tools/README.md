# Tools

Python 3 scripts (numpy) around `r3dmvs` and the external meshing and
texturing programs. R3DMVS writes a point cloud only (decision D3); these
scripts turn it into a textured mesh with free tools.

| Script | Purpose |
|---|---|
| `r3d_mesh.py` | End to end: `r3dmvs dense` → PoissonRecon + SurfaceTrimmer or FSSR + meshclean → texrecon |
| `sample_mesh.py` | Uniform area sampling of a triangle mesh, for point-based benchmarks (ETH3D) |
| `plyio.py` | Minimal PLY reader / writer used by the scripts |

## External programs

* [PoissonRecon / SurfaceTrimmer](https://github.com/mkazhdan/PoissonRecon) (MIT)
* [MVE](https://github.com/simonfuhrmann/mve) `fssrecon`, `meshclean` (BSD 3-clause)
* [mvs-texturing](https://github.com/nmoehrle/mvs-texturing) `texrecon` (BSD 3-clause)

Regard3D ships all of them in its `external_tools` folder; pass that folder
with `--tools` (it is searched recursively), or put the programs on `PATH`.

## End to end

```sh
# Poisson (default): dense cloud, octree depth from the point footprint, trimmed, textured
python tools/r3d_mesh.py sfm_data.bin --out-dir out --tools <external_tools> -- --resolution 0.5 --voxel-factor 2

# FSSR, from an existing cloud
python tools/r3d_mesh.py sfm_data.bin --out-dir out --tools <external_tools> --mesher fssr --cloud cloud.ply
```

Results in `out/`: `cloud.ply`, `<mesher>_in.ply`, `<mesher>_raw.ply`,
`mesh.ply` (trimmed / cleaned), `texrecon/` and `textured.obj` with its
material and texture atlases; `log.txt` holds the programs' output.

Defaults, tested on ETH3D (docs/DECISIONS.md, D42):

* A cloud with `--voxel-factor 2` meshes as well as the full-density cloud
  and is 2.5-4x smaller, so meshing and texturing are several times faster.
* Poisson: octree cells of about 6x the median point footprint
  (`--poisson-cell`), points weighted by their number of fused depths
  (normals scaled to full length at 4, `PoissonRecon --confidence`),
  SurfaceTrimmer at 10. A finer octree does not score better and makes
  texrecon very slow (tens of millions of faces).
* FSSR: sample scale 8x the footprint (`--fssr-scale`); smaller scales are
  slower and noisier. meshclean with its default threshold 1.
* Faces of zero area are removed before texturing: texrecon (2017 builds)
  crashes on them in "Generating texture patches", and meshclean keeps them.
  texrecon can still crash in that step on some Poisson meshes (seen once, on
  ETH3D terrains); texture the FSSR mesh instead in that case.
* PoissonRecon's thread pool spins when the machine is oversubscribed; give
  `--threads` when other heavy jobs run at the same time.

The same steps by hand (without the zero-area face removal):

```sh
r3dmvs dense sfm_data.bin -o cloud.ply --resolution 0.5 --voxel-factor 2
r3dmvs export --cloud cloud.ply --for poisson -o poisson_in.ply
PoissonRecon --in poisson_in.ply --out poisson_raw.ply --depth 11 --density --confidence --color 16
SurfaceTrimmer --in poisson_raw.ply --out mesh.ply --trim 10
r3dmvs export sfm_data.bin --texrecon-dir texrecon
texrecon texrecon mesh.ply textured
```

`r3dmvs dense --for poisson|fssr` writes the mesher layout directly.

## Mesh benchmark

```sh
python tools/sample_mesh.py mesh.ply mesh_points.ply --spacing 0.002
ETH3DMultiViewEvaluation --reconstruction_ply_path mesh_points.ply --ground_truth_mlp_path <scene>/dslr_scan_eval/scan_alignment.mlp --tolerances 0.01,0.02,0.05,0.1
```
