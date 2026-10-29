#!/usr/bin/env python3
"""End to end: dense cloud -> PoissonRecon or FSSR -> trimmed mesh -> texrecon.

  python r3d_mesh.py SCENE --out-dir DIR [--image-dir IMAGES] [--tools TOOLS]
                     [--mesher poisson|fssr] [--cloud CLOUD.ply] [--no-texture]
                     [-- DENSE OPTIONS]

SCENE is what r3dmvs reads (OpenMVG sfm_data, or a COLMAP text model
directory). Without --cloud, "r3dmvs dense" runs first; arguments after "--"
go to it (e.g. -- --resolution 0.5). The external programs (PoissonRecon,
SurfaceTrimmer, fssrecon, meshclean, texrecon) are looked up in --tools
(searched recursively, e.g. Regard3D's external_tools folder), then on PATH.

Outputs in DIR: cloud.ply (the dense cloud), <mesher>_in.ply (the cloud in
the mesher's layout), <mesher>_raw.ply, mesh.ply (trimmed and cleaned),
texrecon/ (images and cameras) and textured.obj (+ .mtl and textures).

Poisson: the octree depth defaults to the level whose cells are about
--poisson-cell x the median point footprint, points are weighted by their
number of fused depths (--poisson-confidence), the mesh is trimmed with
SurfaceTrimmer at --trim (density, in octree levels). FSSR: meshclean
removes vertices below --clean-threshold confidence and components with
fewer than --component-size vertices. Both: faces of zero area are removed
(texrecon crashes on them).
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plyio  # noqa: E402

EXE = ".exe" if os.name == "nt" else ""


def find_tool(name, tools_dir, required=True):
    if tools_dir:
        for root, _, files in os.walk(tools_dir):
            for f in files:
                if f.lower() == (name + EXE).lower():
                    return str(Path(root) / f)
    found = shutil.which(name)
    if found is None and required:
        sys.exit(f"cannot find {name} (give --tools or put it on PATH)")
    return found


def run(cmd, log):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    start = time.time()
    with open(log, "ab") as f:
        f.write(("$ " + " ".join(str(c) for c in cmd) + "\n").encode())
        f.flush()
        result = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"{Path(cmd[0]).name} failed with code {result.returncode}, see {log}")
    seconds = time.time() - start
    print(f"  {seconds:.1f} s", flush=True)
    return seconds


def poisson_depth(cloud_path, cell_factor):
    """Octree depth whose cells are about cell_factor x the median footprint, at most 12.

    PoissonRecon's cube is the bounding box enlarged by --scale (1.1), so far
    outliers ask for a deeper octree; beyond 12 the meshes did not improve on
    ETH3D but grew 4x per level."""
    vertex = plyio.read_ply(cloud_path)["vertex"]
    xyz = plyio.positions(vertex)
    extent = float((xyz.max(axis=0) - xyz.min(axis=0)).max()) * 1.1
    footprint = float(np.median(vertex["value"])) if "value" in vertex.dtype.names else extent / 4096
    depth = round(math.log2(extent / (cell_factor * footprint)))
    return max(6, min(depth, 12)), extent, footprint


def remove_degenerate_faces(in_path, out_path):
    """Drops faces of zero area (in float precision) and unused vertices.

    texrecon crashes on zero-area faces (zero-length face normals); meshclean
    keeps them."""
    ply = plyio.read_ply(in_path)
    vertex, face = ply["vertex"], ply["face"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    tri = plyio.triangles(ply)
    normal = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])
    keep = np.linalg.norm(normal, axis=1) > 0
    used = np.unique(tri[keep])
    remap = np.full(len(vertex), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    face = face[keep].copy()
    name = "vertex_indices" if "vertex_indices" in face.dtype.names else "vertex_index"
    face[name] = remap[tri[keep]]
    plyio.write_ply(out_path, {"vertex": vertex[used], "face": face})
    print(f"  removed {int((~keep).sum())} zero-area faces, {len(vertex) - len(used)} unused vertices")


def main():
    argv = sys.argv[1:]
    dense_args = []
    if "--" in argv:
        k = argv.index("--")
        argv, dense_args = argv[:k], argv[k + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-dir")
    parser.add_argument("--tools", help="directory with the external programs (searched recursively)")
    parser.add_argument("--r3dmvs", help="r3dmvs executable (default: --tools, PATH, next to this script's build)")
    parser.add_argument("--mesher", choices=["poisson", "fssr"], default="poisson")
    parser.add_argument("--cloud", help="existing dense cloud from 'r3dmvs dense' (skips densification)")
    parser.add_argument("--no-texture", action="store_true")
    parser.add_argument("--threads", type=int, default=0, help="PoissonRecon threads (0: default)")
    parser.add_argument("--depth", type=int, default=0, help="Poisson octree depth (0: automatic)")
    parser.add_argument("--poisson-cell", type=float, default=6.0, help="automatic depth: cell / footprint")
    parser.add_argument("--poisson-confidence", type=float, default=4.0,
                        help="fused depths per point for full Poisson weight (0: all points alike)")
    parser.add_argument("--point-weight", type=float, default=4.0, help="Poisson screening weight")
    parser.add_argument("--trim", type=float, default=10.0, help="SurfaceTrimmer density threshold (0: no trimming)")
    parser.add_argument("--fssr-scale", type=float, default=8.0, help="FSSR sample scale / footprint")
    parser.add_argument("--clean-threshold", type=float, default=1.0, help="meshclean confidence threshold")
    parser.add_argument("--component-size", type=int, default=10000, help="meshclean minimum component size")
    args = parser.parse_args(argv)

    out = Path(args.out_dir).resolve()  # texrecon resolves relative scene paths wrongly
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.txt"
    r3dmvs = args.r3dmvs or find_tool("r3dmvs", args.tools, required=False)
    if r3dmvs is None:
        sys.exit("cannot find r3dmvs (give --r3dmvs)")
    r3dmvs = str(Path(r3dmvs).resolve())  # Windows does not search relative paths with "/"
    scene = [args.scene] + (["--image-dir", args.image_dir] if args.image_dir else [])
    times = {}

    cloud = Path(args.cloud) if args.cloud else out / "cloud.ply"
    if not args.cloud:
        times["dense"] = run([r3dmvs, "dense", *scene, "-o", cloud, *dense_args], log)

    mesh = out / "mesh.ply"
    raw = out / f"{args.mesher}_raw.ply"
    if args.mesher == "poisson":
        poisson_in = out / "poisson_in.ply"
        times["export"] = run([r3dmvs, "export", "--cloud", cloud, "--for", "poisson", "--poisson-full-confidence",
                               args.poisson_confidence, "-o", poisson_in], log)
        depth = args.depth
        if depth <= 0:
            depth, extent, footprint = poisson_depth(cloud, args.poisson_cell)
            print(f"  Poisson depth {depth} (extent {extent:.4g}, median footprint {footprint:.4g})")
        cmd = [find_tool("PoissonRecon", args.tools), "--in", poisson_in, "--out", raw, "--depth", depth,
               "--pointWeight", args.point_weight, "--density", "--color", 16]
        if args.poisson_confidence > 0:
            cmd.append("--confidence")
        if args.threads > 0:
            cmd += ["--threads", args.threads]
        times["poisson"] = run(cmd, log)
        trimmed = out / "poisson_trimmed.ply"
        if args.trim > 0:
            times["trim"] = run([find_tool("SurfaceTrimmer", args.tools), "--in", raw, "--out", trimmed,
                                 "--trim", args.trim], log)
        else:
            shutil.copyfile(raw, trimmed)
        remove_degenerate_faces(trimmed, mesh)
    else:
        fssr_in = out / "fssr_in.ply"
        times["export"] = run([r3dmvs, "export", "--cloud", cloud, "--for", "fssr", "--fssr-scale", args.fssr_scale,
                               "-o", fssr_in], log)
        times["fssr"] = run([find_tool("fssrecon", args.tools), fssr_in, raw], log)
        cleaned = out / "fssr_cleaned.ply"
        times["clean"] = run([find_tool("meshclean", args.tools), "-t", args.clean_threshold, "-c",
                              args.component_size, raw, cleaned], log)
        remove_degenerate_faces(cleaned, mesh)

    if not args.no_texture:
        tex_dir = out / "texrecon"
        times["texrecon scene"] = run([r3dmvs, "export", *scene, "--texrecon-dir", tex_dir], log)
        times["texrecon"] = run([find_tool("texrecon", args.tools), tex_dir, mesh, out / "textured"], log)

    print("times: " + ", ".join(f"{k} {v:.1f} s" for k, v in times.items()))
    print(f"mesh: {mesh}" + ("" if args.no_texture else f"\ntextured: {out / 'textured.obj'}"))


if __name__ == "__main__":
    main()
