#!/usr/bin/env python3
"""Samples a triangle mesh uniformly by area, for point-cloud benchmarks
(the ETH3D evaluation takes points only).

  python sample_mesh.py mesh.ply points.ply [--spacing 0.002] [--seed 1]

About one point per spacing^2 of surface; triangles get floor(area /
spacing^2 + u) points, u uniform in [0, 1), at uniform random barycentric
positions. Deterministic for a given seed.
"""

import argparse
import sys
import time

import numpy as np

import plyio


def sample(xyz, tri, spacing, seed, out_path, chunk=1 << 20):
    """Writes the samples to out_path (float x y z) in chunks of triangles; returns (points, area)."""
    rng = np.random.default_rng(seed)
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    count = np.floor(area / (spacing * spacing) + rng.random(len(area))).astype(np.int64)
    total = int(count.sum())
    with open(out_path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {total}\n"
                 "property float x\nproperty float y\nproperty float z\nend_header\n").encode("ascii"))
        for begin in range(0, len(tri), chunk):
            end = min(len(tri), begin + chunk)
            idx = np.repeat(np.arange(begin, end), count[begin:end])
            r1 = np.sqrt(rng.random(len(idx)))
            r2 = rng.random(len(idx))
            w0, w1, w2 = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
            points = w0[:, None] * a[idx] + w1[:, None] * b[idx] + w2[:, None] * c[idx]
            f.write(points.astype("<f4").tobytes())
    return total, float(area.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh")
    parser.add_argument("output")
    parser.add_argument("--spacing", type=float, default=0.002, help="mean point distance (scene units)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    start = time.time()
    ply = plyio.read_ply(args.mesh)
    xyz = plyio.positions(ply["vertex"])
    tri = plyio.triangles(ply)
    if len(tri) == 0:
        sys.exit(f"{args.mesh}: no triangles")
    points, area = sample(xyz, tri, args.spacing, args.seed, args.output)
    print(f"{len(xyz)} vertices, {len(tri)} triangles, area {area:.4g} -> {points} points, "
          f"{time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
