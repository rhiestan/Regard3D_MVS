# Benchmarks

`bench_stereo [width height disparities]` measures the Census-SGM stages on random images (default
1024x768, 128 disparities): Census transform, cost rows, the 4-path row kernel on a cache-resident row,
cost + 8-path aggregation of one tile, and the complete matcher (WTA, left-right check, filters, tiling)
on 1 and on all threads. One cell is one pixel x one disparity.

## Results

2026-10-25, AVX2 target, 16-core / 32-thread desktop CPU, Windows 11. Mcell/s, best of 2–3 runs.

| Stage (1 thread unless noted) | MSVC 19.44 | clang-cl 22 |
|---|---|---|
| Census transform (Mpixel/s) | 238 | 351 |
| Cost rows | 2651 | 2639 |
| 4-path row kernel, cache-resident | 2387 | 3308 |
| Cost + 8-path aggregation | 377 | 450 |
| Complete matcher | 280 | 326 |
| Complete matcher, 32 threads | 609 | 705 |
| Scalar reference: aggregation | 29 | 28 |

Notes:
- The aggregation runs at about 1/3 of the cache-resident kernel speed: path buffers and the aggregated
  volume (2 bytes per cell) stream through L2/L3 and memory.
- Within one pair, parallelism is limited to the number of tiles (4 at 1024x768); the MVS pipeline
  processes pairs in parallel.
- The large gains are expected from Phase 3 (hierarchical, per-pixel disparity ranges), which cuts the
  number of cells by roughly an order of magnitude on real scenes.
