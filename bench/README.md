# Benchmarks

`bench_stereo [-v] [-m] [-H] [width height disparities]` measures the Census-SGM stages on synthetic images
(default 1024x768, 128 disparities): Census transform, cost rows, the 4-path row kernel on a cache-resident
row, cost + 8-path aggregation of one tile, and the complete matcher over the full range (`levels = 0`) and
hierarchically (automatic levels), on 1 and on all threads. `-m` runs only the complete matchers, `-H` only
the hierarchical one, `-v` logs per-stage timings.

The left image is multi-scale noise (texture on every pyramid level); the right image shows it shifted by a
smooth disparity field from 1/8 to 5/8 of the range. Columns without a match (where the shift leaves the
image) keep wide ranges, as occlusions do in real pairs.

One cell is one pixel x one disparity. Mcell/s of the complete matchers counts the full-range cells
(width x height x disparities), so hierarchical and full-range numbers compare run times directly; "cells
per pixel" are the cells actually aggregated, over all levels and both matching directions.

## Results

2026-10-26, AVX2 target, 16-core / 32-thread desktop CPU, Windows 11. Mcell/s, best of 2–3 runs.

| Stage (1 thread unless noted) | MSVC 19.44 | clang-cl 22 |
|---|---|---|
| Census transform (Mpixel/s) | 230 | 413 |
| Cost rows | 2576 | 2666 |
| 4-path row kernel, cache-resident | 1810 | 2258 |
| Cost + 8-path aggregation | 295 | 354 |
| Complete matcher, full range (315 cells per pixel) | 130 | 150 |
| Complete matcher, full range, 32 threads | 315 | 363 |
| Complete matcher, hierarchical (2 levels, 53 cells per pixel) | 216 | 277 |
| Complete matcher, hierarchical, 32 threads | 401 | 515 |
| Scalar reference: aggregation | 26 | 27 |

Large pair, `bench_stereo -m 4096 3072 1024` (12.6 MP, 1024 disparities), clang-cl:

| Complete matcher | Cells per pixel | 1 thread | 32 threads |
|---|---|---|---|
| Full range | 2757 | 68.1 s | 34.0 s |
| Hierarchical (5 levels) | 70 | 6.6 s | 1.2 s |

On real scenes (`r3dmvs pair`, 32 threads) a 24.7 MP pair with 2528 disparities takes 4.9 s instead of 348 s.

Notes:
- Since Phase 3 the kernels take a label range per pixel. Compared with the Phase 2 kernels (fixed range;
  2387 / 3308 Mcell/s for the row kernel, 377 / 450 for the aggregation) the full-range case is 10-30%
  slower; the hierarchy more than makes up for it.
- The complete matchers include the full right-to-left match of the default left-right check (D25), which
  doubles the work against Phase 2's diagonal check.
- At small label ranges (16-48 labels per pixel) the per-pixel overhead of the row kernel (path pointers,
  predecessor re-alignment, horizontal minima) is comparable to the label loop itself; this is where further
  speed-ups would come from.
- Within one pair, parallelism is limited to the number of tiles (4 at 1024x768); the MVS pipeline
  processes pairs in parallel.
