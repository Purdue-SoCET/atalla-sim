# Blocked M/N Reuse Roofline Sweeps

This note documents the blocked `1024x1024` GEMM reuse-policy sweep driven by:

- `tools/run_blocked_mn_reuse_roofline_sweep.py`
- `tools/plot_blocked_mn_reuse_roofline.py`

The sweep compares these reuse policies on the tiled TPU harness in
`tests/atalla/test_scratchpad_vector_core_sysarr_tpu_tiled_1024_blocked_mn.py`:

- `Naive:1:1`
- `m2_n4:2:4`
- `m2_n8:2:8`
- `m4_n4:4:4`
- `m4_n8:4:8`

Each point runs as a full simulator pass and writes roofline-ready metrics for
that reuse policy.

When present, the plotter also overlays a supplemental `m8_n32` reference point
from `logs/blocked_m8_n32_spad_frontend_queue_sweep/m8_n32_spad_frontend_queue_size_4.json`.
That row comes from the same blocked harness with the default frontend queue
size, and the plotter derives the algorithmic roofline fields from the raw
`cycles` and `bytes_transmitted` stats so the reuse sweep does not need to be
rerun just to place that point.

## Runner

Run the default sequential sweep:

```bash
python3.11 tools/run_blocked_mn_reuse_roofline_sweep.py
```

The runner is sequential on purpose. It executes one case at a time so the
progress log, output directory, and simulator resource usage stay easy to
reason about.

By default, each invocation creates a fresh timestamped run directory under:

```text
logs/roofline_study/blocked_mn_reuse_1024/
```

The runner also updates `LATEST.txt` in that root so the plotter can find the
most recent completed or partial run automatically.

Override the matrix size, tile size, or case list if needed:

```bash
python3.11 tools/run_blocked_mn_reuse_roofline_sweep.py \
  --matrix-size 1024 \
  --tile-size 16 \
  --cases Naive:1:1 m2_n4:2:4 m2_n8:2:8 m4_n4:4:4 m4_n8:4:8
```

If you also want the harness logs for each run, add:

```bash
python3.11 tools/run_blocked_mn_reuse_roofline_sweep.py --write-run-logs
```

Per-run logs are written under `<output-dir>/<case-name>/`.

If you want to pin a specific run directory instead of using a timestamped one,
pass `--output-dir`. The runner refuses to reuse a non-empty explicit output
directory unless you also pass `--overwrite-output-dir`.

## Outputs

Each run directory contains:

- `results.csv`
- `results.json`
- `manifest.json`
- one JSON file per policy, for example `naive.json` and `m4_n8.json`
- one stdout log per policy, for example `naive.stdout.log`
- one stderr log per policy, for example `naive.stderr.log`

The per-policy JSON files are the main plotting inputs. They also make it easy
to inspect failures without rerunning the full sweep.

The manifest records sweep-wide metadata, including:

- matrix size and tile size
- the case list and case order
- theoretical peak compute in FLOPs per cycle
- empirical peak external bandwidth from the successful runs
- run status, completed case count, and finish timestamp

The runner updates `results.csv`, `results.json`, and `manifest.json` after
every completed case. If a long run is interrupted, the partial outputs are
still usable for inspection.

## Key Metrics

The runner enriches the raw harness stats with the roofline fields used by the
plotter:

- `flops_algo`: algorithmic GEMM work, computed as `2 * N^3`
- `throughput_algo_flops_per_cycle`: algorithmic throughput
- `arithmetic_intensity_external_algo_flops_per_byte`: algorithmic FLOPs per
  external byte
- `arithmetic_intensity_external_micro_flops_per_byte`: microarchitectural
  FLOPs per external byte
- `arithmetic_intensity_internal_algo_flops_per_byte`: algorithmic FLOPs per
  internal byte
- `external_bandwidth_avg_bytes_per_cycle`: average external bandwidth over the
  full run
- `external_bandwidth_active_bytes_per_cycle`: external bandwidth over active
  transfer windows
- `internal_bandwidth_bytes_per_cycle`: estimated internal bandwidth observed by
  the harness

For the roofline plot, the most important axes are:

- x-axis: `arithmetic_intensity_external_algo_flops_per_byte`
- y-axis: `throughput_algo_flops_per_cycle`

## Plotter

Install matplotlib if needed:

```bash
python3.11 -m pip install matplotlib
```

Plot the roofline from the default output directory:

```bash
python3.11 tools/plot_blocked_mn_reuse_roofline.py
```

With no `--input-dir`, the plotter resolves the latest run from
`logs/roofline_study/blocked_mn_reuse_1024/LATEST.txt`.

Choose a different input directory or output PNG explicitly:

```bash
python3.11 tools/plot_blocked_mn_reuse_roofline.py \
  --input-dir logs/roofline_study/blocked_mn_reuse_1024/run_YYYYMMDD_HHMMSS_utc \
  --output logs/roofline_study/blocked_mn_reuse_1024/run_YYYYMMDD_HHMMSS_utc/plots/blocked_mn_reuse_roofline.png
```

The plotter supports two bandwidth ceilings:

- `--bandwidth-source active`
- `--bandwidth-source avg`

The default is `active`, which typically gives the more useful external-memory
roof for comparing these reuse policies.

By default, the plotter auto-loads the `m8_n32` queue-size-4 reference above
when that JSON file exists. If you want to suppress that overlay, add:

```bash
python3.11 tools/plot_blocked_mn_reuse_roofline.py --no-default-references
```

You can also add other points explicitly from raw or roofline-ready per-run
JSON files:

```bash
python3.11 tools/plot_blocked_mn_reuse_roofline.py \
  --reference-json logs/blocked_m8_n32_spad_frontend_queue_sweep/m8_n32_spad_frontend_queue_size_4.json
```

Plots are written to `<input-dir>/plots/` by default.

## Reading the Sweep

- Higher `throughput_algo_flops_per_cycle` means the reuse policy completed the
  same GEMM with fewer cycles.
- Higher `arithmetic_intensity_external_algo_flops_per_byte` means the policy
  extracted more algorithmic work from each external byte.
- If a point moves right more than it moves up, the policy improved external
  traffic reuse more than raw throughput.
- If a point approaches the flat roof, the run is becoming compute-limited.
- If a point stays on the sloped roof, the run is still primarily limited by
  external bandwidth.

## Typical Workflow

Run the sweep, then plot it:

```bash
python3.11 tools/run_blocked_mn_reuse_roofline_sweep.py
python3.11 tools/plot_blocked_mn_reuse_roofline.py
```