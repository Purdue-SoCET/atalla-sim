# sysarr TPU Sweeps

These scripts automate the parameter sweeps for
`tests/atalla/test_scratchpad_vector_core_sysarr_tpu.py` without modifying the
test itself.

## Runner

Run a quick pass:

```bash
python3.11 tools/run_sysarr_tpu_sweeps.py --quick
```

Run the full talk-oriented sweep set:

```bash
python3.11 tools/run_sysarr_tpu_sweeps.py
```

Choose only a subset of sweeps:

```bash
python3.11 tools/run_sysarr_tpu_sweeps.py --sweeps tile dram_latency spad_frontend_queue_size
```

Outputs land in `logs/sysarr_tpu_sweeps/` by default:

- `results.csv`
- `results.json`
- one JSON file per run

The runner retries timeout-like failures with larger `max_cycles` budgets.
If a configuration still fails, it is recorded with `status=failed` instead of
aborting the entire sweep.

## Plotter

The plotter reads `results.csv` and writes one PNG per sweep.

```bash
python3.11 -m pip install matplotlib
python3.11 tools/plot_sysarr_tpu_sweeps.py
```

Plots are written to `logs/sysarr_tpu_sweeps/plots/`.
The plotter ignores rows whose `status` is not `ok`.

## Blocked M/N Queue Sweep

For the blocked tiled TPU harness, there is a dedicated runner for the
`M=8, N=32` frontend-queue sweep:

```bash
python3.11 tools/run_blocked_mn_spad_frontend_queue_sweep.py
```

By default it runs `weight_reuse_m=8`, `activation_reuse_n=32`, and sweeps
`spad_frontend_queue_size` from `1` through `8`.

Outputs land in `logs/blocked_m8_n32_spad_frontend_queue_sweep/`:

- `results.csv`
- `results.json`
- one JSON file per queue-size point

If you also want the per-run harness logs and PNGs, add:

```bash
python3.11 tools/run_blocked_mn_spad_frontend_queue_sweep.py --write-run-logs
```

To graph total cycles and GSAU RD queue depth versus frontend queue size, use:

```bash
python3.11 tools/plot_blocked_mn_spad_frontend_queue_sweep.py \
	--input-dir logs/blocked_m8_n32_spad_frontend_queue_sweep
```

This writes three PNGs under `logs/blocked_m8_n32_spad_frontend_queue_sweep/plots/`:

- `cycles_vs_spad_frontend_queue_size.png`
- `gsau_rd_queue_vs_spad_frontend_queue_size.png`
- `blocked_mn_spad_frontend_queue_sweep_summary.png`

For the GSAU RD queue plots, the average line is normalized by valid MAC cycles rather than total end-to-end cycles.
That metric is emitted directly by the updated harness; older sweep JSONs do not contain it and should be rerun before plotting.

## Blocked M/N Roofline Sweep

For the `1024x1024` blocked reuse-policy roofline comparison, use:

```bash
python3.11 tools/run_blocked_mn_reuse_roofline_sweep.py
python3.11 tools/plot_blocked_mn_reuse_roofline.py
```

The detailed workflow, outputs, and metric definitions are documented in
`docs/sweeps/blocked-mn-roofline.md`.
The runner now writes each invocation into a fresh timestamped run directory
under `logs/roofline_study/blocked_mn_reuse_1024/`, and the plotter resolves
the latest run automatically when `--input-dir` is omitted.

## What Gets Graphed

Each sweep plot includes:

- total end-to-end cycles
- stacked phase breakdown
- MAC utilization and throughput
- max queue depths for key bottleneck queues
