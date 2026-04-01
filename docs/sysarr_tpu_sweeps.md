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

## What Gets Graphed

Each sweep plot includes:

- total end-to-end cycles
- stacked phase breakdown
- MAC utilization and throughput
- max queue depths for key bottleneck queues
