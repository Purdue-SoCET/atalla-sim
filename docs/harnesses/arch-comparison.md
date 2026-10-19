# Comparing the two architectures

The same harness runs both. `TiledTPUCosim` and its reuse subclass
`MNReuseBlockedTPUCosim` take `systolic_array` and `spad_num_tiles`, so the work
decomposition, tile scheduling and reuse policy are **identical** across a
comparison and any cycle difference is the hardware.

| | old (default) | new |
|---|---|---|
| systolic array | TPU, grouped MAC + 4-input adder | MEISSA, multiplier grid + column adder trees |
| scratchpad | 2 pads x 1 MB | 4 pads x 0.5 MB |
| prefetch slots | 2 | 4 |
| VLSUs reached | 2 | 4 |
| total capacity | 2 MB | 2 MB |

Capacity is a property of the chip, not of how it is cut up, so the pad count
splits the same 2 MB into more, smaller pads — each with its own frontend,
backend and VLSU. The default is the old configuration, which is what every
number in `docs/results/` was measured on.

## Measured

Verified at sizes that finish in seconds. Both configurations produce the
correct GEMM against the tiled-fp16 reference.

| harness | size | old | new | delta |
|---|---|---|---|---|
| tiled | 64 | 7,368 | 7,328 | −0.5% |
| tiled | 128 | 50,400 | 50,032 | −0.7% |
| reuse 2x2 | 64 | 5,244 | 4,983 | **−5.0%** |
| reuse 2x2 | 128 | 39,964 | 37,887 | **−5.2%** |

The plain tiled harness barely moves; the reuse harness gains ~5% and does so
consistently at both sizes. That shape makes sense: the win comes from four
prefetch slots instead of two, which only helps when there is enough resident
work to overlap. DRAM bandwidth is unchanged by design — one burst per cycle
across all backends — so this is a latency-hiding gain, not a bandwidth one.

## Running the full 1024x1024

Roughly 22M cycles and ~75-85 minutes per run on one core. The four runs below
are independent and can go in parallel.

Build the native kernel once per node:

```bash
cd /home/asicfab/a/socet149/atalla-sim && make native
```

Plain tiled, old and new:

```bash
PYTHONPATH=src python tests/atalla/test_scratchpad_vector_core_sysarr_tpu_tiled_1024.py --matrix-size 1024 --tile-size 32 --systolic-array tpu --spad-pads 2 --log-dir logs/tiled1024_tpu_2pad
```

```bash
PYTHONPATH=src python tests/atalla/test_scratchpad_vector_core_sysarr_tpu_tiled_1024.py --matrix-size 1024 --tile-size 32 --systolic-array meissa --spad-pads 4 --log-dir logs/tiled1024_meissa_4pad
```

Blocked M/N with weight and activation reuse:

```bash
PYTHONPATH=src python tests/atalla/test_scratchpad_vector_core_sysarr_tpu_tiled_1024_blocked_mn.py --matrix-size 1024 --tile-size 32 --weight-reuse-m 4 --activation-reuse-n 4 --systolic-array tpu --spad-pads 2 --log-dir logs/blocked1024_tpu_2pad
```

```bash
PYTHONPATH=src python tests/atalla/test_scratchpad_vector_core_sysarr_tpu_tiled_1024_blocked_mn.py --matrix-size 1024 --tile-size 32 --weight-reuse-m 4 --activation-reuse-n 4 --systolic-array meissa --spad-pads 4 --log-dir logs/blocked1024_meissa_4pad
```

Each writes `stats.log`, `gemm_cycles.log`, `sdma_load_cycles.log`, `gantt.log`
and `schedule.log` into its log directory. Both runs raise on a wrong GEMM, so a
run that finishes has already checked itself.

To compare afterwards:

```bash
for d in logs/*1024_*; do echo "$d $(grep -m1 '^\[stats\] cycles' $d/stats.log)"; done
```

## Reading the stats

`stats.log` records `systolic_array`, `spad_pads`, `spad_bank_size`,
`vls_count` and `prefetch_slots`, so a log says which machine produced it.

It also carries `sa_stats_not_applicable`. MEISSA has no partial-sum adder and
nothing accumulates between cells, so `pe_psum_adds` and the `psum_shift` byte
counter are listed there rather than reported as measurements of zero. Every
other array counter — multiply/add ops, active PEs, peak PEs, internal shift
bytes, output bytes, saturation and overflow — is real on both.
