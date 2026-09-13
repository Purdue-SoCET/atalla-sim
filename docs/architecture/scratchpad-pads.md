# Scratchpad Pads

The chip's local memory is **one 2 MB scratchpad split into four 0.5 MB pads**.
Each pad has its own bank array, its own crossbar pair, its own frontend and its
own DRAM backend, and is paired 1:1 with one of the vector core's four VLSUs.

## Why

The vector core has four VLSUs (`rtl/params.py: NUM_SCPADS = 4`). Before this,
the platform built a scratchpad with two frontends, so
`min(len(vc.vls_units), len(spad.frontends))` was 2 — **VLSU 2 and VLSU 3 issued
into nothing.** One pad per VLSU gives every load/store unit a private path to
memory.

## Shape

```
   DRAM  ── one channel, one burst per cycle, shared by all four ──┐
                                                                   │
   ┌────────────┬────────────┬────────────┬────────────┐           │
   │ Backend 0  │ Backend 1  │ Backend 2  │ Backend 3  │ ◄─────────┘
   ├────────────┼────────────┼────────────┼────────────┤
   │  pad 0     │  pad 1     │  pad 2     │  pad 3     │  0.5 MB each
   │  32 banks  │  32 banks  │  32 banks  │  32 banks  │  8192 slots
   │  xbar pair │  xbar pair │  xbar pair │  xbar pair │
   ├────────────┼────────────┼────────────┼────────────┤
   │ Frontend 0 │ Frontend 1 │ Frontend 2 │ Frontend 3 │
   ├────────────┼────────────┼────────────┼────────────┤
   │  VLSU 0    │  VLSU 1    │  VLSU 2    │  VLSU 3    │
   └────────────┴────────────┴────────────┴────────────┘
                          │
              Vector Register File (Veggie)
                          │
                        GSAU
                          │
                  SystolicArrayTPU
```

The pad index is the same integer everywhere:

```
pad p == spad.tiles[p] == spad.frontends[p] == spad.backends[p]
      == platform.backends[p] == platform.vls_bridges[p]
      == vc.vls_units[p] == the "vls": p field of an enqueue_memory op
```

## Addressing

Each pad has its **own local slot space**, `0 .. bank_size-1`. The bridge passes
`tile_id=frontend_id` on every access, so a VLSU only ever sees its own pad — slot
0 of pad 0 and slot 0 of pad 3 are different storage.

Backends build their own addresses, unchanged:

```python
platform.backends[PAD_WGT].driver_to_backend_start_load(
    base_sp_addr=0, base_dram_addr=0x2000, rows=32, cols=32)
```

`base_sp_addr` is a slot index inside that pad. There is no address-generator
layer; callers compute `base_sp` and `base_dram` themselves.

## Roles

`PAD_ACT=0`, `PAD_WGT=1`, `PAD_PSUM=2`, `PAD_OUT=3` in
`atalla/sysarr_tpu_system.py`. A convention for how software lays the chip out —
the pads are identical and nothing in the hardware model checks them.

`SysArrTPUSystem` follows it: activations load through VLSU 0, weights through
VLSU 1, outputs store through VLSU 3.

The 1024 GEMM harness does **not**. It uses pads as double-buffer prefetch slots
and pins its own two-pad geometry (`spad_num_tiles=SPAD_NUM_TILES`), so its
published results still describe the configuration they were measured on.

## DRAM bandwidth does not scale with pad count

This is the thing to understand before reading a sweep.

`SharedDRAMBurstChannel` is a single launch slot: **one burst per cycle across all
backends**, whatever their number. So:

| | 2 pads | 4 pads |
|---|---|---|
| aggregate DRAM bandwidth | `dram_burst_bytes`/cycle | **same** |
| share per backend | 1/2 | 1/4 |
| time to fill one pad | t | **2t** |

Four pads buy *scratchpad-side* parallelism — four frontends, four crossbar pairs,
four VLSUs feeding the register file — not more DRAM bandwidth. A workload already
DRAM-bound will not get faster.

Fairness comes from `RoundRobinBackendTicker`, which rotates the starting backend
each cycle (`cycle % len(backends)`). The platform must tick the backends *through*
it: children tick in registration order, so adding backends to the tree directly
would hand backend 0 the channel every cycle and let backend 3 issue only when the
other three are idle.

## Configuring

```python
platform = build_tpu_platform()                      # 4 pads x 0.5 MB = 2 MB
platform = build_tpu_platform(spad_num_tiles=2,      # a different geometry
                              spad_bank_size=16384)  # still 2 MB, two pads
platform = build_tpu_platform(spad_bank_size=128)    # small, for wiring tests
```

The default allocates `4 x 32 x 8192` slots, which costs about 0.26 s and 8.6 MB
per build (`SRAMBank.mem` is a dense list). Tests that only check wiring should
pass a small `spad_bank_size`.

`spad.total_bytes` and `spad.tile_bytes` report the built capacity.
