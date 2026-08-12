# Presentation Graph Notes

This note explains the four PNGs emitted by the blocked
`M/N` tiled TPU harness and by:

```bash
python3.11 tools/plot_tiled_sysarr_tpu_gantt.py \
  --input logs/tiled_1024_m8_n32/gantt.log \
  --tj 0 --tk 0 \
  --presentation-set
```

## Plot Set

The presentation set currently writes these files:

- `presentation_block_overview_tj00_tk00.png`
- `presentation_weight_flow_tj00_tk00.png`
- `presentation_weight_compute_tj00_tk00.png`
- `presentation_reuse_balance_tk00.png`

## 1. Block Overview

- `presentation_block_overview_tjXX_tkXX.png`

- one full reuse block at a high level
- each row is one resident weight tile in that block
- the dark bar at the front is the weight tile being loaded once
- the long lighter bar is the period where activations stream past that same
  resident weight
- the hatched tail is the accumulate and write-back phase after the streamed
  activations finish

- the schedule is weight-stationary at the block level
- we pay to load a weight tile once, then keep it resident while multiple
  activation tiles use it
- the important asymmetry is deliberate: a small set of weights stays put while
  a larger set of activation tiles moves through

## 2. One Weight Flow

- `presentation_weight_flow_tjXX_tkXX.png`


- the life of one resident weight tile and all activation tiles that use it
- the top row is the chosen resident weight tile
- each lower row is one activation tile
- light blue shows DRAM to scratchpad preload
- darker blue shows scratchpad to TPU feed
- gray bars show the coarse compute window for that activation tile
- the hatched tail shows accumulate and write-back after compute


- this is the best plot for explaining reuse under one weight tile
- the weight stays in place while activation tiles arrive one after another
- the activation tiles do not wait for the previous one to fully finish writing
  back before the next one begins moving through the TPU path
- it shows overlap at the schedule level without drowning the audience in queue
  names


## 3. One Weight Compute Detail

- `presentation_weight_compute_tjXX_tkXX.png`


- the same one-weight story as the previous figure, but with the compute phase
  split into more detailed stages
- light blue shows the earlier block-prefetch into scratchpad
- dark blue shows the later scratchpad-to-TPU stream under the resident weight
- light gray shows the coarse kernel envelope
- the weight-colored bar shows only the systolic-array multiply span
- green shows the TPU response returning
- the dotted gray gap shows the result waiting in `pending_accum_rows` for a
  psum read slot on the scratchpad frontend
- brown shows psum reload before the vector-core add path starts
- the hatched gold bar shows the datapath-add and store tail after reload


- this is the plot to use when someone asks where the real multiplication
  happens
- it separates early prefetch from the later TPU stream, so the long blue block
  is no longer mistaken for one continuous arrival path
- it exposes the long wait between TPU response and psum reload, which is where
  the blocked schedule can stall when activation and psum traffic contend for
  the same scratchpad-tile read window
- it still makes pipeline stagger visible, which helps explain why overlap
  exists but overall utilization can still be below the active-window peak

## 4. Reuse Balance

- `presentation_reuse_balance_tkXX.png`

- why the schedule looks asymmetric and whether the traffic numbers agree with
  that choice
- the left chart shows the programmed block shape:
  - resident weight tiles per block (`weight_reuse_m`)
  - activation tiles streamed through that block (`activation_reuse_n`)
- the right chart shows the measured traffic reuse from `stats.log`:
  - weight reuse before going back to DRAM
  - activation reuse before going back to DRAM