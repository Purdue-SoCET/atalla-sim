# What Else to Take From gem5

Atalla-Sim is explicitly gem5-inspired, and says so in the README. But the
borrowing stopped partway: the event/clock/tick skeleton came across, and almost
nothing else did. This is a ranked catalogue of what is still worth taking, what
each one would cost, and — just as important — what should never be ported.

Every claim below cites a file and line. The counts were measured, not estimated.

Nothing here is a commitment. It is a menu.

---

## 1. What we already took, and where it drifted

| gem5 | here | how it differs today |
|---|---|---|
| `EventQueue` | [`src/base/eventq.py`](../../src/base/eventq.py) | `cancel()` is an unimplemented stub (`:30`); `Event.cancelled` is declared and never read |
| `ClockDomain` | [`src/base/clock_domain.py`](../../src/base/clock_domain.py) | `period=1.0` at **all 21** instantiation sites; `start()` (`:24`) is dead — harnesses call `schedule_next` directly |
| `ClockedObject` | [`src/base/clocked_object.py`](../../src/base/clocked_object.py) | no name, no params; `_consume_tick` treats a tick and a cycle as the same thing |
| `Root` | [`src/base/core.py`](../../src/base/core.py) | **dead.** `Core.domains` is appended to at `:15` and read nowhere — those are the only two references in the repo |
| `DPRINTF` + debug flags | [`src/base/debug.py`](../../src/base/debug.py) | 6 hardcoded flags, no per-instance attribution |
| `Port` / `Packet` | `SimQueue` + callbacks + bridges | not borrowed at all |
| `Stats` | 7 unrelated ad-hoc mechanisms | not borrowed at all |
| `Drainable` | per-harness bespoke idle checks | not borrowed at all |
| `SimObject` params | 18 kwargs threaded by hand | not borrowed at all |

The skeleton is sound. What is missing is everything gem5 layers *on top* of it
to make a simulator maintainable at scale.

---

## 2. Tier 1 — recommended

### 2.1 Hierarchical naming (`SimObject::name()`)

**What gem5 does.** Every SimObject knows its own path — `system.cpu0.dcache` —
and that one string is the backbone of stats naming, debug output, checkpoints,
and config dumps.

**What it costs us not to have it.** No component in Atalla-Sim has a name. All
four `Xbar` instances write into the same `Xbar.log` with no way to tell which
one emitted a line ([`crossbar.py:120`](../../src/memory/crossbar.py)):

```python
dprintf("Xbar", f"complete op={tail['op']}")
```

Stats have the same problem from the other direction: structure is either a list
index buried in a nested dict, or a prefix glued on with string concatenation at
write time (`f"queue_max_{name}"`) and pulled back off by string-stripping at
read time in the plotter. Per-bank, per-lane and per-bridge numbers are
aggregated away before naming and are unreachable from any output file.

**One caution.** The tree must be an **ownership** tree, not the
`CompositeClocked` scheduling tree. The scheduling tree is incomplete by design:
`dram` is never a child, and the tiled-1024 harness bypasses `platform.tick()`
entirely with its own tick loop. Naming off it would leave the most important
harness unnamed.

Small, inert, no risk to simulation results. It is the prerequisite for §2.2.

### 2.2 A stats framework — the biggest single win

**What gem5 does.** `Stats::Scalar`, `Vector`, `Formula`, registered against a
named object, dumped by one writer.

**What we have instead.** Seven unrelated mechanisms: 4 `get_stats()` dict
accessors, bare integer attributes, string-keyed dicts, a hand-rolled
`TPUMetrics` class, a numpy array with integer slot constants for the native
kernel ABI, harness-local accumulator dicts, and append-only record lists.

The consequences are concrete:

- **The same ~30 derived metrics are implemented three times** — in
  `sysarr_tpu_experiment.py`, the tiled-1024 harness, and the 32×32 test.
- **`_algo_flops()` (2·M·N·K) is copied four times**, once in the simulator
  ([`systolic_array_tpu.py:289`](../../src/systolic_array/systolic_array_tpu.py))
  and three times in tools — where two of the copies *recompute* it as a
  fallback when a CSV column is missing, leaving no record of which number a
  given figure was drawn from.
- **Arithmetic intensity is derived six ways under five different key names.**
- **The four `get_stats()` methods reach no output file at all.** Grepping the
  harnesses and all of `tools/` for them returns zero hits. Every bank, crossbar
  and backend counter is computed and thrown away.

A `Formula`, defined once against a named object, collapses all of that.

**Effort:** ~16 days for the full migration, staged so each step is shippable
and gated on the existing golden-trace diff. The compatibility trick that makes
it safe: stats own canonical hierarchical names, and a single manifest table
maps the legacy flat CSV column names onto them, so no plotter moves until you
choose to move it.

### 2.3 Drain / quiescence (`Drainable`)

**What gem5 does.** Every SimObject answers `drain()` with "I'm done" or "not
yet"; the system is quiescent when all of them agree.

**Why this one is nearly free.** We already built the hard half. `next_wake()`
returning `None` *is* a component declaring itself quiescent — see
[base-classes.md §5](base-classes.md#5-scheduling-simclock-wakegroup-compositeclocked).
A `Drainable` protocol composes directly from it.

**What it replaces.** `_compute_path_idle()` in the tiled-1024 harness is 30
lines that hand-enumerate 14 private attributes across 5 components:

```python
if any(self.vc._build_packet[key] for key in ("gsau","vlsu","datapath")): return False
if len(self.vc.gsau.to_systolic) > 0 or len(self.vc.gsau.from_systolic) > 0: return False
if self.sa._algo_out_pending > 0: return False
if self.sysarr_bridge._pending_meta: return False
```

It covers only the compute path; the memory side needs a separate
`_spad_write_path_idle()`, which is duplicated verbatim between
`sysarr_tpu_experiment.py` and `test_..._sysarr_tpu.py`. Termination otherwise
leans on scattered watchdog constants — 1000, 2000, 4000, 8000, 400_000 — and
the sweep runner reruns whole experiments with escalating `max_cycles` because
there is no way to distinguish "deadlocked" from "needed more cycles".

Smallest of the three Tier-1 builds, and the one with the clearest boundary.

---

## 3. Tier 2 — worth it, but bigger

### 3.1 Ports and Packets

**What gem5 does.** Components expose typed ports; a request travels as a
`Packet`; backpressure is a uniform `sendTimingReq` → `false` → `recvReqRetry`
protocol.

**What we have.** ~330 lines across the three bridge classes in `src/`
(`VLSFrontendBridge` 110, `GSAUTPUBridge` 109, `MetricsVLSFrontendBridge` 110),
almost all of it mechanical request/response plumbing — plus **four separate
copies of `VLSFrontendBridge`** across the repo, forked rather than imported.

Backpressure is currently expressed three incompatible ways:

1. Peeking into the consumer's internal FIFO — `spad.frontends[id].writeq.is_full()`
2. A callback returning `False` as a nack — [`crossbar.py:117`](../../src/memory/crossbar.py)
3. `assert` as flow control — **17 sites in `src/`** of the form
   `assert self.vc.push_scratchpad_response(...)`, where a refusal is a crash
   rather than a retry. That is only safe because step 1 already peeked.

A uniform port subsumes all of it. But it touches every component boundary in
the simulator, so it is the most invasive item on this list.

### 3.2 Params and config objects

**What gem5 does.** Each SimObject declares typed, documented, defaulted
parameters; machines are described in Python config scripts; the whole
configuration is dumped with the results.

**What we have.** `build_tpu_platform` takes 18 keyword arguments, and
**21 constructor parameters are unreachable through it** — `SRAMBank.queue_size`
is permanently 1 because `SRAMBanks` never forwards it, and the entire GSAU and
VLSU timing configuration is fixed at its constructor defaults.

Worse, the same logical parameter carries different defaults at each layer.
Bank read latency is 1 in `SRAMBank`, 2 in `SRAMBanks`, 2 in `Scratchpad`, 1 in
`build_tpu_platform` — so the default platform is a machine nobody actually
simulates; every real caller passes 2.

"Queue depth" has **fifteen distinct spellings**:

```
boundary_buffer_depth  capacity      dram_q_depth   dst_fifo_depth
fu_capacity            issue_queue_depth            max_size
queue_size             req_depth     req_queue_depth
rsp_depth              rsp_queue_depth              scheduler_depth
sink_capacity          wb_depth
```

There is no config file format at all — every machine is Python code, and each
of the four sweep runners redeclares its own axis, its own baseline, and its own
CSV column list. Validation is 8 constructor checks total; `Scratchpad`,
`Backend`, `SRAMBank`, `Xbar` and `SystolicArrayTPU` validate nothing, and the
dominant pattern is silent `max(1, ...)` coercion, which hides bad config rather
than rejecting it.

---

## 4. Tier 3 — only when a specific need arrives

### 4.1 Tick vs Cycles, and multiple clock domains

gem5 separates a global `Tick` (picoseconds) from per-domain `Cycles`, with
`clockEdge()` and `cyclesToTicks()` conversions. Atalla-Sim conflates them.

You need this the day you want to model components at different frequencies —
DRAM slower than the core, say, which is realistic for this SoC. **Nothing needs
it today**, and it is worth knowing the price in advance:

- `period=1.0` at all 21 instantiation sites; `Time` is a bare `float`.
- `_consume_tick` does `cycle = int(time)`, so with `period=0.5` two ticks
  collapse to one cycle and the second is **silently dropped**. Every major
  component routes through it.
- `Backend.tick` hand-rolls the same truncation rather than calling it.
- Latencies are added straight onto time values as if they were cycles —
  `frontend.py:18`, `crossbar.py:149`, `backend.py:142`, `sc_sram_banks.py:109`.
- `SharedDRAMBurstChannel` arbitrates by integer-tick equality
  (`backend.py:53`), so a sub-unit period makes two launches collide.
- ~30 tests encode "cycle N" as `until=N.0`.

This is a genuine architectural limit, not a wart. Decide it deliberately.

### 4.2 Event lifecycle — `deschedule` / `reschedule`

`EventQueue.cancel()` is a stub. Nothing needs it while the clock domain keeps
exactly one event pending, but the moment per-object events land in the event
queue, lazy tombstoning becomes mandatory. Cheap when the need arrives.

### 4.3 ProbePoints

gem5's decoupled instrumentation: a component publishes notifications, listeners
attach without the component knowing. `bridge.trace_hook` is the hand-rolled
version — roughly 84 lines of duplicated dict literals across three bridge
classes. Pairs naturally with §2.2.

### 4.4 Checkpointing

Would let a 70-minute 1024×1024 run resume past its preload phase instead of
redoing it. A large lift in Python — the object graph is full of closures and
callbacks that do not serialise. Revisit if run times grow again.

### 4.5 Atomic / functional access modes

gem5 switches between a fast functional model and a detailed timing model.
`TPUReference` (`sysarr_tpu_system.py:207`) is already a functional mirror of
the systolic array; mode switching is the general form of that idea.

---

## 5. What not to port, and why

Stated plainly so nobody relitigates it:

- **Ruby and the coherence protocol machinery (SLICC).** There is no coherence
  in this design — one scratchpad, explicit DMA. Enormous complexity for zero
  applicable behaviour.
- **SE/FS mode, ISA decoding, syscall emulation, `m5ops`.** No CPU is modelled
  and there is no guest software.
- **Bucketed `Distribution` / `Histogram` / `SparseHistogram`.** Nothing needs
  buckets; min/mean/max covers every current consumer, and keeping raw span rows
  makes any percentile computable offline.
- **Stat display flags** (`pdf`, `cdf`, `nozero`, `precision`) **and HDF5
  output.** Presentation policy with no consumer here.
- **`Vector2d`.** The only plausible use is per-PE-per-cycle, which nothing wants.
- **Unit *algebra*.** Keep a plain `unit=` string instead — cheap, and the repo
  genuinely does confuse `bytes/cycle` with `flop/byte`. Full dimensional
  analysis is not worth the machinery.
- **gem5's C++/Python split with generated bindings.** Atalla-Sim is Python-first
  by design, and the hot path is already solved differently — see the SIMD
  kernels in `src/native/`.
- **Multi-queue parallel event simulation.** Single-threaded is fine at this
  scale; the 1024 suite runs in ~69 minutes.

---

## 6. Quick wins, independent of all of the above

Three defects found while surveying. They are worth fixing on their own merits.

**1. `mac_utilization` is two different quantities under one CSV column name.**

```python
# src/atalla/sysarr_tpu_experiment.py:793   — utilisation over ACTIVE cycles
mac_utilization = active_pe_sum / (valid_mac_cycles * tile * tile)

# tests/.../test_..._tiled_1024.py:469      — utilisation over WALL cycles
mac_utilization = active_pe_sum / (self.global_cycle * tile_size * tile_size)
```

`tools/plot_sysarr_tpu_sweeps.py:90` plots whichever it receives, labelled
"overall utilization". The tiled harness already emits the active-cycle variant
separately as `mac_utilization_sysarr_active`, so the experiment harness is the
mislabelled one. Fix: emit both under unambiguous names everywhere, keep the
legacy column as a per-harness alias.

**2. `logs/sysarr_gemm_tpu/stats.log` cannot be read by its own plotter.** It is
written without the `[stats] ` prefix every other stats log uses, so
`tools/plot_final_report_reuse_roofline.py:103` skips every line and exits with
`no [stats] entries found`. A two-line fix.

**3. `Core` is 17 dead lines.** `self.domains` is appended to and never read.
Either give it gem5 `Root`'s job — owning the domains, driving instantiate and
reset — or delete it.

---

## Where to start

If you take one thing: **§2.1 naming, then §2.2 stats.** Naming is small and
inert, stats is the largest payoff, and the first is the prerequisite for the
second.

If you want the cheapest real win: **§2.3 drain**, because `next_wake()` already
did the hard part.

A note on how this document came to exist: the ASCII diagrams in
[docs/README.md](../README.md) and [platform-uml.md](platform-uml.md) originally
omitted the vector core's SIMD datapath entirely, showing only the path to the
systolic array. That is a good argument for §2.1 and §2.2 on its own — a named
component tree would have made the missing subtree obvious, and the datapath's
`op_counts` / `total_ops` / `reduce_ops` are exactly the per-lane numbers that
are currently aggregated away before they reach any output file. What you cannot
see, you forget to document.

---

## Related

- [base-classes.md](base-classes.md) — the tick/cycle model and the scheduler
- [platform-uml.md](platform-uml.md) — runtime dataflow and queue ownership
- [queue-glossary.md](queue-glossary.md) — every FIFO and its payload
