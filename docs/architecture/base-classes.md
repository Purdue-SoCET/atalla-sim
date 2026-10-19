# Base Classes, Clocking, and Ticking

Everything in Atalla-Sim that advances with time is built from a handful of
classes in `src/base/`. This document covers what each one does, how a cycle is
actually executed, and how a component joins the simulation.

The design is gem5-inspired: components are *clocked objects* driven by a
*clock domain*, which is itself driven by events on a global *event queue*.

---

## 1. The pieces

```
   src/base/
   ├── eventq.py          EventQueue   – time-ordered callback heap
   ├── clock_domain.py    ClockDomain  – turns events into per-cycle ticks
   ├── clocked_object.py  Clocked      – base class for anything with a tick()
   ├── sched.py           WakeGroup    – ticks only members that have work
   │                      CompositeClocked – a component made of components
   ├── core.py            Core         – holds the clock domains
   ├── sim.py             Sim          – run/stop entry point
   ├── queue.py           SimQueue     – fixed-capacity FIFO, the standard
   │                                     inter-component connector
   ├── dtype.py           DType        – fp16 / bf16 / int8 numerics
   └── debug.py           dprintf      – flag-gated tracing
```

Each layer drives the one below it, and the bottom layer feeds work back up:

```
       Sim.run()
          │  drains the event queue
          ▼
   ┌─────────────────┐   schedule(t+period, tick)
   │   EventQueue    │◄──────────────────────────┐
   │  (min-heap by   │                           │
   │   time)         │                           │
   └────────┬────────┘                           │
            │ pop earliest event, call it        │
            ▼                                    │
   ┌─────────────────┐                           │
   │   ClockDomain   │───────────────────────────┘
   │  period = 1.0   │  reschedules itself every cycle
   └────────┬────────┘
            │ obj.tick(t) for each registered object
            ▼
   ┌─────────────────┐
   │   Clocked       │  the component: one tick == one modeled cycle
   └─────────────────┘
```

---

## 2. EventQueue

A min-heap of `(time, seq, callback, args, kwargs)`. `seq` is a monotonically
increasing integer, so events scheduled for the same time fire in the order
they were scheduled — the queue is deterministic.

```python
eq = EventQueue()
eq.schedule(10.0, my_callback, arg)   # returns a handle
eq.now()                              # current simulated time
eq.run_until(100.0)                   # drain everything up to t=100
eq.run_all()                          # drain until the queue is empty
```

`run_until` advances `_time` to each popped event's timestamp before invoking
it, so a callback always sees the correct `now()`.

- **`run_all()` terminates only when the queue empties.** Since `ClockDomain`
  reschedules itself forever, a run ends when something calls `clk.stop()`.

---

## 3. Clocked

The base class for every component that advances with time.

```python
class Clocked:
    def tick(self, time=None) -> None: ...
    def next_wake(self, now: int) -> Optional[int]: ...
    def wake(self) -> None: ...
    def request_wake(self, cycle: int) -> None: ...
    def add_child(self, child): ...
    def tick_children(self, time=None) -> None: ...
    def _consume_tick(self, time, attr_name="_clocked_last_tick") -> Optional[int]: ...
```

### `tick(time)`

Advance this component by one modeled cycle. Overriding this is the whole job
of writing a component.

### `_consume_tick(time, attr_name)`

The guard nearly every component opens its `tick()` with:

```python
def tick(self, time=None):
    cycle = self._consume_tick(time, attr_name="_curr_tick")
    if cycle is None:
        return          # already ran this cycle; do nothing
    ...
```

1. **Quantises time to integer cycles.** Float times within `1e-6` of an
   integer snap to it, so float drift never produces a half cycle.
2. **Makes a repeated tick idempotent.** Called twice for the same cycle, the
   second call returns `None`. This is a genuine safety net — a spurious double
   wake is harmless rather than corrupting.
3. **Tolerates a time reset,** so an object can be reused across runs.

`time=None` means "just advance by one", used by unit tests that drive a
component directly.

### `next_wake`, `wake`, `request_wake`, `add_child`, `tick_children`

These let idle components sleep; see §5.

---

## 4. ClockDomain

Converts the event queue into a steady cycle drumbeat.

```python
eq   = EventQueue()
clk  = ClockDomain(eq, period=1.0)
core = Core(eq); core.add_clock_domain(clk)
sim  = Sim(); sim.init(eq, core)

clk.add_clocked(component)
clk.schedule_next(0.0)
sim.run()
```

Each firing:

```
   ClockDomain.tick(t):
       if not running: return
       for obj in objects:          ← every registered object, in list order
           obj.tick(t)
       schedule_next(t)             ← push one event at t + period
```

- **Order is list order, and it is load-bearing.** A component that reads
  another's output queue in the same cycle only sees it if it ticks later. Any
  reordering changes results.
- **The event queue is used as a metronome, not a scheduler.** Exactly one
  event is ever pending. Skipping idle cycles happens a level down, in the
  wake groups.

`clk.stop()` clears `running`, so the next tick does not reschedule and
`sim.run()` returns. That is how a run ends.

---

## 5. Scheduling: WakeGroup, CompositeClocked

Ticking every component every cycle wastes most of the work: a VLSU with
empty queues, a DMA backend with no transfer, the lanes between vector
instructions. A component can sleep while it has nothing to do, and be woken
when work reaches it. Results never change; only the simulation gets faster
(18-37% on the kernels and GEMM co-simulations, measured when this was added).

### The contract

```python
def next_wake(self, now: int) -> Optional[int]:
    return now + 1        # default: "tick me every cycle"
```

After each tick, a component says the next cycle it has work, or `None` to
sleep. Whoever hands it work from outside its own tick calls `wake()`.

- The **default is `now + 1`**: a component that does nothing special is
  ticked every cycle, exactly as before. Components opt in one at a time.
- Return `None` **only when a tick would change nothing** -- no value, no
  counter -- until someone calls `wake()`. Too early costs a wasted tick; too
  late silently loses work.
- Call `wake()` **everywhere work arrives** from outside: an `enqueue`, a
  `push_response`, a transfer start. One missed call is a lost wake.
- A sleeping component **stops counting cycles**, so it must not time anything
  from its own last tick while asleep, and **a counter it increments every
  tick** (busy cycles, say) is wrong once it sleeps. Audit both first.

`wake()` means "as soon as you would have seen it had you been ticked every
cycle": this cycle if your group has not reached you yet, the next one if it
has. So a hand-off forward in the tick order lands the same cycle and one
backward lands the next -- the same as ticking everything.

### Children: `add_child` and `tick_children`

Any Clocked can hold children:

```python
class VectorCore(Clocked):
    def __init__(...):
        for unit in (self.datapath, *self.vls_units, self.gsau, self.transpose):
            self.add_child(unit)         # ticked in this order

    def tick_units(self, time=None):
        self.tick_children(cycle)        # only the ones that are due
```

The children live in a `WakeGroup`: due members are ticked in the order they
were added (`CHILDREN_ROTATE = True` on the parent for round robin, starting
from child `cycle mod count`, as `RoundRobinBackendTicker` does). A child's
`next_wake` is asked after all the children have ticked, so work handed to it
later in the same cycle is seen.

### What sleeps today

| Component | Asleep when | Woken by |
|---|---|---|
| `VectorDatapath` | nothing queued, in the lanes or the collector | `enqueue` |
| `VectorLoadStoreUnit` | every queue empty | `enqueue_issue`, `push_response` |
| `GSAU` | no array result waiting | `push_systolic_response` |
| `TransposeUnit` | idle, no request | `push`, `pop` |
| `Backend` | no transfer queued, active or in flight | `driver_to_backend_start_load/store`, `queue_sram_read_response` |
| `Scratchpad` | every pad idle | its own `submit_*` |

The systolic arrays do not sleep: some of their statistics count every cycle,
so a sleeping array would report less. Instead each has an idle fast path
inside its tick (`SystolicArrayMEISSA._idle_tick`, `SystolicArrayTPU._idle_tick`):
once drained, or settled into a fixed point, an idle cycle adds to those
counters directly instead of running the datapath.
`tests/systolic_array/test_idle_fast_path.py` checks them against the full
datapath after every cycle.

### CompositeClocked

A component whose `tick()` ticks its children in a declared phase order; the
platforms' root.

```python
root = CompositeClocked("tpu_platform")
root.add_child(vc,            phase=PHASE_CORE)     # 10
root.add_child(vls_bridge,    phase=PHASE_VLS)      # 20
root.add_child(sysarr_bridge, phase=PHASE_SYSARR)   # 30
root.add_child(spad,          phase=PHASE_SPAD)     # 40
root.tick(now)     # advances all of them, in phase order
```

Children sort by `(phase, insertion order)`, so ties keep registration order
and the ordering stays deterministic. Children that sleep are skipped.

This exists because the order used to be re-specified as a list literal at
every call site -- which is how the DRAM and the backends came to be omitted
from some driving lists entirely. Stating it once makes that class of bug
structurally impossible.

### Turning scheduling off

```bash
ATALLA_SCHED=legacy pytest tests/     # tick everything, every cycle
pytest tests/                         # default: skip idle components
```

Both must produce identical results.
`tests/equivalence/test_equivalence.py::test_legacy_scheduling_gives_the_same_results`
runs a kernel, a harness experiment, the transpose bench and a tiled GEMM
both ways against the same snapshot.

---

## 6. SimQueue — how components connect

Components do not call each other directly; they pass work through
fixed-capacity FIFOs. This is what makes backpressure emerge naturally instead
of having to be modeled explicitly.

```python
q = SimQueue(max_size=4)
q.enqueue(item)   # False if full  ← the producer stalls
q.dequeue()       # None if empty  ← the consumer idles
q.peek()          # look without consuming
q.is_full(); q.is_empty(); len(q)
```

The `enqueue` returning `False` *is* the stall signal:

```
   producer                    SimQueue(4)                consumer
      │                     ┌───┬───┬───┬───┐                │
      │  enqueue() ────────►│ a │ b │ c │ d │                │
      │  ◄──── False        └───┴───┴───┴───┘ ──── dequeue()──►
      │  (full: stall,           full                         │
      │   retry next cycle)                                   │
```

`peek()` then `dequeue()` is the standard idiom for "only consume if I can
actually accept it this cycle".

---

## 7. Core and Sim

Thin. `Core` holds clock domains (`add_clock_domain`); `reset()` is a stub.
`Sim` binds an event queue and a core, and exposes:

```python
sim.run()              # run_all()  – until the queue drains
sim.run(until=5000.0)  # run_until()
sim.close_debug()
```

Debug logging is turned on with `base.debug.configure_debug(flags=[...])`.

`dprintf(flag, msg)` writes to `logs/<flag>.log` when that flag is enabled; an
empty flag set enables everything.

---

## 8. Writing a component

Minimum viable clocked object:

```python
from base.clocked_object import Clocked
from base.queue import SimQueue

class MyUnit(Clocked):
    def __init__(self, latency=2, depth=4):
        super().__init__()
        self.latency = latency
        self.in_q  = SimQueue(depth)
        self.out_q = SimQueue(depth)
        self._pending = []          # (due_cycle, payload)
        self._tick = -1

    def submit(self, payload) -> bool:
        if not self.in_q.enqueue(payload):
            return False            # full: caller must retry
        self.wake()                 # revive us if asleep
        return True

    def next_wake(self, now):
        if self._pending:
            return self._pending[0][0]      # deadline-driven
        if not self.in_q.is_empty():
            return now + 1                  # work waiting to start
        return None                         # quiescent

    def tick(self, time=None):
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        while self._pending and self._pending[0][0] <= cycle:
            _, payload = self._pending.pop(0)
            self.out_q.enqueue(payload)
        item = self.in_q.dequeue()
        if item is not None:
            self._pending.append((cycle + self.latency, item))
```

Then attach it, either to a clock domain directly:

```python
clk.add_clocked(unit)
```

or, preferably, into the platform tree so it is driven with everything else:

```python
platform.add_component(unit, phase=PHASE_SPAD)
```

### Checklist

- [ ] Call `super().__init__()`.
- [ ] Open `tick()` with `_consume_tick` and bail on `None`.
- [ ] Implement `next_wake` **conservatively**, or leave it alone (the default
      is always correct, just slower).
- [ ] Call `wake()` wherever this object is handed work.
- [ ] If it can sleep: no counter incremented every tick, and no "now"
      taken from its own last tick (callers pass the cycle they are in).
- [ ] Verify with `ATALLA_SCHED=legacy` and confirm results are unchanged.

---

## 9. One cycle, end to end

Cycle *t* of the TPU platform, showing where each class acts:

```
 EventQueue pops (t, ClockDomain.tick)
   │
   └─► ClockDomain.tick(t)
         │
         └─► CompositeClocked "tpu_platform" .tick(t)
               │
               ├─ phase 10  VectorCore.tick(t)
               │              issues one VLIW packet, up to three slots:
               │                vlsu     ──► memory requests
               │                datapath ──► SIMD lanes + tree reduction
               │                gsau     ──► systolic array
               │
               ├─ phase 20  VLSFrontendBridge.tick(t)
               │              VLSU req_q  ──► Frontend.read/write  (+request_wake)
               │
               ├─ phase 30  GSAUTPUBridge.tick(t)
               │              pops systolic request, drives SystolicArrayTPU.tick(t)
               │
               ├─ phase 40  Scratchpad.tick(t)
               │              each busy pad: enable its read and write channels
               │              when free, hand back reads that are due
               │              (idle pads return at once)
               │
               ├─ phase 50  Backend.tick(t)          [when DRAM is attached]
               │
               └─ phase 90  harness.tick(t)
                              observes writeback, issues next work,
                              calls clk.stop() when the run is complete
   │
   └─► ClockDomain.schedule_next(t) ─► EventQueue.schedule(t+1, ...)
```

---

## Related

- [platform-uml.md](platform-uml.md) — full runtime dataflow, queue ownership
- [queue-glossary.md](queue-glossary.md) — every FIFO and its payload
- `tests/test_base_classes.py` — clock domain and event queue behaviour
- `tests/base/test_sched.py` — wake groups, composition, legacy equivalence
