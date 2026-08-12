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
   ├── sched.py           SimClock     – shared "what cycle is it"
   │                      WakeGroup    – ticks only members that have work
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

Two things to know:

- **`cancel()` is an unimplemented stub.** The `Event.cancelled` field exists
  but is never consulted. Nothing currently needs cancellation because the
  clock domain keeps exactly one event in flight; if you add per-object events
  you will have to implement it (lazy tombstoning is the usual approach).
- **`run_all()` terminates only when the queue empties.** Since `ClockDomain`
  reschedules itself forever, a run ends when something calls `clk.stop()`.

---

## 3. Clocked

The base class for every component that advances with time.

```python
class Clocked:
    def tick(self, time=None) -> None: ...
    def next_wake(self, now: int) -> Optional[int]: ...
    def request_wake(self, cycle: int) -> None: ...
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

It does three things:

1. **Quantises time to integer cycles.** Float times within `1e-6` of an
   integer snap to it, so float drift never produces a half cycle.
2. **Makes a repeated tick idempotent.** Called twice for the same cycle, the
   second call returns `None`. This is a genuine safety net — a spurious double
   wake is harmless rather than corrupting.
3. **Tolerates a time reset,** so an object can be reused across runs.

`time=None` means "just advance by one", used by unit tests that drive a
component directly.

### `next_wake(now)` and `request_wake(cycle)`

These drive scheduling; see §5.

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

Each firing does exactly this:

```
   ClockDomain.tick(t):
       if not running: return
       for obj in objects:          ← every registered object, in list order
           obj.tick(t)
       schedule_next(t)             ← push one event at t + period
```

Two consequences worth stating plainly:

- **Order is list order, and it is load-bearing.** A component that reads
  another's output queue in the same cycle only sees it if it ticks later. Any
  reordering changes results.
- **The event queue is used as a metronome, not a scheduler.** Exactly one
  event is ever pending. Skipping idle cycles happens a level down, in the
  wake groups.

`clk.stop()` clears `running`, so the next tick does not reschedule and
`sim.run()` returns. That is how a run ends.

---

## 5. Scheduling: SimClock, WakeGroup, CompositeClocked

Ticking every component every cycle wastes most of its work. Measured on a
16×16 GEMM, **92% of tick calls changed no state at all** — an SRAM bank whose
ready-heap is empty, a functional-unit pipeline with nothing in flight. `sched.py`
lets a parent skip those.

### `next_wake(now)` — the contract

```python
def next_wake(self, now: int) -> Optional[int]:
    return now + 1        # default: "tick me every cycle"
```

Return the earliest cycle at which this object *must* run, or `None` if it is
quiescent. Two rules:

- The **default is `now + 1`**, which reproduces the old tick-everything
  behaviour exactly. An unconverted component keeps working untouched, so
  components can be converted one at a time.
- An override may only ever be **conservative**. Waking too early costs one
  wasted tick. Waking too late drops work silently, and nothing will catch it.

### WakeGroup

A set of sibling components plus a min-heap of wake cycles.

```
   WakeGroup.tick(now):

     heap: (wake_cycle, member_index)
       ┌──────────────────────────────┐
       │ (7, bank12) (7, bank30) (91, bank3) ...
       └──────────┬───────────────────┘
                  │ pop everything with wake_cycle <= now
                  ▼
            due = [12, 30]           ← sorted into REGISTRATION order,
                  │                     so a partially-woken group ticks
                  │                     in the same order the old flat
                  ▼                     loop did
            members[12].tick(now)
            members[30].tick(now)
                  │
                  ▼
            reinsert each at its new next_wake(now)
            (None ⇒ not reinserted; it sleeps until someone wakes it)
```

A producer revives a sleeping consumer with `consumer.request_wake(cycle)`,
which only ever moves a member *earlier* — a later request cannot push a
pending wake back.

### The ordering invariant

> A member may wake a member of a group that ticks **later** in the same cycle,
> or anyone for a **future** cycle — but never an earlier-ticking group for the
> current cycle.

Waking an already-passed group for the current cycle would run it one cycle
sooner than the unscheduled code did. The memory pipeline satisfies this
naturally, because its groups are ticked in the direction data flows:

```
   Scratchpad.tick(now):
        clock.advance_to(now)
        ┌──────────────┐   enqueue    ┌──────────────┐  enqueue   ┌───────────┐
        │ frontends    │ ───────────► │ crossbars    │ ─────────► │ banks     │
        │ (group 1)    │              │ (group 2)    │            │ (group 3) │
        └──────────────┘              └──────────────┘            └───────────┘
             ticked first                  then                      then
```

### Stale time — the trap to know about

A sleeping object stops observing cycles, so anything it derives from *"when
did I last run"* goes stale. `SimClock` is a one-field shared object giving any
component the current cycle in O(1) without being ticked.

`SRAMBank` is the worked example. It computes an operation's due cycle as
`base + latency`, where `base` used to be its own last-tick counter. Once the
bank can sleep, that counter lags and every latency computed from it is wrong —
silently, in the direction of completing *too early*. It now reads the shared
clock instead.

The same hazard hits **any counter incremented once per tick**. `cycles_busy`
was `+= 1` per tick while ops were outstanding; sleeping through the wait would
under-report utilisation. It now accumulates by elapsed cycles:

```python
if self._pending:
    start = max(prev_tick + 1, self._busy_from)
    if start <= cycle:
        self.cycles_busy += cycle - start + 1     # reduces to += 1 when
                                                  # ticked every cycle
```

**When converting a component, audit every per-tick counter first.** These
counters are the simulator's output; getting them wrong corrupts results
without failing anything.

### CompositeClocked

A component whose `tick()` ticks its children in a declared phase order.

```python
root = CompositeClocked("tpu_platform")
root.add_child(vc,            phase=PHASE_CORE)     # 10
root.add_child(vls_bridge,    phase=PHASE_VLS)      # 20
root.add_child(sysarr_bridge, phase=PHASE_SYSARR)   # 30
root.add_child(spad,          phase=PHASE_SPAD)     # 40
root.tick(now)     # advances all of them, in phase order
```

Children sort by `(phase, insertion order)`, so ties keep registration order
and the ordering stays deterministic.

This exists because the order used to be re-specified as a list literal at
every call site — which is how the DRAM and the backends came to be omitted
from some driving lists entirely. Stating it once makes that class of bug
structurally impossible.

### Turning scheduling off

```bash
ATALLA_SCHED=legacy pytest tests/     # tick everything, every cycle
pytest tests/                         # default: skip idle components
```

Both must produce identical results. `tests/base/test_sched.py` asserts this
end-to-end by comparing per-bank utilisation counters, which shift if any
operation completes even one cycle differently.

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
sim.configure_debug(flags=["Xbar", "Clocked"], log_dir="logs")
sim.close_debug()
```

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
        self.request_wake(self._tick + 1)   # revive us if asleep
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
- [ ] Call `request_wake` wherever you hand this object work.
- [ ] Convert any per-tick counter to accumulate by elapsed cycles.
- [ ] Never derive "now" from your own last tick if you can sleep — use the
      shared `SimClock`.
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
               │              issues memory / GSAU requests into SimQueues
               │
               ├─ phase 20  VLSFrontendBridge.tick(t)
               │              VLSU req_q  ──► Frontend.read/write  (+request_wake)
               │
               ├─ phase 30  GSAUTPUBridge.tick(t)
               │              pops systolic request, drives SystolicArrayTPU.tick(t)
               │
               ├─ phase 40  Scratchpad.tick(t)
               │              clock.advance_to(t)
               │              WakeGroup frontends ─► WakeGroup xbars ─► WakeGroup banks
               │              (only members whose next_wake <= t actually run)
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
