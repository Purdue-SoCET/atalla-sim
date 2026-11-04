"""Run a model's operation plan on atalla-sim, op by op: `atalla-sim model`.

**A stub ahead of atalla-graph.** atalla-graph will lower a PyTorch model to
Linalg and describe it as a JSON plan (schema `atalla-graph-plan-v1`:
function inputs, operations in order with their operands, result types,
attributes, and the names of the scalar ops in each region). That export,
and choosing kernels for Linalg ops, are atalla-graph's work and not done
upstream yet; this runner is the atalla-sim side, written against the plan
format of a local atalla-graph build so it is ready when they land.

    atalla-sim model examples/model/mlp.plan.json --inputs mlp.inputs.npz \\
        --reference torch_out.npy --report-json report.json

For each operation in order:

- **on the accelerator** (a kernel runs on atalla-sim): `linalg.matmul`
  (the gemm kernel, which computes C = A W^T + C on square tiles up to 32,
  given W = B^T), `linalg.generic`
  whose region is one `arith.addf` (the add kernel; a broadcast operand is
  expanded first) or one `arith.maximumf` with one input (the relu kernel);
- **on the host** (numpy, counted as such): `linalg.transpose`;
- **no work**: `tensor.empty`, `linalg.fill`, `arith.constant` (a scalar).

Anything else stops the run with the operation's name and position. The
kernels are the functional simulator's hand-written ones
(third_party/atalla-functional-sim/kernels/build_{gemm,add,relu}.py), built
at the operation's size; they read their tensor addresses from the table at
0x3C, so the runner places each operation's tensors and reads the result.

Each accelerator operation is checked against numpy on the same inputs
(worst error in BF16 steps), the chained output against numpy running the
whole plan, and, with --reference, against the framework's own output.

Limits of the stub, all to do with what the plan doesn't carry yet:
constants other than scalars need their data in --inputs (key `const<id>`);
a region's operands aren't recorded, so `arith.maximumf` is taken to be
max(x, 0) (ReLU); dimensions above 32 need tiling, which isn't done.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from atalla import tensors

SCHEMA = "atalla-graph-plan-v1"
TILE = 32
CFG = 0x3C
BASE = 0x1000


class PlanError(ValueError):
    pass


# -- the plan -----------------------------------------------------------------------------

_TENSOR = re.compile(r"^tensor<((?:\d+x)*)([a-z0-9]+)>$")


def parse_type(t: str) -> Tuple[Tuple[int, ...], str]:
    m = _TENSOR.match(t.strip())
    if m:
        dims = tuple(int(d) for d in m.group(1).split("x") if d)
        return dims, m.group(2)
    if re.fullmatch(r"[a-z]+[0-9]*", t.strip()):
        return (), t.strip()
    raise PlanError("unsupported type %r (static-shape tensors and scalars only)" % t)


def _segments(attrs: str, n: int) -> Tuple[int, int]:
    """(ins, outs) from operandSegmentSizes, else all but the last are ins."""
    m = re.search(r"operandSegmentSizes = array<i32: (\d+), (\d+)>", attrs)
    if m:
        return int(m.group(1)), int(m.group(2))
    return n - 1, 1


def _maps(attrs: str) -> List[List[str]]:
    """indexing_maps as the result dims of each map, e.g. [['d0','d1'], ['d1'], ...]."""
    m = re.search(r"indexing_maps = \[(.*?)\]\s*[,}]", attrs)
    if not m:
        return []
    return [[d.strip() for d in res.split(",") if d.strip()]
            for res in re.findall(r"->\s*\(([^)]*)\)", m.group(1))]


@dataclass
class Plan:
    path: str
    inputs: List[Tuple[int, Tuple[int, ...], str]]
    ops: List[dict]
    returns: List[int]

    @classmethod
    def load(cls, path) -> "Plan":
        doc = json.loads(Path(path).read_text())
        if doc.get("schema") != SCHEMA:
            raise PlanError("%s: schema %r, expected %r" % (path, doc.get("schema"), SCHEMA))
        funcs = doc.get("functions", [])
        if len(funcs) != 1:
            raise PlanError("%s: %d functions; the runner takes one" % (path, len(funcs)))
        f = funcs[0]
        inputs = [(int(a["id"]),) + parse_type(a["type"]) for a in f["inputs"]]
        return cls(str(path), inputs, list(f["operations"]), [int(r) for r in f["returns"]])


# -- operations -----------------------------------------------------------------------------

@dataclass
class OpRecord:
    index: int
    name: str
    where: str                      # "accelerator", "host", "none"
    kernel: str = ""
    shape: str = ""
    cycles: int = 0
    check: Dict[str, Any] = field(default_factory=dict)


def _bf16(x) -> np.ndarray:
    return tensors.bf16_round(np.asarray(x, dtype=np.float32))


def _scalar_constant(attrs: str) -> float:
    m = re.search(r"value = ([-+0-9.eE]+|0x[0-9A-Fa-f]+) : ", attrs)
    if not m:
        raise PlanError("a constant without scalar data: %s" % attrs[:80])
    v = m.group(1)
    return float(int(v, 16)) if v.startswith("0x") else float(v)


class Runner:
    """Executes a Plan on one backend: "atalla-sim" or "numpy" (the reference)."""

    def __init__(self, plan: Plan, backend: str = "atalla-sim", platform_kw=None,
                 kernel_builder=None):
        self.plan = plan
        self.backend = backend
        self.platform_kw = dict(platform_kw or {})
        self.records: List[OpRecord] = []
        self._build = kernel_builder
        self._images: Dict[Tuple, Dict[int, int]] = {}
        #: With activity: (layer name, activity trace) per accelerator operation.
        self.activity = bool(self.platform_kw.pop("activity", False))
        self.traces: List[Tuple[str, dict]] = []
        self._op = None

    # The functional sim's hand-written kernels, built at a size.
    def _kernel(self, name: str, args: List[str]) -> Dict[int, int]:
        key = (name, tuple(args))
        if key not in self._images:
            self._images[key] = self._build(name, args)
        return self._images[key]

    def _sim(self, name, args, inputs: List[Tuple[int, np.ndarray]], table: List[int],
             out_addr: int, out_shape) -> Tuple[np.ndarray, int]:
        from atalla.kernel import Kernel
        k = Kernel(program=(self._kernel(name, args), {}), c_entry=False)
        for i, a in enumerate(table):
            k.write(CFG + 4 * i, np.array([a]), "u32")
        for addr, x in inputs:
            k.write(addr, x)
        r = k.run(activity=self.activity, **self.platform_kw)
        if self.activity:
            label = "#%d %s" % (self._op, name)
            self.traces.append((label, r.plat.activity.trace({"program": label})))
        return r.read(out_addr, out_shape), r.cycles

    def matmul(self, a, b, c) -> Tuple[np.ndarray, int, str]:
        m, kdim = a.shape
        k2, n = b.shape
        if kdim != k2 or c.shape != (m, n):
            raise PlanError("matmul shapes %s x %s -> %s" % (a.shape, b.shape, c.shape))
        if self.backend == "numpy":
            return _bf16(a.astype(np.float32) @ b.astype(np.float32) + c), 0, ""
        t = max(m, kdim, n)
        if t > TILE:
            raise PlanError("matmul %dx%dx%d: dimensions above %d need tiling (not done)"
                            % (m, kdim, n, TILE))
        pad = lambda x: np.pad(x, ((0, t - x.shape[0]), (0, t - x.shape[1])))  # noqa: E731
        size = t * t * 2
        A, W, C = BASE, BASE + size, BASE + 2 * size
        # The kernel computes C = A W^T + C: lw.vi shifts each weight row in at
        # column 0, so row j of W ends up as output column j (nn.Linear's
        # weight layout). build_gemm.py's own numpy reference (A @ W) is stale.
        out, cycles = self._sim("gemm", ["--rows", str(t), "--cols", str(t)],
                                [(A, pad(a)), (W, pad(b.T)), (C, pad(c))],
                                [A, 0, W, 0, C, 0], C, (t, t))
        return out[:m, :n], cycles, "gemm %dx%d" % (t, t)

    def add(self, a, b) -> Tuple[np.ndarray, int, str]:
        if self.backend == "numpy":
            return _bf16(a + b), 0, ""
        rows, width = a.shape if a.ndim == 2 else (1, a.size)
        if width > TILE:
            raise PlanError("add of width %d: above %d needs tiling (not done)" % (width, TILE))
        size = rows * width * 2
        A, B, C = BASE, BASE + size, BASE + 2 * size
        out, cycles = self._sim("add", ["--rows", str(rows), "--width", str(width)],
                                [(A, a.reshape(rows, width)), (B, b.reshape(rows, width))],
                                [A, B, C], C, (rows, width))
        return out.reshape(a.shape), cycles, "add %dx%d" % (rows, width)

    def relu(self, x) -> Tuple[np.ndarray, int, str]:
        if self.backend == "numpy":
            return _bf16(np.maximum(x, 0)), 0, ""
        rows, width = x.shape if x.ndim == 2 else (1, x.size)
        if width > TILE or rows > TILE:
            raise PlanError("relu of %dx%d: above %d needs tiling (not done)" % (rows, width, TILE))
        size = rows * width * 2
        I, O = BASE, BASE + size
        out, cycles = self._sim("relu", ["--rows", str(rows), "--width", str(width)],
                                [(I, x.reshape(rows, width))], [I, O], O, (rows, width))
        return out.reshape(x.shape), cycles, "relu %dx%d" % (rows, width)

    # -- the plan --------------------------------------------------------------------------
    def run(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        vals: Dict[int, Any] = {}
        for pos, (vid, shape, _) in enumerate(self.plan.inputs):
            key = "arg%d" % pos
            if key not in inputs:
                raise PlanError("no input %s (value %d, shape %s)" % (key, vid, shape))
            x = np.asarray(inputs[key], dtype=np.float32)
            if x.shape != shape:
                raise PlanError("input %s has shape %s, the plan says %s" % (key, x.shape, shape))
            vals[vid] = _bf16(x)
        for i, op in enumerate(self.plan.ops):
            self._op = i
            name, attrs = op["name"], op.get("attributes", "")
            args = [vals[o] for o in op["operands"]]
            res = op["results"]
            shape = parse_type(res[0]["type"])[0] if res else ()
            rec = OpRecord(i, name, "none", shape="x".join(map(str, shape)))
            if name == "arith.constant":
                key = "const%d" % res[0]["id"]
                out = (_bf16(inputs[key]) if key in inputs else
                       np.float32(_scalar_constant(attrs)) if shape == () else None)
                if out is None:
                    raise PlanError("op %d: arith.constant %s has no data in the plan; give it "
                                    "as %s in --inputs" % (i, rec.shape, key))
            elif name == "tensor.empty":
                out = np.zeros(shape, dtype=np.float32)
            elif name == "linalg.fill":
                out = np.full(shape, args[0], dtype=np.float32)
            elif name == "linalg.transpose":
                perm = [int(p) for p in re.search(r"array<i64: ([0-9, ]+)>", attrs)
                        .group(1).split(",")]
                out = np.transpose(args[0], perm).copy()
                rec.where = "host"
            elif name == "linalg.matmul":
                nin, _ = _segments(attrs, len(args))
                out, rec.cycles, rec.kernel = self.matmul(args[0], args[1], args[nin])
                rec.where = "accelerator"
                rec.check = tensors.compare(out, _bf16(args[0] @ args[1] + args[nin]))
            elif name == "linalg.generic":
                out = self._generic(i, op, args, shape, rec)
            else:
                raise PlanError("op %d: %s has no kernel or host implementation" % (i, name))
            vals[res[0]["id"]] = out
            self.records.append(rec)
        return [vals[r] for r in self.plan.returns]

    def _generic(self, i, op, args, shape, rec) -> np.ndarray:
        attrs = op.get("attributes", "")
        region = [r["name"] for r in op.get("region_operations", [])]
        nin, _ = _segments(attrs, len(args))
        ins = args[:nin]
        maps = _maps(attrs)
        if "reduction" in attrs:
            raise PlanError("op %d: a reducing linalg.generic (%s) has no kernel yet" % (i, region))
        out_dims = maps[-1] if maps else ["d%d" % k for k in range(len(shape))]
        expanded = []
        for x, m in zip(ins, maps[:nin] or [out_dims] * nin):
            if m != out_dims:                         # a broadcast operand
                idx = [out_dims.index(d) for d in m]
                view = np.expand_dims(x, [k for k in range(len(out_dims)) if k not in idx])
                x = np.broadcast_to(view, shape).copy()
            expanded.append(x)
        if region == ["arith.addf"] and nin == 2:
            out, rec.cycles, rec.kernel = self.add(*expanded)
            rec.check = tensors.compare(out, _bf16(expanded[0] + expanded[1]))
        elif region == ["arith.maximumf"] and nin == 1:
            out, rec.cycles, rec.kernel = self.relu(expanded[0])
            rec.check = tensors.compare(out, _bf16(np.maximum(expanded[0], 0)))
        else:
            raise PlanError("op %d: linalg.generic with region %s and %d inputs has no kernel"
                            % (i, region, nin))
        rec.where = "accelerator"
        return out


def build_with_functional_sim(name: str, args: List[str]) -> Dict[int, int]:
    """A hand-written kernel's program, from the functional sim's build script."""
    import subprocess
    import sys
    import tempfile
    from scheduler import golden
    from scheduler.core import load_program_text
    script = golden.SUBMODULE_PATH / "kernels" / ("build_%s.py" % name)
    if not script.is_file():
        raise PlanError("no kernel build script %s (the functional-sim submodule)" % script)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / ("%s.in" % name)
        env = dict(__import__("os").environ, PYTHONPATH="%s:%s" % (golden.SUBMODULE_PATH,
                                                                  script.parent))
        p = subprocess.run([sys.executable, str(script), "-o", str(out), *args],
                           cwd=str(golden.SUBMODULE_PATH), env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
        if p.returncode:
            raise PlanError("building %s %s failed:\n%s" % (name, " ".join(args), p.stdout[-2000:]))
        instr, _data = load_program_text(out.read_text())
    return instr


def run_model(plan_path, inputs: Dict[str, np.ndarray], platform_kw=None,
              reference: Optional[np.ndarray] = None, config_summary: str = "") -> dict:
    plan = Plan.load(plan_path)
    sim = Runner(plan, "atalla-sim", platform_kw, build_with_functional_sim)
    outs = sim.run(inputs)
    ref = Runner(plan, "numpy").run(inputs)
    report = {
        "plan": str(plan_path), "machine": config_summary,
        "operations": [dict(index=r.index, name=r.name, where=r.where, kernel=r.kernel,
                            shape=r.shape, cycles=r.cycles,
                            max_bf16_steps=r.check.get("max_bf16_steps"),
                            max_abs=r.check.get("max_abs"))
                       for r in sim.records],
        "counts": {w: sum(r.where == w for r in sim.records)
                   for w in ("accelerator", "host", "none")},
        "cycles": sum(r.cycles for r in sim.records),
        "output_vs_numpy": _jsonable(tensors.compare(outs[0], ref[0])),
    }
    if reference is not None:
        report["output_vs_reference"] = _jsonable(tensors.compare(outs[0], reference))
    report["outputs"] = [o.tolist() for o in outs]
    if sim.activity:
        report["activity"] = {"version": 1, "workload": Path(str(plan_path)).name,
                              "meta": {"plan": str(plan_path)},
                              "layers": [{"name": n, "trace": t} for n, t in sim.traces]}
    return report


def _jsonable(m: dict) -> dict:
    return {k: (list(map(int, v)) if isinstance(v, tuple) else v) for k, v in m.items()}


def summary(report: dict) -> str:
    lines = ["%s on %s" % (report["plan"], report["machine"] or "atalla-sim"),
             "  %3s  %-18s %-12s %-10s %8s  %s" % ("#", "operation", "where", "kernel",
                                                  "cycles", "worst BF16 steps vs numpy")]
    for o in report["operations"]:
        if o["where"] == "none":
            continue
        lines.append("  %3d  %-18s %-12s %-10s %8s  %s"
                     % (o["index"], o["name"], o["where"], o["kernel"] or "-",
                        o["cycles"] or "-", "%.1f" % o["max_bf16_steps"]
                        if o["max_bf16_steps"] is not None else "-"))
    c = report["counts"]
    lines.append("%d operations on the accelerator, %d on the host, %d with no work; "
                 "%d cycles" % (c["accelerator"], c["host"], c["none"], report["cycles"]))
    for key, what in (("output_vs_numpy", "numpy running the plan"),
                      ("output_vs_reference", "the reference output")):
        if key in report:
            m = report[key]
            lines.append("output vs %s: worst %.1f BF16 steps, max abs %.3g, rel_l2 %.3g, "
                         "cosine %.6f" % (what, m["max_bf16_steps"], m["max_abs"], m["rel_l2"],
                                          m["cosine"]))
    return "\n".join(lines)
