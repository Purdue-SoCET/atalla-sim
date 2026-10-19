"""Helpers the blocked-GEMM sweep and plot scripts share.

    run_blocked_mn_*_sweep.py      run the blocked M/N co-simulation over a
                                   parameter and write results.csv / .json
    plot_blocked_mn_*.py           read those results back and plot them
    plot_final_report_*.py         the report's roofline figure

Only code that is the same in every script lives here; what differs per
script (its CSV columns, its plots) stays in the script. To add a helper:
put it here only if at least two scripts need exactly the same thing.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
#: The blocked M/N co-simulation the sweeps drive. It lives with its tests.
BLOCKED_HARNESS_PATH = (
    REPO_ROOT / "tests" / "atalla" / "test_scratchpad_vector_core_sysarr_tpu_tiled_1024_blocked_mn.py"
)


# -- running sweeps --------------------------------------------------------------------

def load_blocked_harness(module_name: str):
    """Import the blocked M/N co-simulation module under `module_name`."""
    spec = importlib.util.spec_from_file_location(module_name, BLOCKED_HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load blocked harness from {BLOCKED_HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """One CSV row per result, with these columns in this order."""
    import csv
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -- reading results back ----------------------------------------------------------------

def load_json_rows(input_dir: Path) -> List[Dict[str, Any]]:
    """Every per-run result JSON in a sweep's output directory."""
    rows: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"results.json", "manifest.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def ok_rows(rows: List[Dict[str, Any]], sort_field: str) -> List[Dict[str, Any]]:
    """The runs that succeeded, ordered by an integer column (the swept one)."""
    filtered = [row for row in rows if str(row.get("status", "ok")) == "ok"]
    filtered.sort(key=lambda row: int(row.get(sort_field, 0) or 0))
    return filtered


# -- plotting ----------------------------------------------------------------------------

def label_for_int(value: int) -> str:
    return str(int(value))


def format_cycles_in_millions(ax: Any) -> None:
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: str(int(round(value / 1_000_000.0)))))
    ax.set_ylabel("cycles (x 1M)")


def algo_flops(matrix_size: int) -> float:
    """FLOPs of an N x N x N matrix product: a multiply and an add per MAC."""
    size = int(matrix_size)
    return float(2 * size * size * size)


def theoretical_peak_compute(tile_size: int) -> float:
    """The array's peak FLOPs a cycle: every PE does a multiply and an add."""
    tile = int(tile_size)
    return float(2 * tile * tile)
