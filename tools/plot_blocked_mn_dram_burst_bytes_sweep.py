#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `/home/asicfab/a/socet149/sc_env_new/bin/python -m pip install matplotlib`."
    ) from exc


def _load_json_rows(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"results.json", "manifest.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _ok_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("status", "ok")) == "ok"]
    filtered.sort(key=lambda row: int(row.get("backend_dram_burst_bytes", 0) or 0))
    return filtered


def _label_for_int(value: int) -> str:
    return str(int(value))


def _annotate_points(ax: Any, x: List[int], y: List[float], *, dy: float = 0.0) -> None:
    if not y:
        return
    y_span = max(y) - min(y) if len(y) > 1 else max(abs(y[0]), 1.0)
    offset = dy if dy else 0.02 * max(y_span, 1.0)
    for x_val, y_val in zip(x, y):
        if math.isnan(y_val):
            continue
        label = f"{y_val:.2f}" if not float(y_val).is_integer() else str(int(y_val))
        ax.text(x_val, y_val + offset, label, ha="center", va="bottom", fontsize=8)


def _format_cycles_in_millions(ax: Any) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: str(int(round(value / 1_000_000.0)))))
    ax.set_ylabel("cycles (x 1M)")


def plot_cycles(rows: List[Dict[str, Any]], output_path: Path) -> Path:
    x = [int(row["backend_dram_burst_bytes"]) for row in rows]
    cycles = [float(row["cycles"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(x, cycles, marker="o", linewidth=2.4, color="#4e79a7")
    ax.set_title("Blocked M=8, N=32: Total Cycles vs DRAM Burst Bytes")
    ax.set_xlabel("backend dram burst bytes")
    _format_cycles_in_millions(ax)
    ax.set_xticks(x, [_label_for_int(value) for value in x])
    ax.grid(True, alpha=0.28)
    _annotate_points(ax, x, cycles)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_external_bandwidth(rows: List[Dict[str, Any]], output_path: Path) -> Path:
    x = [int(row["backend_dram_burst_bytes"]) for row in rows]
    bandwidth = [float(row.get("external_bandwidth_avg_bytes_per_cycle", 0.0) or 0.0) for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(x, bandwidth, marker="o", linewidth=2.4, color="#e15759")
    ax.set_title("Blocked M=8, N=32: External Bandwidth vs DRAM Burst Bytes")
    ax.set_xlabel("backend dram burst bytes")
    ax.set_ylabel("external bandwidth (bytes/cycle)")
    ax.set_xticks(x, [_label_for_int(value) for value in x])
    ax.grid(True, alpha=0.28)
    _annotate_points(ax, x, bandwidth)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_summary(rows: List[Dict[str, Any]], output_path: Path) -> Path:
    x = [int(row["backend_dram_burst_bytes"]) for row in rows]
    cycles = [float(row["cycles"]) for row in rows]
    bandwidth = [float(row.get("external_bandwidth_avg_bytes_per_cycle", 0.0) or 0.0) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    fig.suptitle("Blocked M=8, N=32 DRAM Burst Byte Sweep")

    axes[0].plot(x, cycles, marker="o", linewidth=2.4, color="#4e79a7")
    axes[0].set_title("Total Cycles")
    axes[0].set_xlabel("backend dram burst bytes")
    _format_cycles_in_millions(axes[0])
    axes[0].set_xticks(x, [_label_for_int(value) for value in x])
    axes[0].grid(True, alpha=0.28)

    axes[1].plot(x, bandwidth, marker="o", linewidth=2.4, color="#e15759")
    axes[1].set_title("External Bandwidth")
    axes[1].set_xlabel("backend dram burst bytes")
    axes[1].set_ylabel("bytes/cycle")
    axes[1].set_xticks(x, [_label_for_int(value) for value in x])
    axes[1].grid(True, alpha=0.28)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot total cycles and external bandwidth for blocked M/N DRAM burst byte sweeps."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing per-run JSON files from run_blocked_mn_dram_burst_bytes_sweep.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG outputs. Defaults to <input-dir>/plots",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = (
        Path(args.input_dir)
        if args.input_dir
        else repo_root / "logs" / "blocked_m8_n32_dram_burst_bytes_sweep"
    )
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"

    rows = _ok_rows(_load_json_rows(input_dir))
    if not rows:
        raise SystemExit(f"no successful run JSON files found in {input_dir}")

    written = [
        plot_cycles(rows, output_dir / "cycles_vs_dram_burst_bytes.png"),
        plot_external_bandwidth(rows, output_dir / "external_bandwidth_vs_dram_burst_bytes.png"),
        plot_summary(rows, output_dir / "blocked_mn_dram_burst_bytes_sweep_summary.png"),
    ]
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())