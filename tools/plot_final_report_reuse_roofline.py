#!/usr/bin/env python3

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `/home/asicfab/a/socet149/sc_env_new/bin/python -m pip install matplotlib`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = REPO_ROOT / "logs" / "roofline_study" / "blocked_mn_reuse_1024" / "run_20260420_034142_utc"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "final_report" / "figures" / "reuse_policy_roofline.png"
NAIVE_STATS_LOG = REPO_ROOT / "logs" / "sysarr_gemm_tpu_tiled_1024" / "stats.log"
M2_N32_STATS_LOG = REPO_ROOT / "logs" / "m2_n32" / "stats.log"
M8_N32_JSON = REPO_ROOT / "logs" / "blocked_m8_n32_spad_frontend_queue_sweep" / "m8_n32_spad_frontend_queue_size_4.json"

POINT_SPECS = [
    {
        "label": "Naive",
        "kind": "stats",
        "path": NAIVE_STATS_LOG,
        "order": 0,
        "weight_reuse_m": 1,
        "activation_reuse_n": 1,
    },
    {"label": "m2_n4", "kind": "json", "filename": "m2_n4.json", "order": 1},
    {"label": "m2_n8", "kind": "json", "filename": "m2_n8.json", "order": 2},
    {"label": "m4_n4", "kind": "json", "filename": "m4_n4.json", "order": 3},
    {"label": "m4_n8", "kind": "json", "filename": "m4_n8.json", "order": 4},
    {
        "label": "m2_n32",
        "kind": "stats",
        "path": M2_N32_STATS_LOG,
        "order": 5,
        "weight_reuse_m": 2,
        "activation_reuse_n": 32,
    },
    {"label": "m8_n32", "kind": "json", "path": M8_N32_JSON, "order": 6},
]

POINT_STYLES = {
    "Naive": {"color": "#b33951", "offset": (-24, -12)},
    "m2_n4": {"color": "#f28e2b", "offset": (20, 4)},
    "m2_n8": {"color": "#edc948", "offset": (18, -18)},
    "m4_n4": {"color": "#59a14f", "offset": (-20, 12)},
    "m4_n8": {"color": "#4e79a7", "offset": (20, 14)},
    "m2_n32": {"color": "#af7aa1", "offset": (-28, -4)},
    "m8_n32": {"color": "#76b7b2", "offset": (22, 22)},
}

ANNOTATION_BBOX = {
    "boxstyle": "round,pad=0.18",
    "facecolor": "#fffdfa",
    "edgecolor": "none",
    "alpha": 0.92,
}


def _algo_flops(matrix_size: int) -> float:
    size = int(matrix_size)
    return float(2 * size * size * size)


def _theoretical_peak_compute(tile_size: int) -> float:
    tile = int(tile_size)
    return float(2 * tile * tile)


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _load_stats_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing stats log: {path}")

    stats: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("[stats] "):
            continue
        payload = line[len("[stats] "):]
        key, separator, value = payload.partition(" ")
        if not separator:
            continue
        stats[key] = _parse_scalar(value)
    if not stats:
        raise SystemExit(f"no [stats] entries found in {path}")
    return stats


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing json input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return payload


def _normalize_row(payload: Dict[str, Any], *, label: str, order: int) -> Dict[str, Any]:
    row = dict(payload)
    row["policy_label"] = label
    row["name"] = label
    row["policy_order"] = int(order)
    row["status"] = "ok"

    matrix_size = int(row.get("matrix_size", 0) or 0)
    tile_size = int(row.get("tile_size", 0) or 0)
    cycles = float(row.get("cycles", 0.0) or 0.0)
    flops_algo = float(row.get("flops_algo", 0.0) or 0.0)
    flops_micro = float(row.get("flops_micro", 0.0) or 0.0)
    bytes_transmitted = float(row.get("bytes_transmitted", 0.0) or 0.0)
    bytes_internal = float(row.get("bytes_internal", 0.0) or 0.0)

    if flops_algo <= 0.0 and matrix_size > 0:
        flops_algo = _algo_flops(matrix_size)

    throughput_algo = float(row.get("throughput_algo_flops_per_cycle", 0.0) or 0.0)
    if throughput_algo <= 0.0 and flops_algo > 0.0 and cycles > 0.0:
        throughput_algo = flops_algo / cycles

    arithmetic_intensity_external_algo = float(
        row.get("arithmetic_intensity_external_algo_flops_per_byte", 0.0) or 0.0
    )
    if arithmetic_intensity_external_algo <= 0.0 and flops_algo > 0.0 and bytes_transmitted > 0.0:
        arithmetic_intensity_external_algo = flops_algo / bytes_transmitted

    arithmetic_intensity_external_micro = float(
        row.get("arithmetic_intensity_external_micro_flops_per_byte", 0.0) or 0.0
    )
    if arithmetic_intensity_external_micro <= 0.0 and flops_micro > 0.0 and bytes_transmitted > 0.0:
        arithmetic_intensity_external_micro = flops_micro / bytes_transmitted

    arithmetic_intensity_internal_algo = float(
        row.get("arithmetic_intensity_internal_algo_flops_per_byte", 0.0) or 0.0
    )
    if arithmetic_intensity_internal_algo <= 0.0 and flops_algo > 0.0 and bytes_internal > 0.0:
        arithmetic_intensity_internal_algo = flops_algo / bytes_internal

    theoretical_peak_compute = float(row.get("theoretical_peak_compute_flops_per_cycle", 0.0) or 0.0)
    if theoretical_peak_compute <= 0.0 and tile_size > 0:
        theoretical_peak_compute = _theoretical_peak_compute(tile_size)

    row.update(
        {
            "flops_algo": flops_algo,
            "throughput_algo_flops_per_cycle": throughput_algo,
            "arithmetic_intensity_external_algo_flops_per_byte": arithmetic_intensity_external_algo,
            "arithmetic_intensity_external_micro_flops_per_byte": arithmetic_intensity_external_micro,
            "arithmetic_intensity_internal_algo_flops_per_byte": arithmetic_intensity_internal_algo,
            "theoretical_peak_compute_flops_per_cycle": theoretical_peak_compute,
        }
    )
    return row


def _load_rows(sweep_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in POINT_SPECS:
        if spec["kind"] == "json":
            source_path = Path(spec.get("path") or sweep_dir / spec["filename"])
            payload = _load_json(source_path)
        else:
            source_path = Path(spec["path"])
            payload = _load_stats_log(source_path)
            if "weight_reuse_m" in spec:
                payload["weight_reuse_m"] = spec["weight_reuse_m"]
            if "activation_reuse_n" in spec:
                payload["activation_reuse_n"] = spec["activation_reuse_n"]

        rows.append(_normalize_row(payload, label=spec["label"], order=int(spec["order"])))
    return rows


def _resolve_peak_bandwidth(rows: List[Dict[str, Any]], source: str) -> Tuple[float, str]:
    if source == "naive-active":
        for row in rows:
            if row["policy_label"] == "Naive":
                value = float(row.get("external_bandwidth_active_bytes_per_cycle", 0.0) or 0.0)
                if value > 0.0:
                    return value, "baseline active external bandwidth roof"
        raise SystemExit("could not derive naive active bandwidth from the baseline stats log")

    if source == "max-active":
        values = [
            float(row.get("external_bandwidth_active_bytes_per_cycle", 0.0) or 0.0)
            for row in rows
            if float(row.get("external_bandwidth_active_bytes_per_cycle", 0.0) or 0.0) > 0.0
        ]
        if not values:
            raise SystemExit("no active external-bandwidth samples found in the selected rows")
        return max(values), "max active external bandwidth roof"

    values = [
        float(row.get("external_bandwidth_avg_bytes_per_cycle", 0.0) or 0.0)
        for row in rows
        if float(row.get("external_bandwidth_avg_bytes_per_cycle", 0.0) or 0.0) > 0.0
    ]
    if not values:
        raise SystemExit("no average external-bandwidth samples found in the selected rows")
    return max(values), "max average external bandwidth roof"


def _logspace(start: float, stop: float, count: int) -> List[float]:
    lo = math.log10(start)
    hi = math.log10(stop)
    return [10 ** (lo + (hi - lo) * idx / (count - 1)) for idx in range(count)]


def _axis_bounds(rows: List[Dict[str, Any]], peak_compute: float, peak_bandwidth: float) -> Tuple[float, float, float, float]:
    x_values = [float(row["arithmetic_intensity_external_algo_flops_per_byte"]) for row in rows]
    y_values = [float(row["throughput_algo_flops_per_cycle"]) for row in rows]
    knee_x = peak_compute / peak_bandwidth
    x_min = min(x_values) * 0.82
    x_max = max(max(x_values) * 1.2, knee_x * 1.22)
    y_min = min(y_values) * 0.72
    y_max = peak_compute * 1.55
    return x_min, x_max, y_min, y_max


def _annotate_point(ax: Any, label: str, x_val: float, y_val: float) -> None:
    style = POINT_STYLES[label]
    dx, dy = style["offset"]
    ax.annotate(
        label,
        (x_val, y_val),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=10,
        weight="bold",
        color="#2f2a24",
        bbox=ANNOTATION_BBOX,
        arrowprops={"arrowstyle": "-", "color": "#91887b", "linewidth": 0.8},
        ha="left" if dx >= 0 else "right",
        va="bottom" if dy >= 0 else "top",
        clip_on=False,
    )


def plot_roofline(rows: List[Dict[str, Any]], output_path: Path, bandwidth_source: str) -> Path:
    peak_compute = float(rows[0]["theoretical_peak_compute_flops_per_cycle"])
    peak_bandwidth, roof_label = _resolve_peak_bandwidth(rows, bandwidth_source)
    knee_x = peak_compute / peak_bandwidth

    x_lo, x_hi, y_lo, y_hi = _axis_bounds(rows, peak_compute, peak_bandwidth)
    roof_x = _logspace(x_lo, x_hi, 256)
    roof_y = [min(peak_compute, peak_bandwidth * intensity) for intensity in roof_x]

    plt.rcParams.update(
        {
            "axes.facecolor": "#fbf7ef",
            "axes.edgecolor": "#2f2a24",
            "figure.facecolor": "#fffdfa",
            "grid.color": "#cfc7b6",
            "grid.alpha": 0.35,
            "axes.grid": True,
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(10.25, 6.25))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.plot(roof_x, roof_y, color="#1f2a44", linewidth=2.9, label=roof_label)
    ax.axhline(
        peak_compute,
        color="#1f2a44",
        linewidth=1.6,
        linestyle="--",
        label=f"theoretical compute roof ({peak_compute:.0f} FLOPs/cycle)",
    )

    ax.scatter([knee_x], [peak_compute], color="#1f2a44", s=26, zorder=4)
    ax.annotate(
        f"knee\nAI={knee_x:.2f}",
        (knee_x, peak_compute),
        xytext=(14, -18),
        textcoords="offset points",
        fontsize=9,
        color="#1f2a44",
        ha="left",
        va="top",
    )

    for row in rows:
        label = str(row["policy_label"])
        x_val = float(row["arithmetic_intensity_external_algo_flops_per_byte"])
        y_val = float(row["throughput_algo_flops_per_cycle"])
        color = POINT_STYLES[label]["color"]
        ax.scatter([x_val], [y_val], s=105, color=color, edgecolors="#1f1f1f", linewidths=0.85, zorder=5)
        _annotate_point(ax, label, x_val, y_val)

    matrix_size = int(rows[0]["matrix_size"])
    tile_size = int(rows[0]["tile_size"])
    ax.set_title(f"{matrix_size}x{matrix_size} GEMM Reuse-Policy Roofline (tile={tile_size})")
    ax.set_xlabel("operational intensity (algorithmic FLOPs / external byte)")
    ax.set_ylabel("performance (algorithmic FLOPs / cycle)")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot the final-report roofline using the exact mixed-source policy points referenced in evaluation.tex."
    )
    parser.add_argument(
        "--sweep-dir",
        default=str(DEFAULT_SWEEP_DIR),
        help="Directory containing the m2_n4/m2_n8/m4_n4/m4_n8 JSON files that match the report table.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="PNG output path. Defaults to logs/final_report/figures/reuse_policy_roofline.png",
    )
    parser.add_argument(
        "--bandwidth-source",
        choices=("naive-active", "max-active", "max-avg"),
        default="naive-active",
        help="Bandwidth ceiling used for the roofline slope.",
    )
    args = parser.parse_args()

    rows = _load_rows(Path(args.sweep_dir))
    written = plot_roofline(rows, Path(args.output), args.bandwidth_source)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())