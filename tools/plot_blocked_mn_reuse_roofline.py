#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `/home/asicfab/a/socet149/sc_env_new/bin/python -m pip install matplotlib`."
    ) from exc


CASE_COLORS = {
    "Naive": "#b33951",
    "m2_n4": "#f28e2b",
    "m2_n8": "#edc948",
    "m4_n4": "#59a14f",
    "m4_n8": "#4e79a7",
    "m8_n32": "#76b7b2",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "logs" / "roofline_study" / "blocked_mn_reuse_1024"
LEGACY_INPUT_DIR = REPO_ROOT / "logs" / "blocked_mn_reuse_roofline_1024"
LATEST_RUN_FILE = "LATEST.txt"
DEFAULT_REFERENCE_JSONS = [
    REPO_ROOT
    / "logs"
    / "blocked_m8_n32_spad_frontend_queue_sweep"
    / "m8_n32_spad_frontend_queue_size_4.json",
]

ANNOTATION_OFFSETS = {
    "Naive": (-30, -12),
    "m2_n4": (18, 2),
    "m2_n8": (18, -18),
    "m4_n4": (-18, 10),
    "m4_n8": (12, 10),
    "m8_n32": (14, 20),
}

ANNOTATION_BBOX = {
    "boxstyle": "round,pad=0.18",
    "facecolor": "#fffdfa",
    "edgecolor": "none",
    "alpha": 0.9,
}


def _load_manifest(input_dir: Path) -> Dict[str, Any]:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_latest_run(input_root: Path) -> Optional[Path]:
    latest_path = input_root / LATEST_RUN_FILE
    if latest_path.exists():
        raw = latest_path.read_text(encoding="utf-8").strip()
        if raw:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = input_root / candidate
            if candidate.exists():
                return candidate

    candidates = [
        path for path in input_root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    ] if input_root.exists() else []
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _resolve_input_dir(input_dir: Optional[str], input_root: str) -> Path:
    if input_dir:
        return Path(input_dir)

    root_path = Path(input_root)
    latest = _resolve_latest_run(root_path)
    if latest is not None:
        return latest
    if LEGACY_INPUT_DIR.exists():
        return LEGACY_INPUT_DIR
    raise SystemExit(
        f"no roofline sweep run found under {root_path}; run tools/run_blocked_mn_reuse_roofline_sweep.py first or pass --input-dir"
    )


def _load_json_rows(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"results.json", "manifest.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.append(payload)
    if rows:
        return rows

    results_path = input_dir / "results.json"
    if not results_path.exists():
        return rows
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return rows
    for item in payload:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _algo_flops(matrix_size: int) -> float:
    size = int(matrix_size)
    return float(2 * size * size * size)


def _theoretical_peak_compute(tile_size: int) -> float:
    tile = int(tile_size)
    return float(2 * tile * tile)


def _derive_policy_label(row: Dict[str, Any]) -> str:
    policy_label = str(row.get("policy_label") or "").strip()
    if policy_label:
        return policy_label

    weight_reuse_m = int(row.get("weight_reuse_m", 0) or 0)
    activation_reuse_n = int(row.get("activation_reuse_n", 0) or 0)
    if weight_reuse_m > 0 and activation_reuse_n > 0:
        if weight_reuse_m == 1 and activation_reuse_n == 1:
            return "Naive"
        return f"m{weight_reuse_m}_n{activation_reuse_n}"

    name = str(row.get("name") or "case").strip()
    if name.lower() == "naive":
        return "Naive"
    return name or "case"


def _row_identity(row: Dict[str, Any]) -> str:
    return _derive_policy_label(row)


def _normalize_row(row: Dict[str, Any], *, default_policy_order: int) -> Dict[str, Any]:
    payload = dict(row)
    payload.setdefault("status", "ok")
    payload["policy_label"] = _derive_policy_label(payload)
    if payload.get("policy_order") in (None, ""):
        payload["policy_order"] = int(default_policy_order)

    matrix_size = int(payload.get("matrix_size", 0) or 0)
    tile_size = int(payload.get("tile_size", 0) or 0)
    cycles = float(payload.get("cycles", 0.0) or 0.0)
    flops_algo = float(payload.get("flops_algo", 0.0) or 0.0)
    flops_micro = float(payload.get("flops_micro", 0.0) or 0.0)
    bytes_transmitted = float(payload.get("bytes_transmitted", 0.0) or 0.0)
    bytes_internal = float(payload.get("bytes_internal", 0.0) or 0.0)

    if flops_algo <= 0.0 and matrix_size > 0:
        flops_algo = _algo_flops(matrix_size)

    throughput_algo = float(payload.get("throughput_algo_flops_per_cycle", 0.0) or 0.0)
    if throughput_algo <= 0.0 and flops_algo > 0.0 and cycles > 0.0:
        throughput_algo = flops_algo / cycles

    arithmetic_intensity_external_algo = float(
        payload.get("arithmetic_intensity_external_algo_flops_per_byte", 0.0) or 0.0
    )
    if arithmetic_intensity_external_algo <= 0.0 and flops_algo > 0.0 and bytes_transmitted > 0.0:
        arithmetic_intensity_external_algo = flops_algo / bytes_transmitted

    arithmetic_intensity_external_micro = float(
        payload.get("arithmetic_intensity_external_micro_flops_per_byte", 0.0) or 0.0
    )
    if arithmetic_intensity_external_micro <= 0.0 and flops_micro > 0.0 and bytes_transmitted > 0.0:
        arithmetic_intensity_external_micro = flops_micro / bytes_transmitted

    arithmetic_intensity_internal_algo = float(
        payload.get("arithmetic_intensity_internal_algo_flops_per_byte", 0.0) or 0.0
    )
    if arithmetic_intensity_internal_algo <= 0.0 and flops_algo > 0.0 and bytes_internal > 0.0:
        arithmetic_intensity_internal_algo = flops_algo / bytes_internal

    theoretical_peak_compute = float(payload.get("theoretical_peak_compute_flops_per_cycle", 0.0) or 0.0)
    if theoretical_peak_compute <= 0.0 and tile_size > 0:
        theoretical_peak_compute = _theoretical_peak_compute(tile_size)

    payload.update(
        {
            "flops_algo": flops_algo,
            "throughput_algo_flops_per_cycle": throughput_algo,
            "arithmetic_intensity_external_algo_flops_per_byte": arithmetic_intensity_external_algo,
            "arithmetic_intensity_external_micro_flops_per_byte": arithmetic_intensity_external_micro,
            "arithmetic_intensity_internal_algo_flops_per_byte": arithmetic_intensity_internal_algo,
            "theoretical_peak_compute_flops_per_cycle": theoretical_peak_compute,
        }
    )
    return payload


def _load_reference_rows(reference_paths: List[Path], existing_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    identities = {_row_identity(row) for row in existing_rows}
    next_policy_order = max(
        (int(row.get("policy_order", 0) or 0) for row in existing_rows),
        default=-1,
    ) + 1

    reference_rows: List[Dict[str, Any]] = []
    for path in reference_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue

        row = _normalize_row(payload, default_policy_order=next_policy_order)
        identity = _row_identity(row)
        if identity in identities:
            continue

        reference_rows.append(row)
        identities.add(identity)
        next_policy_order += 1
    return reference_rows


def _ok_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("status", "ok")) == "ok"]
    filtered.sort(key=lambda row: int(row.get("policy_order", 0) or 0))
    return filtered


def _positive_or(default: float, *values: float) -> float:
    positives = [value for value in values if value > 0.0]
    return max(positives) if positives else default


def _logspace(start: float, stop: float, count: int) -> List[float]:
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("logspace bounds must be positive")
    if count <= 1:
        return [start]
    lo = math.log10(start)
    hi = math.log10(stop)
    return [10 ** (lo + (hi - lo) * idx / (count - 1)) for idx in range(count)]


def _axis_bounds(
    rows: List[Dict[str, Any]],
    peak_compute: float,
    peak_bandwidth: float,
    empirical_peak_compute: float = 0.0,
) -> Tuple[float, float, float, float]:
    x_values = [
        float(row.get("arithmetic_intensity_external_algo_flops_per_byte", 0.0) or 0.0)
        for row in rows
        if float(row.get("arithmetic_intensity_external_algo_flops_per_byte", 0.0) or 0.0) > 0.0
    ]
    y_values = [
        float(row.get("throughput_algo_flops_per_cycle", 0.0) or 0.0)
        for row in rows
        if float(row.get("throughput_algo_flops_per_cycle", 0.0) or 0.0) > 0.0
    ]
    knee_x = (peak_compute / peak_bandwidth) if peak_compute > 0.0 and peak_bandwidth > 0.0 else 0.0
    empirical_knee_x = (
        (empirical_peak_compute / peak_bandwidth)
        if empirical_peak_compute > 0.0 and peak_bandwidth > 0.0
        else 0.0
    )
    x_markers = [value for value in (knee_x, empirical_knee_x) if value > 0.0]
    y_markers = [value for value in (peak_compute, empirical_peak_compute) if value > 0.0]
    x_min = min(x_values + x_markers) if x_values or x_markers else 1.0
    x_max = max(x_values + x_markers) if x_values or x_markers else 10.0
    y_min = min(y_values + y_markers) if y_values or y_markers else 1.0
    y_max = max(y_values + y_markers) if y_values or y_markers else 10.0
    return x_min / 1.8, x_max * 1.8, y_min / 1.8, y_max * 1.8


def _set_annotation_alignment(annotation: Any) -> None:
    dx, dy = annotation.get_position()
    annotation.set_ha("left" if dx >= 0 else "right")
    annotation.set_va("bottom" if dy >= 0 else "top")


def _repel_annotations(fig: Any, annotations: List[Any]) -> None:
    if len(annotations) < 2:
        return

    max_abs_offset = 44
    for _ in range(24):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [annotation.get_window_extent(renderer=renderer).expanded(1.04, 1.16) for annotation in annotations]
        moved = False

        for idx in range(1, len(annotations)):
            current = annotations[idx]
            current_bbox = bboxes[idx]
            for prev_idx in range(idx):
                if not current_bbox.overlaps(bboxes[prev_idx]):
                    continue

                previous = annotations[prev_idx]
                dx, dy = current.get_position()
                shift_x = 6 if current.xy[0] >= previous.xy[0] else -6
                shift_y = 10 if current.xy[1] >= previous.xy[1] else -10
                next_dx = dx + shift_x
                next_dy = dy + shift_y

                if abs(next_dx) > max_abs_offset:
                    next_dx = dx - shift_x
                if abs(next_dy) > max_abs_offset:
                    next_dy = dy - shift_y

                current.set_position((next_dx, next_dy))
                _set_annotation_alignment(current)
                moved = True
                break

            if moved:
                break

        if not moved:
            return


def _plot_policy_points(ax: Any, fig: Any, rows: List[Dict[str, Any]]) -> None:
    annotations: List[Any] = []
    for row in rows:
        label = str(row.get("policy_label") or row.get("name") or "case")
        x_val = float(row.get("arithmetic_intensity_external_algo_flops_per_byte", 0.0) or 0.0)
        y_val = float(row.get("throughput_algo_flops_per_cycle", 0.0) or 0.0)
        if x_val <= 0.0 or y_val <= 0.0:
            continue

        color = CASE_COLORS.get(label, "#4e79a7")
        ax.scatter([x_val], [y_val], s=90, color=color, edgecolors="#1f1f1f", linewidths=0.8, zorder=5)

        dx, dy = ANNOTATION_OFFSETS.get(label, (10, 8))
        annotation = ax.annotate(
            label,
            (x_val, y_val),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            color="#2f2a24",
            bbox=ANNOTATION_BBOX,
            clip_on=False,
        )
        _set_annotation_alignment(annotation)
        annotations.append(annotation)

    _repel_annotations(fig, annotations)


def plot_roofline(rows: List[Dict[str, Any]], manifest: Dict[str, Any], output_path: Path, bandwidth_source: str) -> Path:
    peak_compute = float(manifest.get("theoretical_peak_compute_flops_per_cycle", 0.0) or 0.0)
    empirical_peak_compute = float(manifest.get("empirical_peak_compute_algo_flops_per_cycle", 0.0) or 0.0)
    if bandwidth_source == "avg":
        peak_bandwidth = float(manifest.get("empirical_peak_external_bandwidth_avg_bytes_per_cycle", 0.0) or 0.0)
        bandwidth_label = "empirical external bandwidth roof (avg)"
    else:
        peak_bandwidth = float(manifest.get("empirical_peak_external_bandwidth_active_bytes_per_cycle", 0.0) or 0.0)
        bandwidth_label = "empirical external bandwidth roof (active)"

    peak_compute = _positive_or(1.0, peak_compute)
    empirical_peak_compute = _positive_or(0.0, empirical_peak_compute)
    peak_bandwidth = _positive_or(1.0, peak_bandwidth)

    if empirical_peak_compute <= 0.0:
        empirical_peak_compute = _positive_or(
            0.0,
            *(float(row.get("throughput_algo_flops_per_cycle", 0.0) or 0.0) for row in rows),
        )

    x_lo, x_hi, y_lo, y_hi = _axis_bounds(rows, peak_compute, peak_bandwidth, empirical_peak_compute)
    roof_x = _logspace(max(x_lo, 1e-3), max(x_hi, x_lo * 10.0), 256)
    roof_y = [min(peak_compute, peak_bandwidth * intensity) for intensity in roof_x]
    knee_x = peak_compute / peak_bandwidth if peak_bandwidth > 0.0 else 0.0
    empirical_knee_x = empirical_peak_compute / peak_bandwidth if peak_bandwidth > 0.0 else 0.0

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

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(x_lo, 1e-3), max(x_hi, x_lo * 10.0))
    ax.set_ylim(max(y_lo, 1e-3), max(y_hi, y_lo * 10.0))

    ax.plot(roof_x, roof_y, color="#1f2a44", linewidth=2.8, label=bandwidth_label)
    ax.axhline(
        peak_compute,
        color="#1f2a44",
        linewidth=1.5,
        linestyle="--",
        label=f"theoretical compute roof ({peak_compute:.0f} FLOPs/cycle)",
    )
    if empirical_peak_compute > 0.0:
        ax.axhline(
            empirical_peak_compute,
            color="#2e8b57",
            linewidth=1.5,
            linestyle=":",
            label=f"empirical compute roof ({empirical_peak_compute:.0f} FLOPs/cycle)",
        )
    if knee_x > 0.0:
        ax.scatter([knee_x], [peak_compute], color="#1f2a44", s=22, zorder=4)
        ax.annotate(
            f"knee\nAI={knee_x:.2f}",
            (knee_x, peak_compute),
            xytext=(8, -28),
            textcoords="offset points",
            fontsize=9,
            color="#1f2a44",
        )
    if empirical_peak_compute > 0.0 and empirical_knee_x > 0.0:
        ax.scatter([empirical_knee_x], [empirical_peak_compute], color="#2e8b57", s=22, zorder=4)
        ax.annotate(
            f"empirical knee\nAI={empirical_knee_x:.2f}",
            (empirical_knee_x, empirical_peak_compute),
            xytext=(-24, 8),
            textcoords="offset points",
            fontsize=9,
            color="#2e8b57",
        )

    _plot_policy_points(ax, fig, rows)

    matrix_size = manifest.get("matrix_size", "?")
    tile_size = manifest.get("tile_size", "?")
    ax.set_title(f"1024x1024 Blocked GEMM Reuse Roofline (tile={tile_size})")
    ax.set_xlabel("operational intensity (algorithmic FLOPs / external byte)")
    ax.set_ylabel("performance (algorithmic FLOPs / cycle)")
    ax.legend(loc="lower right", fontsize=9)

    subtitle = (
        f"matrix={matrix_size}  theoretical peak={peak_compute:.0f} FLOPs/cycle  "
        f"empirical peak={empirical_peak_compute:.0f} FLOPs/cycle  "
        f"{bandwidth_label}={peak_bandwidth:.2f} B/cycle"
    )
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=10, color="#4b4339")

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot a roofline comparing blocked-M/N GEMM reuse policies from the reuse sweep outputs."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing per-run JSON files from run_blocked_mn_reuse_roofline_sweep.py.",
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Root directory used to auto-discover the latest sweep run when --input-dir is omitted.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="PNG output path. Defaults to <input-dir>/plots/blocked_mn_reuse_roofline.png",
    )
    parser.add_argument(
        "--bandwidth-source",
        choices=("active", "avg"),
        default="active",
        help="Which empirical external-bandwidth ceiling to use for the roofline slope.",
    )
    parser.add_argument(
        "--reference-json",
        nargs="*",
        default=None,
        help=(
            "Optional per-run JSON files to overlay as extra roofline points. "
            "When omitted, the plotter auto-loads the default m8_n32 queue-size-4 reference if it exists."
        ),
    )
    parser.add_argument(
        "--no-default-references",
        action="store_true",
        help="Disable auto-loading the built-in supplemental reference points.",
    )
    args = parser.parse_args()

    input_dir = _resolve_input_dir(args.input_dir, args.input_root)
    output_path = (
        Path(args.output)
        if args.output
        else input_dir / "plots" / "blocked_mn_reuse_roofline.png"
    )

    manifest = _load_manifest(input_dir)
    base_rows = [
        _normalize_row(row, default_policy_order=index)
        for index, row in enumerate(_ok_rows(_load_json_rows(input_dir)))
    ]
    if not base_rows:
        raise SystemExit(f"no successful run JSON files found in {input_dir}")

    reference_paths: List[Path] = []
    if not args.no_default_references:
        reference_paths.extend(DEFAULT_REFERENCE_JSONS)
    if args.reference_json:
        reference_paths.extend(Path(value) for value in args.reference_json)

    rows = _ok_rows(base_rows + _load_reference_rows(reference_paths, base_rows))

    print(f"using {input_dir}")
    written = plot_roofline(rows, manifest, output_path, args.bandwidth_source)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())