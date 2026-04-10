#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `python3.11 -m pip install matplotlib`."
    ) from exc


PATH_ORDER = [
    "kernel_total",
    "prefetch_window",
    "compute_window",
    "sdma_act",
    "sdma_wgt",
    "vls_wgt",
    "gsau_wgt",
    "vls_act",
    "gsau_act",
    "systolic_array",
    "gsau_rsp",
    "vls_acc_load",
    "datapath_add",
    "vls_acc_store",
]


PATH_LABELS = {
    "kernel_total": "kernel total",
    "prefetch_window": "prefetch ready",
    "compute_window": "compute window",
    "sdma_act": "SDMA act",
    "sdma_wgt": "SDMA wgt",
    "vls_wgt": "VLS weight",
    "gsau_wgt": "GSAU weight",
    "vls_act": "VLS act",
    "gsau_act": "GSAU act",
    "systolic_array": "systolic array",
    "gsau_rsp": "GSAU rsp",
    "vls_acc_load": "VLS acc load",
    "datapath_add": "datapath add",
    "vls_acc_store": "VLS acc store",
}


PATH_COLORS = {
    "kernel_total": "#4d4d4d",
    "prefetch_window": "#9c755f",
    "compute_window": "#2f4b7c",
    "sdma_act": "#4c78a8",
    "sdma_wgt": "#f58518",
    "vls_wgt": "#54a24b",
    "gsau_wgt": "#eeca3b",
    "vls_act": "#b279a2",
    "gsau_act": "#ff9da6",
    "systolic_array": "#e45756",
    "gsau_rsp": "#72b7b2",
    "vls_acc_load": "#9d755d",
    "datapath_add": "#8cd17d",
    "vls_acc_store": "#b6992d",
}


ENVELOPE_PATHS = {"kernel_total", "prefetch_window", "compute_window"}
TILE_DRAIN_LABEL = "tile drain to DRAM"
TILE_DRAIN_COLOR = "#bab0ac"

COMPUTE_DONE_RE = re.compile(
    r"cycle\s+(?P<cycle>\d+):\s+compute\s+done\s+ti=(?P<ti>\d+)\s+tj=(?P<tj>\d+)\s+tk=(?P<tk>\d+)\s+slot=(?P<slot>\d+)"
)
OUTPUT_TILE_COMPLETE_RE = re.compile(
    r"cycle\s+(?P<cycle>\d+):\s+output\s+tile\s+complete\s+ti=(?P<ti>\d+)\s+tj=(?P<tj>\d+)"
)
GANTT_LOG_RE = re.compile(
    r"^\[gantt\]\s+"
    r"(?P<tag>\S+)\s+"
    r"(?P<ti>\d+)\s+"
    r"(?P<tj>\d+)\s+"
    r"(?P<tk>\d+)\s+"
    r"(?P<slot>\d+)\s+"
    r"(?P<path>\S+)\s+"
    r"(?P<start_cycle>\d+)\s+"
    r"(?P<end_cycle>\d+)\s+"
    r"(?P<duration_cycles>\d+)\s+"
    r"(?P<touches>\d+)\s*$"
)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            match = GANTT_LOG_RE.match(line.strip())
            if match is None:
                continue
            row = {key: value for key, value in match.groupdict().items()}
            rows.append(row)
    return rows


def _to_int(row: Dict[str, str], key: str) -> int:
    return int(row[key])


def _row_sort_key(row: Dict[str, str]):
    return (
        _to_int(row, "ti"),
        _to_int(row, "tj"),
        _to_int(row, "tk"),
        _to_int(row, "slot"),
        PATH_ORDER.index(row["path"]) if row["path"] in PATH_ORDER else len(PATH_ORDER),
    )


def _tag_sort_key(row: Dict[str, str]):
    return (
        _to_int(row, "ti"),
        _to_int(row, "tj"),
        _to_int(row, "tk"),
        _to_int(row, "slot"),
    )


def _filter_rows(
    rows: Sequence[Dict[str, str]],
    *,
    ti: int | None,
    tj: int | None,
    tags: Sequence[str],
    max_tags: int | None,
    include_envelopes: bool,
) -> List[Dict[str, str]]:
    filtered = [row for row in rows if int(row.get("duration_cycles", "0")) > 0]
    if tags:
        selected = set(tags)
        filtered = [row for row in filtered if row["tag"] in selected]
    else:
        if ti is not None:
            filtered = [row for row in filtered if _to_int(row, "ti") == int(ti)]
        if tj is not None:
            filtered = [row for row in filtered if _to_int(row, "tj") == int(tj)]

    if not include_envelopes:
        filtered = [row for row in filtered if row["path"] not in ENVELOPE_PATHS]

    ordered_tags: List[str] = []
    seen = set()
    for row in sorted(filtered, key=_tag_sort_key):
        tag = row["tag"]
        if tag not in seen:
            seen.add(tag)
            ordered_tags.append(tag)
    if max_tags is not None:
        keep = set(ordered_tags[: int(max_tags)])
        filtered = [row for row in filtered if row["tag"] in keep]

    return sorted(filtered, key=_row_sort_key)


def _selected_tile_key(rows: Sequence[Dict[str, str]]) -> Optional[Tuple[int, int]]:
    keys = sorted({(_to_int(row, "ti"), _to_int(row, "tj")) for row in rows})
    if len(keys) != 1:
        return None
    return keys[0]


def _infer_tile_drain_interval(schedule_path: Path, ti: int, tj: int) -> Optional[Tuple[int, int]]:
    last_compute_done: Optional[int] = None
    tile_complete: Optional[int] = None

    with schedule_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            compute_done = COMPUTE_DONE_RE.search(line)
            if compute_done is not None:
                if int(compute_done.group("ti")) == int(ti) and int(compute_done.group("tj")) == int(tj):
                    cycle = int(compute_done.group("cycle"))
                    if last_compute_done is None or cycle > last_compute_done:
                        last_compute_done = cycle
                continue

            output_done = OUTPUT_TILE_COMPLETE_RE.search(line)
            if output_done is not None:
                if int(output_done.group("ti")) == int(ti) and int(output_done.group("tj")) == int(tj):
                    tile_complete = int(output_done.group("cycle"))

    if last_compute_done is None or tile_complete is None or tile_complete <= last_compute_done:
        return None
    return (last_compute_done, tile_complete)


def plot_gantt(
    rows: Sequence[Dict[str, str]],
    output_path: Path,
    title: str,
    *,
    row_mode: str = "tag",
    tile_drain_interval: Optional[Tuple[int, int]] = None,
) -> Path:
    tags: List[str] = []
    tag_meta: Dict[str, Dict[str, str]] = {}
    for row in rows:
        tag = row["tag"]
        if tag not in tag_meta:
            tag_meta[tag] = row
            tags.append(tag)

    if row_mode == "slot":
        slot_keys = sorted({_to_int(row, "slot") for row in rows})
        fig_height = max(3.4, 0.95 * len(slot_keys) + 1.8)
    else:
        slot_keys = []
        fig_height = max(4.0, 0.38 * len(tags) + 1.8)

    fig, ax = plt.subplots(figsize=(16, fig_height))

    if tile_drain_interval is not None:
        drain_start, drain_end = tile_drain_interval
        ax.axvspan(
            drain_start,
            drain_end,
            facecolor=TILE_DRAIN_COLOR,
            edgecolor="#7f7f7f",
            linewidth=1.0,
            alpha=0.20,
            zorder=0,
        )

    if row_mode == "slot":
        y_positions = {slot: idx for idx, slot in enumerate(slot_keys)}
    else:
        y_positions = {tag: idx for idx, tag in enumerate(tags)}

    used_paths: List[str] = []
    for row in rows:
        path = row["path"]
        if path not in used_paths:
            used_paths.append(path)
        if row_mode == "slot":
            y = y_positions[_to_int(row, "slot")]
        else:
            y = y_positions[row["tag"]]
        start = _to_int(row, "start_cycle")
        width = max(1, _to_int(row, "duration_cycles"))
        color = PATH_COLORS.get(path, "#808080")
        alpha = 0.28 if path in ENVELOPE_PATHS else 0.92
        linewidth = 1.2 if path in ENVELOPE_PATHS else 0.5
        edgecolor = color if path in ENVELOPE_PATHS else "black"
        ax.barh(
            y,
            width,
            left=start,
            height=0.72,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        if row_mode == "slot" and path == "compute_window" and width >= 20:
            ax.text(
                start + (width / 2.0),
                y,
                f"tk={_to_int(row, 'tk'):02d}",
                ha="center",
                va="center",
                fontsize=7,
                color="#1f1f1f",
                clip_on=True,
                zorder=4,
            )

    ax.set_title(title)
    ax.set_xlabel("cycle")
    if row_mode == "slot":
        ax.set_ylabel("prefetch slot")
        ax.set_yticks(range(len(slot_keys)))
        ax.set_yticklabels([f"slot={slot}" for slot in slot_keys])
    else:
        ax.set_ylabel("microkernel tag")
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels(
            [
                f"tk={_to_int(tag_meta[tag], 'tk'):02d} slot={_to_int(tag_meta[tag], 'slot')}"
                for tag in tags
            ]
        )
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)

    legend_handles = [
        Patch(color=PATH_COLORS.get(path, "#808080"), label=PATH_LABELS.get(path, path), alpha=(0.28 if path in ENVELOPE_PATHS else 0.92))
        for path in PATH_ORDER
        if path in used_paths
    ]
    if tile_drain_interval is not None:
        legend_handles.append(Patch(color=TILE_DRAIN_COLOR, label=TILE_DRAIN_LABEL, alpha=0.20))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot per-kernel tagged spans for the tiled TPU harness.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to gantt.log or legacy kernel_gantt_spans.csv. Defaults to logs/sysarr_gemm_tpu_tiled_1024/gantt.log",
    )
    parser.add_argument("--output", default=None, help="PNG output path")
    parser.add_argument("--ti", type=int, default=0, help="Filter to one output-tile row index")
    parser.add_argument("--tj", type=int, default=0, help="Filter to one output-tile column index")
    parser.add_argument("--tag", action="append", default=[], help="Exact kernel tag to include. Repeatable.")
    parser.add_argument("--max-tags", type=int, default=None, help="Maximum number of tags to draw after filtering")
    parser.add_argument(
        "--row-mode",
        choices=["tag", "slot"],
        default="tag",
        help="Draw one row per microkernel tag or collapse all kernels onto one row per prefetch slot.",
    )
    parser.add_argument(
        "--include-envelopes",
        action="store_true",
        help="Include kernel_total, prefetch_window, and compute_window spans in the plot",
    )
    parser.add_argument(
        "--include-tile-drain",
        action="store_true",
        help="Infer and overlay the coarse post-compute output-tile drain-to-DRAM interval from schedule.log.",
    )
    parser.add_argument(
        "--schedule-log",
        default=None,
        help="Path to schedule.log used when --include-tile-drain is requested. Defaults to <input dir>/schedule.log.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input) if args.input else repo_root / "logs" / "sysarr_gemm_tpu_tiled_1024" / "gantt.log"
    rows = _read_rows(input_path)
    if not rows:
        raise SystemExit(f"no rows found in {input_path}")

    filtered = _filter_rows(
        rows,
        ti=(None if args.tag else args.ti),
        tj=(None if args.tag else args.tj),
        tags=args.tag,
        max_tags=args.max_tags,
        include_envelopes=bool(args.include_envelopes),
    )
    if not filtered:
        raise SystemExit("no matching span rows for the requested filter")

    tile_drain_interval: Optional[Tuple[int, int]] = None
    if args.include_tile_drain:
        tile_key = _selected_tile_key(filtered)
        if tile_key is None:
            raise SystemExit("--include-tile-drain requires rows from exactly one output tile")
        schedule_path = Path(args.schedule_log) if args.schedule_log else input_path.parent / "schedule.log"
        if not schedule_path.exists():
            raise SystemExit(f"schedule log not found: {schedule_path}")
        tile_drain_interval = _infer_tile_drain_interval(schedule_path, tile_key[0], tile_key[1])
        if tile_drain_interval is None:
            raise SystemExit(
                f"could not infer tile drain interval for ti={tile_key[0]} tj={tile_key[1]} from {schedule_path}"
            )

    if args.output:
        output_path = Path(args.output)
    elif args.tag:
        output_path = input_path.parent / "kernel_gantt_selected_tags.png"
    else:
        stem = f"kernel_gantt_ti{int(args.ti):02d}_tj{int(args.tj):02d}"
        if args.row_mode == "slot":
            stem += "_slots"
        if args.include_tile_drain:
            stem += "_tile_drain"
        output_path = input_path.parent / f"{stem}.png"

    if args.tag:
        title = "Tiled TPU tagged microkernel Gantt"
    else:
        if args.row_mode == "slot":
            title = f"Tiled TPU slot-collapsed Gantt (ti={int(args.ti)}, tj={int(args.tj)})"
        else:
            title = f"Tiled TPU tagged microkernel Gantt (ti={int(args.ti)}, tj={int(args.tj)})"
    output_path = plot_gantt(
        filtered,
        output_path,
        title,
        row_mode=args.row_mode,
        tile_drain_interval=tile_drain_interval,
    )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
