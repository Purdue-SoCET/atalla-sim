#!/usr/bin/env python3

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    "vls_wgt_shared",
    "gsau_wgt",
    "gsau_wgt_shared",
    "vls_act",
    "gsau_act",
    "systolic_array",
    "gsau_rsp",
    "vls_psum_load",
    "vls_acc_load",
    "datapath_add",
    "vls_psum_store",
    "vls_acc_store",
]


PATH_LABELS = {
    "kernel_total": "kernel total",
    "prefetch_window": "prefetch ready",
    "compute_window": "compute window",
    "sdma_act": "SDMA act",
    "sdma_wgt": "SDMA wgt",
    "vls_wgt": "VLS weight",
    "vls_wgt_shared": "VLS shared weight",
    "gsau_wgt": "GSAU weight",
    "gsau_wgt_shared": "GSAU shared weight",
    "vls_act": "VLS act",
    "gsau_act": "GSAU act",
    "systolic_array": "systolic array",
    "gsau_rsp": "GSAU rsp",
    "vls_psum_load": "VLS psum load",
    "vls_acc_load": "VLS acc load",
    "datapath_add": "datapath add",
    "vls_psum_store": "VLS psum store",
    "vls_acc_store": "VLS acc store",
}


PATH_COLORS = {
    "kernel_total": "#4d4d4d",
    "prefetch_window": "#9c755f",
    "compute_window": "#2f4b7c",
    "sdma_act": "#4c78a8",
    "sdma_wgt": "#f58518",
    "vls_wgt": "#54a24b",
    "vls_wgt_shared": "#6fb95f",
    "gsau_wgt": "#eeca3b",
    "gsau_wgt_shared": "#f1d35d",
    "vls_act": "#b279a2",
    "gsau_act": "#ff9da6",
    "systolic_array": "#e45756",
    "gsau_rsp": "#72b7b2",
    "vls_psum_load": "#9d755d",
    "vls_acc_load": "#9d755d",
    "datapath_add": "#8cd17d",
    "vls_psum_store": "#b6992d",
    "vls_acc_store": "#b6992d",
}


ENVELOPE_PATHS = {"kernel_total", "prefetch_window", "compute_window"}
TILE_DRAIN_LABEL = "tile drain to DRAM"
TILE_DRAIN_COLOR = "#bab0ac"

WEIGHT_INPUT_PATHS = ("sdma_wgt", "vls_wgt", "vls_wgt_shared", "gsau_wgt", "gsau_wgt_shared")
ACTIVATION_INPUT_PATHS = ("sdma_act", "vls_act", "gsau_act")
ACTIVATION_PRELOAD_PATHS = ("sdma_act",)
ACTIVATION_FEED_PATHS = ("vls_act", "gsau_act")
STORE_PATHS = ("gsau_rsp", "vls_psum_load", "vls_acc_load", "datapath_add", "vls_psum_store", "vls_acc_store")
STORE_TAIL_PATHS = ("vls_psum_store", "vls_acc_store")
PSUM_RELOAD_PATHS = ("vls_psum_load", "vls_acc_load")
ADD_STORE_TAIL_PATHS = ("datapath_add", "vls_psum_store", "vls_acc_store")
PRESENTATION_WEIGHT_COLORS = [
    "#c44e52",
    "#4e79a7",
    "#59a14f",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#9c755f",
]

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

STATS_LOG_RE = re.compile(r"^\[stats\]\s+(?P<key>\S+)\s+(?P<value>.*)$")


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
    tk: int | None,
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
        if tk is not None:
            filtered = [row for row in filtered if _to_int(row, "tk") == int(tk)]

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


def _make_tag_label(tag_meta: Dict[str, str], rows: Sequence[Dict[str, str]]) -> str:
    ti_values = sorted({_to_int(row, "ti") for row in rows})
    tj_values = sorted({_to_int(row, "tj") for row in rows})
    tk_values = sorted({_to_int(row, "tk") for row in rows})

    parts: List[str] = []
    if len(ti_values) > 1:
        parts.append(f"ti={_to_int(tag_meta, 'ti'):02d}")
    if len(tj_values) > 1:
        parts.append(f"tj={_to_int(tag_meta, 'tj'):02d}")
    if len(tk_values) > 1 or not parts:
        parts.append(f"tk={_to_int(tag_meta, 'tk'):02d}")
    parts.append(f"slot={_to_int(tag_meta, 'slot')}")
    return " ".join(parts)


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


def _median_duration(rows: Sequence[Dict[str, str]], preferred_paths: Sequence[str]) -> int:
    preferred = [
        _to_int(row, "duration_cycles")
        for row in rows
        if row["path"] in preferred_paths and _to_int(row, "duration_cycles") > 0
    ]
    durations = preferred or [
        _to_int(row, "duration_cycles")
        for row in rows
        if _to_int(row, "duration_cycles") > 0
    ]
    if not durations:
        return 1
    durations = sorted(durations)
    return durations[len(durations) // 2]


def _activity_clusters(
    rows: Sequence[Dict[str, str]],
    *,
    tile_drain_interval: Optional[Tuple[int, int]] = None,
) -> List[Tuple[int, int]]:
    intervals = [
        (_to_int(row, "start_cycle"), _to_int(row, "end_cycle"))
        for row in rows
        if _to_int(row, "duration_cycles") > 0
    ]
    if tile_drain_interval is not None:
        intervals.append(tile_drain_interval)
    if not intervals:
        return []

    gap_threshold = max(4096, 20 * _median_duration(rows, ("kernel_total", "compute_window")))
    merged: List[List[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + gap_threshold:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _parse_stat_value(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def read_stats_log(path: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    if not path.exists():
        return stats
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            match = STATS_LOG_RE.match(line.strip())
            if match is None:
                continue
            stats[match.group("key")] = _parse_stat_value(match.group("value"))
    return stats


def _rows_by_tag(rows: Sequence[Dict[str, str]]) -> List[Tuple[str, List[Dict[str, str]]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["tag"], []).append(row)
    ordered_tags = sorted(grouped.keys(), key=lambda tag: _tag_sort_key(grouped[tag][0]))
    return [(tag, grouped[tag]) for tag in ordered_tags]


def _span_for_paths(rows: Sequence[Dict[str, str]], paths: Sequence[str]) -> Optional[Tuple[int, int]]:
    selected = [row for row in rows if row["path"] in paths and _to_int(row, "duration_cycles") > 0]
    if not selected:
        return None
    return (
        min(_to_int(row, "start_cycle") for row in selected),
        max(_to_int(row, "end_cycle") for row in selected),
    )


def _gap_between(
    earlier: Optional[Tuple[int, int]],
    later: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if earlier is None or later is None:
        return None
    start = earlier[1]
    end = later[0]
    if end <= start:
        return None
    return (start, end)


def _weight_color(weight_id: int) -> str:
    return PRESENTATION_WEIGHT_COLORS[int(weight_id) % len(PRESENTATION_WEIGHT_COLORS)]


def _local_limits(intervals: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    start = min(start for start, _ in intervals)
    end = max(end for _, end in intervals)
    return start, end


def _draw_bar(ax, y: float, interval: Optional[Tuple[int, int]], *, origin: int, height: float, color: str, alpha: float, label: Optional[str] = None, edgecolor: str = "#333333", linewidth: float = 0.8, hatch: Optional[str] = None, text_color: str = "#1f1f1f") -> None:
    if interval is None:
        return
    start, end = interval
    width = max(1, end - start)
    ax.barh(
        y,
        width,
        left=start - origin,
        height=height,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        hatch=hatch,
    )
    if label and width >= 180:
        ax.text(start - origin + (width / 2.0), y, label, ha="center", va="center", fontsize=8, color=text_color)


def _presentation_style(ax, title: str, y_labels: Sequence[str], *, xlabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(list(y_labels))
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.22)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_presentation_block_overview(
    rows: Sequence[Dict[str, str]],
    output_path: Path,
    title: str,
    *,
    activation_count: Optional[int] = None,
) -> Path:
    by_weight: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        by_weight.setdefault(_to_int(row, "tj"), []).append(row)
    weight_ids = sorted(by_weight)
    if not weight_ids:
        raise ValueError("presentation block overview needs at least one weight id")

    summaries: List[Dict[str, Any]] = []
    all_intervals: List[Tuple[int, int]] = []
    for weight_id in weight_ids:
        weight_rows = by_weight[weight_id]
        load_interval = _span_for_paths(weight_rows, WEIGHT_INPUT_PATHS)
        stream_interval = _span_for_paths(weight_rows, ("compute_window", "kernel_total"))
        store_interval = _span_for_paths(weight_rows, STORE_TAIL_PATHS)
        for interval in (load_interval, stream_interval, store_interval):
            if interval is not None:
                all_intervals.append(interval)
        ti_values = sorted({_to_int(row, "ti") for row in weight_rows})
        activation_label = f"A{ti_values[0]:02d}..A{ti_values[-1]:02d}" if ti_values else "activations"
        if activation_count is not None and ti_values and len(ti_values) == activation_count:
            activation_label = f"{activation_count} activation tiles"
        summaries.append(
            {
                "weight_id": weight_id,
                "load": load_interval,
                "stream": stream_interval,
                "store": store_interval,
                "activation_label": activation_label,
            }
        )

    origin, end = _local_limits(all_intervals)
    pad = max(64, int(0.03 * max(1, end - origin)))
    fig_height = max(5.8, 0.72 * len(weight_ids) + 1.8)
    fig, ax = plt.subplots(figsize=(15, fig_height))

    y_labels = [f"Weight W{summary['weight_id']:02d}" for summary in summaries]
    for y, summary in enumerate(summaries):
        weight_color = _weight_color(summary["weight_id"])
        _draw_bar(
            ax,
            y,
            summary["stream"],
            origin=origin,
            height=0.62,
            color=weight_color,
            alpha=0.38,
            label=summary["activation_label"],
            edgecolor=weight_color,
            linewidth=1.0,
        )
        _draw_bar(
            ax,
            y,
            summary["load"],
            origin=origin,
            height=0.62,
            color=weight_color,
            alpha=0.92,
            edgecolor="#2f2f2f",
            linewidth=0.8,
        )
        _draw_bar(
            ax,
            y,
            summary["store"],
            origin=origin,
            height=0.34,
            color="#c8a453",
            alpha=0.95,
            edgecolor="#7f6d26",
            linewidth=0.8,
            hatch="//",
        )

    _presentation_style(ax, title, y_labels, xlabel="cycles inside the first reuse block")
    ax.set_xlim(-pad, (end - origin) + pad)
    legend_handles = [
        Patch(color="#7b8ba3", alpha=0.38, label="activations streamed past one resident weight"),
        Patch(color="#4e79a7", alpha=0.92, label="weight tile moving into the TPU"),
        Patch(color="#c8a453", alpha=0.95, label="accumulate and write-back tail", hatch="//"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_presentation_weight_flow(
    rows: Sequence[Dict[str, str]],
    output_path: Path,
    title: str,
) -> Path:
    grouped = _rows_by_tag(rows)
    if not grouped:
        raise ValueError("presentation weight flow needs at least one tag")

    fixed_weight = _to_int(grouped[0][1][0], "tj")
    weight_load = _span_for_paths([row for _, tag_rows in grouped for row in tag_rows], WEIGHT_INPUT_PATHS)
    resident_span = _span_for_paths([row for _, tag_rows in grouped for row in tag_rows], ("compute_window", "kernel_total"))

    entries: List[Dict[str, Any]] = []
    all_intervals: List[Tuple[int, int]] = []
    for _, tag_rows in grouped:
        activation_id = _to_int(tag_rows[0], "ti")
        preload_interval = _span_for_paths(tag_rows, ACTIVATION_PRELOAD_PATHS)
        feed_interval = _span_for_paths(tag_rows, ACTIVATION_FEED_PATHS)
        compute_interval = _span_for_paths(tag_rows, ("compute_window",))
        store_interval = _span_for_paths(tag_rows, STORE_TAIL_PATHS)
        for interval in (preload_interval, feed_interval, compute_interval, store_interval):
            if interval is not None:
                all_intervals.append(interval)
        entries.append(
            {
                "activation_id": activation_id,
                "preload": preload_interval,
                "feed": feed_interval,
                "compute": compute_interval,
                "store": store_interval,
            }
        )
    for interval in (weight_load, resident_span):
        if interval is not None:
            all_intervals.append(interval)

    origin, end = _local_limits(all_intervals)
    pad = max(64, int(0.03 * max(1, end - origin)))
    fig_height = max(8.0, 0.35 * (len(entries) + 1) + 2.0)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    weight_color = _weight_color(fixed_weight)
    labels = [f"Resident W{fixed_weight:02d}"] + [f"Activation A{entry['activation_id']:02d}" for entry in entries]
    _draw_bar(
        ax,
        0,
        resident_span,
        origin=origin,
        height=0.58,
        color=weight_color,
        alpha=0.32,
        label="stays resident while activations stream",
        edgecolor=weight_color,
        linewidth=1.0,
    )
    _draw_bar(
        ax,
        0,
        weight_load,
        origin=origin,
        height=0.58,
        color=weight_color,
        alpha=0.95,
        label=f"W{fixed_weight:02d}",
        edgecolor="#2f2f2f",
        linewidth=0.8,
    )

    for row_index, entry in enumerate(entries, start=1):
        label = f"A{entry['activation_id']:02d}"
        _draw_bar(
            ax,
            row_index,
            entry["compute"],
            origin=origin,
            height=0.55,
            color="#d2d7df",
            alpha=1.0,
            label=label,
            edgecolor=weight_color,
            linewidth=1.1,
        )
        _draw_bar(
            ax,
            row_index,
            entry["preload"],
            origin=origin,
            height=0.55,
            color="#9ecae1",
            alpha=0.95,
            edgecolor="#2f2f2f",
            linewidth=0.7,
        )
        _draw_bar(
            ax,
            row_index,
            entry["feed"],
            origin=origin,
            height=0.34,
            color="#4c78a8",
            alpha=0.98,
            edgecolor="#2f2f2f",
            linewidth=0.7,
        )
        _draw_bar(
            ax,
            row_index,
            entry["store"],
            origin=origin,
            height=0.30,
            color="#c8a453",
            alpha=0.95,
            edgecolor="#7f6d26",
            linewidth=0.8,
            hatch="//",
        )

    _presentation_style(ax, title, labels, xlabel="cycles for one resident weight tile")
    ax.set_xlim(-pad, (end - origin) + pad)
    ax.legend(
        handles=[
            Patch(color=weight_color, alpha=0.95, label=f"resident weight W{fixed_weight:02d}"),
            Patch(color="#9ecae1", alpha=0.95, label="DRAM to scratchpad preload"),
            Patch(color="#4c78a8", alpha=0.98, label="scratchpad to TPU feed"),
            Patch(color="#d2d7df", alpha=1.0, label="coarse compute window"),
            Patch(color="#c8a453", alpha=0.95, label="accumulate and write-back", hatch="//"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_presentation_weight_compute(
    rows: Sequence[Dict[str, str]],
    output_path: Path,
    title: str,
) -> Path:
    grouped = _rows_by_tag(rows)
    if not grouped:
        raise ValueError("presentation weight compute needs at least one tag")

    fixed_weight = _to_int(grouped[0][1][0], "tj")
    weight_color = _weight_color(fixed_weight)
    all_intervals: List[Tuple[int, int]] = []
    for _, tag_rows in grouped:
        preload = _span_for_paths(tag_rows, ACTIVATION_PRELOAD_PATHS)
        feed = _span_for_paths(tag_rows, ACTIVATION_FEED_PATHS)
        rsp = _span_for_paths(tag_rows, ("gsau_rsp",))
        psum_reload = _span_for_paths(tag_rows, PSUM_RELOAD_PATHS)
        store_tail = _span_for_paths(tag_rows, ADD_STORE_TAIL_PATHS)
        accum_ready = psum_reload if psum_reload is not None else store_tail
        pending_accum = _gap_between(rsp, accum_ready)
        for interval in (
            preload,
            feed,
            _span_for_paths(tag_rows, ("compute_window",)),
            _span_for_paths(tag_rows, ("systolic_array",)),
            rsp,
            pending_accum,
            psum_reload,
            store_tail,
        ):
            if interval is not None:
                all_intervals.append(interval)
    origin, end = _local_limits(all_intervals)
    pad = max(48, int(0.03 * max(1, end - origin)))
    fig_height = max(8.0, 0.32 * len(grouped) + 2.0)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    labels: List[str] = []
    for row_index, (_, tag_rows) in enumerate(grouped):
        activation_id = _to_int(tag_rows[0], "ti")
        labels.append(f"Activation A{activation_id:02d}")
        preload_interval = _span_for_paths(tag_rows, ACTIVATION_PRELOAD_PATHS)
        feed_interval = _span_for_paths(tag_rows, ACTIVATION_FEED_PATHS)
        compute_window = _span_for_paths(tag_rows, ("compute_window",))
        systolic = _span_for_paths(tag_rows, ("systolic_array",))
        rsp = _span_for_paths(tag_rows, ("gsau_rsp",))
        psum_reload = _span_for_paths(tag_rows, PSUM_RELOAD_PATHS)
        store_tail = _span_for_paths(tag_rows, ADD_STORE_TAIL_PATHS)
        accum_ready = psum_reload if psum_reload is not None else store_tail
        pending_accum = _gap_between(rsp, accum_ready)

        _draw_bar(
            ax,
            row_index,
            compute_window,
            origin=origin,
            height=0.52,
            color="#f2f4f7",
            alpha=1.0,
            label=f"A{activation_id:02d}",
            edgecolor="#b8c1cc",
            linewidth=0.8,
        )
        _draw_bar(
            ax,
            row_index,
            preload_interval,
            origin=origin,
            height=0.20,
            color="#9ecae1",
            alpha=0.92,
            edgecolor="#2f2f2f",
            linewidth=0.7,
        )
        _draw_bar(
            ax,
            row_index,
            feed_interval,
            origin=origin,
            height=0.28,
            color="#4c78a8",
            alpha=0.98,
            edgecolor="#2f2f2f",
            linewidth=0.7,
        )
        _draw_bar(
            ax,
            row_index,
            systolic,
            origin=origin,
            height=0.52,
            color=weight_color,
            alpha=0.98,
            edgecolor="#2f2f2f",
            linewidth=0.8,
        )
        _draw_bar(
            ax,
            row_index,
            rsp,
            origin=origin,
            height=0.18,
            color="#6bb0a8",
            alpha=0.95,
            edgecolor="#2f2f2f",
            linewidth=0.6,
        )
        _draw_bar(
            ax,
            row_index,
            pending_accum,
            origin=origin,
            height=0.18,
            color="#7f7f7f",
            alpha=0.60,
            edgecolor="#525252",
            linewidth=0.6,
            hatch="..",
        )
        _draw_bar(
            ax,
            row_index,
            psum_reload,
            origin=origin,
            height=0.18,
            color="#9d755d",
            alpha=0.95,
            edgecolor="#2f2f2f",
            linewidth=0.6,
        )
        _draw_bar(
            ax,
            row_index,
            store_tail,
            origin=origin,
            height=0.18,
            color="#c8a453",
            alpha=0.95,
            edgecolor="#7f6d26",
            linewidth=0.7,
            hatch="//",
        )

    _presentation_style(ax, title, labels, xlabel="cycles for one resident weight tile")
    ax.set_xlim(-pad, (end - origin) + pad)
    ax.legend(
        handles=[
            Patch(facecolor="#9ecae1", edgecolor="#2f2f2f", alpha=0.92, label="block prefetch to scratchpad"),
            Patch(facecolor="#4c78a8", edgecolor="#2f2f2f", alpha=0.98, label="scratchpad to TPU stream"),
            Patch(facecolor="#f2f4f7", edgecolor="#b8c1cc", alpha=1.0, label="coarse kernel envelope"),
            Patch(facecolor=weight_color, edgecolor="#2f2f2f", alpha=0.98, label=f"systolic multiply span with W{fixed_weight:02d}"),
            Patch(facecolor="#6bb0a8", edgecolor="#2f2f2f", alpha=0.95, label="TPU result returns"),
            Patch(facecolor="#7f7f7f", edgecolor="#525252", alpha=0.60, hatch="..", label="wait in pending_accum_rows for psum read slot"),
            Patch(facecolor="#9d755d", edgecolor="#2f2f2f", alpha=0.95, label="psum reload"),
            Patch(facecolor="#c8a453", edgecolor="#7f6d26", alpha=0.95, label="datapath add + store", hatch="//"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_presentation_reuse_balance(
    stats: Dict[str, Any],
    output_path: Path,
    title: str,
) -> Path:
    weight_reuse_m = float(stats.get("weight_reuse_m", 0) or 0)
    activation_reuse_n = float(stats.get("activation_reuse_n", 0) or 0)
    weight_reuse = float(stats.get("reuse_weight_internal_over_external", 0) or 0)
    activation_reuse = float(stats.get("reuse_act_internal_over_external", 0) or 0)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    fig.suptitle(title)

    axes[0].bar(["resident\nweight tiles", "activation tiles\nstreamed"], [weight_reuse_m, activation_reuse_n], color=["#f28e2b", "#4e79a7"])
    axes[0].set_title("Tiles handled in one reuse block")
    axes[0].set_ylabel("tile count")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(["weight reuse", "activation reuse"], [weight_reuse, activation_reuse], color=["#f28e2b", "#4e79a7"])
    axes[1].set_title("Reuse achieved before going back to DRAM")
    axes[1].set_ylabel("internal / external traffic (x)")
    axes[1].grid(True, axis="y", alpha=0.25)
    if weight_reuse > 0 or activation_reuse > 0:
        ymax = max(weight_reuse, activation_reuse)

    for axis in axes:
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_presentation_plot_set(
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
    *,
    stats: Optional[Dict[str, Any]] = None,
    tj: int = 0,
    tk: int = 0,
) -> List[Path]:
    stats = dict(stats or {})
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_reuse_m = max(1, int(stats.get("weight_reuse_m", 1) or 1))
    activation_reuse_n = max(1, int(stats.get("activation_reuse_n", 1) or 1))

    block_rows = [
        row
        for row in rows
        if _to_int(row, "tk") == int(tk)
        and int(tj) <= _to_int(row, "tj") < int(tj) + weight_reuse_m
    ]
    weight_rows = [row for row in rows if _to_int(row, "tk") == int(tk) and _to_int(row, "tj") == int(tj)]

    written: List[Path] = []
    if block_rows:
        written.append(
            plot_presentation_block_overview(
                block_rows,
                output_dir / f"presentation_block_overview_tj{int(tj):02d}_tk{int(tk):02d}.png",
                f"First reuse block: {weight_reuse_m} resident weights reused across {activation_reuse_n} activation tiles",
                activation_count=activation_reuse_n,
            )
        )
    if weight_rows:
        written.append(
            plot_presentation_weight_flow(
                weight_rows,
                output_dir / f"presentation_weight_flow_tj{int(tj):02d}_tk{int(tk):02d}.png",
                f"Resident weight W{int(tj):02d}: loading, using, and writing back each activation tile",
            )
        )
        written.append(
            plot_presentation_weight_compute(
                weight_rows,
                output_dir / f"presentation_weight_compute_tj{int(tj):02d}_tk{int(tk):02d}.png",
                f"Resident weight W{int(tj):02d}: preload, multiply, and psum-wait per activation tile",
            )
        )
    if stats:
        written.append(
            plot_presentation_reuse_balance(
                stats,
                output_dir / f"presentation_reuse_balance_tk{int(tk):02d}.png",
                "Why this schedule looks asymmetric",
            )
        )
    return written


def plot_gantt(
    rows: Sequence[Dict[str, str]],
    output_path: Path,
    title: str,
    *,
    row_mode: str = "tag",
    tile_drain_interval: Optional[Tuple[int, int]] = None,
    time_mode: str = "absolute",
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
        fig_height = max(3.8, 1.05 * len(slot_keys) + 1.8)
    else:
        slot_keys = []
        fig_height = max(4.8, 0.46 * len(tags) + 1.8)

    if row_mode == "slot":
        y_positions = {slot: idx for idx, slot in enumerate(slot_keys)}
    else:
        y_positions = {tag: idx for idx, tag in enumerate(tags)}

    if time_mode == "clustered":
        clusters = _activity_clusters(rows, tile_drain_interval=tile_drain_interval)
        if not clusters:
            min_cycle = min(_to_int(row, "start_cycle") for row in rows)
            max_cycle = max(_to_int(row, "end_cycle") for row in rows)
            clusters = [(min_cycle, max_cycle)]
        cluster_widths = [max(1, end - start) for start, end in clusters]
        fig_width = max(16.0, 5.4 * len(clusters))
        fig, axes = plt.subplots(
            1,
            len(clusters),
            figsize=(fig_width, fig_height),
            sharey=True,
            gridspec_kw={"width_ratios": cluster_widths},
        )
        axis_list = [axes] if hasattr(axes, "barh") else list(axes)
    else:
        min_cycle = min(_to_int(row, "start_cycle") for row in rows)
        max_cycle = max(_to_int(row, "end_cycle") for row in rows)
        clusters = [(min_cycle, max_cycle)]
        fig, ax = plt.subplots(figsize=(16, fig_height))
        axis_list = [ax]

    used_paths: List[str] = []
    for axis_index, axis in enumerate(axis_list):
        cluster_start, cluster_end = clusters[axis_index]

        if tile_drain_interval is not None:
            drain_start = tile_drain_interval[0]
            drain_end = tile_drain_interval[1]
            if time_mode == "clustered":
                drain_start = max(drain_start, cluster_start)
                drain_end = min(drain_end, cluster_end)
            if drain_end > drain_start:
                axis.axvspan(
                    drain_start,
                    drain_end,
                    facecolor=TILE_DRAIN_COLOR,
                    edgecolor="#7f7f7f",
                    linewidth=1.0,
                    alpha=0.20,
                    zorder=0,
                )

        for row in rows:
            path = row["path"]
            if path not in used_paths:
                used_paths.append(path)

            if row_mode == "slot":
                y = y_positions[_to_int(row, "slot")]
            else:
                y = y_positions[row["tag"]]

            start = _to_int(row, "start_cycle")
            end = _to_int(row, "end_cycle")
            if time_mode == "clustered":
                if end <= cluster_start or start >= cluster_end:
                    continue
                start = max(start, cluster_start)
                end = min(end, cluster_end)

            width = max(1, end - start)
            color = PATH_COLORS.get(path, "#808080")
            alpha = 0.28 if path in ENVELOPE_PATHS else 0.92
            linewidth = 1.2 if path in ENVELOPE_PATHS else 0.5
            edgecolor = color if path in ENVELOPE_PATHS else "black"
            axis.barh(
                y,
                width,
                left=start,
                height=0.80,
                color=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )

            if row_mode == "slot" and path == "compute_window" and width >= 20:
                axis.text(
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

        if row_mode == "slot":
            axis.set_yticks(range(len(slot_keys)))
            if axis_index == 0:
                axis.set_ylabel("prefetch slot")
                axis.set_yticklabels([f"slot={slot}" for slot in slot_keys])
            else:
                axis.tick_params(axis="y", labelleft=False)
        else:
            axis.set_yticks(range(len(tags)))
            if axis_index == 0:
                axis.set_ylabel("microkernel tag")
                axis.set_yticklabels([_make_tag_label(tag_meta[tag], rows) for tag in tags])
            else:
                axis.tick_params(axis="y", labelleft=False)

        axis.invert_yaxis()
        axis.grid(True, axis="x", alpha=0.25)
        if time_mode == "clustered":
            cluster_span = max(1, cluster_end - cluster_start)
            cluster_pad = max(64, int(0.03 * cluster_span))
            axis.set_xlim(cluster_start - cluster_pad, cluster_end + cluster_pad)
            if len(axis_list) > 1:
                axis.set_title(f"{cluster_start:,}..{cluster_end:,}", fontsize=9)

    if len(axis_list) == 1:
        axis_list[0].set_title(title)
        axis_list[0].set_xlabel("cycle")
    else:
        fig.suptitle(title)
        fig.supxlabel("cycle")

    legend_handles = [
        Patch(
            color=PATH_COLORS.get(path, "#808080"),
            label=PATH_LABELS.get(path, path),
            alpha=(0.28 if path in ENVELOPE_PATHS else 0.92),
        )
        for path in PATH_ORDER
        if path in used_paths
    ]
    if tile_drain_interval is not None:
        legend_handles.append(Patch(color=TILE_DRAIN_COLOR, label=TILE_DRAIN_LABEL, alpha=0.20))
    if legend_handles:
        axis_list[-1].legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=2)

    if len(axis_list) == 1:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
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
    parser.add_argument(
        "--view",
        choices=["tile", "weight-reuse", "activation-reuse"],
        default="tile",
        help=(
            "Select the slice to plot: one output tile, all kernels sharing a resident weight tile, "
            "or all kernels sharing a streamed activation tile."
        ),
    )
    parser.add_argument("--ti", type=int, default=0, help="Filter to one output-tile row index")
    parser.add_argument("--tj", type=int, default=0, help="Filter to one output-tile column index")
    parser.add_argument("--tk", type=int, default=0, help="Filter to one K tile index for reuse views")
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
    parser.add_argument(
        "--presentation-set",
        action="store_true",
        help="Write a presentation-friendly plot set: block overview, one-weight flow, compute detail, and reuse balance.",
    )
    parser.add_argument(
        "--stats-log",
        default=None,
        help="Path to stats.log used for the presentation plot set. Defaults to <input dir>/stats.log.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input) if args.input else repo_root / "logs" / "sysarr_gemm_tpu_tiled_1024" / "gantt.log"
    rows = _read_rows(input_path)
    if not rows:
        raise SystemExit(f"no rows found in {input_path}")

    if args.presentation_set:
        stats_path = Path(args.stats_log) if args.stats_log else input_path.parent / "stats.log"
        stats = read_stats_log(stats_path)
        written = write_presentation_plot_set(
            rows,
            input_path.parent,
            stats=stats,
            tj=int(args.tj),
            tk=int(args.tk),
        )
        if not written:
            raise SystemExit("no presentation plots were generated for the requested slice")
        for path in written:
            print(f"wrote {path}")
        return 0

    filter_ti: Optional[int]
    filter_tj: Optional[int]
    filter_tk: Optional[int]
    if args.tag:
        filter_ti = None
        filter_tj = None
        filter_tk = None
    elif args.view == "tile":
        filter_ti = int(args.ti)
        filter_tj = int(args.tj)
        filter_tk = None
    elif args.view == "weight-reuse":
        filter_ti = None
        filter_tj = int(args.tj)
        filter_tk = int(args.tk)
    else:
        filter_ti = int(args.ti)
        filter_tj = None
        filter_tk = int(args.tk)

    filtered = _filter_rows(
        rows,
        ti=filter_ti,
        tj=filter_tj,
        tk=filter_tk,
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
        if args.view == "tile":
            stem = f"kernel_gantt_ti{int(args.ti):02d}_tj{int(args.tj):02d}"
        elif args.view == "weight-reuse":
            stem = f"kernel_gantt_weight_reuse_tj{int(args.tj):02d}_tk{int(args.tk):02d}"
        else:
            stem = f"kernel_gantt_activation_reuse_ti{int(args.ti):02d}_tk{int(args.tk):02d}"
        if args.row_mode == "slot":
            stem += "_slots"
        if args.include_tile_drain:
            stem += "_tile_drain"
        output_path = input_path.parent / f"{stem}.png"

    if args.tag:
        title = "Tiled TPU tagged microkernel Gantt"
    else:
        if args.view == "tile":
            if args.row_mode == "slot":
                title = f"Tiled TPU slot-collapsed Gantt (ti={int(args.ti)}, tj={int(args.tj)})"
            else:
                title = f"Tiled TPU tagged microkernel Gantt (ti={int(args.ti)}, tj={int(args.tj)})"
        elif args.view == "weight-reuse":
            if args.row_mode == "slot":
                title = f"Blocked TPU weight-reuse Gantt, slot-collapsed (tj={int(args.tj)}, tk={int(args.tk)})"
            else:
                title = f"Blocked TPU weight-reuse Gantt (tj={int(args.tj)}, tk={int(args.tk)})"
        else:
            if args.row_mode == "slot":
                title = f"Blocked TPU activation-reuse Gantt, slot-collapsed (ti={int(args.ti)}, tk={int(args.tk)})"
            else:
                title = f"Blocked TPU activation-reuse Gantt (ti={int(args.ti)}, tk={int(args.tk)})"

    output_path = plot_gantt(
        filtered,
        output_path,
        title,
        row_mode=args.row_mode,
        tile_drain_interval=tile_drain_interval,
        time_mode=("clustered" if args.view != "tile" else "absolute"),
    )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
