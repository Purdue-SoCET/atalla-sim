#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import textwrap
from typing import List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image


Tile = Tuple[int, int]

ACT_IDLE = {"face": "#f5f5f5", "edge": "#c7c7c7", "text": "#6b6b6b", "lw": 1.1, "hatch": None}
ACT_SELECTED = {"face": "#d9e7f5", "edge": "#4c78a8", "text": "#1f2d3a", "lw": 1.6, "hatch": None}
ACT_ACTIVE = {"face": "#4c78a8", "edge": "#2f4b7c", "text": "#ffffff", "lw": 2.3, "hatch": None}

WGT_IDLE = {"face": "#f6f6f6", "edge": "#c7c7c7", "text": "#6b6b6b", "lw": 1.1, "hatch": None}
WGT_SELECTED = {"face": "#fde1ca", "edge": "#f58518", "text": "#52320b", "lw": 1.6, "hatch": None}
WGT_ACTIVE = {"face": "#f58518", "edge": "#9b5d14", "text": "#ffffff", "lw": 2.3, "hatch": None}
WGT_NEXT = {"face": "#fff4e5", "edge": "#d08700", "text": "#6a4a00", "lw": 1.6, "hatch": "//"}

OUT_IDLE = {"face": "#f5f5f5", "edge": "#c7c7c7", "text": "#7a7a7a", "lw": 1.1, "hatch": None}
OUT_TARGET = {"face": "#ececec", "edge": "#bab0ac", "text": "#595959", "lw": 1.5, "hatch": None}
OUT_DONE = {"face": "#d3bf55", "edge": "#9a8620", "text": "#382d00", "lw": 1.6, "hatch": None}
OUT_ACTIVE = {"face": "#8cd17d", "edge": "#4f8c46", "text": "#10300c", "lw": 2.3, "hatch": None}

SPAD_EMPTY = {"face": "#fbfbfb", "edge": "#c7c7c7", "text": "#999999", "lw": 1.0}
SPAD_ACT = {"face": "#d9e7f5", "edge": "#4c78a8", "text": "#1f2d3a", "lw": 1.4}
SPAD_ACT_ACTIVE = {"face": "#4c78a8", "edge": "#2f4b7c", "text": "#ffffff", "lw": 2.3}
SPAD_WGT = {"face": "#fde1ca", "edge": "#f58518", "text": "#52320b", "lw": 1.4}
SPAD_WGT_ACTIVE = {"face": "#f58518", "edge": "#9b5d14", "text": "#ffffff", "lw": 2.3}


@dataclass(frozen=True)
class FrameSpec:
    caption: str
    act_selected: Set[Tile] = field(default_factory=set)
    act_active: Optional[Tile] = None
    wgt_selected: Set[Tile] = field(default_factory=set)
    wgt_active: Optional[Tile] = None
    wgt_next: Set[Tile] = field(default_factory=set)
    out_targets: Set[Tile] = field(default_factory=set)
    out_done: Set[Tile] = field(default_factory=set)
    out_active: Optional[Tile] = None
    act_buffers: Sequence[Optional[str]] = field(default_factory=list)
    wgt_buffers: Sequence[Optional[str]] = field(default_factory=list)
    active_act_slot: Optional[int] = None
    active_wgt_slot: Optional[int] = None
    next_preview: Sequence[str] = field(default_factory=list)


def _tile_name(prefix: str, row: int, col: int) -> str:
    return f"{prefix}{row}{col}"


def _wrap_caption(text: str, width: int = 92) -> str:
    return textwrap.fill(text, width=width)


def _draw_tile_grid(
    ax: plt.Axes,
    *,
    title: str,
    prefix: str,
    row_prefix: str,
    col_prefix: str,
    rows: int,
    cols: int,
    default_style: dict,
    selected: Set[Tile],
    selected_style: dict,
    active: Optional[Tile],
    active_style: dict,
    done: Set[Tile],
    done_style: Optional[dict],
    next_tiles: Set[Tile],
    next_style: Optional[dict],
) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks([index + 0.5 for index in range(cols)], [f"{col_prefix}{index}" for index in range(cols)])
    ax.set_yticks([index + 0.5 for index in range(rows)], [f"{row_prefix}{index}" for index in range(rows)])
    ax.tick_params(length=0, labelsize=10)
    ax.set_aspect("equal")
    ax.grid(False)

    for row in range(rows):
        for col in range(cols):
            tile = (row, col)
            style = default_style
            if tile in selected:
                style = selected_style
            if done_style is not None and tile in done:
                style = done_style
            if next_style is not None and tile in next_tiles:
                style = next_style
            if active is not None and tile == active:
                style = active_style

            rect = Rectangle(
                (col, row),
                1.0,
                1.0,
                facecolor=style["face"],
                edgecolor=style["edge"],
                linewidth=style["lw"],
                hatch=style.get("hatch"),
            )
            ax.add_patch(rect)
            ax.text(
                col + 0.5,
                row + 0.5,
                _tile_name(prefix, row, col),
                ha="center",
                va="center",
                fontsize=11,
                color=style["text"],
                fontweight="bold" if active is not None and tile == active else None,
            )

    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.9)


def _draw_scratchpad(ax: plt.Axes, frame: FrameSpec) -> None:
    ax.set_title("Scratchpad Reuse Block", fontsize=13, pad=10)
    ax.set_xlim(0, 8.3)
    ax.set_ylim(6.2, 0)
    ax.axis("off")

    ax.text(1.5, 0.55, "activation buffers", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(4.65, 0.55, "resident weights", ha="center", va="center", fontsize=11, fontweight="bold")
    if frame.next_preview:
        ax.text(7.0, 0.55, "next k-slice", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#6a4a00")

    for index in range(4):
        label = frame.act_buffers[index] if index < len(frame.act_buffers) else None
        style = SPAD_EMPTY
        if label is not None:
            style = SPAD_ACT_ACTIVE if frame.active_act_slot == index else SPAD_ACT
        rect = Rectangle((0.45, 1.0 + index), 2.15, 0.82, facecolor=style["face"], edgecolor=style["edge"], linewidth=style["lw"])
        ax.add_patch(rect)
        ax.text(1.525, 1.41 + index, label or "empty", ha="center", va="center", fontsize=11, color=style["text"], fontweight="bold" if frame.active_act_slot == index else None)

    for index in range(2):
        label = frame.wgt_buffers[index] if index < len(frame.wgt_buffers) else None
        style = SPAD_EMPTY
        if label is not None:
            style = SPAD_WGT_ACTIVE if frame.active_wgt_slot == index else SPAD_WGT
        rect = Rectangle((3.55, 1.35 + 1.35 * index), 2.15, 1.02, facecolor=style["face"], edgecolor=style["edge"], linewidth=style["lw"])
        ax.add_patch(rect)
        ax.text(4.625, 1.86 + 1.35 * index, label or "empty", ha="center", va="center", fontsize=11, color=style["text"], fontweight="bold" if frame.active_wgt_slot == index else None)

    if frame.next_preview:
        for index, label in enumerate(frame.next_preview[:2]):
            rect = Rectangle((6.15, 1.35 + 0.72 * index), 1.65, 0.52, facecolor=WGT_NEXT["face"], edgecolor=WGT_NEXT["edge"], linewidth=WGT_NEXT["lw"], hatch=WGT_NEXT["hatch"])
            ax.add_patch(rect)
            ax.text(6.975, 1.61 + 0.72 * index, label, ha="center", va="center", fontsize=9.7, color=WGT_NEXT["text"])

    ax.text(
        4.1,
        5.82,
        "Prefetch fills the block first. Then each W[k0,j] stays resident while A[i,k0] tiles stream.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )


def _frame_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3])


def _build_frames(tile_grid: int, weight_reuse_m: int, activation_reuse_n: int, k_index: int) -> Tuple[List[FrameSpec], List[int]]:
    act_tiles = {_tile for _tile in [(row, k_index) for row in range(activation_reuse_n)]}
    wgt_tiles = {_tile for _tile in [(k_index, col) for col in range(weight_reuse_m)]}
    out_tiles = {(row, col) for row in range(activation_reuse_n) for col in range(weight_reuse_m)}

    act_buffers = [_tile_name("A", row, k_index) for row in range(activation_reuse_n)]
    wgt_buffers = [_tile_name("W", k_index, col) for col in range(weight_reuse_m)]

    frames: List[FrameSpec] = [
        FrameSpec(
            caption=(
                f"Split A and W into {tile_grid}x{tile_grid} tiles. For k{k_index}, one reuse block picks "
                f"A[:,k{k_index}] and W[k{k_index},:]: {activation_reuse_n} activation tiles and {weight_reuse_m} resident weights."
            ),
            act_selected=act_tiles,
            wgt_selected=wgt_tiles,
            out_targets=out_tiles,
            act_buffers=[None] * activation_reuse_n,
            wgt_buffers=[None] * weight_reuse_m,
        ),
        FrameSpec(
            caption=(
                f"Prefetch policy: load all selected A[:,k{k_index}] and W[k{k_index},:] tiles into scratchpad first. "
                "No C partial starts until the full reuse block is resident."
            ),
            act_selected=act_tiles,
            wgt_selected=wgt_tiles,
            out_targets=out_tiles,
            act_buffers=act_buffers,
            wgt_buffers=wgt_buffers,
        ),
    ]
    durations = [1300, 1400]

    done_outputs: Set[Tile] = set()
    for col in range(weight_reuse_m):
        frames.append(
            FrameSpec(
                caption=(
                    f"Resident weight {_tile_name('W', k_index, col)} stays in the TPU. "
                    f"The selected A[:,k{k_index}] tiles now stream past it and accumulate into output column j{col}."
                ),
                act_selected=act_tiles,
                wgt_selected=wgt_tiles,
                wgt_active=(k_index, col),
                out_targets=out_tiles,
                out_done=set(done_outputs),
                act_buffers=act_buffers,
                wgt_buffers=wgt_buffers,
                active_wgt_slot=col,
            )
        )
        durations.append(1000)
        for row in range(activation_reuse_n):
            active_output = (row, col)
            frames.append(
                FrameSpec(
                    caption=(
                        f"Accumulate the k{k_index} contribution into {_tile_name('C', row, col)}: "
                        f"{_tile_name('C', row, col)} += {_tile_name('A', row, k_index)} @ {_tile_name('W', k_index, col)}. "
                        f"The final {_tile_name('C', row, col)} still needs the other k slices."
                    ),
                    act_selected=act_tiles,
                    act_active=(row, k_index),
                    wgt_selected=wgt_tiles,
                    wgt_active=(k_index, col),
                    out_targets=out_tiles,
                    out_done=set(done_outputs),
                    out_active=active_output,
                    act_buffers=act_buffers,
                    wgt_buffers=wgt_buffers,
                    active_act_slot=row,
                    active_wgt_slot=col,
                )
            )
            durations.append(700)
            done_outputs.add(active_output)

    next_k_index = k_index + 1
    next_weight_tiles = {(next_k_index, col) for col in range(weight_reuse_m)} if next_k_index < tile_grid else set()
    next_preview = [_tile_name("W", next_k_index, col) for col in range(weight_reuse_m)] if next_k_index < tile_grid else []
    next_act_preview = [_tile_name("A", row, next_k_index) for row in range(activation_reuse_n)] if next_k_index < tile_grid else []
    frames.append(
        FrameSpec(
            caption=(
                f"Block complete: the k{k_index} partials are stored in the C tiles. "
                + (
                    f"Next, prefetch A[:,k{next_k_index}] and W[k{next_k_index},:] before the next accumulation pass."
                    if next_preview
                    else "All k slices for this toy example are complete."
                )
            ),
            act_selected=act_tiles,
            wgt_selected=wgt_tiles,
            wgt_next=next_weight_tiles,
            out_targets=out_tiles,
            out_done=set(done_outputs),
            act_buffers=act_buffers,
            wgt_buffers=wgt_buffers,
            next_preview=next_preview,
        )
    )
    durations.append(1600)
    return frames, durations


def _render_frame(tile_grid: int, weight_reuse_m: int, activation_reuse_n: int, k_index: int, frame: FrameSpec) -> Image.Image:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Blocked prefetch + reuse policy (4x4 tiles, m={weight_reuse_m}, n={activation_reuse_n}, k={k_index})",
        fontsize=16.2,
        y=0.992,
    )
    fig.text(
        0.5,
        0.958,
        f"At one reduction slice: C[i,j] += A[i,k{k_index}] @ W[k{k_index},j]",
        ha="center",
        va="center",
        fontsize=10.9,
        color="#2f4b7c",
    )
    fig.text(
        0.5,
        0.922,
        _wrap_caption(frame.caption),
        ha="center",
        va="center",
        fontsize=10.3,
        color="#333333",
        multialignment="center",
        linespacing=1.15,
    )

    _draw_tile_grid(
        axes[0, 0],
        title="Activation Matrix A",
        prefix="A",
        row_prefix="i",
        col_prefix="k",
        rows=tile_grid,
        cols=tile_grid,
        default_style=ACT_IDLE,
        selected=frame.act_selected,
        selected_style=ACT_SELECTED,
        active=frame.act_active,
        active_style=ACT_ACTIVE,
        done=set(),
        done_style=None,
        next_tiles=set(),
        next_style=None,
    )
    _draw_tile_grid(
        axes[1, 0],
        title="Weight Matrix W",
        prefix="W",
        row_prefix="k",
        col_prefix="j",
        rows=tile_grid,
        cols=tile_grid,
        default_style=WGT_IDLE,
        selected=frame.wgt_selected,
        selected_style=WGT_SELECTED,
        active=frame.wgt_active,
        active_style=WGT_ACTIVE,
        done=set(),
        done_style=None,
        next_tiles=frame.wgt_next,
        next_style=WGT_NEXT,
    )
    _draw_scratchpad(axes[0, 1], frame)
    _draw_tile_grid(
        axes[1, 1],
        title="Partial-Sum / Output Tiles C",
        prefix="C",
        row_prefix="i",
        col_prefix="j",
        rows=tile_grid,
        cols=tile_grid,
        default_style=OUT_IDLE,
        selected=frame.out_targets,
        selected_style=OUT_TARGET,
        active=frame.out_active,
        active_style=OUT_ACTIVE,
        done=frame.out_done,
        done_style=OUT_DONE,
        next_tiles=set(),
        next_style=None,
    )

    fig.legend(
        handles=[
            Patch(facecolor=ACT_SELECTED["face"], edgecolor=ACT_SELECTED["edge"], label="prefetched activations"),
            Patch(facecolor=WGT_ACTIVE["face"], edgecolor=WGT_ACTIVE["edge"], label="resident weight in use"),
            Patch(facecolor=OUT_ACTIVE["face"], edgecolor=OUT_ACTIVE["edge"], label="current k-slice partial"),
            Patch(facecolor=OUT_DONE["face"], edgecolor=OUT_DONE["edge"], label="psum tile after this k slice"),
            Patch(facecolor=WGT_NEXT["face"], edgecolor=WGT_NEXT["edge"], hatch=WGT_NEXT["hatch"], label="next k-slice weight prefetch"),
        ],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        fontsize=10.2,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.84), h_pad=3.0, w_pad=2.6)
    image = _frame_to_image(fig)
    plt.close(fig)
    return image


def make_gif(output_path: Path, *, tile_grid: int, weight_reuse_m: int, activation_reuse_n: int, k_index: int) -> Path:
    frames, durations = _build_frames(tile_grid, weight_reuse_m, activation_reuse_n, k_index)
    images = [
        _render_frame(
            tile_grid=tile_grid,
            weight_reuse_m=weight_reuse_m,
            activation_reuse_n=activation_reuse_n,
            k_index=k_index,
            frame=frame,
        )
        for frame in frames
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a conceptual GIF for blocked scratchpad prefetch and M/N reuse.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/animations/blocked_prefetch_reuse_m2_n4.gif"),
        help="Output GIF path.",
    )
    parser.add_argument("--tile-grid", type=int, default=4, help="Number of tiles along each matrix axis.")
    parser.add_argument("--weight-reuse-m", type=int, default=2, help="Resident weight tiles per block.")
    parser.add_argument("--activation-reuse-n", type=int, default=4, help="Activation tiles streamed per resident weight.")
    parser.add_argument("--k-index", type=int, default=0, help="Reduction tile index to illustrate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tile_grid <= 0:
        raise SystemExit("--tile-grid must be > 0")
    if args.weight_reuse_m <= 0 or args.activation_reuse_n <= 0:
        raise SystemExit("--weight-reuse-m and --activation-reuse-n must be > 0")
    if args.weight_reuse_m > args.tile_grid:
        raise SystemExit("--weight-reuse-m cannot exceed --tile-grid")
    if args.activation_reuse_n > args.tile_grid:
        raise SystemExit("--activation-reuse-n cannot exceed --tile-grid")
    if args.k_index < 0 or args.k_index >= args.tile_grid:
        raise SystemExit("--k-index must be within the tile grid")

    output_path = make_gif(
        args.output,
        tile_grid=args.tile_grid,
        weight_reuse_m=args.weight_reuse_m,
        activation_reuse_n=args.activation_reuse_n,
        k_index=args.k_index,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()