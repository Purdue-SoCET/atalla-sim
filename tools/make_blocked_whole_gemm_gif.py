#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from make_blocked_reuse_policy_gif import (
    ACT_ACTIVE,
    ACT_IDLE,
    ACT_SELECTED,
    OUT_ACTIVE,
    OUT_DONE,
    OUT_IDLE,
    OUT_TARGET,
    SPAD_ACT,
    SPAD_ACT_ACTIVE,
    SPAD_EMPTY,
    SPAD_WGT,
    SPAD_WGT_ACTIVE,
    WGT_ACTIVE,
    WGT_IDLE,
    WGT_SELECTED,
    _draw_tile_grid,
    _frame_to_image,
    _tile_name,
    _wrap_caption,
)


Tile = Tuple[int, int]

OUT_PARTIAL = {"face": "#dcefd7", "edge": "#77a65a", "text": "#25411e", "lw": 1.6, "hatch": ".."}
SPAD_PSUM = {"face": "#dcefd7", "edge": "#77a65a", "text": "#25411e", "lw": 1.4}
SPAD_PSUM_ACTIVE = {"face": "#8cd17d", "edge": "#4f8c46", "text": "#10300c", "lw": 2.3}


@dataclass(frozen=True)
class WholeGemmFrame:
    caption: str
    act_selected: Set[Tile] = field(default_factory=set)
    act_active: Optional[Tile] = None
    wgt_selected: Set[Tile] = field(default_factory=set)
    wgt_active: Optional[Tile] = None
    current_outputs: Set[Tile] = field(default_factory=set)
    partial_outputs: Set[Tile] = field(default_factory=set)
    completed_outputs: Set[Tile] = field(default_factory=set)
    out_active: Optional[Tile] = None
    act_buffers: Sequence[Optional[str]] = field(default_factory=list)
    wgt_buffers: Sequence[Optional[str]] = field(default_factory=list)
    psum_buffers: Sequence[Optional[str]] = field(default_factory=list)
    active_act_slot: Optional[int] = None
    active_wgt_slot: Optional[int] = None
    active_psum_slot: Optional[int] = None
    ti0: int = 0
    tj0: int = 0
    tk: Optional[int] = None
    block_index: int = 0
    total_blocks: int = 0
    completed_tiles: int = 0
    total_output_tiles: int = 0


def _status_line(frame: WholeGemmFrame, *, tile_grid: int) -> str:
    if not frame.current_outputs:
        return (
            f"{frame.completed_tiles}/{frame.total_output_tiles} output tiles complete. "
            f"Tile grid is {tile_grid}x{tile_grid}."
        )

    rows = sorted({row for row, _ in frame.current_outputs})
    cols = sorted({col for _, col in frame.current_outputs})
    k_text = f"k{frame.tk}" if frame.tk is not None else "setup"
    return (
        f"block {frame.block_index + 1}/{frame.total_blocks}: "
        f"C[i{rows[0]}..i{rows[-1]}, j{cols[0]}..j{cols[-1]}], {k_text}, "
        f"completed output tiles {frame.completed_tiles}/{frame.total_output_tiles}"
    )


def _draw_schedule_panel(ax: plt.Axes, frame: WholeGemmFrame, *, tile_grid: int) -> None:
    row_count = max(len(frame.act_buffers), len(frame.wgt_buffers), len(frame.psum_buffers), 1)
    top = 1.25
    slot_height = 0.72
    panel_height = top + row_count * slot_height + 1.75

    ax.set_title("Scratchpad", fontsize=13, pad=10)
    ax.set_xlim(0, 9.6)
    ax.set_ylim(panel_height, 0)
    ax.axis("off")

    headings = [
        (1.55, "activation buffers"),
        (4.75, "resident weights"),
        (7.9, "psum tiles in scratchpad"),
    ]
    for x_pos, title in headings:
        ax.text(x_pos, 0.6, title, ha="center", va="center", fontsize=10.8, fontweight="bold")

    for index in range(row_count):
        y_pos = top + index * slot_height

        act_label = frame.act_buffers[index] if index < len(frame.act_buffers) else None
        act_style = SPAD_EMPTY
        if act_label is not None:
            act_style = SPAD_ACT_ACTIVE if frame.active_act_slot == index else SPAD_ACT
        act_rect = Rectangle(
            (0.45, y_pos),
            2.2,
            0.56,
            facecolor=act_style["face"],
            edgecolor=act_style["edge"],
            linewidth=act_style["lw"],
        )
        ax.add_patch(act_rect)
        ax.text(
            1.55,
            y_pos + 0.28,
            act_label or "empty",
            ha="center",
            va="center",
            fontsize=10,
            color=act_style["text"],
            fontweight="bold" if frame.active_act_slot == index else None,
        )

        wgt_label = frame.wgt_buffers[index] if index < len(frame.wgt_buffers) else None
        wgt_style = SPAD_EMPTY
        if wgt_label is not None:
            wgt_style = SPAD_WGT_ACTIVE if frame.active_wgt_slot == index else SPAD_WGT
        wgt_rect = Rectangle(
            (3.65, y_pos),
            2.2,
            0.56,
            facecolor=wgt_style["face"],
            edgecolor=wgt_style["edge"],
            linewidth=wgt_style["lw"],
        )
        ax.add_patch(wgt_rect)
        ax.text(
            4.75,
            y_pos + 0.28,
            wgt_label or "empty",
            ha="center",
            va="center",
            fontsize=10,
            color=wgt_style["text"],
            fontweight="bold" if frame.active_wgt_slot == index else None,
        )

        psum_label = frame.psum_buffers[index] if index < len(frame.psum_buffers) else None
        psum_style = SPAD_EMPTY
        if psum_label is not None:
            psum_style = SPAD_PSUM_ACTIVE if frame.active_psum_slot == index else SPAD_PSUM
        psum_rect = Rectangle(
            (6.85, y_pos),
            2.35,
            0.56,
            facecolor=psum_style["face"],
            edgecolor=psum_style["edge"],
            linewidth=psum_style["lw"],
        )
        ax.add_patch(psum_rect)
        ax.text(
            8.025,
            y_pos + 0.28,
            psum_label or "empty",
            ha="center",
            va="center",
            fontsize=10,
            color=psum_style["text"],
            fontweight="bold" if frame.active_psum_slot == index else None,
        )

    ax.text(
        4.8,
        top + row_count * slot_height + 0.45,
        _status_line(frame, tile_grid=tile_grid),
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )
    ax.text(
        4.8,
        top + row_count * slot_height + 0.95,
        "Outer loops walk output blocks in (ti0, tj0) order. Inside a block, tk sweeps across all reduction tiles.",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#555555",
    )
    ax.text(
        4.8,
        top + row_count * slot_height + 1.35,
        "Each psum tile stays live in scratchpad until every k slice for that output block has accumulated.",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#555555",
    )


def _build_frames(tile_grid: int, weight_reuse_m: int, activation_reuse_n: int) -> Tuple[List[WholeGemmFrame], List[int]]:
    blocks: List[Tuple[int, int, int, int]] = []
    for ti0 in range(0, tile_grid, activation_reuse_n):
        act_tiles = min(activation_reuse_n, tile_grid - ti0)
        for tj0 in range(0, tile_grid, weight_reuse_m):
            wgt_tiles = min(weight_reuse_m, tile_grid - tj0)
            blocks.append((ti0, tj0, act_tiles, wgt_tiles))

    total_blocks = len(blocks)
    total_output_tiles = tile_grid * tile_grid
    completed_outputs: Set[Tile] = set()
    frames: List[WholeGemmFrame] = [
        WholeGemmFrame(
            caption=(
                f"Whole blocked GEMM: partition A, W, and C into a {tile_grid}x{tile_grid} tile grid. "
                "The scheduler visits one output reuse block at a time, reusing resident weights across streamed activations."
            ),
            completed_tiles=0,
            total_output_tiles=total_output_tiles,
            total_blocks=total_blocks,
        )
    ]
    durations = [1500]

    for block_index, (ti0, tj0, act_tiles, wgt_tiles) in enumerate(blocks):
        current_outputs = {(ti0 + row_idx, tj0 + col_idx) for row_idx in range(act_tiles) for col_idx in range(wgt_tiles)}
        psum_labels = [
            _tile_name("C", ti0 + row_idx, tj0 + col_idx)
            for row_idx in range(act_tiles)
            for col_idx in range(wgt_tiles)
        ]
        partial_outputs: Set[Tile] = set()

        frames.append(
            WholeGemmFrame(
                caption=(
                    f"Start output block {block_index + 1}/{total_blocks}: target C[i{ti0}..i{ti0 + act_tiles - 1}, "
                    f"j{tj0}..j{tj0 + wgt_tiles - 1}]. These psum tiles stay resident while all k slices accumulate."
                ),
                current_outputs=current_outputs,
                completed_outputs=set(completed_outputs),
                psum_buffers=psum_labels,
                ti0=ti0,
                tj0=tj0,
                block_index=block_index,
                total_blocks=total_blocks,
                completed_tiles=len(completed_outputs),
                total_output_tiles=total_output_tiles,
            )
        )
        durations.append(950)

        for tk in range(tile_grid):
            act_selected = {(ti0 + row_idx, tk) for row_idx in range(act_tiles)}
            wgt_selected = {(tk, tj0 + col_idx) for col_idx in range(wgt_tiles)}
            act_buffers = [_tile_name("A", ti0 + row_idx, tk) for row_idx in range(act_tiles)]
            wgt_buffers = [_tile_name("W", tk, tj0 + col_idx) for col_idx in range(wgt_tiles)]

            frames.append(
                WholeGemmFrame(
                    caption=(
                        f"Prefetch reduction slice k{tk} for this output block: load A[i{ti0}..i{ti0 + act_tiles - 1}, k{tk}] "
                        f"and W[k{tk}, j{tj0}..j{tj0 + wgt_tiles - 1}] before the batch starts."
                    ),
                    act_selected=act_selected,
                    wgt_selected=wgt_selected,
                    current_outputs=current_outputs,
                    partial_outputs=set(partial_outputs),
                    completed_outputs=set(completed_outputs),
                    act_buffers=act_buffers,
                    wgt_buffers=wgt_buffers,
                    psum_buffers=psum_labels,
                    ti0=ti0,
                    tj0=tj0,
                    tk=tk,
                    block_index=block_index,
                    total_blocks=total_blocks,
                    completed_tiles=len(completed_outputs),
                    total_output_tiles=total_output_tiles,
                )
            )
            durations.append(800)

            for col_idx in range(wgt_tiles):
                wgt_tile = (tk, tj0 + col_idx)
                frames.append(
                    WholeGemmFrame(
                        caption=(
                            f"Keep {_tile_name('W', wgt_tile[0], wgt_tile[1])} resident and stream the selected activation tiles past it. "
                            f"This updates output column j{tj0 + col_idx} inside the current block."
                        ),
                        act_selected=act_selected,
                        wgt_selected=wgt_selected,
                        wgt_active=wgt_tile,
                        current_outputs=current_outputs,
                        partial_outputs=set(partial_outputs),
                        completed_outputs=set(completed_outputs),
                        act_buffers=act_buffers,
                        wgt_buffers=wgt_buffers,
                        psum_buffers=psum_labels,
                        active_wgt_slot=col_idx,
                        ti0=ti0,
                        tj0=tj0,
                        tk=tk,
                        block_index=block_index,
                        total_blocks=total_blocks,
                        completed_tiles=len(completed_outputs),
                        total_output_tiles=total_output_tiles,
                    )
                )
                durations.append(600)

                for row_idx in range(act_tiles):
                    act_tile = (ti0 + row_idx, tk)
                    out_tile = (ti0 + row_idx, tj0 + col_idx)
                    psum_slot = row_idx * wgt_tiles + col_idx
                    frames.append(
                        WholeGemmFrame(
                            caption=(
                                f"Run one tiled GEMM: {_tile_name('C', out_tile[0], out_tile[1])} += "
                                f"{_tile_name('A', act_tile[0], act_tile[1])} @ {_tile_name('W', wgt_tile[0], wgt_tile[1])}."
                            ),
                            act_selected=act_selected,
                            act_active=act_tile,
                            wgt_selected=wgt_selected,
                            wgt_active=wgt_tile,
                            current_outputs=current_outputs,
                            partial_outputs=set(partial_outputs),
                            completed_outputs=set(completed_outputs),
                            out_active=out_tile,
                            act_buffers=act_buffers,
                            wgt_buffers=wgt_buffers,
                            psum_buffers=psum_labels,
                            active_act_slot=row_idx,
                            active_wgt_slot=col_idx,
                            active_psum_slot=psum_slot,
                            ti0=ti0,
                            tj0=tj0,
                            tk=tk,
                            block_index=block_index,
                            total_blocks=total_blocks,
                            completed_tiles=len(completed_outputs),
                            total_output_tiles=total_output_tiles,
                        )
                    )
                    durations.append(420)

                    if tk == tile_grid - 1:
                        completed_outputs.add(out_tile)
                        partial_outputs.discard(out_tile)
                    else:
                        partial_outputs.add(out_tile)

            if tk < tile_grid - 1:
                frames.append(
                    WholeGemmFrame(
                        caption=(
                            f"k{tk} is complete for this output block. The psum tiles stay live in scratchpad while the scheduler "
                            f"moves on to prefetch k{tk + 1}."
                        ),
                        current_outputs=current_outputs,
                        partial_outputs=set(partial_outputs),
                        completed_outputs=set(completed_outputs),
                        psum_buffers=psum_labels,
                        ti0=ti0,
                        tj0=tj0,
                        tk=tk,
                        block_index=block_index,
                        total_blocks=total_blocks,
                        completed_tiles=len(completed_outputs),
                        total_output_tiles=total_output_tiles,
                    )
                )
                durations.append(700)
            else:
                frames.append(
                    WholeGemmFrame(
                        caption=(
                            f"This block is now complete: C[i{ti0}..i{ti0 + act_tiles - 1}, j{tj0}..j{tj0 + wgt_tiles - 1}] "
                            "has received every k slice and can drain from scratchpad as final output tiles."
                        ),
                        current_outputs=current_outputs,
                        completed_outputs=set(completed_outputs),
                        psum_buffers=psum_labels,
                        ti0=ti0,
                        tj0=tj0,
                        tk=tk,
                        block_index=block_index,
                        total_blocks=total_blocks,
                        completed_tiles=len(completed_outputs),
                        total_output_tiles=total_output_tiles,
                    )
                )
                durations.append(950)

    frames.append(
        WholeGemmFrame(
            caption=(
                "Whole GEMM complete: every output tile has accumulated all k slices. "
                "The animation shows how blocked prefetching keeps data resident long enough to overlap load latency with compute."
            ),
            completed_outputs=set(completed_outputs),
            completed_tiles=len(completed_outputs),
            total_output_tiles=total_output_tiles,
            total_blocks=total_blocks,
        )
    )
    durations.append(1800)
    return frames, durations


def _render_frame(tile_grid: int, weight_reuse_m: int, activation_reuse_n: int, frame: WholeGemmFrame) -> "Image.Image":
    panel_rows = max(len(frame.act_buffers), len(frame.wgt_buffers), len(frame.psum_buffers), 1)
    fig_height = max(8.8, 6.8 + 0.34 * panel_rows)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, fig_height))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Whole blocked GEMM schedule ({tile_grid}x{tile_grid} tiles, m={weight_reuse_m}, n={activation_reuse_n})",
        fontsize=16.2,
        y=0.992,
    )
    fig.text(
        0.5,
        0.958,
        "Outer loops walk output reuse blocks. Inside each block, every k slice is prefetched, multiplied, and accumulated into resident psums.",
        ha="center",
        va="center",
        fontsize=10.6,
        color="#2f4b7c",
    )
    fig.text(
        0.5,
        0.922,
        _wrap_caption(frame.caption),
        ha="center",
        va="center",
        fontsize=10.2,
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
        next_tiles=set(),
        next_style=None,
    )
    _draw_schedule_panel(axes[0, 1], frame, tile_grid=tile_grid)
    _draw_tile_grid(
        axes[1, 1],
        title="Output Tiles C",
        prefix="C",
        row_prefix="i",
        col_prefix="j",
        rows=tile_grid,
        cols=tile_grid,
        default_style=OUT_IDLE,
        selected=frame.current_outputs,
        selected_style=OUT_TARGET,
        active=frame.out_active,
        active_style=OUT_ACTIVE,
        done=frame.partial_outputs,
        done_style=OUT_PARTIAL,
        next_tiles=frame.completed_outputs,
        next_style=OUT_DONE,
    )

    fig.legend(
        handles=[
            Patch(facecolor=ACT_SELECTED["face"], edgecolor=ACT_SELECTED["edge"], label="selected activation tiles"),
            Patch(facecolor=WGT_ACTIVE["face"], edgecolor=WGT_ACTIVE["edge"], label="resident weight in use"),
            Patch(facecolor=OUT_TARGET["face"], edgecolor=OUT_TARGET["edge"], label="current output block"),
            Patch(facecolor=OUT_PARTIAL["face"], edgecolor=OUT_PARTIAL["edge"], hatch=OUT_PARTIAL["hatch"], label="partial sum resident"),
            Patch(facecolor=OUT_ACTIVE["face"], edgecolor=OUT_ACTIVE["edge"], label="active tiled GEMM"),
            Patch(facecolor=OUT_DONE["face"], edgecolor=OUT_DONE["edge"], label="final output tile complete"),
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        fontsize=10.0,
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.84), h_pad=3.0, w_pad=2.6)
    image = _frame_to_image(fig)
    plt.close(fig)
    return image


def make_gif(output_path: Path, *, tile_grid: int, weight_reuse_m: int, activation_reuse_n: int, max_frames: int) -> Path:
    frames, durations = _build_frames(tile_grid, weight_reuse_m, activation_reuse_n)
    if max_frames > 0 and len(frames) > max_frames:
        raise SystemExit(
            f"estimated {len(frames)} frames, which exceeds --max-frames={max_frames}. "
            "Use a smaller toy tile grid or raise the frame budget explicitly."
        )

    images = [
        _render_frame(
            tile_grid=tile_grid,
            weight_reuse_m=weight_reuse_m,
            activation_reuse_n=activation_reuse_n,
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
    parser = argparse.ArgumentParser(description="Create a conceptual GIF for the full blocked tiled GEMM schedule.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/animations/blocked_whole_gemm_m2_n4.gif"),
        help="Output GIF path.",
    )
    parser.add_argument("--tile-grid", type=int, default=4, help="Number of tiles along each matrix axis.")
    parser.add_argument("--weight-reuse-m", type=int, default=2, help="Resident weight tiles per block.")
    parser.add_argument("--activation-reuse-n", type=int, default=4, help="Activation tiles streamed per resident weight.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=240,
        help="Safety limit for conceptual animations. Use 0 to disable the cap.",
    )
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

    output_path = make_gif(
        args.output,
        tile_grid=args.tile_grid,
        weight_reuse_m=args.weight_reuse_m,
        activation_reuse_n=args.activation_reuse_n,
        max_frames=args.max_frames,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()