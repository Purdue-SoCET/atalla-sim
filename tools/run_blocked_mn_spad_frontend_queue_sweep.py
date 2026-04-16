#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
BLOCKED_HARNESS_PATH = (
    REPO_ROOT / "tests" / "atalla" / "test_scratchpad_vector_core_sysarr_tpu_tiled_1024_blocked_mn.py"
)
DEFAULT_QUEUE_SIZES = list(range(1, 9))

if sys.version_info < (3, 7):
    raise SystemExit("This runner needs Python 3.7+. Try `python3.11`.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_blocked_harness_module():
    spec = importlib.util.spec_from_file_location("_blocked_mn_sweep_harness", BLOCKED_HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load blocked harness from {BLOCKED_HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


blocked = _load_blocked_harness_module()


def _build_run_name(weight_reuse_m: int, activation_reuse_n: int, queue_size: int) -> str:
    return f"m{int(weight_reuse_m)}_n{int(activation_reuse_n)}_spad_frontend_queue_size_{int(queue_size)}"


def _safe_json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    queue_max_depths = dict(result.get("queue_max_depths") or {})
    queue_avg_depths = dict(result.get("queue_avg_depths") or {})
    return {
        "name": result.get("name"),
        "status": result.get("status"),
        "matrix_size": result.get("matrix_size"),
        "tile_size": result.get("tile_size"),
        "weight_reuse_m": result.get("weight_reuse_m"),
        "activation_reuse_n": result.get("activation_reuse_n"),
        "spad_frontend_queue_size": result.get("spad_frontend_queue_size"),
        "cycles": result.get("cycles"),
        "mac_utilization": result.get("mac_utilization"),
        "mac_utilization_sysarr_active": result.get("mac_utilization_sysarr_active"),
        "avg_active_pes_when_active": result.get("avg_active_pes_when_active"),
        "avg_active_pes_during_compute_window": result.get("avg_active_pes_during_compute_window"),
        "max_active_pes_in_any_cycle": result.get("max_active_pes_in_any_cycle"),
        "throughput_float_operations_per_cycle": result.get("throughput_float_operations_per_cycle"),
        "external_bandwidth_avg_bytes_per_cycle": result.get("external_bandwidth_avg_bytes_per_cycle"),
        "internal_bandwidth_bytes_per_cycle": result.get("internal_bandwidth_bytes_per_cycle"),
        "reuse_weight_internal_over_external": result.get("reuse_weight_internal_over_external"),
        "reuse_act_internal_over_external": result.get("reuse_act_internal_over_external"),
        "reuse_psum_internal_over_external": result.get("reuse_psum_internal_over_external"),
        "gsau_rd_queue_max": queue_max_depths.get("gsau_rd_queue"),
        "vlsu_dst_fifo_max": queue_max_depths.get("vlsu_dst_fifo"),
        "gsau_rd_queue_avg": queue_avg_depths.get("gsau_rd_queue"),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
    }


def _write_results_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "status",
        "matrix_size",
        "tile_size",
        "weight_reuse_m",
        "activation_reuse_n",
        "spad_frontend_queue_size",
        "cycles",
        "mac_utilization",
        "mac_utilization_sysarr_active",
        "avg_active_pes_when_active",
        "avg_active_pes_during_compute_window",
        "max_active_pes_in_any_cycle",
        "throughput_float_operations_per_cycle",
        "external_bandwidth_avg_bytes_per_cycle",
        "internal_bandwidth_bytes_per_cycle",
        "reuse_weight_internal_over_external",
        "reuse_act_internal_over_external",
        "reuse_psum_internal_over_external",
        "gsau_rd_queue_max",
        "vlsu_dst_fifo_max",
        "gsau_rd_queue_avg",
        "error_type",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep scratchpad frontend queue size for the blocked tiled TPU harness. "
            "Defaults target the M=8, N=32 weight-stationary schedule."
        )
    )
    parser.add_argument("--matrix-size", type=int, default=blocked.MATRIX)
    parser.add_argument("--tile-size", type=int, default=blocked.TILE)
    parser.add_argument("--weight-reuse-m", type=int, default=8)
    parser.add_argument("--activation-reuse-n", type=int, default=32)
    parser.add_argument("--queue-sizes", nargs="*", type=int, default=DEFAULT_QUEUE_SIZES)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "logs" / "blocked_m8_n32_spad_frontend_queue_sweep"),
    )
    parser.add_argument(
        "--write-run-logs",
        action="store_true",
        help="Also emit per-run stats/schedule/gantt logs under <output-dir>/<run-name>/.",
    )
    args = parser.parse_args()

    queue_sizes = sorted({int(value) for value in args.queue_sizes if int(value) > 0})
    if not queue_sizes:
        raise SystemExit("queue sizes must contain at least one positive integer")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for index, queue_size in enumerate(queue_sizes, start=1):
        run_name = _build_run_name(args.weight_reuse_m, args.activation_reuse_n, queue_size)
        print(f"[{index}/{len(queue_sizes)}] running {run_name}", flush=True)
        try:
            _, stats, runner = blocked.run_blocked_mn_tpu_cosim(
                matrix_size=args.matrix_size,
                tile_size=args.tile_size,
                weight_reuse_m=args.weight_reuse_m,
                activation_reuse_n=args.activation_reuse_n,
                spad_frontend_queue_size=queue_size,
            )
            result: Dict[str, Any] = dict(stats)
            result.update(
                {
                    "name": run_name,
                    "status": "ok",
                }
            )
            if args.write_run_logs:
                blocked._write_logs(runner, stats, output_dir / run_name)
            print(
                "  completed cycles=%s mac_util=%.6f"
                % (
                    result.get("cycles"),
                    float(result.get("mac_utilization", 0.0) or 0.0),
                ),
                flush=True,
            )
        except Exception as exc:
            result = {
                "name": run_name,
                "status": "failed",
                "matrix_size": int(args.matrix_size),
                "tile_size": int(args.tile_size),
                "weight_reuse_m": int(args.weight_reuse_m),
                "activation_reuse_n": int(args.activation_reuse_n),
                "spad_frontend_queue_size": int(queue_size),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(
                "  failed error=%s"
                % result["error_type"],
                flush=True,
            )

        results.append(result)
        summary_rows.append(_summary_row(result))
        _safe_json_dump(output_dir / f"{run_name}.json", result)

    _write_results_csv(output_dir / "results.csv", summary_rows)
    _safe_json_dump(output_dir / "results.json", results)
    _safe_json_dump(
        output_dir / "manifest.json",
        {
            "repo_root": str(REPO_ROOT),
            "harness": str(BLOCKED_HARNESS_PATH),
            "matrix_size": int(args.matrix_size),
            "tile_size": int(args.tile_size),
            "weight_reuse_m": int(args.weight_reuse_m),
            "activation_reuse_n": int(args.activation_reuse_n),
            "queue_sizes": queue_sizes,
            "write_run_logs": bool(args.write_run_logs),
            "run_count": len(results),
        },
    )

    print(f"wrote {output_dir / 'results.csv'}", flush=True)
    print(f"wrote {output_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())