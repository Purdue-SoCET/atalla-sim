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
DEFAULT_BURST_BYTES = [4, 8, 16, 32, 64]

if sys.version_info < (3, 7):
    raise SystemExit("This runner needs Python 3.7+. Try `python3.11`.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_blocked_harness_module():
    spec = importlib.util.spec_from_file_location("_blocked_mn_burst_sweep_harness", BLOCKED_HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load blocked harness from {BLOCKED_HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


blocked = _load_blocked_harness_module()


def _build_run_name(weight_reuse_m: int, activation_reuse_n: int, burst_bytes: int) -> str:
    return (
        f"m{int(weight_reuse_m)}_n{int(activation_reuse_n)}_"
        f"backend_dram_burst_bytes_{int(burst_bytes)}"
    )


def _safe_json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": result.get("name"),
        "status": result.get("status"),
        "matrix_size": result.get("matrix_size"),
        "tile_size": result.get("tile_size"),
        "weight_reuse_m": result.get("weight_reuse_m"),
        "activation_reuse_n": result.get("activation_reuse_n"),
        "spad_frontend_queue_size": result.get("spad_frontend_queue_size"),
        "backend_dram_burst_bytes": result.get("backend_dram_burst_bytes"),
        "cycles": result.get("cycles"),
        "mac_utilization": result.get("mac_utilization"),
        "mac_utilization_sysarr_active": result.get("mac_utilization_sysarr_active"),
        "throughput_float_operations_per_cycle": result.get("throughput_float_operations_per_cycle"),
        "external_bandwidth_avg_bytes_per_cycle": result.get("external_bandwidth_avg_bytes_per_cycle"),
        "internal_bandwidth_bytes_per_cycle": result.get("internal_bandwidth_bytes_per_cycle"),
        "reuse_weight_internal_over_external": result.get("reuse_weight_internal_over_external"),
        "reuse_act_internal_over_external": result.get("reuse_act_internal_over_external"),
        "reuse_psum_internal_over_external": result.get("reuse_psum_internal_over_external"),
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
        "backend_dram_burst_bytes",
        "cycles",
        "mac_utilization",
        "mac_utilization_sysarr_active",
        "throughput_float_operations_per_cycle",
        "external_bandwidth_avg_bytes_per_cycle",
        "internal_bandwidth_bytes_per_cycle",
        "reuse_weight_internal_over_external",
        "reuse_act_internal_over_external",
        "reuse_psum_internal_over_external",
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
            "Sweep backend dram burst width for the blocked tiled TPU harness. "
            "Defaults target the M=8, N=32 weight-stationary schedule."
        )
    )
    parser.add_argument("--matrix-size", type=int, default=blocked.MATRIX)
    parser.add_argument("--tile-size", type=int, default=blocked.TILE)
    parser.add_argument("--weight-reuse-m", type=int, default=8)
    parser.add_argument("--activation-reuse-n", type=int, default=32)
    parser.add_argument("--spad-frontend-queue-size", type=int, default=4)
    parser.add_argument("--burst-bytes", nargs="*", type=int, default=DEFAULT_BURST_BYTES)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "logs" / "blocked_m8_n32_dram_burst_bytes_sweep"),
    )
    parser.add_argument(
        "--write-run-logs",
        action="store_true",
        help="Also emit per-run stats/schedule/gantt logs under <output-dir>/<run-name>/.",
    )
    args = parser.parse_args()

    burst_bytes_values = sorted({int(value) for value in args.burst_bytes if int(value) > 0})
    if not burst_bytes_values:
        raise SystemExit("burst-bytes must contain at least one positive integer")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for index, burst_bytes in enumerate(burst_bytes_values, start=1):
        run_name = _build_run_name(args.weight_reuse_m, args.activation_reuse_n, burst_bytes)
        print(f"[{index}/{len(burst_bytes_values)}] running {run_name}", flush=True)
        try:
            _, stats, runner = blocked.run_blocked_mn_tpu_cosim(
                matrix_size=args.matrix_size,
                tile_size=args.tile_size,
                weight_reuse_m=args.weight_reuse_m,
                activation_reuse_n=args.activation_reuse_n,
                spad_frontend_queue_size=args.spad_frontend_queue_size,
                backend_dram_burst_bytes=burst_bytes,
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
                "  completed cycles=%s ext_bw=%.6f"
                % (
                    result.get("cycles"),
                    float(result.get("external_bandwidth_avg_bytes_per_cycle", 0.0) or 0.0),
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
                "spad_frontend_queue_size": int(args.spad_frontend_queue_size),
                "backend_dram_burst_bytes": int(burst_bytes),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            print("  failed error=%s" % result["error_type"], flush=True)

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
            "spad_frontend_queue_size": int(args.spad_frontend_queue_size),
            "burst_bytes": burst_bytes_values,
            "write_run_logs": bool(args.write_run_logs),
            "run_count": len(results),
        },
    )

    print(f"wrote {output_dir / 'results.csv'}", flush=True)
    print(f"wrote {output_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())