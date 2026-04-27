#!/usr/bin/env python3

import argparse
import contextlib
import csv
import importlib.util
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
BLOCKED_HARNESS_PATH = (
    REPO_ROOT / "tests" / "atalla" / "test_scratchpad_vector_core_sysarr_tpu_tiled_1024_blocked_mn.py"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "roofline_study" / "blocked_mn_reuse_1024"
LATEST_RUN_FILE = "LATEST.txt"

DEFAULT_CASE_SPECS = [
    "Naive:1:1",
    "m2_n4:2:4",
    "m2_n8:2:8",
    "m4_n4:4:4",
    "m4_n8:4:8",
]


if sys.version_info < (3, 7):
    raise SystemExit("This runner needs Python 3.7+. Try `python3.11`.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_blocked_harness_module():
    spec = importlib.util.spec_from_file_location("_blocked_mn_roofline_harness", BLOCKED_HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load blocked harness from {BLOCKED_HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


blocked = _load_blocked_harness_module()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_utc")


def _safe_slug(text: str) -> str:
    lowered = text.strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch for ch in lowered if ch.isalnum() or ch == "_")


def _parse_case_spec(spec: str, order: int) -> Dict[str, Any]:
    parts = [part.strip() for part in str(spec).split(":") if part.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"invalid case spec {spec!r}; expected LABEL:WEIGHT_REUSE_M:ACTIVATION_REUSE_N"
        )
    label = parts[0]
    weight_reuse_m = int(parts[1])
    activation_reuse_n = int(parts[2])
    if weight_reuse_m <= 0 or activation_reuse_n <= 0:
        raise ValueError(f"reuse factors must be positive in case spec {spec!r}")
    return {
        "name": _safe_slug(label),
        "policy_label": label,
        "policy_order": int(order),
        "weight_reuse_m": weight_reuse_m,
        "activation_reuse_n": activation_reuse_n,
    }


def _algo_flops(matrix_size: int) -> int:
    size = int(matrix_size)
    return 2 * size * size * size


def _theoretical_peak_compute(tile_size: int) -> int:
    tile = int(tile_size)
    return 2 * tile * tile


def _safe_json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_output_dir(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    if args.output_dir:
        return Path(args.output_dir), None
    output_root = Path(args.output_root)
    return output_root / _run_stamp(), output_root


def _clear_output_dir(output_dir: Path) -> None:
    for child in output_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _prepare_output_dir(output_dir: Path, overwrite_output_dir: bool) -> None:
    if output_dir.exists():
        has_existing_content = any(output_dir.iterdir())
        if has_existing_content and not overwrite_output_dir:
            raise SystemExit(
                f"output directory {output_dir} already exists and is not empty; use --output-dir with a new path or pass --overwrite-output-dir"
            )
        if has_existing_content and overwrite_output_dir:
            _clear_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_latest_run(output_root: Optional[Path], output_dir: Path) -> None:
    if output_root is None:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    latest_path = output_root / LATEST_RUN_FILE
    latest_path.write_text(str(output_dir.resolve()), encoding="utf-8")


def _enrich_result(
    result: Dict[str, Any],
    case: Dict[str, Any],
    *,
    matrix_size: int,
    tile_size: int,
    spad_frontend_queue_size: int,
) -> Dict[str, Any]:
    payload = dict(result)
    matrix_size = int(payload.get("matrix_size", matrix_size) or matrix_size)
    tile_size = int(payload.get("tile_size", tile_size) or tile_size)
    cycles = float(payload.get("cycles", 0) or 0)
    flops_micro = float(payload.get("flops_micro", 0) or 0)
    bytes_transmitted = float(payload.get("bytes_transmitted", 0) or 0)
    bytes_internal = float(payload.get("bytes_internal", 0) or 0)
    flops_algo = float(_algo_flops(matrix_size))

    throughput_algo = (flops_algo / cycles) if cycles else 0.0
    arithmetic_intensity_external_algo = (flops_algo / bytes_transmitted) if bytes_transmitted else 0.0
    arithmetic_intensity_external_micro = (flops_micro / bytes_transmitted) if bytes_transmitted else 0.0
    arithmetic_intensity_internal_algo = (flops_algo / bytes_internal) if bytes_internal else 0.0

    payload.update(case)
    payload.update(
        {
            "matrix_size": matrix_size,
            "tile_size": tile_size,
            "spad_frontend_queue_size": int(spad_frontend_queue_size),
            "flops_algo": flops_algo,
            "throughput_algo_flops_per_cycle": throughput_algo,
            "arithmetic_intensity_external_algo_flops_per_byte": arithmetic_intensity_external_algo,
            "arithmetic_intensity_external_micro_flops_per_byte": arithmetic_intensity_external_micro,
            "arithmetic_intensity_internal_algo_flops_per_byte": arithmetic_intensity_internal_algo,
            "theoretical_peak_compute_flops_per_cycle": float(_theoretical_peak_compute(tile_size)),
        }
    )
    return payload


def _execute_case(
    case: Dict[str, Any],
    *,
    matrix_size: int,
    tile_size: int,
    spad_frontend_queue_size: int,
    write_run_logs: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    blocked_module = _load_blocked_harness_module()
    stdout_log_path = output_dir / f"{case['name']}.stdout.log"
    stderr_log_path = output_dir / f"{case['name']}.stderr.log"
    wall_start = time.perf_counter()
    try:
        with stdout_log_path.open("w", encoding="utf-8") as stdout_fh:
            with stderr_log_path.open("w", encoding="utf-8") as stderr_fh:
                with contextlib.redirect_stdout(stdout_fh), contextlib.redirect_stderr(stderr_fh):
                    _, stats, runner = blocked_module.run_blocked_mn_tpu_cosim(
                        matrix_size=matrix_size,
                        tile_size=tile_size,
                        weight_reuse_m=case["weight_reuse_m"],
                        activation_reuse_n=case["activation_reuse_n"],
                        spad_frontend_queue_size=spad_frontend_queue_size,
                    )

        result = _enrich_result(
            dict(stats),
            case,
            matrix_size=matrix_size,
            tile_size=tile_size,
            spad_frontend_queue_size=spad_frontend_queue_size,
        )
        result.update(
            {
                "status": "ok",
                "wall_seconds": round(time.perf_counter() - wall_start, 3),
                "stdout_log": stdout_log_path.name,
                "stderr_log": stderr_log_path.name,
            }
        )
        if write_run_logs:
            blocked_module._write_logs(runner, stats, output_dir / case["name"])
            result["run_log_dir"] = case["name"]
        return result
    except Exception as exc:
        return {
            "name": case["name"],
            "policy_label": case["policy_label"],
            "policy_order": case["policy_order"],
            "status": "failed",
            "matrix_size": int(matrix_size),
            "tile_size": int(tile_size),
            "weight_reuse_m": int(case["weight_reuse_m"]),
            "activation_reuse_n": int(case["activation_reuse_n"]),
            "spad_frontend_queue_size": int(spad_frontend_queue_size),
            "theoretical_peak_compute_flops_per_cycle": float(_theoretical_peak_compute(tile_size)),
            "wall_seconds": round(time.perf_counter() - wall_start, 3),
            "stdout_log": stdout_log_path.name,
            "stderr_log": stderr_log_path.name,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }


def _print_case_result(result: Dict[str, Any]) -> None:
    wall_seconds = float(result.get("wall_seconds", 0.0) or 0.0)
    if result.get("status") == "ok":
        print(
            (
                "  completed cycles=%s throughput=%.3f ai_ext=%.3f wall=%.1fs"
                % (
                    result.get("cycles"),
                    float(result.get("throughput_algo_flops_per_cycle", 0.0) or 0.0),
                    float(result.get("arithmetic_intensity_external_algo_flops_per_byte", 0.0) or 0.0),
                    wall_seconds,
                )
            ),
            flush=True,
        )
        return
    print(
        "  failed error=%s wall=%.1fs stderr=%s"
        % (
            result.get("error_type"),
            wall_seconds,
            result.get("stderr_log", ""),
        ),
        flush=True,
    )


def _summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": result.get("name"),
        "policy_label": result.get("policy_label"),
        "policy_order": result.get("policy_order"),
        "status": result.get("status"),
        "matrix_size": result.get("matrix_size"),
        "tile_size": result.get("tile_size"),
        "weight_reuse_m": result.get("weight_reuse_m"),
        "activation_reuse_n": result.get("activation_reuse_n"),
        "spad_frontend_queue_size": result.get("spad_frontend_queue_size"),
        "cycles": result.get("cycles"),
        "wall_seconds": result.get("wall_seconds"),
        "flops_algo": result.get("flops_algo"),
        "flops_micro": result.get("flops_micro"),
        "throughput_algo_flops_per_cycle": result.get("throughput_algo_flops_per_cycle"),
        "throughput_float_operations_per_cycle": result.get("throughput_float_operations_per_cycle"),
        "bytes_transmitted": result.get("bytes_transmitted"),
        "bytes_internal": result.get("bytes_internal"),
        "arithmetic_intensity_external_algo_flops_per_byte": result.get(
            "arithmetic_intensity_external_algo_flops_per_byte"
        ),
        "arithmetic_intensity_external_micro_flops_per_byte": result.get(
            "arithmetic_intensity_external_micro_flops_per_byte"
        ),
        "arithmetic_intensity_internal": result.get("arithmetic_intensity_internal"),
        "arithmetic_intensity_internal_algo_flops_per_byte": result.get(
            "arithmetic_intensity_internal_algo_flops_per_byte"
        ),
        "external_bandwidth_avg_bytes_per_cycle": result.get("external_bandwidth_avg_bytes_per_cycle"),
        "external_bandwidth_active_bytes_per_cycle": result.get("external_bandwidth_active_bytes_per_cycle"),
        "internal_bandwidth_bytes_per_cycle": result.get("internal_bandwidth_bytes_per_cycle"),
        "mac_utilization": result.get("mac_utilization"),
        "mac_utilization_sysarr_active": result.get("mac_utilization_sysarr_active"),
        "reuse_weight_internal_over_external": result.get("reuse_weight_internal_over_external"),
        "reuse_act_internal_over_external": result.get("reuse_act_internal_over_external"),
        "reuse_psum_internal_over_external": result.get("reuse_psum_internal_over_external"),
        "theoretical_peak_compute_flops_per_cycle": result.get("theoretical_peak_compute_flops_per_cycle"),
        "stdout_log": result.get("stdout_log", ""),
        "stderr_log": result.get("stderr_log", ""),
        "run_log_dir": result.get("run_log_dir", ""),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
    }


def _write_results_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "policy_label",
        "policy_order",
        "status",
        "matrix_size",
        "tile_size",
        "weight_reuse_m",
        "activation_reuse_n",
        "spad_frontend_queue_size",
        "cycles",
        "wall_seconds",
        "flops_algo",
        "flops_micro",
        "throughput_algo_flops_per_cycle",
        "throughput_float_operations_per_cycle",
        "bytes_transmitted",
        "bytes_internal",
        "arithmetic_intensity_external_algo_flops_per_byte",
        "arithmetic_intensity_external_micro_flops_per_byte",
        "arithmetic_intensity_internal",
        "arithmetic_intensity_internal_algo_flops_per_byte",
        "external_bandwidth_avg_bytes_per_cycle",
        "external_bandwidth_active_bytes_per_cycle",
        "internal_bandwidth_bytes_per_cycle",
        "mac_utilization",
        "mac_utilization_sysarr_active",
        "reuse_weight_internal_over_external",
        "reuse_act_internal_over_external",
        "reuse_psum_internal_over_external",
        "theoretical_peak_compute_flops_per_cycle",
        "stdout_log",
        "stderr_log",
        "run_log_dir",
        "error_type",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_manifest(args: argparse.Namespace, cases: Sequence[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    return {
        "repo_root": str(REPO_ROOT),
        "harness": str(BLOCKED_HARNESS_PATH),
        "output_dir": str(output_dir),
        "matrix_size": int(args.matrix_size),
        "tile_size": int(args.tile_size),
        "spad_frontend_queue_size": int(args.spad_frontend_queue_size),
        "cases": list(cases),
        "write_run_logs": bool(args.write_run_logs),
        "run_count": len(cases),
        "completed_run_count": 0,
        "successful_run_count": 0,
        "failed_run_count": 0,
        "status": "running",
        "started_at_utc": _utc_now_iso(),
        "theoretical_peak_compute_flops_per_cycle": float(_theoretical_peak_compute(args.tile_size)),
        "empirical_peak_compute_algo_flops_per_cycle": 0.0,
        "empirical_peak_external_bandwidth_avg_bytes_per_cycle": 0.0,
        "empirical_peak_external_bandwidth_active_bytes_per_cycle": 0.0,
    }


def _refresh_manifest(manifest: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> None:
    ok_results = [row for row in results if row.get("status") == "ok"]
    failed_results = [row for row in results if row.get("status") != "ok"]
    manifest["completed_run_count"] = len(results)
    manifest["successful_run_count"] = len(ok_results)
    manifest["failed_run_count"] = len(failed_results)
    manifest["empirical_peak_compute_algo_flops_per_cycle"] = max(
        (float(row.get("throughput_algo_flops_per_cycle", 0.0) or 0.0) for row in ok_results),
        default=0.0,
    )
    manifest["empirical_peak_external_bandwidth_avg_bytes_per_cycle"] = max(
        (float(row.get("external_bandwidth_avg_bytes_per_cycle", 0.0) or 0.0) for row in ok_results),
        default=0.0,
    )
    manifest["empirical_peak_external_bandwidth_active_bytes_per_cycle"] = max(
        (float(row.get("external_bandwidth_active_bytes_per_cycle", 0.0) or 0.0) for row in ok_results),
        default=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a blocked-M/N GEMM reuse sweep and write roofline-ready metrics for each case."
        )
    )
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=blocked.TILE)
    parser.add_argument("--spad-frontend-queue-size", type=int, default=4)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=DEFAULT_CASE_SPECS,
        help="Case specs in LABEL:WEIGHT_REUSE_M:ACTIVATION_REUSE_N form.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for timestamped sweep runs when --output-dir is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit run directory. By default the runner creates a fresh timestamped directory under --output-root.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow reusing a non-empty --output-dir by clearing its existing contents first.",
    )
    parser.add_argument(
        "--write-run-logs",
        action="store_true",
        help="Also emit per-run stats and gantt logs under <output-dir>/<run-name>/.",
    )
    args = parser.parse_args()

    cases = [_parse_case_spec(spec, order=index) for index, spec in enumerate(args.cases)]
    if not cases:
        raise SystemExit("at least one case spec is required")

    output_dir, output_root = _resolve_output_dir(args)
    _prepare_output_dir(output_dir, overwrite_output_dir=bool(args.overwrite_output_dir))
    _write_latest_run(output_root, output_dir)
    print(f"writing sweep outputs to {output_dir}", flush=True)

    manifest = _build_manifest(args, cases, output_dir)
    _safe_json_dump(output_dir / "manifest.json", manifest)

    results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        print(
            (
                f"[{index}/{len(cases)}] running {case['policy_label']} "
                f"(m={case['weight_reuse_m']} n={case['activation_reuse_n']})"
            ),
            flush=True,
        )
        result = _execute_case(
            case,
            matrix_size=args.matrix_size,
            tile_size=args.tile_size,
            spad_frontend_queue_size=args.spad_frontend_queue_size,
            write_run_logs=args.write_run_logs,
            output_dir=output_dir,
        )
        _print_case_result(result)

        results.append(result)
        summary_rows.append(_summary_row(result))
        _safe_json_dump(output_dir / f"{case['name']}.json", result)
        _write_results_csv(output_dir / "results.csv", summary_rows)
        _safe_json_dump(output_dir / "results.json", results)
        _refresh_manifest(manifest, results)
        _safe_json_dump(output_dir / "manifest.json", manifest)

    _refresh_manifest(manifest, results)
    manifest["status"] = "completed" if manifest["failed_run_count"] == 0 else "completed_with_failures"
    manifest["finished_at_utc"] = _utc_now_iso()
    _safe_json_dump(output_dir / "manifest.json", manifest)

    print(f"wrote {output_dir / 'results.csv'}", flush=True)
    print(f"wrote {output_dir / 'results.json'}", flush=True)
    print(f"wrote {output_dir / 'manifest.json'}", flush=True)
    return 0 if manifest["failed_run_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())