#!/usr/bin/env python3

import argparse
import csv
import json
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if sys.version_info < (3, 7):
    raise SystemExit("This runner needs Python 3.7+ because the simulator sources use postponed annotations. Try `python3.11`.")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atalla.sysarr_tpu_experiment import PHASE_ORDER, SysArrTPUExperimentConfig, run_sysarr_tpu_experiment


DEFAULT_SWEEPS = ["tile", "dram_latency", "dram_burst_bytes", "dram_q_depth", "spad_frontend_queue_size"]

SWEEP_TO_CONFIG_KEY = {
    "tile": "tile",
    "dram_latency": "backend_dram_latency",
    "dram_burst_bytes": "backend_dram_burst_bytes",
    "dram_q_depth": "backend_dram_q_depth",
    "spad_frontend_queue_size": "spad_frontend_queue_size",
}


def _baseline() -> Dict[str, int]:
    return {
        "tile": 32,
        "spad_num_banks": 32,
        "spad_bank_size": 128,
        "spad_read_latency": 2,
        "spad_write_latency": 2,
        "spad_xbar_delay": 3,
        "spad_frontend_queue_size": 4,
        "backend_dram_latency": 24,
        "backend_dram_q_depth": 16,
        "backend_dram_burst_bytes": 32,
    }


def _values_for_sweep(name: str, quick: bool) -> List[int]:
    if name == "tile":
        return [16, 32] if quick else [8, 16, 24, 32]
    if name == "dram_latency":
        return [12, 24, 36] if quick else [8, 16, 24, 32, 40]
    if name == "dram_burst_bytes":
        return [16, 32, 64]
    if name == "dram_q_depth":
        return [8, 16, 32] if quick else [4, 8, 16, 32]
    if name == "spad_frontend_queue_size":
        return [1, 4, 8] if quick else [1, 2, 4, 8]
    raise ValueError(f"unsupported sweep: {name}")


def build_experiment_configs(sweeps: Iterable[str], quick: bool) -> List[SysArrTPUExperimentConfig]:
    baseline = _baseline()
    configs: List[SysArrTPUExperimentConfig] = []
    for sweep in sweeps:
        config_key = SWEEP_TO_CONFIG_KEY.get(sweep)
        if config_key is None:
            raise ValueError(f"unsupported sweep: {sweep}")
        for value in _values_for_sweep(sweep, quick):
            cfg = deepcopy(baseline)
            cfg[config_key] = value
            if sweep == "tile":
                cfg["spad_num_banks"] = value
                cfg["spad_bank_size"] = max(128, value * 4)
            name = f"{sweep}_{value}"
            configs.append(
                SysArrTPUExperimentConfig(
                    name=name,
                    sweep=sweep,
                    param_name=sweep,
                    param_value=value,
                    **cfg,
                )
            )
    return configs


def _ordered_fieldnames(rows: List[Dict[str, object]]) -> List[str]:
    leading = [
        "name",
        "sweep",
        "param_name",
        "param_value",
        "status",
        "attempt_count",
        "final_max_cycles",
        "tile",
        "lane_count",
        "vls_count",
        "spad_num_banks",
        "spad_bank_size",
        "spad_read_latency",
        "spad_write_latency",
        "spad_xbar_delay",
        "spad_frontend_queue_size",
        "backend_dram_latency",
        "backend_dram_q_depth",
        "backend_dram_burst_bytes",
        "cycles",
        "flops_micro",
        "bytes_transmitted",
        "bytes_internal",
        "flops_algo",
        "bytes_algo",
        "mac_utilization",
        "throughput_float_operations_per_cycle",
        "external_bandwidth_avg_bytes_per_cycle",
        "internal_bandwidth_bytes_per_cycle",
        "reuse_act_internal_over_external",
        "reuse_psum_internal_over_external",
        "fp16_saturation_count",
        "fp16_overflow_count",
        "max_abs_error",
        "mean_abs_error",
    ]
    extras = sorted({key for row in rows for key in row.keys() if key not in leading and key not in {"config", "vec_op_counts"}})
    return leading + extras


def _run_with_retries(config: SysArrTPUExperimentConfig, retry_max_cycles: List[int]) -> Dict[str, object]:
    attempts = [config.max_cycles] + [value for value in retry_max_cycles if value > config.max_cycles]
    last_error = None
    for attempt_idx, max_cycles in enumerate(attempts, start=1):
        run_config = SysArrTPUExperimentConfig(**dict(config.to_dict(), max_cycles=max_cycles))
        if attempt_idx > 1:
            print(
                "  retrying with max_cycles=%s (attempt %s/%s)"
                % (max_cycles, attempt_idx, len(attempts)),
                flush=True,
            )
        try:
            result = run_sysarr_tpu_experiment(run_config)
            result["status"] = "ok"
            result["attempt_count"] = attempt_idx
            result["final_max_cycles"] = max_cycles
            return result
        except Exception as exc:
            last_error = exc
            message = str(exc)
            timed_out = (
                "timed out waiting" in message
                or "store did not commit to scratchpad" in message
            )
            if (not timed_out) or (attempt_idx >= len(attempts)):
                failure = run_config.to_dict()
                failure.update(
                    {
                        "status": "failed",
                        "attempt_count": attempt_idx,
                        "final_max_cycles": max_cycles,
                        "error_type": type(exc).__name__,
                        "error_message": message,
                        "traceback": traceback.format_exc(),
                    }
                )
                return failure
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parameter sweeps for the sysarr TPU harness.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "logs" / "sysarr_tpu_sweeps"))
    parser.add_argument("--sweeps", nargs="+", default=DEFAULT_SWEEPS)
    parser.add_argument("--quick", action="store_true", help="Run a smaller sweep set for faster turnaround.")
    parser.add_argument(
        "--retry-max-cycles",
        nargs="*",
        type=int,
        default=[40000, 80000],
        help="Extra max-cycle budgets to try if a run times out.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_experiment_configs(args.sweeps, quick=args.quick)
    results = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] running {config.name}", flush=True)
        result = _run_with_retries(config, args.retry_max_cycles)
        results.append(result)
        status = result.get("status", "unknown")
        if status == "ok":
            print(
                "  completed cycles=%s attempts=%s max_cycles=%s"
                % (
                    result.get("cycles"),
                    result.get("attempt_count"),
                    result.get("final_max_cycles"),
                ),
                flush=True,
            )
        else:
            print(
                "  failed error=%s attempts=%s max_cycles=%s"
                % (
                    result.get("error_type"),
                    result.get("attempt_count"),
                    result.get("final_max_cycles"),
                ),
                flush=True,
            )

        run_path = output_dir / f"{config.name}.json"
        run_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = output_dir / "results.csv"
    fieldnames = _ordered_fieldnames(results)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            flat = {key: value for key, value in row.items() if key not in {"config", "vec_op_counts"}}
            writer.writerow(flat)

    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "sweeps": list(args.sweeps),
        "quick": bool(args.quick),
        "run_count": len(results),
        "phase_columns": [f"phase_{name}" for name in PHASE_ORDER],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
