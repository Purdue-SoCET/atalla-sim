import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from atalla.sysarr_tpu_experiment import SysArrTPUExperimentConfig, run_sysarr_tpu_experiment


def test_sysarr_tpu_experiment_reports_two_shared_backends():
    cfg = SysArrTPUExperimentConfig(
        name="shared_backend_unit",
        sweep="unit",
        param_name="backend_count",
        param_value="2",
        tile=8,
        spad_num_banks=8,
        spad_bank_size=128,
        max_cycles=8000,
    )

    result = run_sysarr_tpu_experiment(cfg)

    assert result["backend_count"] == 2
    assert result["backend_slots_attached"] == 2
    assert result["backend_shared_dram"] is True
    assert result["cycles"] > 0


if __name__ == "__main__":
    test_sysarr_tpu_experiment_reports_two_shared_backends()