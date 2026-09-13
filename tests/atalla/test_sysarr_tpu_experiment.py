import pytest

from atalla.sysarr_tpu_experiment import SysArrTPUExperimentConfig, run_sysarr_tpu_experiment
from atalla.sysarr_tpu_system import SPAD_NUM_PADS


def test_sysarr_tpu_experiment_reports_one_backend_per_pad():
    """One backend per scratchpad pad, all on the same DRAM."""
    cfg = SysArrTPUExperimentConfig(
        name="shared_backend_unit",
        sweep="unit",
        param_name="backend_count",
        param_value=str(SPAD_NUM_PADS),
        tile=8,
        spad_num_banks=8,
        spad_bank_size=128,
        max_cycles=8000,
    )

    result = run_sysarr_tpu_experiment(cfg)

    assert result["backend_count"] == SPAD_NUM_PADS
    assert result["backend_slots_attached"] == SPAD_NUM_PADS
    assert result["backend_shared_dram"] is True
    assert result["cycles"] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
