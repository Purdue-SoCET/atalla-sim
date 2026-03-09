import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from base.debug import close_debug, configure_debug
from memory.crossbar import Xbar


def test_crossbar_writes_debug_logs(tmp_path):
    log_dir = tmp_path / "logs"
    configure_debug(flags=["Xbar"], log_dir=str(log_dir))

    try:
        xbar = Xbar(delay=1, num_banks=4, max_size=1)

        shift_mask = [1, 0, None, None]
        vals = [10, 11, 12, 13]

        ok_op = xbar.enqueue(shift_mask, vals)
        dropped_op = xbar.enqueue(shift_mask, vals)
        completed = xbar.tick()

        assert ok_op > 0
        assert dropped_op == -1
        assert completed and completed[0][0] == ok_op

        log_path = log_dir / "Xbar.log"
        assert log_path.exists(), "expected Xbar debug log file"

        text = log_path.read_text(encoding="utf-8")
        assert "enqueue op=" in text
        assert "enqueue dropped: queue full" in text
        assert "complete op=" in text
    finally:
        close_debug()


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[2] / "logs" / "crossbar_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_crossbar_writes_debug_logs(out_dir)
