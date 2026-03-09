import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from base.debug import close_debug, configure_debug
from memory.dram import DRAM


def test_dram_writes_debug_logs(tmp_path):
    log_dir = tmp_path / "logs"
    configure_debug(flags=["DRAM"], log_dir=str(log_dir))

    try:
        dram = DRAM(block_bytes=16)
        dram.write(0x1003, b"\xAA\xBB\xCC")
        got = dram.read(0x1000, 8)
        dram.snapshot_addr(0x1000, 0x1007)

        dram.write(0x2000, bytes([1, 2, 3, 4, 5, 6]))
        dram.snapshot_tile(0x2000, m=2, n=3, elem_bytes=1)

        assert got == b"\x00\x00\x00\xAA\xBB\xCC\x00\x00"

        log_path = log_dir / "DRAM.log"
        assert log_path.exists(), "expected DRAM debug log file"

        text = log_path.read_text(encoding="utf-8")
        assert "write addr=0x1003 len=3" in text
        assert "read addr=0x1000 len=8" in text
        assert "snapshot_addr start=0x1000 end=0x1007 bytes=8" in text
        assert "0x00001000: 00 00 00 aa bb cc 00 00" in text
        assert "snapshot_tile start=0x2000 m=2 n=3 elem_bytes=1 row_stride=3" in text
        assert "row 00: 0x01 0x02 0x03" in text
        assert "row 01: 0x04 0x05 0x06" in text
    finally:
        close_debug()


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[2] / "logs" / "dram_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dram_writes_debug_logs(out_dir)
