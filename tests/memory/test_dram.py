import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from memory.dram import DRAM


def test_sparse_dram_is_lazily_allocated_and_zero_initialized():
    dram = DRAM(block_bytes=16)

    assert dram.allocated_blocks == 0
    assert dram.read(0x1000, 8) == b"\x00" * 8
    assert dram.allocated_blocks == 0

    dram.write(0x1003, b"\xAA\xBB\xCC")
    assert dram.allocated_blocks == 1

    assert dram.read(0x1000, 8) == b"\x00\x00\x00\xAA\xBB\xCC\x00\x00"


def test_sparse_dram_cross_block_write_and_overwrite():
    dram = DRAM(block_bytes=8)

    dram.write(6, b"\x01\x02\x03\x04")
    assert dram.allocated_blocks == 2
    assert dram.read(0, 12) == b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x00\x00"

    dram.write(7, b"\xAA\xBB\xCC")
    assert dram.read(6, 4) == b"\x01\xAA\xBB\xCC"
