from typing import Dict, Iterable, Optional, Tuple, Union
from base.debug import dprintf


class DRAM:
    """
    Sparse DRAM model backed by a hashmap.

    Memory is allocated lazily in fixed-size blocks and only for addresses that
    are written. Reads from untouched locations return zeros.
    """

    def __init__(self, block_bytes: int = 256):
        if int(block_bytes) <= 0:
            raise ValueError("block_bytes must be > 0")
        self.block_bytes = int(block_bytes)
        self._blocks: Dict[int, bytearray] = {}

    def _split_addr(self, addr: int) -> Tuple[int, int]:
        if int(addr) < 0:
            raise ValueError("address must be >= 0")
        block_idx = int(addr) // self.block_bytes
        offset = int(addr) % self.block_bytes
        return block_idx, offset

    def _get_block(self, block_idx: int, create: bool) -> Optional[bytearray]:
        block = self._blocks.get(block_idx)
        if block is None and create:
            block = bytearray(self.block_bytes)
            self._blocks[block_idx] = block
        return block

    def _read_impl(self, addr: int, length: int, log: bool) -> bytes:
        if int(length) < 0:
            raise ValueError("length must be >= 0")
        if int(length) == 0:
            return b""

        out = bytearray(int(length))
        remaining = int(length)
        cur = int(addr)
        dst = 0
        while remaining > 0:
            block_idx, offset = self._split_addr(cur)
            step = min(remaining, self.block_bytes - offset)
            block = self._get_block(block_idx, create=False)
            if block is not None:
                out[dst : dst + step] = block[offset : offset + step]
            cur += step
            dst += step
            remaining -= step
        if log:
            dprintf("DRAM", f"read addr=0x{int(addr):x} len={int(length)}")
        return bytes(out)

    def read(self, addr: int, length: int) -> bytes:
        return self._read_impl(addr, length, log=True)

    def write(self, addr: int, data: Union[bytes, bytearray, memoryview, Iterable[int]]) -> None:
        blob = bytes(data)
        if not blob:
            return

        remaining = len(blob)
        cur = int(addr)
        src = 0
        while remaining > 0:
            block_idx, offset = self._split_addr(cur)
            step = min(remaining, self.block_bytes - offset)
            block = self._get_block(block_idx, create=True)
            assert block is not None
            block[offset : offset + step] = blob[src : src + step]
            cur += step
            src += step
            remaining -= step
        dprintf("DRAM", f"write addr=0x{int(addr):x} len={len(blob)}")

    def snapshot_addr(self, start_addr: int, end_addr: int, bytes_per_line: int = 16) -> None:
        """
        Debug dump of address range [start_addr, end_addr] (inclusive).
        """
        start = int(start_addr)
        end = int(end_addr)
        if start < 0 or end < 0:
            raise ValueError("addresses must be >= 0")
        if end < start:
            raise ValueError("end_addr must be >= start_addr")
        if int(bytes_per_line) <= 0:
            raise ValueError("bytes_per_line must be > 0")

        line_w = int(bytes_per_line)
        total = end - start + 1
        dprintf("DRAM", f"snapshot_addr start=0x{start:x} end=0x{end:x} bytes={total}")
        cur = start
        while cur <= end:
            chunk_len = min(line_w, end - cur + 1)
            chunk = self._read_impl(cur, chunk_len, log=False)
            hex_bytes = " ".join(f"{b:02x}" for b in chunk)
            dprintf("DRAM", f"0x{cur:08x}: {hex_bytes}")
            cur += chunk_len

    def snapshot_tile(
        self,
        start_addr: int,
        m: int,
        n: int,
        elem_bytes: int = 1,
        row_stride_bytes: Optional[int] = None,
    ) -> None:
        """
        Debug dump of an m x n tile in row-major layout from start_addr.
        """
        base = int(start_addr)
        rows = int(m)
        cols = int(n)
        ebytes = int(elem_bytes)
        if base < 0:
            raise ValueError("start_addr must be >= 0")
        if rows <= 0 or cols <= 0:
            raise ValueError("m and n must be > 0")
        if ebytes <= 0:
            raise ValueError("elem_bytes must be > 0")

        row_stride = int(row_stride_bytes) if row_stride_bytes is not None else cols * ebytes
        if row_stride <= 0:
            raise ValueError("row_stride_bytes must be > 0")

        dprintf(
            "DRAM",
            f"snapshot_tile start=0x{base:x} m={rows} n={cols} elem_bytes={ebytes} row_stride={row_stride}",
        )
        for r in range(rows):
            row_vals = []
            row_base = base + r * row_stride
            for c in range(cols):
                elem_addr = row_base + c * ebytes
                raw = self._read_impl(elem_addr, ebytes, log=False)
                val = int.from_bytes(raw, "little", signed=False)
                row_vals.append(f"0x{val:0{2 * ebytes}x}")
            dprintf("DRAM", f"row {r:02d}: {' '.join(row_vals)}")

    def clear(self) -> None:
        self._blocks.clear()

    @property
    def allocated_blocks(self) -> int:
        return len(self._blocks)

    @property
    def allocated_bytes(self) -> int:
        return len(self._blocks) * self.block_bytes
