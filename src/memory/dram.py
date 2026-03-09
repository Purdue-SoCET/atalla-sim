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

    def read(self, addr: int, length: int) -> bytes:
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
        dprintf("DRAM", f"read addr=0x{int(addr):x} len={int(length)}")
        return bytes(out)

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

    def clear(self) -> None:
        self._blocks.clear()

    @property
    def allocated_blocks(self) -> int:
        return len(self._blocks)

    @property
    def allocated_bytes(self) -> int:
        return len(self._blocks) * self.block_bytes
