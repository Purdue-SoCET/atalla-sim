"""The banked vector register file, VEGGIE v2 (plan section 4).

B = 2S banks of `regs_per_bank` full vectors. Register r lives in bank
r % B at address r // B (Q1, "mod"), so vd[2:0] is the bank at B = 8. Each
bank is 1R1W (Q2); the read and write ports are arbitrated by VectorCoreV2,
this class only stores. Writes take a per-element enable (Q10b): a masked
write changes only its active elements, with no read of the old value.

The untimed API (read / write / read_with_dtype) is what harness setup,
golden comparison and decode 2's storage adapter use.
"""

from typing import List, Optional, Sequence, Tuple

from base.dtype import DType, cast_vector


class SlotContractError(ValueError):
    """A slot op the M x N hardware cannot do, e.g. a source outside its bank
    pair in strict read mode."""


class BankedVRF:

    def __init__(self, num_banks: int = 8, regs_per_bank: int = 32, vector_len: int = 32):
        self.num_banks = int(num_banks)
        self.regs_per_bank = int(regs_per_bank)
        self.vector_len = int(vector_len)
        zero = [0.0] * self.vector_len
        self.banks: List[List[List[float]]] = [
            [list(zero) for _ in range(self.regs_per_bank)] for _ in range(self.num_banks)]
        self.reads = [0] * self.num_banks
        self.writes = [0] * self.num_banks

    @property
    def num_regs(self) -> int:
        return self.num_banks * self.regs_per_bank

    def bank_addr(self, reg: int) -> Tuple[int, int]:
        reg = int(reg)
        if not 0 <= reg < self.num_regs:
            raise ValueError("vector register out of range: %d" % reg)
        return reg % self.num_banks, reg // self.num_banks

    def bank_of(self, reg: int) -> int:
        return self.bank_addr(reg)[0]

    # -- untimed ------------------------------------------------------------------
    def read(self, reg: int) -> List[float]:
        b, a = self.bank_addr(reg)
        return list(self.banks[b][a])

    def read_with_dtype(self, reg: int) -> Tuple[List[float], DType]:
        return self.read(reg), DType.BF16

    def write(self, reg: int, values: Sequence[Optional[float]],
              enable: Optional[Sequence[bool]] = None) -> None:
        """Write `values` (rounded to BF16) where `enable` is set; every
        element when `enable` is None. A None value is never written."""
        b, a = self.bank_addr(reg)
        old = self.banks[b][a]
        vals = list(values)[: self.vector_len]
        vals += [None] * (self.vector_len - len(vals))
        en = list(enable) if enable is not None else [True] * self.vector_len
        new = [float(v) if (e and v is not None) else o for o, v, e in zip(old, vals, en)]
        self.banks[b][a] = cast_vector(new, DType.BF16)


class BankedVRFStorage:
    """Decode 2's vector register file kept in the banked VRF, so the
    registers exist once (as core.VeggieStorage does for v1)."""

    def __init__(self, vrf: BankedVRF):
        self.vrf = vrf

    def read(self, r: int):
        return self.vrf.read(r)

    def write(self, r: int, value) -> None:
        self.vrf.write(r, [float(x) for x in value])
