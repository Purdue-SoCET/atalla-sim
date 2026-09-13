"""The chip's local memory is four independent pads, one per VLSU.

Proves the system-level path end to end:

    DRAM -> Backend[p] -> SRAM banks[p] -> Frontend[p] -> VLSU[p] -> VRF

for every pad, and that the four pads are genuinely separate address spaces --
the same local slot in each holds different data.
"""
import pytest

from atalla.sysarr_tpu_system import (
    PAD_ACT, PAD_OUT, PAD_PSUM, PAD_WGT, SPAD_NUM_PADS, build_tpu_platform)
from memory.sc_sram_banks import _xor_bank

VEC = 8


def _small_platform(**kw):
    """A 4-pad platform small enough to be instant."""
    opts = dict(size=VEC, dtype="fp16", lane_count=2, spad_num_banks=VEC,
                spad_bank_size=64, spad_read_latency=1, spad_write_latency=1,
                spad_xbar_delay=1, backend_dram_latency=4,
                backend_dram_burst_bytes=32)
    opts.update(kw)
    return build_tpu_platform(**opts)


def _read_pad_row(spad, pad, slot):
    """Undo the bank swizzle and read one row straight out of a pad's banks."""
    out = []
    for lane in range(spad.num_banks):
        bank = _xor_bank(slot, lane, spad.num_banks)
        blob = bytes(spad.tiles[pad].banks[bank].mem[slot] or b"")
        blob = blob + b"\x00" * (2 - len(blob))
        out.append(int.from_bytes(blob[:2], "little", signed=False))
    return out


def _pad_row(pad):
    """A payload that identifies the pad it belongs to."""
    return [(pad + 1) * 10 + i for i in range(VEC)]


def _dma_every_pad(platform, base_slot=0):
    """Fill the same local slot of all four pads through their own backends."""
    done = set()
    for pad in range(SPAD_NUM_PADS):
        dram_addr = 0x1000 + pad * 0x100
        platform.dram.write(
            dram_addr,
            b"".join(int(v).to_bytes(2, "little") for v in _pad_row(pad)))
        tx = platform.backends[pad].driver_to_backend_start_load(
            base_sp_addr=base_slot, base_dram_addr=dram_addr, rows=1, cols=VEC,
            callback=lambda _tx, _p=pad: done.add(_p))
        assert tx > 0, "pad %d refused its DMA" % pad

    # The callback fires when the DRAM side is done; the row still has to cross
    # the write crossbar and commit into the banks, so wait for the data itself.
    for cycle in range(1, 2000):
        platform.tick(float(cycle))
        if len(done) == SPAD_NUM_PADS and all(
            _read_pad_row(platform.spad, pad, base_slot) == _pad_row(pad)
            for pad in range(SPAD_NUM_PADS)
        ):
            return cycle
    raise AssertionError(
        "DMA never landed in the banks; callbacks done for %s" % sorted(done))


def test_each_pad_is_its_own_address_space():
    """Same slot index, four backends, four different payloads."""
    platform = _small_platform()
    _dma_every_pad(platform)

    for pad in range(SPAD_NUM_PADS):
        assert _read_pad_row(platform.spad, pad, 0) == _pad_row(pad), (
            "pad %d holds the wrong row" % pad)

    # and they really are distinct storage, not four views of one array
    rows = [tuple(_read_pad_row(platform.spad, p, 0)) for p in range(SPAD_NUM_PADS)]
    assert len(set(rows)) == SPAD_NUM_PADS


def test_every_vlsu_loads_from_its_own_pad_into_the_register_file():
    """banks -> Frontend -> VLSU -> vector register file, on all four pads."""
    platform = _small_platform()
    vc = platform.spad and platform.vc
    _dma_every_pad(platform)

    # One destination register per pad, each in a different VRF bank
    # (bank = reg % 4), so the writeback buffer never refuses one for a bank
    # conflict and all four can be in flight together.
    dst_of = {pad: 4 + pad for pad in range(SPAD_NUM_PADS)}
    assert len({r % vc.veggie.bank_count for r in dst_of.values()}) == SPAD_NUM_PADS

    for pad, dst in dst_of.items():
        assert vc.enqueue_memory({"kind": "load", "vls": pad, "dst": dst,
                                  "addr": 0, "dtype": "fp16"})

    seen = {}
    for cycle in range(1, 2000):
        platform.tick(float(cycle))
        wb = vc.last_wb
        if vc.wb_valid and wb is not None and wb.get("source") == "vlsu":
            seen[int(wb["vls"])] = [int(x) for x in wb["data"][:VEC]]
        if len(seen) == SPAD_NUM_PADS:
            break
    else:
        raise AssertionError("only pads %s wrote back" % sorted(seen))

    for pad in range(SPAD_NUM_PADS):
        assert seen[pad] == _pad_row(pad), "VLSU %d read the wrong pad" % pad
        assert [int(x) for x in vc.read_vreg(dst_of[pad])[:VEC]] == _pad_row(pad)

    # every bridge actually moved bytes -- no pad was skipped
    for pad, bridge in enumerate(platform.vls_bridges):
        assert bridge.bytes_load > 0, "bridge %d never loaded" % pad


def test_a_vlsu_store_lands_in_its_own_pad_and_reads_back():
    """The psum pad is written and re-read through VLSU 2 alone."""
    platform = _small_platform()
    vc = platform.vc
    payload = [7 * i + 3 for i in range(VEC)]

    assert vc.enqueue_memory({"kind": "store", "vls": PAD_PSUM,
                              "data": [float(v) for v in payload],
                              "addr": 5, "dtype": "fp16"})
    for cycle in range(1, 400):
        platform.tick(float(cycle))
        if _read_pad_row(platform.spad, PAD_PSUM, 5) == payload:
            break
    else:
        raise AssertionError("store never reached the psum pad")

    # nothing else was disturbed
    for other in (PAD_ACT, PAD_WGT, PAD_OUT):
        assert _read_pad_row(platform.spad, other, 5) == [0] * VEC

    assert vc.enqueue_memory({"kind": "load", "vls": PAD_PSUM, "dst": 9,
                              "addr": 5, "dtype": "fp16"})
    for cycle in range(400, 900):
        platform.tick(float(cycle))
        if vc.wb_valid and vc.last_wb and vc.last_wb.get("source") == "vlsu":
            break
    else:
        raise AssertionError("psum load never wrote back")
    assert [int(x) for x in vc.read_vreg(9)[:VEC]] == payload


def test_the_reference_system_spreads_a_gemm_across_three_pads():
    """SysArrTPUSystem drives the whole chain -- activations, weights and
    outputs each in their own pad, each through its own VLSU, with the result
    checked against the reference model."""
    from atalla.sysarr_tpu_system import SysArrTPUSystem, _act_u16, _weights_u16

    size = 8
    system = SysArrTPUSystem(size=size, dtype="fp16", mirror=True,
                             spad_num_banks=size, spad_bank_size=64)
    assert (system.ACT_PAD, system.WGT_PAD, system.OUT_PAD) == (PAD_ACT, PAD_WGT, PAD_OUT)

    system.load_inputs(_act_u16(size), _weights_u16(size))
    observed, expected, cycles, _metrics = system.run(max_cycles=20000)

    assert expected is not None, "the mirror must produce a reference"
    assert observed == expected
    assert cycles > 0

    # each role pad carried its own traffic
    assert system.platform.vls_bridges[PAD_WGT].bytes_load > 0
    assert system.platform.vls_bridges[PAD_ACT].bytes_load > 0
    assert system.platform.vls_bridges[PAD_OUT].bytes_store > 0
