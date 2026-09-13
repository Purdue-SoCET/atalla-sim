from vector_core.vector_lanes import LaneFUContext, VectorLane


class RecordingCollector:
    def __init__(self):
        self.dispatched = []
        self.reduce_accum = []
        self.results = []
        self.done = []

    def can_accept_result(self):
        return True

    def lane_dispatched(self, inst_id, lane_id):
        self.dispatched.append((inst_id, lane_id))

    def lane_reduce_accum(self, inst_id, lane_id, value):
        self.reduce_accum.append((inst_id, lane_id, value))

    def lane_result(self, inst_id, lane_id, lane_elem_idx, value):
        self.results.append((inst_id, lane_id, lane_elem_idx, value))

    def lane_done(self, inst_id, lane_id):
        self.done.append((inst_id, lane_id))


def _run_lane(lane, collector, cycles=12):
    for t in range(cycles):
        lane.tick(float(t), collector)


def test_vector_lane_unit_dispatch_and_result_order():
    lane = VectorLane(lane_id=1, lane_count=4, fu_latencies={"alu": 2})
    collector = RecordingCollector()

    src0 = [10, 11, 12, 13, 14, 15, 16, 17]
    src1 = [1, 2, 3, 4, 5, 6, 7, 8]
    ctx = LaneFUContext(
        inst_id=9,
        op="add",
        src0=src0,
        src1=src1,
        mask=[True] * len(src0),
        indices=lane._lane_indices(len(src0)),
        dst=0,
        reduce=False,
    )
    assert lane.issue(ctx, "alu")

    _run_lane(lane, collector)

    assert collector.done == [(9, 1)]
    assert collector.dispatched == [(9, 1), (9, 1)]
    assert collector.results == [
        # The slicer gives lane 1 of 4 the contiguous pair [2, 3].
        (9, 1, 0, 15),  # vector index 2 => 12 + 3
        (9, 1, 1, 17),  # vector index 3 => 13 + 4
    ]
    assert collector.reduce_accum == []


def test_vector_lane_unit_mask_and_reduction_callbacks():
    lane = VectorLane(lane_id=0, lane_count=4, fu_latencies={"alu": 1})
    collector = RecordingCollector()

    src0 = [2, 3, 4, 5, 6, 7, 8, 9]
    src1 = [10, 10, 10, 10, 10, 10, 10, 10]
    # The slicer gives lane 0 of 4 the contiguous pair [0, 1]. Mask out index 1.
    mask = [True, False, True, True, True, True, True, True]
    ctx = LaneFUContext(
        inst_id=3,
        op="mul",
        src0=src0,
        src1=src1,
        mask=mask,
        indices=lane._lane_indices(len(src0)),
        dst=1,
        reduce=True,
    )
    assert lane.issue(ctx, "alu")

    _run_lane(lane, collector, cycles=8)

    assert collector.done == [(3, 0)]
    assert collector.dispatched == [(3, 0)]
    assert collector.results == [(3, 0, 0, 20)]
    assert collector.reduce_accum == [(3, 0, 20)]


def test_slicer_cuts_the_vector_into_contiguous_lane_slices():
    """Each lane owns an adjacent run of elements, not a stride."""
    from vector_core.vector_lanes import (
        lane_slice_indices, slice_to_lane, slice_width)

    assert slice_width(32, 16) == 2
    assert slice_width(32, 4) == 8

    # 8 elements over 4 lanes: [0,1] [2,3] [4,5] [6,7]
    slice_w = slice_width(8, 4)
    assert [lane_slice_indices(l, slice_w) for l in range(4)] == [
        [0, 1], [2, 3], [4, 5], [6, 7]]

    # Every element lands in exactly one lane, and the map inverts.
    for elem in range(8):
        lane, pos = slice_to_lane(elem, slice_w)
        assert lane_slice_indices(lane, slice_w)[pos] == elem


def test_slicer_and_collector_agree_on_the_mapping():
    """The collector must reassemble in the order the slicer cut."""
    from vector_core.vector_lanes import ResultCollector, slice_width

    vector_len, lane_count = 32, 8
    rc = ResultCollector(lane_count, vector_len)
    assert rc.slice_w == slice_width(vector_len, lane_count)
    for lane in range(lane_count):
        for pos in range(rc.slice_w):
            assert lane * rc.slice_w + pos < vector_len


def test_lane_count_must_divide_the_vector():
    from vector_core.vector_lanes import VectorDatapath
    import pytest as _pytest

    VectorDatapath(veggie_size=32 * 16, lane_count=16)      # 32 / 16 == 2
    with _pytest.raises(ValueError, match="does not divide"):
        VectorDatapath(veggie_size=32 * 16, lane_count=7)
