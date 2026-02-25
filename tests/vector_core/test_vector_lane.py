import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

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
        (9, 1, 0, 13),  # vector index 1 => 11 + 2
        (9, 1, 1, 21),  # vector index 5 => 15 + 6
    ]
    assert collector.reduce_accum == []


def test_vector_lane_unit_mask_and_reduction_callbacks():
    lane = VectorLane(lane_id=0, lane_count=4, fu_latencies={"alu": 1})
    collector = RecordingCollector()

    src0 = [2, 3, 4, 5, 6, 7, 8, 9]
    src1 = [10, 10, 10, 10, 10, 10, 10, 10]
    # Lane 0 visits vector indices [0, 4]. Mask out index 4.
    mask = [True, True, True, True, False, True, True, True]
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
