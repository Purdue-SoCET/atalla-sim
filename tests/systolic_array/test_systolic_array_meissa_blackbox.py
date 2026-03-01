import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from systolic_array.systolic_array_meissa_blackbox import SystolicArrayMEISSABlackbox


def test_meissa_blackbox_latency_and_matmul():
    sa = SystolicArrayMEISSABlackbox(n=2, m=3, p=2)

    # B (m x p) loaded top-to-bottom as m rows.
    assert sa.issue({"vdata": [1.0, 2.0], "is_weight": True, "expect_output": False})
    assert sa.issue({"vdata": [3.0, 4.0], "is_weight": True, "expect_output": False})
    assert sa.issue({"vdata": [5.0, 6.0], "is_weight": True, "expect_output": False})

    # A (n x m) streamed left-to-right as n rows.
    assert sa.issue({"vdata": [1.0, 0.0, 1.0], "is_weight": False, "expect_output": True, "meta": {"dst": 10}})
    assert sa.issue({"vdata": [2.0, 1.0, 0.0], "is_weight": False, "expect_output": True, "meta": {"dst": 11}})

    # t_total = n + m + ceil(log2(m)) + p - 1 = 2 + 3 + 2 + 2 - 1 = 8.
    # Since outputs stream one row/cycle, first output appears at t_total - (n-1) = 7.
    for _ in range(6):
        sa.tick()
        assert sa.pop_response() is None

    sa.tick()  # cycle 7
    rsp0 = sa.pop_response()
    assert rsp0 is not None
    assert rsp0["vdata"] == [6.0, 8.0]
    assert rsp0["meta"]["dst"] == 10

    sa.tick()  # cycle 8
    rsp1 = sa.pop_response()
    assert rsp1 is not None
    assert rsp1["vdata"] == [5.0, 8.0]
    assert rsp1["meta"]["dst"] == 11


def test_meissa_blackbox_gsau_packet_shape():
    sa = SystolicArrayMEISSABlackbox(n=1, m=2, p=1)

    # One weight row for m=2? No, B is (m x p), so 2 rows of length 1.
    assert sa.issue({"vdata": [2.0], "is_weight": True, "expect_output": False})
    assert sa.issue({"vdata": [3.0], "is_weight": True, "expect_output": False})
    assert sa.issue({"vdata": [4.0, 5.0], "is_weight": False, "expect_output": True, "meta": {"tag": "req0"}})

    # Total latency: 1 + 2 + ceil(log2(2)) + 1 - 1 = 4 cycles.
    for _ in range(4):
        sa.tick()
    rsp = sa.pop_response()
    assert rsp is not None
    assert rsp["vdata"] == [23.0]
    assert rsp["meta"]["tag"] == "req0"
    assert rsp["meta"]["model"] == "meissa_blackbox"
