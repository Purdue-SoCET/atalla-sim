import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..//..', 'src')))

from systolic_array.systolic_array_tpu import TPUCell4Input


def test_tpu_cell_latches_group_inputs_and_weights():
    cell = TPUCell4Input(group_size=4)

    cell._input([1.0, 2.0])
    cell._weight([3.0, 4.0, 5.0])
    cell._accumulation(9.75)

    assert cell.activation_latch == [1.0, 2.0, 0.0, 0.0]
    assert cell.weight == [3.0, 4.0, 5.0, 0.0]
    assert cell.accumulation == 9.75


def test_tpu_cell_counts_grouped_mac_ops():
    cell = TPUCell4Input(group_size=4)

    cell.count_mul(4)
    cell.count_add(3)
    cell.count_add(1, is_psum=True)

    assert cell.mul_ops == 4
    assert cell.mac_ops == 4
    assert cell.add_ops == 4
    assert cell.psum_adds == 1


if __name__ == '__main__':
    test_tpu_cell_latches_group_inputs_and_weights()
    test_tpu_cell_counts_grouped_mac_ops()
