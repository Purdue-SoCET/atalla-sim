"""atalla-sim new-component (src/atalla/new_component.py): what it writes runs and passes."""
import subprocess
import sys
from pathlib import Path

import pytest

from atalla import new_component

ROOT = Path(__file__).resolve().parents[2]


def test_the_generated_component_and_test_pass(tmp_path):
    src, test = new_component.write("DotProductUnit", "widgets", root=tmp_path)
    assert src == tmp_path / "src" / "widgets" / "dot_product_unit.py"
    assert test == tmp_path / "tests" / "widgets" / "test_dot_product_unit.py"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        str(test)], cwd=str(tmp_path), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True,
                       env={"PYTHONPATH": "%s:%s" % (tmp_path / "src", ROOT / "src"),
                            "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stdout
    assert "2 passed" in p.stdout


def test_names_are_checked_and_files_not_overwritten(tmp_path):
    with pytest.raises(ValueError):
        new_component.write("dot_product", "widgets", root=tmp_path)
    new_component.write("Thing", "widgets", root=tmp_path)
    with pytest.raises(FileExistsError):
        new_component.write("Thing", "widgets", root=tmp_path)
    assert new_component.snake("GSAUBridge") == "gsau_bridge"
