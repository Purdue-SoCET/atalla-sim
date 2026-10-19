"""Shared pytest setup.

@pytest.mark.golden marks a test that compares against the functional
simulator (the third_party/atalla-functional-sim submodule). When the
submodule isn't checked out, those tests are skipped rather than failing.
Mark any new test that calls scheduler.golden.run_golden_state.
"""
import pytest

from scheduler import golden


def pytest_collection_modifyitems(config, items):
    if golden.HAVE_GOLDEN:
        return
    skip = pytest.mark.skip(reason="functional sim unavailable (%s)" % golden.load_error)
    for item in items:
        if "golden" in item.keywords:
            item.add_marker(skip)
