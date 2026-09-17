# Copyright (c) 2026, QuACK team.
"""Tests for CUDA capture-info compatibility across binding versions."""

import pytest

from quack.distributed.all_gather_gemm import _capture_dependencies


@pytest.mark.parametrize(
    ("capture_info", "expected"),
    [
        ((0, 11, 22, [33], 44, 55), ([33], 55)),
        ((0, 11, 22, [33], 55), ([33], 55)),
    ],
)
def test_capture_dependencies_supports_five_and_six_value_bindings(capture_info, expected):
    assert _capture_dependencies(capture_info) == expected


def test_capture_dependencies_rejects_unknown_binding_shape():
    with pytest.raises(RuntimeError, match="unexpected number of values: 4"):
        _capture_dependencies((0, 1, 2, 3))
