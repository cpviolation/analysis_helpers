import math

import pytest

from analysis_helpers import evaluate_threshold, evaluate_threshold_dacp


def test_cpp_binding_exports_are_consistent():
    if evaluate_threshold is None or evaluate_threshold_dacp is None:
        pytest.skip("Native _cpp binding is not available in this Python environment")

    v1 = evaluate_threshold(2.0, 1.0, 1.5)
    v2 = evaluate_threshold_dacp(2.0, 1.0, 1.5, 2.0)

    assert math.isfinite(v1)
    assert math.isfinite(v2)
    assert v1 > 0.0
    assert v2 > 0.0
