import numpy as np
import pytest

from analysis_helpers.binning import (
    adaptive_bin_edges,
)


def test_adaptive_bin_edges_respects_requested_range():
    data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    edges = adaptive_bin_edges(data, bins=4, bins_range=(1.0, 3.0))

    assert edges[0] == pytest.approx(1.0)
    assert edges[-1] == pytest.approx(3.0)
    assert np.all(np.diff(edges) >= 0)
