import numpy as np
import pytest

from analysis_helpers import root_helpers


def test_ttree2array_handles_in_memory_tree():
    if not root_helpers.is_root_available():
        pytest.skip("PyROOT not available")

    tree = root_helpers.DefineTree({"w_nsig": "F", "w_nbkg": "F"}, "ntp_sweights", "tree with s-weights")['tree']
    for val1, val2 in [(0.5, 1.5), (1.25, -0.25)]:
        tree.w_nsig = val1
        tree.w_nbkg = val2
        tree.Fill()

    arr = root_helpers.TTree2Array(tree)
    assert arr.shape == (2, 2)
    assert np.allclose(arr[:, 0], [0.5, 1.25])
    assert np.allclose(arr[:, 1], [1.5, -0.25])


def test_is_root_available_returns_boolean():
    assert isinstance(root_helpers.is_root_available(), bool)


def test_require_root_matches_environment():
    if root_helpers.is_root_available():
        root_helpers.require_root()
    else:
        with pytest.raises(ImportError, match="PyROOT is not available"):
            root_helpers.require_root()


def test_root_backed_function_fails_cleanly_without_root():
    if root_helpers.is_root_available():
        pytest.skip("PyROOT is available; optional-ROOT fallback path is not active")

    with pytest.raises(ImportError, match="PyROOT is not available"):
        root_helpers.DefineTree({"x": "F"})
