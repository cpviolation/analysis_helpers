import pytest

from analysis_helpers import root_helpers


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
