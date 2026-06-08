import pytest
import numpy as np

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


def test_ttree2array_handles_in_memory_tree(monkeypatch):
    if not root_helpers.is_root_available():
        pytest.skip("PyROOT is not available; in-memory ROOT fallback cannot be exercised")

    class FakeLeaf:
        def __init__(self, name):
            self._name = name

        def GetName(self):
            return self._name

    class FakeTree:
        def __init__(self, rows):
            self._rows = rows

        def GetListOfLeaves(self):
            return [FakeLeaf("a"), FakeLeaf("b")]

        def GetEntries(self):
            return len(self._rows)

        def GetCurrentFile(self):
            return None

        def GetEntry(self, entry_idx):
            self.a, self.b = self._rows[entry_idx]
            return 1

    class FailIfUsed:
        def EnableImplicitMT(self):
            raise AssertionError("RDataFrame path should not be used for in-memory trees")

        def RDataFrame(self, *_args, **_kwargs):
            raise AssertionError("RDataFrame path should not be used for in-memory trees")

    monkeypatch.setattr(root_helpers, "r", FailIfUsed())

    arr = root_helpers.TTree2Array(FakeTree([(1.0, 2.0), (3.0, 4.0)]))

    assert arr.shape == (2, 2)
    assert np.allclose(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_root2mpltext_keeps_plain_text_unchanged():
    assert root_helpers.ROOT2MPLText("Mass [MeV]") == "Mass [MeV]"


def test_root2mpltext_converts_root_math_syntax():
    label = "#Delta m = m(D^{*+}) - m(D^{0})"
    converted = root_helpers.ROOT2MPLText(label)
    assert converted == "$\\Delta m = m(D^{*+}) - m(D^{0})$"


def test_root2mpltext_converts_root_text_styles():
    label = "#it{Signal} #bf{Yield}"
    converted = root_helpers.ROOT2MPLText(label)
    assert converted == "$\\mathit{Signal} \\mathbf{Yield}$"
