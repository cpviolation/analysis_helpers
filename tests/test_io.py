from pathlib import Path

import pandas as pd

from analysis_helpers import io


class _FakeTree:
    def __init__(self, chunks):
        self._chunks = chunks

    def iterate(self, branches, library, step_size):
        assert library == "pd"
        for chunk in self._chunks:
            yield chunk[list(branches)]


class _FakeFile:
    def __init__(self, tree_name, chunks):
        self._tree_name = tree_name
        self._tree = _FakeTree(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getitem__(self, key):
        if key != self._tree_name:
            raise KeyError(key)
        return self._tree


def test_load_df_incremental_combines_rows(monkeypatch):
    branches = ["x", "y"]
    tree_name = "Events"

    chunks = [
        pd.DataFrame({"x": [1, 2], "y": [10, 20]}),
        pd.DataFrame({"x": [3], "y": [30]}),
    ]

    def fake_open(path):
        assert path == "fake.root"
        return _FakeFile(tree_name, chunks)

    monkeypatch.setattr(io.uproot, "open", fake_open)

    out = io.load_df_incremental(["fake.root"], branches, tree_name, progress=False)

    assert out.to_dict(orient="list") == {"x": [1, 2, 3], "y": [10, 20, 30]}


def test_cache_is_valid_false_when_input_is_newer(tmp_path):
    cache = tmp_path / "cache.parquet"
    cache.write_text("cache")

    source = tmp_path / "input.root"
    source.write_text("input")

    # Touch source after cache to simulate stale cache.
    source.touch()

    assert io.cache_is_valid([str(source)], str(cache)) is False


def test_cache_is_valid_true_when_missing_inputs_are_ignored(tmp_path):
    cache = tmp_path / "cache.parquet"
    cache.write_text("cache")

    missing = tmp_path / "missing.root"

    assert io.cache_is_valid([str(missing)], str(cache)) is True
