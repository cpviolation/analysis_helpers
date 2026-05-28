from pathlib import Path

import pandas as pd
import pytest

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


def test_load_data_reads_lines(tmp_path):
    data_file = tmp_path / "sample.txt"
    data_file.write_text("a\nb\nc\n")

    out = io.load_data(str(data_file))

    assert out == ["a", "b", "c"]


def test_load_data_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        io.load_data(str(missing))


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


def test_load_df_incremental_parallel_combines_multiple_files(monkeypatch):
    branches = ["x", "y"]
    tree_name = "Events"

    chunks_by_path = {
        "a.root": [pd.DataFrame({"x": [1], "y": [10]})],
        "b.root": [pd.DataFrame({"x": [2, 3], "y": [20, 30]})],
    }

    def fake_open(path):
        return _FakeFile(tree_name, chunks_by_path[path])

    monkeypatch.setattr(io.uproot, "open", fake_open)

    out = io.load_df_incremental(
        ["a.root", "b.root"],
        branches,
        tree_name,
        progress=False,
        n_workers=2,
    )

    assert out.to_dict(orient="list") == {"x": [1, 2, 3], "y": [10, 20, 30]}


def test_load_df_incremental_raises_for_invalid_worker_count():
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        _ = io.load_df_incremental(
            [],
            ["x"],
            "Events",
            progress=False,
            n_workers=0,
        )


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
