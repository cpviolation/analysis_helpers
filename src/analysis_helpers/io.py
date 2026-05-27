from tqdm import tqdm
import uproot
import pandas as pd
import os
from pathlib import Path


def iter_file_dfs(paths, branches, tree_name, chunk_size=100_000, progress=True):
    """Yield one DataFrame per ROOT file.

    The function opens each file with uproot, iterates a tree in chunks,
    and concatenates chunks from the same file into a single DataFrame.
    Files that cannot be opened/read are skipped.

    Args:
        paths: Iterable of ROOT file paths.
        branches: Branch names to read from the tree.
        tree_name: Name of the TTree inside each file.
        chunk_size: Number of entries per chunk while iterating.
        progress: If True, wrap file iteration with a tqdm progress bar.

    Yields:
        tuple[str, pandas.DataFrame]: Pair ``(path, dataframe)`` for each
        readable file containing at least one chunk.
    """
    iterator = tqdm(paths, desc='Loading files') if progress else paths
    for p in iterator:
        try:
            with uproot.open(p) as f:
                tree = f[tree_name]
                parts = []
                for part in tree.iterate(branches, library='pd', step_size=chunk_size):
                    parts.append(part)
                if parts:
                    yield p, pd.concat(parts, ignore_index=True)
        except Exception:
            # skip unreadable files
            continue


def load_df_incremental(paths, branches, tree_name, chunk_size=100_000, progress=True):
    """Load and concatenate data from multiple ROOT files.

    This function consumes :func:`iter_file_dfs` and performs one final
    concatenation across files.

    Args:
        paths: Iterable of ROOT file paths.
        branches: Branch names to read from each tree.
        tree_name: Name of the TTree inside each file.
        chunk_size: Number of entries per chunk while iterating.
        progress: If True, show a file-level progress bar.

    Returns:
        pandas.DataFrame: Concatenated dataframe for all readable files.
        If no data is read, returns an empty dataframe with ``branches`` as
        columns.
    """
    # Efficient final concatenation, while still processing one file at a time
    all_parts = []
    for _, file_df in iter_file_dfs(paths, branches, tree_name, chunk_size=chunk_size, progress=progress):
        all_parts.append(file_df)
    if all_parts:
        return pd.concat(all_parts, ignore_index=True)
    return pd.DataFrame(columns=branches)


def cache_is_valid(paths, cache_path):
    """Check whether a cache file is up to date with respect to inputs.

    The cache is considered valid when it exists and no readable input file has
    a modification time newer than the cache file.

    Missing input files are ignored, while other stat/mtime failures force the
    cache to be considered invalid.

    Args:
        paths: Iterable of input file paths.
        cache_path: Path to the cache file.

    Returns:
        bool: True if cache can be reused, False if it should be rebuilt.
    """
    cp = Path(cache_path)
    if not cp.exists():
        return False
    try:
        cache_mtime = cp.stat().st_mtime
    except Exception:
        return False
    for p in paths:
        try:
            if os.path.getmtime(p) > cache_mtime:
                return False
        except FileNotFoundError:
            # if a file is missing, ignore it for cache validation
            continue
        except Exception:
            # if mtime fails for any file, fall back to rebuild
            return False
    return True