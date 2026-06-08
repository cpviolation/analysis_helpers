from ._root_import import import_root_with_proxy
r, _HAS_ROOT = import_root_with_proxy()
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import Iterable, Iterator
from tqdm import tqdm
import uproot
import pandas as pd
import os
from pathlib import Path
from .root_helpers import require_root
from .utils import remove_newlines
import yaml


def load_data(filename):
    """Load data from a file

    Args:
        filename (str): the name of the file to load

    Returns:
        list: the data from the file
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'File not found: {filename}')
    with open(filename, 'r') as f:
        data = f.read().splitlines()
    return data


def _load_single_file_df(path, branches, tree_name, chunk_size) -> tuple[str, pd.DataFrame] | None:
    """Read one ROOT file into a dataframe.

    Returns:
        tuple[str, pandas.DataFrame] | None: ``(path, dataframe)`` when data
        is read successfully, otherwise ``None``.
    """
    try:
        with uproot.open(path) as f:
            tree = f[tree_name]
            parts = []
            for part in tree.iterate(branches, library='pd', step_size=chunk_size):
                parts.append(part)
            if parts:
                return path, pd.concat(parts, ignore_index=True)
    except Exception:
        # skip unreadable files
        return None
    return None


def iter_file_dfs(
    paths: Iterable[str],
    branches,
    tree_name,
    chunk_size=100_000,
    progress=True,
    n_workers=1,
) -> Iterator[tuple[str, pd.DataFrame]]:
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
        n_workers: Number of worker threads used to load files. Use ``1`` for
            sequential loading.

    Yields:
        tuple[str, pandas.DataFrame]: Pair ``(path, dataframe)`` for each
        readable file containing at least one chunk.
    """
    if n_workers < 1:
        raise ValueError('n_workers must be >= 1')

    if n_workers == 1:
        iterator = tqdm(paths, desc='Loading files') if progress else paths
        for p in iterator:
            result = _load_single_file_df(p, branches, tree_name, chunk_size)
            if result is not None:
                yield result
        return

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        mapped = executor.map(
            _load_single_file_df,
            paths,
            repeat(branches),
            repeat(tree_name),
            repeat(chunk_size),
        )
        iterator = tqdm(mapped, desc='Loading files') if progress else mapped
        for result in iterator:
            if result is not None:
                yield result


def get_concat_df(name, data_files,
               branches=None, filter=None, 
               nfiles=None, nstart=0, library=None):
    """Get a concatenation of tuples from data files

    Args:
        name (str): the tuples name
        data_files (list): list of data files.
        branches (list, optional): the list of branches to load. Defaults to None.
        filter (str, optional): the filter to apply. Defaults to None.
        nfiles (int, optional): the number of files to load. Defaults to None.
        nstart (int, optional): the index of the file to start the list with. Defaults to 0.
        library (str, optional): the library to convert the data to. Defaults to None.

    Raises:
        ValueError: Invalid data type for source
        ValueError: Invalid list for data type in source
        ValueError: Invalid name for data type in source

    Returns:
        array: An array with all or the specified variables
    """
    if type(data_files) is not list:
        raise ValueError('Invalid data (not a list)')
    if library not in [None, 'pd', 'np']:
        raise ValueError(f'Invalid library: {library} (must be in [None, "pd", "np"])')
    data_files = reduce_data_files(data_files, nfiles, nstart)
    df = uproot.concatenate([f"{fname}:{name}" for fname in data_files],
                                   step_size = '200 MB',
                                   expressions=branches, filter_name=filter)
    if library == 'pd':
        # Convert to pandas DataFrame
        pd_df = pd.DataFrame(ak.to_list(df))
        return pd_df
    if library == 'np':
        # Convert to numpy array
        np_df = df.to_numpy()
        return np_df
    return df
    

def load_df_incremental(
    paths,
    branches,
    tree_name,
    chunk_size=100_000,
    progress=True,
    n_workers=1,
):
    """Load and concatenate data from multiple ROOT files.

    This function consumes :func:`iter_file_dfs` and performs one final
    concatenation across files.

    Args:
        paths: Iterable of ROOT file paths.
        branches: Branch names to read from each tree.
        tree_name: Name of the TTree inside each file.
        chunk_size: Number of entries per chunk while iterating.
        progress: If True, show a file-level progress bar.
        n_workers: Number of worker threads used to load files. Use ``1`` for
            sequential loading.

    Returns:
        pandas.DataFrame: Concatenated dataframe for all readable files.
        If no data is read, returns an empty dataframe with ``branches`` as
        columns.
    """
    # Efficient final concatenation, while still processing one file at a time
    all_parts = []
    for _, file_df in iter_file_dfs(
        paths,
        branches,
        tree_name,
        chunk_size=chunk_size,
        progress=progress,
        n_workers=n_workers,
    ):
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


def reduce_data_files(data_files, nfiles=None, nstart=None):
    """Given a lis of data files, reduce the list to a subset

    Args:
        data_files (list): the list of data files.
        nfiles (int, optional): the number of files to keep. Defaults to None.
        nstart (int, optional): the number of files to skip. Defaults to None.

    Returns:
        list: the subset of data files
    """
    if type(data_files) is not list:
        raise ValueError('Invalid data (not a list)')
    if nstart is not None and type(nstart) is not int:
        raise ValueError(f'Invalid nstart: {nstart} (not an integer)')
    if nfiles is not None and type(nfiles) is not int:
        raise ValueError(f'Invalid nfiles: {nfiles} (not an integer)')
    if nstart is None:
        nstart = 0
    if nfiles is None:
        nfiles = len(data_files)-nstart
    if nfiles < 0 or nstart < 0:
        raise ValueError(f'Invalid parameters: {nstart} and {nfiles} (must be positive)')
    if nstart > len(data_files):
        raise ValueError(f'Invalid parameters: {nstart} (greater than the number of files)')
    if nfiles > len(data_files)-nstart:
        print(f'Invalid parameters: {nfiles} (greater than the number of files). Reducing to {len(data_files)-nstart}')
        nfiles = len(data_files)-nstart
    return data_files[nstart:nstart+nfiles]


def get_rdf(name, data,
            nfiles=None, nstart=0, daskclient=None, verbose=False):
    """Gets a `ROOT.TChain` from data files

    Args:
        name (str): the tuple name.
        data (list): list of data files.
        nfiles (int, optional): the number of files to load. Defaults to None.
        nstart (int, optional): the index of the file to start the list with. Defaults to 0.
        daskclient (dask.distributed.Client, optional): the dask client to use. Defaults to None.

    Raises:
        ValueError: Invalid data type for source
        ValueError: Invalid list for data type in source

    Returns:
        ROOT.RDataFrame: A `ROOT.RDataFrame` with all the data
    """
    if not type(data) is list:
        raise ValueError('Invalid data (not a list)')
    require_root()
    data_files = reduce_data_files(data, nfiles, nstart)
    if verbose: 
        print(data_files)
    rdf = r.RDataFrame(name, data_files) if daskclient is None  else \
          r.RDF.Experimental.Distributed.Dask.RDataFrame(name, data_files, daskclient=daskclient)
    return rdf


def load_selection_options(filename):
    """Load selection options from a YAML file

    Args:
        filename (str): the name of the file to load

    Returns:
        dict: the selection options
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'File not found: {filename}')
    with open(filename, 'r') as f:
        options = remove_newlines(yaml.safe_load(f))
    return options