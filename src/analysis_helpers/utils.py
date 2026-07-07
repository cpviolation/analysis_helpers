import fnmatch
import os
import subprocess
import tempfile
import awkward as ak
import numpy as np


def run_command(command, input_text=None):
    """Execute a shell command and return the completed subprocess result.

    Args:
        command (str): the shell command to execute
        input_text (str, optional): the input text to the command. Defaults to None.

    Returns:
        subprocess.CompletedProcess: completed subprocess result with captured output.

    Raises:
        subprocess.CalledProcessError: if the command exits with a non-zero status.
    """
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def get_temporary_file_name():
    """Creates a temporary file name

    Returns:
        str: the temporary file name
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_name = temp_file.name
    return temp_file_name


def get_matching_files(directory, pattern):
    """Get all files in the directory matching the given pattern.

    Args:
        directory (str): The directory to search.
        pattern (str): The pattern to match files.
    Returns:
        list: List of matching file paths.
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            file_path = os.path.join(root, filename)
            matching_files.append(file_path)
    return matching_files


def remove_newlines(data):
    """Remove newlines from a dictionary, list, or string

    Args:
        data (dict, list, str): a dictionary, list, or string to remove newlines from

    Returns:
        dict: data with newlines removed
    """    
    if isinstance(data, dict):
        # Recursively go through dictionary values
        return {key: remove_newlines(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Recursively go through each element of the list
        return [remove_newlines(item) for item in data]
    elif isinstance(data, str):
        # Strip newlines from strings
        return data.replace('\n', '')
    else:
        # Return the data unchanged if it's not a dict, list, or string
        return data
    

def to_numpy(a):
    """Transforms an array to numpy.array if it is an awkward array or list.

    Args:
        a (array): The input array.

    Returns:
        np.ndarray: The transformed numpy array.
    """
    if type(a) == ak.highlevel.Array:
        return a.to_numpy()
    return np.asarray(a)


def get_mask(data, range=None):
    """Obtain a selection filter mask

    Args:
        data (np.ndarray): The input data array.
        range (tuple): The range (min, max) to filter the data.

    Raises:
        ValueError: If the input data is not 1D.
        ValueError: If the range is not a tuple of (min, max).

    Returns:
        np.ndarray: A boolean mask array.
    """    
    if data.ndim != 1:
        raise ValueError("Input data must be 1D.")
    if range is None: return np.full_like(data, True, dtype=bool)
    if len(range) != 2:
        raise ValueError("Range 'range' must be a tuple of (min, max)")
    return (data >= range[0]) & (data <= range[1])