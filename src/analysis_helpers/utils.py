import fnmatch
import os
import subprocess
import tempfile

def run_command(command, input_text=None):
    """execute a shell command and print the output

    Args:
        command (str): the shell command to execute
        input_text (str, optional): the input text to the command. Defaults to None.
    """    
    try:
        # Execute the command, providing input if necessary
        result = subprocess.run(
            command, 
            input=input_text, 
            text=True, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(e.stderr)
        exit(1)


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