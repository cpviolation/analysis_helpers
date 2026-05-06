from pathlib import Path
import subprocess
import sys

import pytest

from analysis_helpers.utils import get_matching_files, get_temporary_file_name, run_command


def test_run_command_returns_completed_process():
    result = run_command(f'"{sys.executable}" -c "print(123)"')

    assert result.stdout.strip() == "123"
    assert result.returncode == 0


def test_run_command_raises_on_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(f'"{sys.executable}" -c "import sys; sys.exit(3)"')


def test_get_temporary_file_name_returns_existing_path():
    temp_name = get_temporary_file_name()
    temp_path = Path(temp_name)

    assert temp_path.exists()

    temp_path.unlink()


def test_get_matching_files_filters_by_pattern(tmp_path):
    (tmp_path / "alpha.txt").write_text("a")
    (tmp_path / "beta.py").write_text("b")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "gamma.txt").write_text("c")

    result = get_matching_files(str(tmp_path), "*.txt")

    assert sorted(Path(path).name for path in result) == ["alpha.txt", "gamma.txt"]
