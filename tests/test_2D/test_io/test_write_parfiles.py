"""
Tests for writing parfile to a data directory.
"""

import pytest
from specster import Control2d


@pytest.fixture(scope="class")
def written_control(control_2d, tmp_path_factory) -> Control2d:
    """Write control 2D to disk."""
    path = tmp_path_factory.mktemp("control_test")
    out = control_2d.write(path, overwrite=True)
    return out


class TestWriteProject:
    """Tests for writing the project to disk."""

    def test_directories_exist(self, written_control):
        """Ensure the required directories were created."""
        paths = written_control.get_input_paths()
        for path in paths.values():
            assert path.exists()
