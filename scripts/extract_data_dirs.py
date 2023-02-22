"""
Extract the data directories from the specfem folder into the test
data directory.
"""

import shutil
from pathlib import Path

if __name__ == "__main__":
    # find example par files
    spec_path = Path("/media/data/Gits/specfem2d")
    examples_path = spec_path / "EXAMPLES"
    assert examples_path.exists()
    # create test data directory
    test_data_path = Path(__file__).parent.parent / "tests" / "test_data"
    test_data_path.mkdir(parents=True, exist_ok=True)

    for example_path in examples_path.glob("*"):
        if not example_path.is_dir():
            continue
        expected = test_data_path / example_path.name
        shutil.copytree(example_path, expected)
