"""
Tests for reading and writing material models to DATA directory.
"""
import pandas as pd
import pytest

import specster


@pytest.fixture(scope="class")
def control_with_datafiles(control_2d_default, tmp_path_factory) -> specster.Control2d:
    """Get the control with data files created."""
    new_path = tmp_path_factory.mktemp("temp_base")
    new = control_2d_default.copy(new_path)
    # set generate material models to true
    new.par.mesh.setup_with_binary_database = "0"
    new.par.mesh.save_model = "binary"
    new.write(overwrite=True)
    new.run()
    return new


@pytest.fixture(scope="class")
def control_with_datafiles(control_2d_default, tmp_path_factory) -> specster.Control2d:
    """Get the control with data files created."""
    new_path = tmp_path_factory.mktemp("temp_base")
    new = control_2d_default.copy(new_path)
    # set generate material models to true
    new.par.mesh.setup_with_binary_database = "0"
    new.par.mesh.save_model = "binary"
    new.write(overwrite=True)
    new.run()
    return new


@pytest.fixture(scope="class")
def materials_df(control_with_datafiles):
    """return material dataframe created."""
    return control_with_datafiles.get_material_model_df()


class TestGetMaterial:
    """Tests for getting material from data"""

    def test_read_materials(self, materials_df):
        """Ensure materials can be read"""
        assert isinstance(materials_df, pd.DataFrame)
