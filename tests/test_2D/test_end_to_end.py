"""
End to end tests.
"""

import matplotlib.pyplot as plt
import pytest
import specster as sp


@pytest.fixture(scope="class")
def acoustic_control(tmp_path_factory):
    """An acoustic model."""
    path = tmp_path_factory.mktemp("acoustic_control")
    cont = sp.load_2d_example("acoustic_reflector").copy(path)
    cont.prepare_fwi_forward().run()
    return cont


@pytest.mark.slow
class TestAcoustic2D:
    """Test case for acoustic 2D."""

    def test_acoustic(self, acoustic_control):
        """Simply run the acoustic example"""
        out = acoustic_control.output
        assert isinstance(out, sp.OutPut2D)


class TestWriteVelocity:
    """Ensure a velocity value can be written."""

    @pytest.fixture(scope="class")
    def original_control_st(self, control_inclusion_2d):
        """Run the original simulation, return streams."""
        control = control_inclusion_2d
        control.par.nstep = 1000
        control.xmeshfem2d()
        control.xspecfem2d()
        return control.output.get_waveforms()

    @pytest.fixture(scope="class")
    def control_new_p(self, control_inclusion_2d):
        """Create a control file with a new p velocity."""
        control = control_inclusion_2d
        control.par.nstep = 1000
        df = control.get_material_model_df()
        # get coords of each gll point and location of center
        df["vp"] = df["vp"] * 0.8
        df["vs"] = df["vs"] * 0.8
        # now set model and plot
        control.set_material_model_df(df)
        control.xmeshfem2d()
        control.xspecfem2d()
        return control

    @pytest.fixture(scope="class")
    def control_new_p_st(self, control_new_p):
        """Get the stream from the new control."""
        return control_new_p.output.get_waveforms()

    def test_can_plot(self, control_new_p):
        """Ensure the new control can be plotted."""
        fig, *_ = control_new_p.plot_geometry()
        assert isinstance(fig, plt.Figure)

    def test_waveforms_different(self, control_new_p_st, original_control_st):
        """Ensure modifying the velocity changes the streams."""
        st1 = control_new_p_st
        st2 = original_control_st
        assert st1 != st2
