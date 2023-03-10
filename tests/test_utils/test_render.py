"""
Tests for rendering logic.
"""
import re

import pytest

from specster.utils.misc import load_templates_text_from_directory


@pytest.fixture(scope="class")
def parfile_characters(control_2d_default):
    """Return a list of parameters requested in the par file."""
    texts = load_templates_text_from_directory(control_2d_default._template_path)
    full = texts["par_file"]
    regex = "\{\{(.*?)\}"
    matches = list(re.finditer(regex, full, re.MULTILINE | re.DOTALL))
    out = []
    for match in matches:
        for group in match.groups():
            txt = group.strip()
            out.append(txt)
    return out


@pytest.fixture(scope="class")
def base_disp(control_2d_default):
    """Return the basic display thing."""
    return control_2d_default.par.disp


class TestDisp:
    """Tests for disp object."""

    def test_base_attribute(self, base_disp):
        """Ensure all base attributes work."""
        p_sv = base_disp.p_sv
        assert "p_sv" in p_sv
        assert ".true." in p_sv

    def test_2d_parfile(self, base_disp, parfile_characters):
        """Iterate all requested params and resolve."""
        for text in parfile_characters:
            if not text.startswith("dis"):
                continue
            val = base_disp
            for attr in text.split(".")[1:]:
                val = getattr(val, attr)
