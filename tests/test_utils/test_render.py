"""
Tests for rendering logic.
"""
import re

import pytest

from specster.utils.misc import load_templates_text_from_directory


def extract_requested_attrs(text):
    """Extract the requested attributes."""
    regex = "\{\{(.*?)\}"
    matches = list(re.finditer(regex, text, re.MULTILINE | re.DOTALL))
    out = []
    for match in matches:
        for group in match.groups():
            txt = group.strip()
            # skipping jinga stuff
            if "loop." in txt:
                continue
            out.append(txt)
    return out


@pytest.fixture(scope="class")
def parfile_attrs(control_2d_default):
    """Return a list of parameters requested in the par file."""
    texts = load_templates_text_from_directory(control_2d_default._template_path)
    return extract_requested_attrs(texts["par_file"])


@pytest.fixture(scope="class")
def source_attrs(control_2d_default):
    """Return a list of parameters requested in the source file."""
    texts = load_templates_text_from_directory(control_2d_default._template_path)
    return extract_requested_attrs(texts["source"])


@pytest.fixture(scope="class")
def base_disp(control_2d_default):
    """Return the basic display thing."""
    return control_2d_default.par.disp


class TestDisp:
    """Tests for disp object."""

    def test_base_attribute(self, base_disp, control_2d_default):
        """Ensure all base attributes work."""
        title = base_disp.title
        assert title == control_2d_default.par.title
        p_sv = base_disp.p_sv
        assert ".true." in p_sv

    def test_2d_parfile(self, base_disp, parfile_attrs):
        """Iterate all requested params and resolve."""
        for text in parfile_attrs:
            if not text.startswith("dis"):
                continue
            val = base_disp
            for attr in text.split(".")[1:]:
                val = getattr(val, attr)

    def test_2d_source_file(self, base_disp, source_attrs):
        """Ensure all the source parameters can be requested."""
        for source in base_disp.sources.sources:
            for text in source_attrs:
                val = source
                for attr in text.split(".")[1:]:
                    val = getattr(val, attr)
