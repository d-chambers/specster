"""
A module for rendering specdata into text.
"""

from __future__ import annotations

from collections.abc import Sequence

import specster


class Displayer:
    """
    A class to produce outputs from the various parameter classes.

    This is essentially just a convenience class for accessing/writing
    string output of various parameters via getattr.
    """

    def __init__(self, model: specster.core.models.SpecsterModel):
        """Init new display for model.

        Parameters
        ----------
        model
            The model which will get parameters from.
        """
        self._model = model

    def __getattr__(self, item):
        # This liberal getattr messes with deep copy.
        if item == "__deepcopy__":
            return None
        key = item.lower()
        if key == "_model":
            return
        # handle case where key is another spec class, need to get new disp
        value = getattr(self._model, key)
        if hasattr(value, "disp"):
            return value.disp
        elif isinstance(value, Sequence) and not isinstance(value, str):
            return value
        out = self._model.write_model_data(key)
        return out


def format_bool(bool_like):
    """Format boolean value to specfem style."""
    return ".true." if bool_like else ".false."


def number_to_spec_str(value: int | float) -> str:
    """
    Write float to string.

    This uses specfem's d notation, eg (2700.d0) means (2700e0)
    """
    # handle "special cases" which are special enough to break the rules
    if value == 0:
        return "0"
    elif value == 9999:
        return "9999"
    e_str = f"{value:.04e}"
    exp = e_str.split("e")[1]
    num = float(e_str.split("e")[0])
    return f"{num}d{int(exp)}"


def dict_to_description(param_name, somedict):
    """Convert a dict of enums to a description string."""
    param_str = "".join([f"{i}: {v}\n" for i, v in somedict.items()])
    out = f"{param_name} supports several params including:\n" f"{param_str}"
    return out
