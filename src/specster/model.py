"""
Module for models.
"""
from specster.utils import SpecsterModel


class AbstractParameterModel(SpecsterModel):
    """Abstract class for defining specfem parameter models."""

    @classmethod
    def init_from_dict(cls, data_dict):
        """Init class, and subclasses, from a dict of values"""
        my_fields = set(cls.__fields__)
        nested_models = {
            k: v.type_
            for k, v in cls.__fields__.items()
            if hasattr(v.type_, "init_from_dict")
            # if the key is already the right type we skip it
            and not isinstance(data_dict.get(k), v.type_)
        }
        # get inputs for this model
        needed_inputs = {k: v for k, v in data_dict.items() if k in my_fields}
        # add nested models
        for field_name, model in nested_models.items():
            needed_inputs[field_name] = model.init_from_dict(data_dict)
        return cls(**needed_inputs)
