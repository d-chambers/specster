"""
Module for models.
"""
from functools import cached_property
from typing import Optional, Annotated

from pydantic import ConfigDict, BaseModel, TypeAdapter, PlainValidator


class SpecsterModel(BaseModel):
    """Abstract model in case we need to modify base behavior."""
    model_config = ConfigDict(extra="ignore", validate_assignment=True, ignored_types=(cached_property,))

    @classmethod
    def read_line(cls, params):
        """Read params from a sequence into pydantic model."""
        if isinstance(params, str):
            params = params.split("#")[0].split()
        field_names = list(cls.model_fields)
        assert len(params) == len(field_names), "names should match args"
        input_dict = {i: v for i, v in zip(field_names, params)}
        return cls(**input_dict)

    @cached_property
    def disp(self):
        """Return a displayer for nicely rendering contents."""
        from specster.core.render import Displayer

        return Displayer(self)

    def get_formatted_str(self, key):
        """Format key into string format."""
        value = getattr(self, key)
        # handle special types that need formatting
        field = self.model_fields.get(key, None)
        formatter_dict = self._parser_dict
        if field and field.annotation in formatter_dict:
            value = formatter_dict[field.annotation](value)
        return str(value)

    def write_model_data(self, key: Optional[str] = None):
        """Write the data contained in key to a string."""
        if key is None:
            msg = f"{self.__class__.__name__} requires a specified field"
            raise ValueError(msg)
        value = self.get_formatted_str(key)
        return str(value)

    @cached_property
    def _parser_dict(self):
        """Return the dict used for parsing."""
        from specster.core.render import format_bool, number_to_spec_str

        out = {bool: format_bool, SpecFloat: number_to_spec_str}
        return out


class AbstractParameterModel(SpecsterModel):
    """Abstract class for defining specfem parameter models."""

    @classmethod
    def init_from_dict(cls, data_dict):
        """Init class, and subclasses, from a dict of values"""
        my_fields = set(cls.model_fields)
        nested_models = {
            k: v.annotation
            for k, v in cls.model_fields.items()
            if hasattr(v.annotation, "init_from_dict")
            # if the key is already the right type we skip it
            and not isinstance(data_dict.get(k), v.annotation)
        }
        # get inputs for this model
        needed_inputs = {k: v for k, v in data_dict.items() if k in my_fields}
        # add nested models
        for field_name, model in nested_models.items():
            needed_inputs[field_name] = model.init_from_dict(data_dict)
        return cls(**needed_inputs)


# class SimpleValidator:
#     """
#     A custom class for getting simple validation behavior in pydantic.
#
#     Subclass, then define function to be used as validator. func
#     """
#
#     @classmethod
#     def func(cls, value):
#         """A method to overwrite with custom validation."""
#         return value
#
#     @classmethod
#     # TODO[pydantic]: We couldn't refactor `__get_validators__`, please create the `__get_pydantic_core_schema__` manually.
#     # Check https://docs.pydantic.dev/latest/migration/#defining-custom-types for more information.
#     def __get_validators__(cls):
#         """Hook used by pydantic."""
#         yield cls.validate
#
#     @classmethod
#     def validate(cls, validator):
#         """Simply call func."""
#         return cls.func(validator)



def spec_str_to_float(value):
    """Remove silly d, cast to float"""
    if isinstance(value, str) and "d" in value:
        value = value.replace("d", "e")
    return float(value)


SpecFloat = Annotated[float, PlainValidator(spec_str_to_float)]
