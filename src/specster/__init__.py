"""
Specster is a python harness to tame the specfem beast.
"""
from ._settings import Settings
from .d2.control2d import Control2d, load_2d_example
from .d2.output2d import OutPut2D
from .version import __version__

settings = Settings()
