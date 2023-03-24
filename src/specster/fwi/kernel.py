"""
Class for handling kernels.
"""


class KernelKeeper:
    """
    A class to apply processing to kernels.
    """

    def __init__(self, kernel_dict, stations=None, events=None):
        self._kernel_dict = kernel_dict
        self.stations = stations
        self.events = events
