"""
Constant values for specster.
"""
# number of zeroes to use after proc in binary file names
PROC_ZEROES = 6
IGNORE_BINS = {"NSPECibool"}

XYZ = ("x", "z", "y")

# parameter value map
_SUB_VALUES = {".false.": False, ".true.": True}

# nested dict mapping int values to their meanings in par file
_ENUM_MAP = {
    "simulation_type": {1: "forward", 2: "adjoint", 3: "both"},
    "noise_tomography": {
        0: "regular wave propagation",
        1: "noise simulation",
        2: "noise simulation",
        3: "noise simulation",
    },
    "time_stepping_scheme": {
        1: "Newmark (2nd order)",
        2: "LDDRK4-6",
        3: "classical RK4 4th-order",
        4: "stage Runge-Kutta",
    },
    "partitioning_type": {
        3: "Scotch",
        1: "ascending order",
    },
    "noise_source_time_function_type": {
        0: "external",
        1: "Ricker (second derivative)",
        2: "Ricker (first derivative)",
        3: "Gaussian",
        4: "Figure 2a of Tromp et al. 2010",
    },
    "setup_with_binary_database": {
        0: "does not read/create database",
        1: "creates database",
        2: "reads database",
    },
    "seismotype": {
        1: "displacement",
        2: "velocity",
        3: "acceleration",
        4: "pressure",
        5: "curl of displacement",
        6: "fluid potential",
    },
    "imagetype_jpeg": {
        "1": "displacement_ux",
        "2": "displacement_uz",
        "3": "displacement_norm",
        "4": "velocity_vx",
        "5": "velocity_vz",
        "6": "velocity_norm",
        "7": "acceleration_ax",
        "8": "acceleration_az",
        "9": "acceleration_norm",
        "10": "pressure",
    },
    "imagetype_postscript": {
        "1": "displacement vector",
        "2": "velocity vectory",
        "3": "acceleration vector",
    },
    "imagetype_wavefield_dumps": {
        "1": "displacement vector",
        "2": "velocity vector",
        "3": "acceleration vector",
        "4": "pressure",
    },
    "source_type": {
        "1": "elastic force or acoustic pressure, or P wave for initial field",
        "2": "moment tensor or S wave if initial field",
        "3": "Rayleigh Wave, must set initialfield",
        "4": "Plane P wave, no converted/reflected waves at surface",
        "5": "Plane S wave, no converted/reflected waves at surface",
        "6": "mode (2,3) of a rectangular membrane",
    },
    "time_function_type": {
        "1": "second derivative of Gaussian",
        "2": "First derivative of a Gaussian",
        "3": "Gaussian",
        "4": "Dirac (probably produce noise data due to freqs above mesh res)",
        "5": "Heaviside (probably produce noise data due to freqs above mesh res)",
        "6": "Ocean acoustics type I",
        "7": "Ocean acoustics type II",
        "8": "External source time function",
        "9": "Burst",
        "10": "Sinus source time function",
        "11": "Marmousi Ormsby wavelet",
    },
}

# extended meaning for certain parameters
_MEANING_MAP = {
    "model": {
        "default": "defines model using nbmodels",
        "ascii": "read model from ascii database model",
        "binary": "read model from the binary database file",
        "binary_voigt": "read Voigt model from binary database",
        "external": "read model using define_external_module subroutine",
        "gll": "read GLL model from binary database file",
        "legacy": "read model from model_velocity.dat_input",
    },
    "save_model": {
        False: "dont save the model",
        "default": "dont save the model",
        "ascii": "Save model as ascii",
        "binary": "save model as binary",
        "gll": "save model as gll",
        "legacy": "save model as legacy format",
    },
}

# type of modules specfem supports
_MODEL_TYPES = ("default", "ascii", "binary", "binary_boigt", "external", "gll")


# Special directory names
special_dirs = frozenset(["DATA", "OUTPUT_FILES", "SEM"])
