"""
Module for reading/writing parfiles.
"""
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import Field, FilePath

from specster.constants import _ENUM_MAP, _MEANING_MAP, _SUB_VALUES
from specster.exceptions import UnhandledParFileLine
from specster.utils import SpecFloat, SpecsterModel, dict_to_description

# --- Material Models


def extract_parline_key_value(line):
    """Extract key/value pairs from a single line of the par file."""
    key_value = line.split("=")
    key = key_value[0].strip().lower()
    value = key_value[1].split("#")[0].strip()
    return key, _SUB_VALUES.get(value, value)


def parse_params_into_model(model: SpecsterModel, params):
    """Read params from a sequence into pydantic model."""
    field_names = list(model.__fields__)
    assert len(params) == len(field_names), "names should match args"
    input_dict = {i: v for i, v in zip(field_names, params)}
    return model(**input_dict)


class AbstractModelType(SpecsterModel):
    """Abstract type for models."""

    _model_type = None
    _type_cls_map = {}  # {type: cls}

    def __init_subclass__(cls, **kwargs):
        cls._type_cls_map[cls._model_type] = cls

    @classmethod
    def read_line(cls, line):
        """Read lines to create class instance."""
        params = line.split(" ")
        assert int(params[1]) == int(cls._model_type), "Wrong model type!"
        # need to strip out model type, if we got here its already handled.
        params_sub = [params[0]] + params[2:]
        return parse_params_into_model(cls, params_sub)


class ElasticModel(AbstractModelType):
    """
    Covers acoustic and elastic material models.

    Here are some examples from the par file how these look:
    acoustic example:
        model_number 1 rho Vp 0  0 0 QKappa 9999 0 0 0 0 0 0
            (for QKappa use 9999 to ignore it)
    elastic example:
        model_number 1 rho Vp Vs 0 0 QKappa Qmu  0 0 0 0 0 0
            (for QKappa and Qmu use 9999 to ignore them)
    """

    _model_type = 1  # fixed type
    model_number: int
    rho: SpecFloat = Field(description="density")
    Vp: SpecFloat = Field(description="P velocity")
    Vs: SpecFloat = Field(description="S velocity")
    void1_: Literal["0"] = 0
    void2_: Literal["0"] = 0
    QKappa: SpecFloat = Field(9999, description="P quality factor")
    Qmu: SpecFloat = Field(9999, description="S quality factor")
    void3_: Literal["0"] = 0
    void4_: Literal["0"] = 0
    void5_: Literal["0"] = 0
    void6_: Literal["0"] = 0
    void7_: Literal["0"] = 0
    void8_: Literal["0"] = 0


class AnisotropicModel(AbstractModelType):
    """
    Covers anisotropic case

    Here are some examples from the par file how these look:

    anisotropic:
        model_number 2 rho c11 c13 c15 c33 c35 c55 c12 c23 c25   0 QKappa Qmu
    anisotropic in AXISYM:
        model_number 2 rho c11 c13 c15 c33 c35 c55 c12 c23 c25 c22 QKappa Qmu
    """

    _model_type = 2  # fixed type
    model_number: int
    rho: float = Field(description="density")
    c11: float
    c13: float
    c15: float
    c33: float
    c35: float
    c55: float
    c12: float
    c23: float
    c25: float
    c22: float
    QKappa: float = Field(9999, description="P quality factor")
    Qmu: float = Field(9999, description="S quality factor")


class PoroelasticModel(AbstractModelType):
    """Model describing poroelastic material"""

    _model_type = 2  # fixed type
    model_number: int
    rhos: float
    rhof: float
    phi: float
    c: float
    kxx: float
    kxz: float
    kzz: float
    Ks: float
    Kf: float
    Kfr: float
    etaf: float
    mufr: float
    Qmu: float


class MaterialModels(SpecsterModel):
    """
    Class to hold information about material properties.
    """

    nbmodels: Optional[int]
    models: List[Union[ElasticModel, AnisotropicModel, PoroelasticModel]]
    tomography_file: Optional[FilePath] = Field(
        None, description="External tomography file"
    )

    @classmethod
    def read_material_properties(cls, nbmodels, iterator):
        """Read material properties"""
        model_type_key = AbstractModelType._type_cls_map
        out = {
            "nbmodels": int(nbmodels),
            "models": [],
        }
        models = []
        for _ in range(int(nbmodels)):
            line = next(iterator)
            model_type = model_type_key[int(line.split()[1])]
            models.append(model_type.read_line(line))
        # now read tomography (TODO: Is this the right place?)
        expected = {"tomography_file"}
        for _ in range(len(expected)):
            key, value = extract_parline_key_value(next(iterator))
            assert key in expected
            out[key] = value
        return cls(**out)


# --- Region handling


class Region2D(SpecsterModel):
    """Regions in the model."""

    nxmin: int = Field(ge=1, description="starting x element")
    nxmax: int = Field(ge=1, description="ending x element")
    nzmin: int = Field(ge=1, description="starting z element")
    nzmax: int = Field(ge=1, description="ending z element")
    material_number: int = Field(ge=1, description="material applied to region")

    @classmethod
    def read_line(cls, line):
        """Create region2D from an input line"""
        return parse_params_into_model(cls, line.split())


class Regions(SpecsterModel):
    """Tracks regions in the model."""

    nbregions: int
    regions: List[Region2D]

    @classmethod
    def read_regions(cls, nbregions, iterator):
        """Read material properties"""
        regions = []
        for _ in range(int(nbregions)):
            line = next(iterator)
            regions.append(Region2D.read_line(line))
        return cls(nbregions=nbregions, regions=regions)


class Mesh(SpecsterModel):
    """Controls the meshing parameters (under MESH section)."""

    # type of partitioning
    partitioning_type: int = Field(
        description=dict_to_description(
            "partitioning_type", _ENUM_MAP["partitioning_type"]
        )
    )
    ngnod: Literal[4, 9] = Field(
        9,
        description="number of control nodes per element (4 or 9)",
    )
    setup_with_binary_database: Literal[0, 1, 2] = Field(
        0,
        description=dict_to_description(
            "setup_with_binary_database", _ENUM_MAP["setup_with_binary_database"]
        ),
    )
    model: Literal[tuple(_MEANING_MAP["model"])] = Field(
        "default", description=dict_to_description("model", _MEANING_MAP["model"])
    )
    model: Literal[tuple(_MEANING_MAP["save_model"])] = Field(
        "default", description=dict_to_description("model", _MEANING_MAP["model"])
    )


class Attenuation(SpecsterModel):
    """Controls the attenuation parameters (under Attenuation section)."""

    attenuation_viscoelastic: bool = Field(
        False,
        description="attenuation for non-poroelastic solid parts",
    )
    attenuation_viscoacoustic: bool = Field(
        False,
        description="attenuation for non-poroelastic fluid parts",
    )
    n_sls: int = Field(
        3,
        ge=3,
        description="number of standard linear solids for attenuation",
    )
    attenuation_f0_reference: float = Field(
        5.196, description="reference attenuation freq, see docs"
    )
    read_velocities_at_f0: bool = Field(
        False,
        description="see user manual",
    )
    use_solvopt: bool = Field(
        False,
        description="More precise/expansive way to get relaxation times",
    )
    attenuation_poro_fluid_part = Field(
        False, description="attenuation for the fluid part of poroelastic parts"
    )
    q0_poroelastic: float = Field(
        1, description="Quality factor for viscous attenuation"
    )
    freq0_poroelastic: float = Field(
        10,
        description="frequency for viscous attenuation",
    )
    undo_attenuation_and_or_pml: bool = Field(
        False, description="Undo attenuation for sensitivity kernel calc."
    )
    nt_dump_attenuation: float = Field(
        500, description="how often to dump restart files in sensitivity calc."
    )
    no_backward_reconstruction: bool = Field(
        False, description="Reads forward image from disk..."
    )


class Sources(SpecsterModel):
    """Controls the source parameters (under Source section)."""

    nsources: int = Field(1, description="number of sources")
    force_normal_to_surface: bool = Field(
        False,
        description="angleforce normal to surface",
    )
    intialfield: bool = Field(
        False, description="use an existing initial wave field as source"
    )
    add_bielak_conditions_bottom: bool = Field(
        False, description=" add Bielak conditions or not if initial plane wave"
    )
    add_bielak_conditions_right: bool = Field(
        False, description=" add Bielak conditions or not if initial plane wave"
    )
    add_bielak_conditions_top: bool = Field(
        False, description=" add Bielak conditions or not if initial plane wave"
    )
    add_bielak_conditions_left: bool = Field(
        False, description=" add Bielak conditions or not if initial plane wave"
    )
    acoustic_forcing: bool = Field(
        False, description="forcing of an acoustic medium with a rigid interface"
    )
    noise_source_time_function_type: Literal[1, 2, 3, 4] = Field(
        4,
        description=dict_to_description(
            "noise_source_time_function_type",
            _ENUM_MAP["noise_source_time_function_type"],
        ),
    )
    write_moving_sources_database: bool = Field(False, description="See manual")


class ReceiverSet(SpecsterModel):
    """A single receiver set."""

    nrec: int = Field(11, description="number of receivers")
    xdeb: float = Field(
        300.0,
        description="first receiver x in meters",
    )
    zdeb: float = Field(2200.0, description="first receiver z in meters")
    xfin: float = Field(
        3700.0,
        description="last receiver x in meters",
    )
    zfin: float = Field(
        2200.0,
        description="last receiver z in meters",
    )
    record_at_surface_same_vertical: bool = Field(
        True,
        description="fix receivers at the surface",
    )


class ReceiverSets(SpecsterModel):
    """Class containing multiple receiver sets."""

    nreceiversets: int = Field(2, description="Number of receiver sets")
    anglerec: SpecFloat = Field(
        0.0, description="angle to rotate components at receivers"
    )
    rec_normal_to_surface: bool = Field(
        False,
        description="base anglerec normal to surface",
    )
    receiver_sets: List[ReceiverSet]

    @classmethod
    def read_receiver_sets(cls, nreceiversets, iterator):
        """Read the receiver sets from the iterator."""
        receiver_set_count = int(nreceiversets)
        out = dict(
            rec_count=receiver_set_count,
            anglerec=extract_parline_key_value(next(iterator))[1],
            rec_normal_to_surface=extract_parline_key_value(next(iterator))[1],
            receiver_sets=[],
        )
        for _ in range(receiver_set_count):
            rec_dict = {}
            for _ in range(6):  # each receiver set has six lines
                key, value = extract_parline_key_value(next(iterator))
                rec_dict[key] = value
            out["receiver_sets"].append(ReceiverSet(**rec_dict))
        return cls(**out)


class Receivers(SpecsterModel):
    """Controls the receiver parameters (under Receiver section)."""

    seismotype: Literal[1, 2, 3, 4, 5, 6] = Field(
        1, description=dict_to_description("seismotype", _ENUM_MAP["seismotype"])
    )
    nstep_between_output_seismos: int = Field(
        10_000, description="Number of steps before saving seismograms"
    )
    nstep_between_output_sample: int = Field(
        1, description="Downsampling factor for output seismograms"
    )
    use_trick_for_better_pressure: bool = Field(
        False, description="Some hack to make pressure more accurate"
    )
    user_t0: SpecFloat = Field(0.0, description="custom start time for seismograms")
    save_ascii_seismograms: bool = Field(
        True,
        description="save seismograms in ASCII",
    )
    save_binary_seismogram_single: bool = Field(
        True, description="save seismograms in single precision binary format"
    )
    save_binary_seismogram_double: bool = Field(
        False,
        description="save seismograms in double precision binary format",
    )
    su_format: bool = Field(
        False, description="save seismograms in Seismic Unix format"
    )
    use_existing_stations: bool = Field(
        False, description="use an existing STATION file found in ./DATA"
    )
    # parse receiver sets
    receiver_sets: ReceiverSets


class AdjointKernel(SpecsterModel):
    """Controls the adjoint kernel parameters (under adjoint kernel section)."""

    save_ascii_kernels: bool = Field(
        True, description="save sensitivity as ascii files. Save to binary otherwise"
    )
    ntsetp_between_compute_kernels: int = Field(
        1, description="Only compute the adjoint every n steps"
    )
    approximate_hess_kl: bool = Field(
        False, description="approximate hessian for preconditioning"
    )


class BoundaryConditions(SpecsterModel):
    """Controls the boundary condition parameters (under boundary section)."""

    pml_boundary_conditions: bool = Field(
        True, description="perfectly matched layer active"
    )
    nelem_pml_thickness: int = Field(2, description="number of pml elements on edges")
    rotate_pml_activate: bool = Field(False, description="Whether to rotate the PMLs")
    rotate_pml_angle: float = Field(30.0, description="Angle to rotate PML (if at all)")
    k_min_pml: SpecFloat = Field(1.0, description="advanced damping parameter")
    k_max_pml: SpecFloat = Field(1.0, description="advanced damping parameter")
    damping_change_factor_acoustic: SpecFloat = Field(
        0.5, description="advanced damping parameter"
    )
    damping_change_factor_elastic: SpecFloat = Field(
        1.0, description="advanced damping parameter"
    )
    pml_parameter_adjustment: bool = Field(
        False, description="adjust PML to better damp single source"
    )
    stacy_absorbing_conditions: bool = Field(
        False, description="activate stacey absorbing BCs"
    )
    add_periodic_conditions: bool = Field(
        False, description="Add periodic boundary conditions"
    )
    periodic_horiz_dist: SpecFloat = Field(
        4000.0, description="Horizontal distance of periodic damping"
    )


class ExternalMeshing:
    """Controls the external meshing parameters."""

    read_external_mesh: bool = Field(
        False, description="read an external mesh for velocity model"
    )
    mesh_file: Optional[FilePath] = Field(
        "./DATA/mesh_file", description="Path to mesh file"
    )
    node_coords_file: Optional[FilePath] = Field(
        "./DATA/nodes_coords_dile", description="Path to mesh coordinates"
    )
    material_file: Optional[FilePath] = Field(
        "./DATA/material_file", description="Path to material file"
    )
    free_surface_file: Optional[FilePath] = Field(
        "./DATA/free_surface_file", description="Path to free surface file"
    )
    axial_element_file: Optional[FilePath] = Field(
        "./DATA/axial_element_file", description="Path to mesh file"
    )
    absorbing_surface_file: Optional[FilePath] = Field(
        "./DATA/absorbing_surface_file", description="Path to mesh file"
    )
    acoustic_forcing_surface_file: Optional[FilePath] = Field(
        "./DATA/Surf_acforcing_Bottom_enforcing_mesh",
    )
    absorbing_cpml_file: Optional[FilePath] = Field(
        "./DATA/absorbing_cpml_file", description="File with CPML element numbers"
    )
    tangential_dtection_curve_file: Optional[FilePath] = Field(
        "./DATA/courbe_eros_nodes", description="Contains curve delineating model"
    )


class InternalMeshing(SpecsterModel):
    """Controls the internal meshing parameters."""

    interfacesfile: Optional[FilePath] = Field(
        "./DATA/interfaces.dat", description="File with interfaces"
    )
    xmin: SpecFloat = Field(0.0, description="abscissa of left side of the model")
    xmax: SpecFloat = Field(4000, description="abscissa of the right side of model")
    nx: int = Field(80, description="number of elements along X")
    absorbbottom: bool = Field(True, description="absorbing conditions on bottom ")
    absorbright: bool = Field(True, description="absorbing conditions on right ")
    absorbtop: bool = Field(False, description="absorbing conditions on top")
    absorbleft: bool = Field(True, description="absorbing conditions on left ")
    regions: Regions


class Display(SpecsterModel):
    """Controls the display parameters."""

    ntsetp_between_output_info: int = Field(
        100, description="Display info about the displacement this steps"
    )
    output_grid_gnuplot: bool = Field(False, description="generate gnuplot file")
    output_grid_ascii: bool = Field(False, description="output ascii grid")
    output_energy: bool = Field(False, description="plot total energy curves")
    ntstep_between_output_energy: int = Field(
        10, description="time steps for wich energy is computed"
    )
    compute_integrated_energy_field: bool = Field(
        False, description="Compute integrated energy"
    )


class Visualizations(SpecsterModel):
    """Controls Other visualization parameters."""

    ntsetp_between_output_images: int = Field(
        100, description="How often (timestep) output is dumped for visualization"
    )
    cutsnaps: float = Field(
        1.0, description="minimum amplitude kept in % for JPEG/PostScript"
    )
    output_color_image: bool = Field(
        True, description="If true, output jpeg color image"
    )


class RunParameters(SpecsterModel):
    """
    Parameters contained in the Par_file.
    """

    # fixed width for parameter names
    _param_name_padding = 32
    # fixed with for value field
    _value_padding = 16

    title: str = Field("", description="Title of simulation")
    simulation_type: int = Field(
        description=dict_to_description("simulation_type", _ENUM_MAP["simulation_type"])
    )
    noise_tomography: int = Field(
        description=dict_to_description(
            "noise_tomography", _ENUM_MAP["noise_tomography"]
        )
    )
    save_forward: bool = Field(False, description="forward modeling should be saved")
    nproc: int = Field(1, description="number of processors to use.")
    nstep: int = Field(1600, description="total number of time steps")
    dt: float = Field(1.1e-3, description="time increment")
    time_stepping_scheme: int = Field(
        description=dict_to_description(
            "time_stepping_scheme", _ENUM_MAP["time_stepping_scheme"]
        )
    )
    axisum: bool = Field(False, description="type of calc (p-sv or SH/membrane waves")
    # ---- Mesh section
    mesh: Mesh
    # --- Attenuation section
    attenuation: Attenuation
    # --- Sources section
    sources: Sources
    # --- Receivers section
    receivers: Receivers
    # --- Adjoint Kernel section
    adjoint_kernel: AdjointKernel
    # --- Boundary conditions section
    boundary_conditions: BoundaryConditions
    # --- Velocity and Density Models
    material_models: MaterialModels
    # --- External Mesh properties
    external_meshing: ExternalMeshing
    # --- Internal Meshing Parameters
    internal_meshing: InternalMeshing
    # --- Display parameters
    display: Display

    #
    number_of_simultaneous_runs: int = Field(
        1, description="complicated parameter, see specfem docs for details"
    )
    gpu_mode: bool = Field(True, description="If GPUs should be used")


_MULTILINE_KEYS = {
    "nbregions": Regions.read_regions,
    "nbmodels": MaterialModels.read_material_properties,
    "nreceiversets": ReceiverSets.read_receiver_sets,
}


def read_parfile(path: Union[str, Path]) -> RunParameters:
    """
    Read a Par_file.

    Parameters
    ----------
    path
        A path to the parameters file. Should be in the DATA directory
        for standard specfem runs.

    """

    out = {}
    iterator = (x for x in path.read_text().split("\n") if not x.startswith("#") and x)
    for line in iterator:
        # simple key/value
        if "=" in line:
            key, value = extract_parline_key_value(line)
            if key in _MULTILINE_KEYS:
                func = _MULTILINE_KEYS[key]
                out[key] = func(value, iterator)
            else:
                out[key] = value
        else:
            msg = f"Unhandled line: \n{line}\n in {path}"
            raise UnhandledParFileLine(msg)
