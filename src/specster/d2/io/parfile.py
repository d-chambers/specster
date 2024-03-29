"""
Module for reading/writing parfiles.
"""
import fnmatch
import re
from pathlib import Path
from typing import List, Literal, Optional, Self

from pydantic import Field

from specster.constants import _ENUM_MAP, _MEANING_MAP
from specster.core.misc import find_file_startswith, write_model_data
from specster.core.models import AbstractParameterModel, SpecFloat, SpecsterModel
from specster.core.parse import extract_parline_key_value, iter_file_lines
from specster.core.render import dict_to_description
from specster.core.stations import Station2D, read_stations
from specster.exceptions import UnhandledParFileLine

SOURCE_REG = re.compile(fnmatch.translate("source*"), re.IGNORECASE)


# --- Material Models


class AbstractMaterialModelType(SpecsterModel):
    """Abstract type for material models."""

    # model_type: str = "-10"
    _type_cls_map = {}  # {type: cls}

    def __init_subclass__(cls, **kwargs):
        default = cls.__fields__["model_type"].default
        cls._type_cls_map[default] = cls

    @classmethod
    def read_line(cls, line):
        """Read lines to create class instance."""
        params = line.split("#")[0].split()
        model_typ = cls.__fields__["model_type"].default
        assert params[1] == model_typ, "Wrong model type!"
        # need to strip out model type, if we got here its already handled.
        return super().read_line(params)

    write_model_data = write_model_data


class ElasticModel(AbstractMaterialModelType):
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

    model_number: int
    model_type: Literal["1"] = "1"  # fixed type
    rho: SpecFloat = Field(description="density")
    Vp: SpecFloat = Field(description="P velocity")
    Vs: SpecFloat = Field(description="S velocity")
    void1_: str = "0"
    void2_: str = "0"
    QKappa: SpecFloat = Field(9999, description="P quality factor")
    Qmu: SpecFloat = Field(9999, description="S quality factor")
    void3_: str = "0"
    void4_: str = "0"
    void5_: str = "0"
    void6_: str = "0"
    void7_: str = "0"
    void8_: str = "0"


class AnisotropicModel(AbstractMaterialModelType):
    """
    Covers anisotropic case

    Here are some examples from the par file how these look:

    anisotropic:
        model_number 2 rho c11 c13 c15 c33 c35 c55 c12 c23 c25   0 QKappa Qmu
    anisotropic in AXISYM:
        model_number 2 rho c11 c13 c15 c33 c35 c55 c12 c23 c25 c22 QKappa Qmu
    """

    model_number: int
    model_type: Literal["2"] = "2"  # fixed type
    rho: SpecFloat = Field(description="density")
    c11: SpecFloat
    c13: SpecFloat
    c15: SpecFloat
    c33: SpecFloat
    c35: SpecFloat
    c55: SpecFloat
    c12: SpecFloat
    c23: SpecFloat
    c25: SpecFloat
    c22: SpecFloat
    QKappa: SpecFloat = Field(9999, description="P quality factor")
    Qmu: SpecFloat = Field(9999, description="S quality factor")


class PoroelasticModel(AbstractMaterialModelType):
    """Model describing poroelastic material"""

    model_number: int
    model_type: Literal["3"] = "3"  # fixed type
    rhos: SpecFloat
    rhof: SpecFloat
    phi: SpecFloat
    c: SpecFloat
    kxx: SpecFloat
    kxz: SpecFloat
    kzz: SpecFloat
    Ks: SpecFloat
    Kf: SpecFloat
    Kfr: SpecFloat
    etaf: SpecFloat
    mufr: SpecFloat
    Qmu: SpecFloat


class TomoModel(AbstractMaterialModelType):
    """Tomography model?"""

    model_number: int
    model_type: Literal["-1"] = "-1"  # fixed type
    void1_: str = "0"
    void2_: str = "0"
    void3_: str = "0"
    void4_: str = "0"
    void5_: str = "0"
    void6_: str = "0"
    void7_: str = "0"
    void8_: str = "0"
    void9_: str = "0"
    void10_: str = "0"
    void11_: str = "0"
    void12_: str = "0"
    void13_: str = "0"


class MaterialModels(AbstractParameterModel):
    """
    Class to hold information about material properties.
    """

    models: List[AbstractMaterialModelType]
    tomography_file: Optional[Path] = Field(
        None, description="External tomography file"
    )

    @property
    def nbmodels(self) -> int:
        """Get the number of regions."""
        return len(self.models)

    @classmethod
    def read_material_properties(cls, value, iterator, **kwargs):
        """Read material properties"""
        model_type_key = AbstractMaterialModelType._type_cls_map
        out = {
            "nbmodels": int(value),
            "models": [],
        }
        for _ in range(int(value)):
            line = next(iterator)
            model_type = model_type_key[line.split()[1]]
            out["models"].append(model_type.read_line(line))
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

    def write_model_data(self, key: Optional[str] = None):
        """Write data to file."""
        field_names = [f"{getattr(self, x):d}" for x in self.__fields__]
        out = " ".join(field_names)
        return out


class Regions(AbstractParameterModel):
    """Tracks regions in the model."""

    regions: List[Region2D]

    @property
    def nbregions(self) -> int:
        """Get the number of regions."""
        return len(self.regions)

    @classmethod
    def read_regions(cls, value, iterator, **kawrgs):
        """Read material properties"""
        regions = []
        for _ in range(int(value)):
            line = next(iterator)
            regions.append(Region2D.read_line(line))
        return cls(nbregions=value, regions=regions)


class Mesh(AbstractParameterModel):
    """Controls the meshing parameters (under MESH section)."""

    # type of partitioning
    partitioning_type: int = Field(
        description=dict_to_description(
            "partitioning_type", _ENUM_MAP["partitioning_type"]
        )
    )
    ngnod: Literal["4", "9"] = Field(
        "9",
        description="number of control nodes per element (4 or 9)",
    )
    setup_with_binary_database: Literal["0", "1", "2"] = Field(
        "0",
        description=dict_to_description(
            "setup_with_binary_database", _ENUM_MAP["setup_with_binary_database"]
        ),
    )
    model: Literal[tuple(_MEANING_MAP["model"])] = Field(
        "default", description=dict_to_description("model", _MEANING_MAP["model"])
    )
    save_model: Literal[tuple(_MEANING_MAP["save_model"])] = Field(
        "default", description=dict_to_description("model", _MEANING_MAP["model"])
    )


class Attenuation(AbstractParameterModel):
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
    attenuation_f0_reference: SpecFloat = Field(
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
    q0_poroelastic: SpecFloat = Field(
        1, description="Quality factor for viscous attenuation"
    )
    freq0_poroelastic: SpecFloat = Field(
        10,
        description="frequency for viscous attenuation",
    )
    undo_attenuation_and_or_pml: bool = Field(
        False, description="Undo attenuation for sensitivity kernel calc."
    )
    nt_dump_attenuation: int = Field(
        500, description="how often to dump restart files in sensitivity calc."
    )
    no_backward_reconstruction: bool = Field(
        False, description="Reads forward image from disk..."
    )


class Source(SpecsterModel):
    """A single Source"""

    source_surf: bool = Field(False, description="tie source to surface")
    xs: SpecFloat = Field(1_000.0, description="Source x location in meters")
    zs: SpecFloat = Field(1_000.0, description="Source z location in meters")
    source_type: Literal["1", "2", "3", "4", "5", "6"] = Field(
        "source_type",
        description=dict_to_description("source_type", _ENUM_MAP["source_type"]),
    )
    time_function_type: Literal[tuple(_ENUM_MAP["time_function_type"])] = Field(
        "time_function_type",
        description=dict_to_description(
            "time_function_type", _ENUM_MAP["time_function_type"]
        ),
    )
    name_of_source_file: str = Field(
        "YYYYYYYY", description="External source time function to use."
    )
    burst_band_width: SpecFloat = Field(
        0, description="bandwith of burst (for source_time option 9)"
    )
    f0: SpecFloat = Field(1.0, description="Dominant source freq (Hz)")
    tshift: SpecFloat = Field(
        0.0, description="time shift when multisources used (one must be 0)"
    )
    anglesource: SpecFloat = Field(0.0, description="Plane have incident angle")
    mxx: SpecFloat = Field(1.0, description="Mxx component of moment tensor")
    mzz: SpecFloat = Field(1.0, description="Mzz component of moment tensor")
    mxz: SpecFloat = Field(0.0, description="Mxz component of moment tensor")
    factor: SpecFloat = Field(1.000e10, description="amplification factor")
    vx: SpecFloat = Field(0.0, description="Horizontal source velocity (m/s)")
    vz: SpecFloat = Field(0.0, description="Vertical source velocity (m/s)")


class Sources(AbstractParameterModel):
    """Controls the source parameters (under Source section)."""

    @property
    def nsources(self) -> int:
        """Get the number of sources."""
        return len(self.sources)

    force_normal_to_surface: bool = Field(
        False,
        description="angleforce normal to surface",
    )
    initialfield: bool = Field(
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
    noise_source_time_function_type: Literal["1", "2", "3", "4"] = Field(
        "4",
        description=dict_to_description(
            "noise_source_time_function_type",
            _ENUM_MAP["noise_source_time_function_type"],
        ),
    )
    write_moving_sources_database: bool = Field(False, description="See manual")

    sources: List[Source]

    @staticmethod
    def read_sources(value, path, **kwargs):
        """Read the sources"""
        source_count = int(value)
        out = Sources.read_source_file(Path(path).parent, source_count)
        assert len(out) == source_count
        return out

    @staticmethod
    def read_source_file(base_path, file_count=1) -> List[Source]:
        """Read the source file(s)."""
        sources = []
        iterable = (x for x in base_path.glob("*") if SOURCE_REG.match(str(x.name)))
        sorted_source_files = sorted(iterable, key=lambda x: x.name)
        for path in sorted_source_files[:file_count]:
            source_kwargs = {}
            iterator = iter_file_lines(path)
            for line in iterator:
                key, value_ = extract_parline_key_value(line)
                # This marks the start of a new event
                if key == "source_surf" and len(source_kwargs):
                    sources.append(Source(**source_kwargs))
                source_kwargs[key] = value_
            # also need to scoop up last event
            sources.append(Source(**source_kwargs))
        return sources


class ReceiverSet(SpecsterModel):
    """A single receiver set."""

    nrec: int = Field(11, description="number of receivers")
    xdeb: SpecFloat = Field(
        300.0,
        description="first receiver x in meters",
    )
    zdeb: SpecFloat = Field(2200.0, description="first receiver z in meters")
    xfin: SpecFloat = Field(
        3700.0,
        description="last receiver x in meters",
    )
    zfin: SpecFloat = Field(
        2200.0,
        description="last receiver z in meters",
    )
    record_at_surface_same_vertical: bool = Field(
        True,
        description="fix receivers at the surface",
    )
    write_model_data = SpecsterModel.write_model_data
    disp = SpecsterModel.disp


class ReceiverSets(AbstractParameterModel):
    """Class containing multiple receiver sets."""

    anglerec: SpecFloat = Field(
        0.0, description="angle to rotate components at receivers"
    )
    rec_normal_to_surface: bool = Field(
        False,
        description="base anglerec normal to surface",
    )
    receiver_sets: List[ReceiverSet]

    @property
    def nreceiversets(self) -> int:
        """Get the number of receiver sets."""
        return len(self.receiver_sets)

    @classmethod
    def read_receiver_sets(cls, value, iterator, state, **kwargs):
        """Read the receiver sets from the iterator."""
        receiver_set_count = int(value)
        out = dict(
            nreceiversets=receiver_set_count,
            anglerec=extract_parline_key_value(next(iterator))[1],
            rec_normal_to_surface=extract_parline_key_value(next(iterator))[1],
            receiver_sets=[],
        )
        # # Skip populating receivers if we don't use them
        if state.get("use_existing_stations"):
            return cls(**out)
        for _ in range(receiver_set_count):
            rec_dict = {}
            for _ in range(6):  # each receiver set has six lines
                key, val = extract_parline_key_value(next(iterator))
                rec_dict[key] = val
            out["receiver_sets"].append(ReceiverSet(**rec_dict))
        assert len(out["receiver_sets"]) == receiver_set_count
        return cls(**out)


class Receivers(AbstractParameterModel):
    """Controls the receiver parameters (under Receiver section)."""

    seismotype: Literal["1", "2", "3", "4", "5", "6"] = Field(
        "1", description=dict_to_description("seismotype", _ENUM_MAP["seismotype"])
    )
    ntstep_between_output_seismos: int = Field(
        10_000, description="Number of steps before saving seismograms"
    )
    ntstep_between_output_sample: int = Field(
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
    save_binary_seismograms_single: bool = Field(
        True, description="save seismograms in single precision binary format"
    )
    save_binary_seismograms_double: bool = Field(
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

    # stations from external files
    stations: List[Station2D]

    # @pydantic.root_validator(pre=True)
    # def get_stations(cls, values):
    #     """Read in external station file if specified."""
    #     use_stations = values.get('use_existing_stations', False)
    #     if not use_stations:
    #         return values
    #     station_path =


class AdjointKernel(AbstractParameterModel):
    """Controls the adjoint kernel parameters (under adjoint kernel section)."""

    save_ascii_kernels: bool = Field(
        True, description="save sensitivity as ascii files. Save to binary otherwise"
    )
    ntstep_between_compute_kernels: int = Field(
        1, description="Only compute the adjoint every n steps"
    )
    approximate_hess_kl: bool = Field(
        False, description="approximate hessian for preconditioning"
    )


class BoundaryConditions(AbstractParameterModel):
    """Controls the boundary condition parameters (under boundary section)."""

    pml_boundary_conditions: bool = Field(
        True, description="perfectly matched layer active"
    )
    nelem_pml_thickness: int = Field(2, description="number of pml elements on edges")
    rotate_pml_activate: bool = Field(False, description="Whether to rotate the PMLs")
    rotate_pml_angle: SpecFloat = Field(
        30.0, description="Angle to rotate PML (if at all)"
    )
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
    stacey_absorbing_conditions: bool = Field(
        False, description="activate stacey absorbing BCs"
    )
    add_periodic_conditions: bool = Field(
        False, description="Add periodic boundary conditions"
    )
    periodic_horiz_dist: SpecFloat = Field(
        4000.0, description="Horizontal distance of periodic damping"
    )


class ExternalMeshing(AbstractParameterModel):
    """Controls the external meshing parameters."""

    read_external_mesh: bool = Field(
        False, description="read an external mesh for velocity model"
    )
    mesh_file: Optional[Path] = Field(
        "./DATA/mesh_file", description="Path to mesh file"
    )
    nodes_coords_file: Optional[Path] = Field(
        "./DATA/nodes_coords_dile", description="Path to mesh coordinates"
    )
    materials_file: Optional[Path] = Field(
        "./DATA/material_file", description="Path to material file"
    )
    free_surface_file: Optional[Path] = Field(
        "./DATA/free_surface_file", description="Path to free surface file"
    )
    axial_elements_file: Optional[Path] = Field(
        "./DATA/axial_element_file", description="Path to mesh file"
    )
    absorbing_surface_file: Optional[Path] = Field(
        "./DATA/absorbing_surface_file", description="Path to mesh file"
    )
    acoustic_forcing_surface_file: Optional[Path] = Field(
        "./DATA/Surf_acforcing_Bottom_enforcing_mesh",
    )
    absorbing_cpml_file: Optional[Path] = Field(
        "./DATA/absorbing_cpml_file", description="File with CPML element numbers"
    )
    tangential_detection_curve_file: Optional[Path] = Field(
        "./DATA/courbe_eros_nodes", description="Contains curve delineating model"
    )


class Interfaces(SpecsterModel):
    """Interfaces for the internal meshing."""

    layers: List[List[tuple[float, float]]]
    z_elements: List[int]

    @property
    def interface_count(self) -> int:
        """return the number of interfaces."""
        return len(self.layers)

    @staticmethod
    def read_interfaces(value, path, state, **kwargs):
        """Read the sources"""
        if state.get("read_external_mesh", False):
            return Interfaces(layers=[[]], z_elements=[])
        layers = []
        z_elements = []
        interfaces_path = path.parent / value
        assert interfaces_path.exists()
        # read number of interfaces
        lines = (x.strip() for x in iter_file_lines(interfaces_path))
        number_of_interfaces = int(next(lines))
        # parse each interface
        for _ in range(number_of_interfaces):
            point_list = []
            num_points = int(next(lines))
            for _ in range(num_points):
                point_tup = tuple(float(x) for x in next(lines).split())
                point_list.append(point_tup)
            layers.append(point_list)
        # read the number of elements
        for num_elements in lines:
            z_elements.append(int(num_elements))
        return Interfaces(layers=layers, z_elements=z_elements)


class InternalMeshing(AbstractParameterModel):
    """Controls the internal meshing parameters."""

    interfacesfile: Optional[Path] = Field(
        "./DATA/interfaces.dat", description="File with interfaces"
    )
    interfaces: Interfaces
    xmin: SpecFloat = Field(0.0, description="abscissa of left side of the model")
    xmax: SpecFloat = Field(4000, description="abscissa of the right side of model")
    nx: int = Field(80, description="number of elements along X")
    absorbbottom: bool = Field(True, description="absorbing conditions on bottom ")
    absorbright: bool = Field(True, description="absorbing conditions on right ")
    absorbtop: bool = Field(False, description="absorbing conditions on top")
    absorbleft: bool = Field(True, description="absorbing conditions on left ")
    regions: Regions


class Display(AbstractParameterModel):
    """Controls the display parameters."""

    ntstep_between_output_info: int = Field(
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


class JPEGDisplay(AbstractParameterModel):
    """Information for jpeg output."""

    output_color_image: bool = Field(
        True, description="If true, output jpeg color image"
    )
    imagetype_jpeg: Literal["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] = Field(
        1,
        description=dict_to_description("imagetype_jpeg", _ENUM_MAP["imagetype_jpeg"]),
    )
    factor_subsample_image: SpecFloat = Field(
        1.0, description="factor to subsample or oversample (<1)"
    )
    use_constant_max_amplitude: bool = Field(
        False, description="Use global maximum for image normalization"
    )
    constant_max_amplitude_to_use: SpecFloat = Field(
        116e4, description="constant maximum amplitude"
    )
    power_display_color: SpecFloat = Field(
        0.30, description="Non linear display to enhance small features"
    )
    draw_sources_and_receivers: bool = Field(
        True,
        description="Display sources as orange crosses and sources as green squares",
    )
    draw_water_in_blue: bool = Field(
        True, description="Display acoustic layers in blue"
    )
    use_snapshot_number_in_filename: bool = Field(
        False, description="use the snapshot number in file name."
    )


class PostScriptDisplay(AbstractParameterModel):
    """Information for postscript output."""

    output_postscript_snapshot: bool = Field(
        True, description="Output postscript snapshots"
    )
    imagetype_postscript: Literal["1", "2", "3"] = Field(
        2,
        description=dict_to_description(
            "imagetype_postscript", _ENUM_MAP["imagetype_postscript"]
        ),
    )
    meshvect: bool = Field(True, description="Display mesh on postscript.")
    modelvect: bool = Field(False, description="display velocity model on plot")
    boundvect: bool = Field(True, description="Display boundary conditions.")
    interpol: bool = Field(True, description="Interpolate GLL onto regular grid")
    pointsdisp: int = Field(
        6, description="Number of points in each direction for interpolation."
    )
    subsamp_postscript: int = Field(
        1, description="subsampling of velocity model for post script plots"
    )
    sizemax_arrows: SpecFloat = Field(1.0, description="Max size for arrows in cm")
    us_letter: bool = Field(False, description="use US letter or European A4 paper")


class WaveDumpDisplay(AbstractParameterModel):
    """Controls dumping wavefield to disk."""

    output_wavefield_dumps: bool = Field(
        False, description="Output wavefield to disk, creates large files!"
    )
    imagetype_wavefield_dumps: Literal["1", "2", "3", "4"] = Field(
        1,
        description=dict_to_description(
            "imagetype_wavefield_dumps", _ENUM_MAP["imagetype_wavefield_dumps"]
        ),
    )
    use_binary_for_wavefield_dumps: bool = Field(
        False, description="If True, use binary format else ascii format."
    )


class Visualizations(AbstractParameterModel):
    """Controls Other visualization parameters."""

    ntstep_between_output_images: int = Field(
        100, description="How often (timestep) output is dumped for visualization"
    )
    cutsnaps: SpecFloat = Field(
        1.0, description="minimum amplitude kept in % for JPEG/PostScript"
    )
    # --- jpeg parameters
    jpeg: JPEGDisplay
    # --- postscript parameters
    postscript: PostScriptDisplay
    # --- wavedump parameters
    wavefield_dump: WaveDumpDisplay


class SpecParameters2D(AbstractParameterModel):
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
    dt: SpecFloat = Field(1.1e-3, description="time increment")
    time_stepping_scheme: int = Field(
        1,
        description=dict_to_description(
            "time_stepping_scheme", _ENUM_MAP["time_stepping_scheme"]
        ),
    )
    p_sv: bool = Field(False, description="If P_SV should be run, otherwise SH")
    axisym: bool = Field(False, description="type of calc (p-sv or SH/membrane waves")
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
    # --- Visualization parameters (no idea why this is different from display)
    visualizations: Visualizations
    # Values at end of par file, no subsection

    broadcast_same_mesh_and_model: bool = Field(
        True,
        description="see manual",
    )

    number_of_simultaneous_runs: int = Field(
        1, description="complicated parameter, see specfem docs for details"
    )
    gpu_mode: bool = Field(True, description="If GPUs should be used")

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Parameter data from file."""
        data = parse_parfile(path)
        return cls.init_from_dict(data)


# keys that require weird parsing rules. The key triggers
# the calling of the function and the output is stored in
# the second argument
_SPECIAL_KEYS = {
    "nbregions": (Regions.read_regions, "regions"),
    "nbmodels": (MaterialModels.read_material_properties, "material_models"),
    "nreceiversets": (ReceiverSets.read_receiver_sets, "receiver_sets"),
    "nsources": (Sources.read_sources, "sources"),
    "use_existing_stations": (read_stations, "stations"),
    "interfacesfile": (Interfaces.read_interfaces, "interfaces"),
}


def parse_parfile(path: Path) -> dict:
    """
    Read a Par_file into a dictionary.

    Parameters
    ----------
    path
        A path to the parameters file. Should be in the DATA directory
        for standard specfem runs.
    """
    out = {}
    path = find_file_startswith(path)
    iterator = iter_file_lines(path)
    for line in iterator:
        # simple key/value
        if "=" in line:
            key, value = extract_parline_key_value(line)
            out[key] = value
            # Need to handle special parsing of some keys
            if key in _SPECIAL_KEYS:
                func, name = _SPECIAL_KEYS[key]
                out[name] = func(
                    value=value,
                    iterator=iterator,
                    path=path,
                    state=out,
                )
        else:
            msg = f"Unhandled line: \n{line}\n in {path}"
            raise UnhandledParFileLine(msg)
    return out
