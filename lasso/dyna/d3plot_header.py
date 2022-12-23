import enum
from typing import Any, Dict, Tuple, Union

import numpy as np
import rich

from ..io.binary_buffer import BinaryBuffer
from ..logging import get_logger

# We have a lot of docstrings here but even if not so, we want to contain the
# code here.
# pylint: disable=too-many-lines

LOGGER = get_logger(__file__)


def get_digit(number: int, i_digit: int) -> int:
    """Get a digit from a number

    Parameters
    ----------
    number: int
        number to get digit from
    i_digit: int
        index of the digit

    Returns
    -------
    digit: int
        digit or 0 if i_digit is too large

    Notes
    -----
        `i_digit` does refer to a digit from the
        lowest position counting. Thus,
        123 with `i_digit=0` is `3`.
    """
    digit_list = []

    # pylint: disable = inconsistent-return-statements
    def _get_digit_recursive(x: int):
        if x < 10:
            digit_list.append(x)
            return x
        _get_digit_recursive(x // 10)
        digit_list.append(x % 10)

    # do the thing
    _get_digit_recursive(number)

    # revert list from smallest to biggest
    digit_list = digit_list[::-1]

    return digit_list[i_digit] if i_digit < len(digit_list) else 0


class D3plotFiletype(enum.Enum):
    """Enum for the filetype of a D3plot"""

    D3PLOT = 1
    D3PART = 5
    D3EIGV = 11
    INTFOR = 4


def d3plot_filetype_from_integer(value: int) -> D3plotFiletype:
    """Get a D3plotFiletype object from an integer

    Parameters
    ----------
    value: int
        integer value representing the filetype

    Returns
    -------
    filetype: D3plotFiletype
        d3plot filetype object

    Raises
    ------
        RuntimeError if invalid value.
    """

    valid_entries = {
        entry.value: entry
        for entry in D3plotFiletype.__members__.values()
        if entry.value != 4  # no intfor
    }

    if value not in valid_entries:
        valid_filetypes = ",".join(
            f"{key} ({value.value})"
            for key, value in D3plotFiletype.__members__.items()
            if value.value != 4
        )
        err_msg = f"Invalid filetype value of {value}. Expected one of: {valid_filetypes}"
        raise ValueError(err_msg)

    return valid_entries[value]


# pylint: disable = too-many-instance-attributes
class D3plotHeader:
    """Class for reading only header information of a d3plot

    Attributes
    ----------
    filepath: str
        Filepath of the processed file.
    itype: np.dtype
        Integer type of d3plot.
    ftype: np.dtype
        Floating point type of d3plot.
    wordsize: int
        size of words in bytes (4 = single precision, 8 = double precision).
    raw_header: Dict[str, Any]
        Raw header data as dict.
    external_numbers_dtype: np.dtype
        Integer type of user ids.
    n_header_bytes: int
        Number of bytes of header (at least 256 or more).
    title: str
        Main title.
    title2: str
        Optional, secondary title.
    runtime: int
        Runtime of the d3plot as timestamp.
    filetype: D3plotFiletype
        Filetype such as d3plot or d3part.
    source_version: int
        Source version of LS-Dyna.
    release_version: str
        Release version of LS-Dyna.
    version: float
        Version of LS-Dyna.
    extra_long_header: bool
        If header was longer than default.
    n_dimensions: int
        Number of dimensions, usually three.
    n_global_vars: int
        How many global vars for each state.
    n_adapted_element_pairs: int
        How many adapted element pairs.
    has_node_deletion_data: bool
        If node deletion data is present.
    has_element_deletion_data: bool
        If element deletion data is present.
    has_numbering_section: bool
        If a user numbering section is present.
    has_material_type_section: bool
        If material type section was written.
    n_numbering_section_words: int
        Amount of words for numbering section.
    has_invariant_numbering: bool
        If invariant numbering is used whatever that means.
    quadratic_elems_has_full_connectivity: bool
        If quadric elements have full connectivity.
    quadratic_elems_has_data_at_integration_points: bool
        If quadric elements data is at integration points.
    n_post_branches: int
        Unused and unknown.
    n_types: Tuple[int, ...]
        Behind geometry these are integers indicating additional data such as
        part names.
    n_parts: int
        Obviously number of parts.
    n_nodes: int
        Number of nodes.
    has_node_temperatures: bool
        If node temperature is present.
    has_node_temperature_layers: bool
        If node temperatures are layered.
    has_node_heat_flux: bool
        If node heat flux is present.
    has_node_mass_scaling: bool
        Mass scaling is written.
    has_node_displacement: bool
        Node displacement is written.
    has_node_velocity: bool
        Node velocity is written.
    has_node_acceleration: bool
        Node acceleration is written.
    has_node_temperature_gradient: bool
        Node temperature gradient is written.
    has_node_residual_forces: bool
        Node residual forces are written.
    has_node_residual_moments: bool
        Node residual moments are written.
    has_node_max_contact_penetration_absolute: bool
        Node contact penetration info exist.
    has_node_max_contact_penetration_relative: bool
        Node relative contact penetration info was written.
    has_node_contact_energy_density: int
        Node energy density was written.
    n_shell_tshell_layers: int
        Number of layers for shells and thick shells.
    n_shell_tshell_history_vars: int
        Number of history vars for shells and thick shells.
    has_shell_tshell_stress: bool
        If shells and thick shells have stresses.
    has_shell_tshell_pstrain: bool
        If shells and thick shells have eff. plastic strain.
    has_element_strain: bool
        If all elements have strain.
    has_solid_shell_plastic_strain_tensor: bool
        If solids have plastic strain tensor.
    has_solid_shell_thermal_strain_tensor: bool
        If solids have thermal strain tensor.
    n_solids: int
        Number of solids.
    n_solid_vars: int
        Number of solid variables per element and state.
    n_solid_materials: int
        Number of solid materials/parts.
    n_solid_history_vars: int
        Number of solid history variables.
    n_solid_thermal_vars: int
        Number of solid thermal variables.
    n_solids_20_node_hexas: int
        Number of 20-node solid hexas.
    n_solids_27_node_hexas: int
        Number of 27-node solid hexas.
    n_solids_21_node_pentas: int
        Number of 21-node solid pentas.
    n_solids_15_node_tetras: int
        Number of 15-node solid tetras.
    n_solids_20_node_tetras: int
        Number of 20-node solid tetras.
    n_solids_40_node_pentas: int
        Number of 40-node solid pentas.
    n_solids_64_node_hexas: int
        Number of 64-node solid hexas.
    has_solid_2_extra_nodes: bool
        If two extra nodes were written for solids.
    has_solid_stress: bool
        If solid stress is present.
    has_solid_pstrain: bool
        If solid eff. plastic strain is present.
    has_quadratic_solids: bool
        If quadratic solids were used.
    has_cubic_solids: bool
        If cubic solids were used.
    has_solid_internal_energy_density: bool
        If solids have internal energy density.
    n_solid_layers: int
        Number of solid layers.
    n_shells: int
        Number of shells.
    n_shell_vars: int
        Number of shell vars per element and state.
    n_shell_materials: int
        Number of shell materials/parts.
    n_shells_8_nodes: int
        Number of 8-node shells.
    has_shell_four_inplane_gauss_points: bool
        If shells have four inplace gaussian integration points.
    has_shell_forces: bool
        If shell forces are present.
    has_shell_extra_variables: bool
        If extra shell variables such as forces are present.
    has_shell_internal_energy_density: bool
        If shell internal energy density is present.
    n_thick_shells: int
        Number of thick shell elements.
    n_thick_shell_vars: int
        Number of thick shell element vars.
    n_thick_shell_materials: int
        Number of thick shell materials/parts.
    has_thick_shell_energy_density: bool
        If thick shells have energy density.
    thick_shell_energy_density_position: int
        Nnused.
    n_beams: int
        Number of beam elements.
    n_beam_vars: int
        Number of state variables per beam element.
    n_beam_materials: int
        Number of beam materials.
    n_beam_history_vars: int
        Number of beam history variables.
    n_airbags: int
        Number of airbags.
    has_airbag_n_chambers: bool
        If airbags have number of chambers var.
    has_rigid_road_surface: bool
        If rigid road surface was written.
    has_rigid_body_data: bool
        If rigid body section was written.
    has_reduced_rigid_body_data: bool
        If the reduced set of rigid body data was written.
    n_rigid_wall_vars: int
        Number of rigid wall vars.
    n_sph_nodes: int
        Number of sph nodes.
    n_sph_materials: int
        Number of sph materials.
    n_ale_materials: int
        Number of ale materials.
    n_ale_fluid_groups: int
        Number of ale fluid groups.
    has_cfd_data: bool
        If CFD-Data was written.
    has_multi_solver_data: bool
        If multi-solver data was written.
    cfd_extra_data: int
        If cfd data contains extra section.
    legacy_code_type: int
        Originally a code indicator but unused nowadays.
    unused_numst: int
        Unused and not explained in docs.
    """

    # meta
    filepath: str = ""

    # file info
    itype: np.dtype = np.int32
    ftype: np.dtype = np.float32
    wordsize: int = 4
    raw_header: Dict[str, Any] = {}
    external_numbers_dtype = np.int32
    n_header_bytes: int = 0

    # header
    title: str = ""
    title2: str = ""
    runtime: int = 0
    filetype: D3plotFiletype = D3plotFiletype.D3PLOT

    source_version: int = 0
    release_version: str = ""
    version: float = 0.0
    extra_long_header: bool = False

    # general info
    n_dimensions: int = 3
    n_global_vars: int = 0
    n_adapted_element_pairs: int = 0
    has_node_deletion_data: bool = False
    has_element_deletion_data: bool = False
    has_numbering_section: bool = False
    has_material_type_section: bool = False
    n_numbering_section_words: int = 0
    has_invariant_numbering: bool = False
    quadratic_elems_has_full_connectivity: bool = False
    quadratic_elems_has_data_at_integration_points: bool = False
    n_post_branches: int = 0
    n_types: Tuple[int, ...] = tuple()

    # parts
    n_parts: int = 0

    # nodes
    n_nodes: int = 0
    has_node_temperatures: bool = False
    has_node_temperature_layers: bool = False
    has_node_heat_flux: bool = False
    has_node_mass_scaling: bool = False
    has_node_displacement: bool = False
    has_node_velocity: bool = False
    has_node_acceleration: bool = False
    has_node_temperature_gradient: bool = False
    has_node_residual_forces: bool = False
    has_node_residual_moments: bool = False
    has_node_max_contact_penetration_absolute: bool = False
    has_node_max_contact_penetration_relative: bool = False
    has_node_contact_energy_density: int = False

    # elements
    n_shell_tshell_layers: int = 3
    n_shell_tshell_history_vars: int = 0
    has_shell_tshell_stress: bool = False
    has_shell_tshell_pstrain: bool = False
    has_element_strain: bool = False
    has_solid_shell_plastic_strain_tensor: bool = False
    has_solid_shell_thermal_strain_tensor: bool = False

    # solids
    n_solids: int = 0
    n_solid_vars: int = 0
    n_solid_materials: int = 0
    n_solid_history_vars: int = 0
    n_solid_thermal_vars: int = 0
    n_solids_20_node_hexas: int = 0
    n_solids_27_node_hexas: int = 0
    n_solids_21_node_pentas: int = 0
    n_solids_15_node_tetras: int = 0
    n_solids_20_node_tetras: int = 0
    n_solids_40_node_pentas: int = 0
    n_solids_64_node_hexas: int = 0
    has_solid_2_extra_nodes: bool = False
    has_solid_stress: bool = False
    has_solid_pstrain: bool = False
    has_quadratic_solids: bool = False
    has_cubic_solids: bool = False
    has_solid_internal_energy_density: bool = False

    # shells
    n_shells: int = 0
    n_shell_vars: int = 0
    n_shell_materials: int = 0
    n_shells_8_nodes: int = 0
    has_shell_four_inplane_gauss_points: bool = False
    has_shell_forces: bool = False
    has_shell_extra_variables: bool = False
    has_shell_internal_energy_density: bool = False
    # has_shell_internal_energy: bool = False

    # thick shells
    n_thick_shells: int = 0
    n_thick_shell_vars: int = 0
    n_thick_shell_materials: int = 0
    has_thick_shell_energy_density: bool = False
    thick_shell_energy_density_position: int = 0

    # beams
    n_beams: int = 0
    n_beam_vars: int = 0
    n_beam_materials: int = 0
    n_beam_history_vars: int = 0

    # airbags
    n_airbags: int = 0
    has_airbag_n_chambers: bool = False

    # rigid roads
    has_rigid_road_surface: bool = False

    # rigid bodies
    has_rigid_body_data: bool = False
    has_reduced_rigid_body_data: bool = False

    # sph
    n_sph_nodes: int = 0
    n_sph_materials: int = 0

    # ale
    n_ale_materials: int = 0
    n_ale_fluid_groups: int = 0

    # cfd
    has_cfd_data: bool = False

    # multi-solver
    has_multi_solver_data: bool = False
    cfd_extra_data: int = 0

    # historical artifacts
    legacy_code_type: int = 6
    unused_numst: int = 0

    def __init__(self, filepath: Union[str, BinaryBuffer, None] = None):
        """Create a D3plotHeader instance

        Parameters
        ----------
        filepath: Union[str, BinaryBuffer, None]
            path to a d3plot file or a buffer holding d3plot memory

        Returns
        -------
        header: D3plotHeader
            d3plot header instance

        Examples
        --------
            Create an empty header file

            >>> header = D3plotHeader()

            Now load only the header of a d3plot.

            >>> header.load_file("path/to/d3plot")

            Or we can do the above together.

            >>> header = D3plotHeader("path/to/d3plot")

        Notes
        -----
            This class does not load the entire memory of a d3plot
            but merely what is required to parse the header information.
            Thus, it is safe to use on big files.
        """

        if filepath is not None:
            self.load_file(filepath)

    def print(self) -> None:
        """Print the header"""
        rich.print(self.__dict__)

    def _read_file_buffer(self, filepath: str) -> BinaryBuffer:
        """Reads a d3plots header

        Parameters
        ----------
        filepath: str
            path to d3plot

        Returns
        -------
        bb: BinaryBuffer
            buffer holding the exact header data in binary form
        """

        LOGGER.debug("_read_file_buffer start")
        LOGGER.debug("filepath: %s", filepath)

        # load first 64 single words
        n_words_header = 64
        n_bytes_hdr_guessed = 64 * self.wordsize
        bb = BinaryBuffer(filepath, n_bytes_hdr_guessed)

        # check if single or double
        self.wordsize, self.itype, self.ftype = self._determine_file_settings(bb)

        # Oops, seems other wordsize is used
        if self.wordsize != D3plotHeader.wordsize:
            bb = BinaryBuffer(filepath, n_words_header * self.wordsize)

        # check for extra long header
        n_header_bytes = self._determine_n_bytes(bb, self.wordsize)
        if len(bb) <= n_header_bytes:
            bb = BinaryBuffer(filepath, n_bytes=n_header_bytes)

        LOGGER.debug("_read_file_buffer end")

        return bb

    def _determine_n_bytes(self, bb: BinaryBuffer, wordsize: int) -> int:
        """Determines how many bytes the header has

        Returns
        -------
        size: int
            size of the header in bytes
        """

        LOGGER.debug("_determine_n_bytes start")

        n_base_words = 64
        min_n_bytes = n_base_words * wordsize

        if len(bb) < n_base_words * wordsize:
            err_msg = "File or file buffer must have at least '{0}' bytes instead of '{1}'"
            raise RuntimeError(err_msg.format(min_n_bytes, len(bb)))

        n_extra_header_words = int(bb.read_number(57 * self.wordsize, self.itype))

        LOGGER.debug("_determine_n_bytes end")

        return (n_base_words + n_extra_header_words) * wordsize

    def load_file(self, file: Union[str, BinaryBuffer]) -> "D3plotHeader":
        """Load d3plot header from a d3plot file

        Parameters
        ----------
        file: Union[str, BinaryBuffer]
            path to d3plot or `BinaryBuffer` holding memory of d3plot

        Returns
        -------
        self: D3plotHeader
            returning self on success

        Notes
        -----
            This routine only loads the minimal amount of data
            that is neccessary. Thus it is safe to use on huge files.

        Examples
        --------
            >>> header = D3plotHeader().load_file("path/to/d3plot")
            >>> header.n_shells
            19684
        """

        # pylint: disable = too-many-locals, too-many-branches, too-many-statements

        LOGGER.debug("_load_file start")
        LOGGER.debug("file: %s", file)

        if not isinstance(file, (str, BinaryBuffer)):
            err_msg = "Argument 'file' must have type 'str' or 'lasso.io.BinaryBuffer'."
            raise ValueError(err_msg)

        # get the memory
        if isinstance(file, str):
            bb = self._read_file_buffer(file)
            self.n_header_bytes = len(bb)
        else:
            bb = file
            self.wordsize, self.itype, self.ftype = self._determine_file_settings(bb)
            self.n_header_bytes = self._determine_n_bytes(bb, self.wordsize)

        LOGGER.debug("n_header_bytes: %d", self.n_header_bytes)

        # read header
        header_words = {
            "title": [0 * self.wordsize, str, 9 * self.wordsize],
            "runtime": [10 * self.wordsize, self.itype],
            "filetype": [11 * self.wordsize, self.itype],
            "source_version": [12 * self.wordsize, self.itype],
            "release_version": [13 * self.wordsize, str, 1 * self.wordsize],
            "version": [14 * self.wordsize, self.ftype],
            "ndim": [15 * self.wordsize, self.itype],
            "numnp": [16 * self.wordsize, self.itype],
            "icode": [17 * self.wordsize, self.itype],
            "nglbv": [18 * self.wordsize, self.itype],
            "it": [19 * self.wordsize, self.itype],
            "iu": [20 * self.wordsize, self.itype],
            "iv": [21 * self.wordsize, self.itype],
            "ia": [22 * self.wordsize, self.itype],
            "nel8": [23 * self.wordsize, self.itype],
            "nummat8": [24 * self.wordsize, self.itype],
            "numds": [25 * self.wordsize, self.itype],
            "numst": [26 * self.wordsize, self.itype],
            "nv3d": [27 * self.wordsize, self.itype],
            "nel2": [28 * self.wordsize, self.itype],
            "nummat2": [29 * self.wordsize, self.itype],
            "nv1d": [30 * self.wordsize, self.itype],
            "nel4": [31 * self.wordsize, self.itype],
            "nummat4": [32 * self.wordsize, self.itype],
            "nv2d": [33 * self.wordsize, self.itype],
            "neiph": [34 * self.wordsize, self.itype],
            "neips": [35 * self.wordsize, self.itype],
            "maxint": [36 * self.wordsize, self.itype],
            "nmsph": [37 * self.wordsize, self.itype],
            "ngpsph": [38 * self.wordsize, self.itype],
            "narbs": [39 * self.wordsize, self.itype],
            "nelt": [40 * self.wordsize, self.itype],
            "nummatt": [41 * self.wordsize, self.itype],
            "nv3dt": [42 * self.wordsize, self.itype],
            "ioshl1": [43 * self.wordsize, self.itype],
            "ioshl2": [44 * self.wordsize, self.itype],
            "ioshl3": [45 * self.wordsize, self.itype],
            "ioshl4": [46 * self.wordsize, self.itype],
            "ialemat": [47 * self.wordsize, self.itype],
            "ncfdv1": [48 * self.wordsize, self.itype],
            "ncfdv2": [49 * self.wordsize, self.itype],
            "nadapt": [50 * self.wordsize, self.itype],
            "nmmat": [51 * self.wordsize, self.itype],
            "numfluid": [52 * self.wordsize, self.itype],
            "inn": [53 * self.wordsize, self.itype],
            "npefg": [54 * self.wordsize, self.itype],
            "nel48": [55 * self.wordsize, self.itype],
            "idtdt": [56 * self.wordsize, self.itype],
            "extra": [57 * self.wordsize, self.itype],
        }

        header_extra_words = {
            "nel20": [64 * self.wordsize, self.itype],
            "nt3d": [65 * self.wordsize, self.itype],
            "nel27": [66 * self.wordsize, self.itype],
            "neipb": [67 * self.wordsize, self.itype],
            "nel21p": [68 * self.wordsize, self.itype],
            "nel15t": [69 * self.wordsize, self.itype],
            "soleng": [70 * self.wordsize, self.itype],
            "nel20t": [71 * self.wordsize, self.itype],
            "nel40p": [72 * self.wordsize, self.itype],
            "nel64": [73 * self.wordsize, self.itype],
            "quadr": [74 * self.wordsize, self.itype],
            "cubic": [75 * self.wordsize, self.itype],
            "tsheng": [76 * self.wordsize, self.itype],
            "nbranch": [77 * self.wordsize, self.itype],
            "penout": [78 * self.wordsize, self.itype],
            "engout": [79 * self.wordsize, self.itype],
        }

        # read header for real
        self.raw_header = self.read_words(bb, header_words)

        if self.raw_header["extra"] != 0:
            self.read_words(bb, header_extra_words, self.raw_header)
        else:
            for name, (_, dtype) in header_extra_words.items():
                self.raw_header[name] = dtype()

        # PARSE HEADER (no fun ahead)
        if isinstance(file, str):
            self.filepath = file
        elif isinstance(file, BinaryBuffer):
            if isinstance(file.filepath_, str):
                self.filepath = file.filepath_
            elif isinstance(file.filepath_, list) and len(file.filepath_) > 0:
                self.filepath = file.filepath_[0]

        self.title = self.raw_header["title"].strip()
        self.runtime = self.raw_header["runtime"]

        # filetype
        filetype = self.raw_header["filetype"]
        if filetype > 1000:
            filetype -= 1000
            self.external_numbers_dtype = np.int64
        else:
            self.external_numbers_dtype = np.int32

        self.filetype = d3plot_filetype_from_integer(filetype)

        self.source_version = self.raw_header["source_version"]
        self.release_version = self.raw_header["release_version"]  # .split("\0", 1)[0]
        self.version = self.raw_header["version"]

        # ndim
        ndim = self.raw_header["ndim"]
        if ndim in (5, 7):
            self.has_material_type_section = True
            ndim = 3
            # self.raw_header['elem_connectivity_unpacked'] = True
        if ndim == 4:
            ndim = 3
            # self.raw_header['elem_connectivity_unpacked'] = True
        if 5 < ndim < 8:
            ndim = 3
            self.has_rigid_road_surface = True
        if ndim in (8, 9):
            ndim = 3
            self.has_rigid_body_data = True
            if self.raw_header["ndim"] == 9:
                self.has_rigid_road_surface = True
                self.has_reduced_rigid_body_data = True
        if ndim not in (2, 3):
            raise RuntimeError(f"Invalid header entry ndim: {self.raw_header['ndim']}")

        self.n_nodes = self.raw_header["numnp"]
        self.legacy_code_type = self.raw_header["icode"]
        self.n_global_vars = self.raw_header["nglbv"]

        # it
        # - mass scaling
        # - node temperature
        # - node heat flux
        if get_digit(self.raw_header["it"], 1) == 1:
            self.has_node_mass_scaling = True
        it_first_digit = get_digit(self.raw_header["it"], 0)
        if it_first_digit == 1:
            self.has_node_temperatures = True
        elif it_first_digit == 2:
            self.has_node_temperatures = True
            self.has_node_heat_flux = True
        elif it_first_digit == 3:
            self.has_node_temperatures = True
            self.has_node_heat_flux = True
            self.has_node_temperature_layers = True

        # iu iv ia
        self.has_node_displacement = self.raw_header["iu"] != 0
        self.has_node_velocity = self.raw_header["iv"] != 0
        self.has_node_acceleration = self.raw_header["ia"] != 0

        # nel8
        self.n_solids = abs(self.raw_header["nel8"])
        if self.raw_header["nel8"] < 0:
            self.has_solid_2_extra_nodes = True

        # nummat8
        self.n_solid_materials = self.raw_header["nummat8"]

        # numds
        self.has_shell_four_inplane_gauss_points = self.raw_header["numds"] < 0

        # numst
        self.unused_numst = self.raw_header["numst"]

        # nv3d
        self.n_solid_vars = self.raw_header["nv3d"]

        # nel2
        self.n_beams = self.raw_header["nel2"]

        # nummat2
        self.n_beam_materials = self.raw_header["nummat2"]

        # nv1d
        self.n_beam_vars = self.raw_header["nv1d"]

        # nel4
        self.n_shells = self.raw_header["nel4"]

        # nummat4
        self.n_shell_materials = self.raw_header["nummat4"]

        # nv2d
        self.n_shell_vars = self.raw_header["nv2d"]

        # neiph
        self.n_solid_history_vars = self.raw_header["neiph"]

        # neips
        self.n_shell_tshell_history_vars = self.raw_header["neips"]

        # maxint
        maxint = self.raw_header["maxint"]
        if maxint > 0:
            self.n_shell_tshell_layers = maxint
        elif maxint <= -10000:
            self.has_element_deletion_data = True
            self.n_shell_tshell_layers = abs(maxint) - 10000
        elif maxint < 0:
            self.has_node_deletion_data = True
            self.n_shell_tshell_layers = abs(maxint)

        # nmsph
        self.n_sph_nodes = self.raw_header["nmsph"]

        # ngpsph
        self.n_sph_materials = self.raw_header["ngpsph"]

        # narbs
        self.has_numbering_section = self.raw_header["narbs"] != 0
        self.n_numbering_section_words = self.raw_header["narbs"]

        # nelt
        self.n_thick_shells = self.raw_header["nelt"]

        # nummatth
        self.n_thick_shell_materials = self.raw_header["nummatt"]

        # nv3dt
        self.n_thick_shell_vars = self.raw_header["nv3dt"]

        # ioshl1
        if self.raw_header["ioshl1"] == 1000:
            self.has_shell_tshell_stress = True
            self.has_solid_stress = True
        elif self.raw_header["ioshl1"] == 999:
            self.has_solid_stress = True

        # ioshl2
        if self.raw_header["ioshl2"] == 1000:
            self.has_shell_tshell_pstrain = True
            self.has_solid_pstrain = True
        elif self.raw_header["ioshl2"] == 999:
            self.has_solid_pstrain = True

        # ioshl3
        self.has_shell_forces = self.raw_header["ioshl3"] == 1000

        # ioshl4
        self.has_shell_extra_variables = self.raw_header["ioshl4"] == 1000

        # ialemat
        self.n_ale_materials = self.raw_header["ialemat"]

        # ncfdv1
        ncfdv1 = self.raw_header["ncfdv1"]
        if ncfdv1 == 67108864:
            self.has_multi_solver_data = True
        elif ncfdv1 != 0:
            self.has_cfd_data = True

        # ncfdv2
        # unused

        # nadapt
        self.n_adapted_element_pairs = self.raw_header["nadapt"]

        # nmmat
        self.n_parts = self.raw_header["nmmat"]

        # numfluid
        self.n_ale_fluid_groups = self.raw_header["numfluid"]

        # inn
        self.has_invariant_numbering = self.raw_header["inn"] != 0

        # nepfg
        npefg = self.raw_header["npefg"]
        self.n_airbags = npefg % 1000
        self.has_airbag_n_chambers = npefg // 1000 == 4

        # nel48
        self.n_shells_8_nodes = self.raw_header["nel48"]

        # idtdt
        self.has_node_temperature_gradient = get_digit(self.raw_header["idtdt"], 0) == 1
        self.has_node_residual_forces = get_digit(self.raw_header["idtdt"], 1) == 1
        self.has_node_residual_moments = self.has_node_residual_forces
        self.has_solid_shell_plastic_strain_tensor = get_digit(self.raw_header["idtdt"], 2) == 1
        self.has_solid_shell_thermal_strain_tensor = get_digit(self.raw_header["idtdt"], 3) == 1
        if self.raw_header["idtdt"] > 100:
            self.has_element_strain = get_digit(self.raw_header["idtdt"], 4) == 1
        else:
            # took a 1000 years to figure this out ...
            # Warning: 4 gaussian points are not considered
            if self.n_shell_vars > 0:
                if (
                    self.n_shell_vars
                    - self.n_shell_tshell_layers
                    * (
                        6 * self.has_shell_tshell_stress
                        + self.has_shell_tshell_pstrain
                        + self.n_shell_tshell_history_vars
                    )
                    - 8 * self.has_shell_forces
                    - 4 * self.has_shell_extra_variables
                ) > 1:
                    self.has_element_strain = True
                # else:
                # self.has_element_strain = False
            elif self.n_thick_shell_vars > 0:
                if (
                    self.n_thick_shell_vars
                    - self.n_shell_tshell_layers
                    * (
                        6 * self.has_shell_tshell_stress
                        + self.has_shell_tshell_pstrain
                        + self.n_shell_tshell_history_vars
                    )
                ) > 1:
                    self.has_element_strain = True
                # else:
                #     self.has_element_strain = False
            # else:
            #     self.has_element_strain = False

        # internal energy
        # shell_vars_behind_layers = (self.n_shell_vars -
        #                             (self.n_shell_tshell_layers * (
        #                                 6 * self.has_shell_tshell_stress +
        #                                 self.has_shell_tshell_pstrain +
        #                                 self.n_shell_tshell_history_vars) +
        #                                 8 * self.has_shell_forces
        #                                 + 4 * self.has_shell_extra_variables))

        # if not self.has_element_strain:
        #     if shell_vars_behind_layers > 1 and shell_vars_behind_layers < 6:
        #         self.has_shell_internal_energy = True
        #     else:
        #         self.has_shell_internal_energy = False
        # elif self.has_element_strain:
        #     if shell_vars_behind_layers > 12:
        #         self.has_shell_internal_energy = True
        #     else:
        #         self.has_shell_internal_energy = False

        # nel20
        if "nel20" in self.raw_header:
            self.n_solids_20_node_hexas = self.raw_header["nel20"]

        # nt3d
        if "nt3d" in self.raw_header:
            self.n_solid_thermal_vars = self.raw_header["nt3d"]

        # nel27
        if "nel27" in self.raw_header:
            self.n_solids_27_node_hexas = self.raw_header["nel27"]

        # neipb
        if "neipb" in self.raw_header:
            self.n_beam_history_vars = self.raw_header["neipb"]

        # nel21p
        if "nel21p" in self.raw_header:
            self.n_solids_21_node_pentas = self.raw_header["nel21p"]

        # nel15t
        if "nel15t" in self.raw_header:
            self.n_solids_15_node_tetras = self.raw_header["nel15t"]

        # soleng
        if "soleng" in self.raw_header:
            self.has_solid_internal_energy_density = self.raw_header["soleng"]

        # nel20t
        if "nel20t" in self.raw_header:
            self.n_solids_20_node_tetras = self.raw_header["nel20t"]

        # nel40p
        if "nel40p" in self.raw_header:
            self.n_solids_40_node_pentas = self.raw_header["nel40p"]

        # nel64
        if "nel64" in self.raw_header:
            self.n_solids_64_node_hexas = self.raw_header["nel64"]

        # quadr
        if "quadr" in self.raw_header:
            quadr = self.raw_header["quadr"]
            if quadr == 1:
                self.quadratic_elems_has_full_connectivity = True
            elif quadr == 2:
                self.quadratic_elems_has_full_connectivity = True
                self.quadratic_elems_has_data_at_integration_points = True

        # cubic
        if "cubic" in self.raw_header:
            self.has_cubic_solids = self.raw_header["cubic"] != 0

        # tsheng
        if "tsheng" in self.raw_header:
            self.has_thick_shell_energy_density = self.raw_header["tsheng"] != 0

        # nbranch
        if "nbranch" in self.raw_header:
            self.n_post_branches = self.raw_header["nbranch"]

        # penout
        if "penout" in self.raw_header:
            penout = self.raw_header["penout"]
            if penout == 1:
                self.has_node_max_contact_penetration_absolute = True
            if penout == 2:
                self.has_node_max_contact_penetration_absolute = True
                self.has_node_max_contact_penetration_relative = True

        # engout
        if "engout" in self.raw_header:
            self.has_node_contact_energy_density = self.raw_header["engout"] == 1

        return self

    @property
    def has_femzip_indicator(self) -> bool:
        """If the femzip indicator can be found in the header

        Notes
        -----
            Only use on raw files.

            If the header displays a femzip indicator then the file
            is femzipped. If you load the femzip file as such then
            this indicator will not be set, since femzip itself
            corrects the indicator again.
        """
        if "nmmat" in self.raw_header:
            return self.raw_header["nmmat"] == 76_893_465
        return False

    @property
    def n_rigid_wall_vars(self) -> int:
        """number of rigid wall vars

        Notes
        -----
            Depends on lsdyna version.
        """
        return 4 if self.version >= 971 else 1

    @property
    def n_solid_layers(self) -> int:
        """number of solid layers

        Returns
        -------
        n_solid_layers: int
        """
        n_solid_base_vars = (
            6 * self.has_solid_stress + self.has_solid_pstrain + self.n_solid_history_vars
        )

        return 8 if self.n_solid_vars // max(n_solid_base_vars, 1) >= 8 else 1

    def read_words(self, bb: BinaryBuffer, words_to_read: dict, storage_dict: dict = None):
        """Read several words described by a dict

        Parameters
        ----------
        bb: BinaryBuffer
        words_to_read: dict
            this dict describes the words to be read. One entry
            must be a tuple of len two (byte position and dtype)
        storage_dict: dict
            in this dict the read words will be saved

        Returns
        -------
        storage_dict: dict
            the storage dict given as arg or a new dict if none was given
        """

        if storage_dict is None:
            storage_dict = {}

        for name, data in words_to_read.items():

            # check buffer length
            if data[0] >= len(bb):
                continue

            # read data
            if data[1] == self.itype:
                storage_dict[name] = int(bb.read_number(data[0], data[1]))
            elif data[1] == self.ftype:
                storage_dict[name] = float(bb.read_number(data[0], data[1]))
            elif data[1] == str:
                try:
                    storage_dict[name] = bb.read_text(data[0], data[2])
                except UnicodeDecodeError:
                    storage_dict[name] = ""

            else:
                raise RuntimeError(f"Encountered unknown dtype {str(data[1])} during reading.")

        return storage_dict

    @staticmethod
    def _determine_file_settings(
        bb: Union[BinaryBuffer, None] = None
    ) -> Tuple[int, Union[np.int32, np.int64], Union[np.float32, np.float64]]:
        """Determine the precision of the file

        Parameters
        ----------
        bb: Union[BinaryBuffer, None]
            binary buffer from the file

        Returns
        -------
        wordsize: int
            size of each word in bytes
        itype: np.dtype
            type of integers
        ftype: np.dtype
            type of floats
        """

        LOGGER.debug("_determine_file_settings")

        word_size = 4
        itype = np.int32
        ftype = np.float32

        # test file type flag (1=d3plot, 5=d3part, 11=d3eigv)

        if isinstance(bb, BinaryBuffer):

            # single precision
            value = bb.read_number(44, np.int32)
            if value > 1000:
                value -= 1000
            if value in (
                D3plotFiletype.D3PLOT.value,
                D3plotFiletype.D3PART.value,
                D3plotFiletype.D3EIGV.value,
            ):
                word_size = 4
                itype = np.int32
                ftype = np.float32

                LOGGER.debug("wordsize=%d itype=%s ftype=%s", word_size, itype, ftype)
                LOGGER.debug("_determine_file_settings end")

                return word_size, itype, ftype

            # double precision
            value = bb.read_number(88, np.int64)
            if value > 1000:
                value -= 1000
            if value in (
                D3plotFiletype.D3PLOT.value,
                D3plotFiletype.D3PART.value,
                D3plotFiletype.D3EIGV.value,
            ):
                word_size = 8
                itype = np.int64
                ftype = np.float64

                LOGGER.debug("wordsize=%d itype=%s ftype=%s", word_size, itype, ftype)
                LOGGER.debug("_determine_file_settings end")

                return word_size, itype, ftype

            raise RuntimeError(f"Unknown file type '{value}'.")

        LOGGER.debug("wordsize=%d itype=%s ftype=%s", word_size, itype, ftype)
        LOGGER.debug("_determine_file_settings end")

        return word_size, itype, ftype

    def compare(self, other: "D3plotHeader") -> Dict[str, Tuple[Any, Any]]:
        """Compare two headers and get the differences

        Parameters
        ----------
        other: D3plotHeader
            other d3plot header instance

        Returns
        -------
        differences: Dict[str, Tuple[Any, Any]]
            The different entries of both headers in a dict
        """
        assert isinstance(other, D3plotHeader)

        differences = {}
        names = {*self.raw_header.keys(), *other.raw_header.keys()}
        for name in names:
            value1 = self.raw_header[name] if name in self.raw_header else "missing"
            value2 = other.raw_header[name] if name in self.raw_header else "missing"
            if value1 != value2:
                differences[name] = (value1, value2)

        return differences
