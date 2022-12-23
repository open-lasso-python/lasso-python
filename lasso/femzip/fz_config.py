from typing import Dict, Union, Tuple

import enum


def get_last_int_of_line(line: str) -> Tuple[str, Union[None, int]]:
    """Searches an integer in the line

    Parameters
    ----------
    line: str
        line to be searched

    Returns
    -------
    rest_line: str
        rest of line before match
    number: Union[int, None]
        number or None if not found
    """
    for entry in line.split():
        if entry.isdigit():
            return line[: line.rfind(entry)], int(entry)
    return line, None


class FemzipVariableCategory(enum.Enum):
    """Enum for femzip variable categories

    Attributes
    ----------
    GEOMETRY: int
        -5
    PART: int
        -2
    GLOBAL: int
        -1
    NODE: int
        0
    SOLID: int
        1
    THICK_SHELL: int
        2
    BEAM: int
        3
    TOOLS: int
        4
    SHELL: int
        5
    SPH: int
        6
    FPM: int
        7
    CFD: int
        8
    CPM_FLOAT_VAR: int
        9
    CPM_AIRBAG: int
        10
    CPM_INT_VAR: int
        11
    RADIOSS_STATE_DATA: int
        12
    HEXA20: int
        13
    """

    GEOMETRY = -5
    # REST_OF_HEADER_AND_GEOMETRY_UNCOMPRESSED = -3
    # ALL_STATE_EXCEPT_GEOMETRY_POSITION = -2
    # REST_OF_HEADER_AND_GEOMETRY_COMPRESSED = -1
    # EXTERNAL_NODE_IDS = 1
    # NODE_COORDINATES = 2
    # SOLID_ELEMENT_IDS = 3
    # SOLID_NEIGHBORS = 4
    # SOLID_MATERIALS = 5
    # THICK_SHELLS = (6, 7, 8)
    # BEAMS = (9, 10, 11)
    # TOOL_ELEMENTS = (12, 13, 14)
    # SHELL_ELEMENTS = (15, 16, 17)
    # HEADER_AND_PART_TITLES = -4
    # TIME = -3
    PART = -2
    GLOBAL = -1
    NODE = 0
    SOLID = 1
    THICK_SHELL = 2
    BEAM = 3
    TOOLS = 4
    SHELL = 5
    SPH = 6
    FPM = 7
    CFD = 8
    CPM_FLOAT_VAR = 9
    CPM_AIRBAG = 10
    CPM_INT_VAR = 11
    RADIOSS_STATE_DATA = 12
    HEXA20 = 13

    @staticmethod
    def from_int(number: int) -> "FemzipVariableCategory":
        """Deserializes an integer into an enum

        Parameters
        ----------
        number: int
            number to turn into an enum

        Returns
        -------
        enum_value: FemzipVariableCategory
        """
        if number not in FEMZIP_CATEGORY_TRANSL_DICT:
            err_msg = f"Error: Unknown femzip variable category: '{number}'"
            raise RuntimeError(err_msg)

        return FEMZIP_CATEGORY_TRANSL_DICT[number]


FEMZIP_CATEGORY_TRANSL_DICT: Dict[int, FemzipVariableCategory] = {
    entry.value: entry for entry in FemzipVariableCategory.__members__.values()
}


class FemzipArrayType(enum.Enum):
    """Enum for femzip array types"""

    GLOBAL_DATA = "global"
    PART_RESULTS = "Parts: Energies and others"
    # nodes
    NODE_DISPLACEMENT = "coordinates"
    NODE_TEMPERATURES = "temperatures"
    NODE_ACCELERATIONS = "accelerations"
    NODE_HEAT_FLUX = "heat_flux"
    NODE_MASS_SCALING = "mass_scaling"
    NODE_TEMPERATURE_GRADIENT = "dtdt"
    NODE_VELOCITIES = "velocities"

    # beam
    BEAM_S_SHEAR_RESULTANT = "s_shear_resultant"
    BEAM_T_SHEAR_RESULTANT = "t_shear_resultant"
    BEAM_S_BENDING_MOMENT = "s_bending_moment"
    BEAM_T_BENDING_MOMENT = "t_bending_moment"
    BEAM_AXIAL_FORCE = "axial_force"
    BEAM_TORSIONAL_MOMENT = "torsional_resultant"
    BEAM_AXIAL_STRESS = "axial_stress"
    BEAM_SHEAR_STRESS_RS = "RS_shear_stress"
    BEAM_SHEAR_STRESS_TR = "TR_shear_stress"
    BEAM_PLASTIC_STRAIN = "plastic_strain"
    BEAM_AXIAL_STRAIN = "axial_strain"

    # airbag
    AIRBAG_STATE_GEOM = "CPMs_state_geometry"
    AIRBAG_PARTICLE_POS_X = "Pos x"
    AIRBAG_PARTICLE_POS_Y = "Pos y"
    AIRBAG_PARTICLE_POS_Z = "Pos z"
    AIRBAG_PARTICLE_VEL_X = "Vel x"
    AIRBAG_PARTICLE_VEL_Y = "Vel y"
    AIRBAG_PARTICLE_VEL_Z = "Vel z"
    AIRBAG_PARTICLE_MASS = "Mass"
    AIRBAG_PARTICLE_RADIUS = "Radius"
    AIRBAG_PARTICLE_SPIN_ENERGY = "Spin En"
    AIRBAG_PARTICLE_TRAN_ENERGY = "Tran En"
    AIRBAG_PARTICLE_NEIGHBOR_DIST = "NS dist"
    AIRBAG_PARTICLE_GAS_CHAMBER_ID = "GasC ID"
    AIRBAG_PARTICLE_CHAMBER_ID = "Cham ID"
    AIRBAG_PARTICLE_LEAKAGE = "Leakage"

    STRESS_X = "Sigma-x"
    STRESS_Y = "Sigma-y"
    STRESS_Z = "Sigma-z"
    STRESS_XY = "Sigma-xy"
    STRESS_YZ = "Sigma-yz"
    STRESS_XZ = "Sigma-zx"
    EFF_PSTRAIN = "Effective plastic strain"
    HISTORY_VARS = "extra_value_per_element"
    BENDING_MOMENT_MX = "bending_moment Mx"
    BENDING_MOMENT_MY = "bending_moment My"
    BENDING_MOMENT_MXY = "bending_moment Mxy"
    SHEAR_FORCE_X = "shear_resultant Qx"
    SHEAR_FORCE_Y = "shear_resultant Qy"
    NORMAL_FORCE_X = "normal_resultant Nx"
    NORMAL_FORCE_Y = "normal_resultant Ny"
    NORMAL_FORCE_XY = "normal_resultant Nxy"
    THICKNESS = "thickness"
    UNKNOWN_1 = "element_dependent_variable_1"
    UNKNOWN_2 = "element_dependent_variable_2"
    STRAIN_INNER_X = "Epsilon-x  (inner)"
    STRAIN_INNER_Y = "Epsilon-y  (inner)"
    STRAIN_INNER_Z = "Epsilon-z  (inner)"
    STRAIN_INNER_XY = "Epsilon-xy (inner)"
    STRAIN_INNER_YZ = "Epsilon-yz (inner)"
    STRAIN_INNER_XZ = "Epsilon-zx (inner)"
    STRAIN_OUTER_X = "Epsilon-x (outer)"
    STRAIN_OUTER_Y = "Epsilon-y (outer)"
    STRAIN_OUTER_Z = "Epsilon-z (outer)"
    STRAIN_OUTER_XY = "Epsilon-xy (outer)"
    STRAIN_OUTER_YZ = "Epsilon-yz (outer)"
    STRAIN_OUTER_XZ = "Epsilon-zx (outer)"
    INTERNAL_ENERGY = "internal_energy"

    STRAIN_X = "Epsilon-x (IP    1)"
    STRAIN_Y = "Epsilon-y (IP    1)"
    STRAIN_Z = "Epsilon-z (IP    1)"
    STRAIN_XY = "Epsilon-xy (IP    1)"
    STRAIN_YZ = "Epsilon-yz (IP    1)"
    STRAIN_XZ = "Epsilon-zx (IP    1)"

    @staticmethod
    def from_string(femzip_name: str) -> "FemzipArrayType":
        """Converts a variable name to an array type string

        Parameters
        ----------
        femzip_name: str
            name of the variable given by femzip

        Returns
        -------
        femzip_array_type: FemzipArrayType
        """
        for fz_array_type in FemzipArrayType.__members__.values():
            if fz_array_type.value in femzip_name.strip():
                return fz_array_type

        err_msg = "Unknown femzip variable name: '{0}'"
        raise ValueError(err_msg.format(femzip_name))
