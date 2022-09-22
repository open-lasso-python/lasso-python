from typing import Dict, Union, Tuple

import enum


def get_last_int_of_line(line: str) -> Tuple[str, Union[None, int]]:
    """ Searches an integer in the line

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
            return line[:line.rfind(entry)], int(entry)
    return line, None


class FemzipVariableCategory(enum.Enum):
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
    def from_int(number: int) -> 'FemzipVariableCategory':
        if number not in FEMZIP_CATEGORY_TRANSL_DICT:
            err_msg = "Error: Unknown femzip variable category: '{0}'"
            raise RuntimeError(err_msg.format(number))

        return FEMZIP_CATEGORY_TRANSL_DICT[number]


FEMZIP_CATEGORY_TRANSL_DICT: Dict[int, FemzipVariableCategory] = {
    entry.value: entry for entry in FemzipVariableCategory.__members__.values()
}


class FemzipArrayType(enum.Enum):
    global_data = "global"
    part_results = "Parts: Energies and others"
    # nodes
    node_displacement = "coordinates"
    node_temperatures = "temperatures"
    node_accelerations = "accelerations"
    node_heat_flux = "heat_flux"
    node_mass_scaling = "mass_scaling"
    node_temperature_gradient = "dtdt"
    node_velocities = "velocities"

    # beam
    beam_s_shear_resultant = "s_shear_resultant"
    beam_t_shear_resultant = "t_shear_resultant"
    beam_s_bending_moment = "s_bending_moment"
    beam_t_bending_moment = "t_bending_moment"
    beam_axial_force = "axial_force"
    beam_torsional_moment = "torsional_resultant"
    beam_axial_stress = "axial_stress"
    beam_shear_stress_rs = "RS_shear_stress"
    beam_shear_stress_tr = "TR_shear_stress"
    beam_plastic_strain = "plastic_strain"
    beam_axial_strain = "axial_strain"

    # airbag
    airbag_state_geom = "CPMs_state_geometry"
    airbag_particle_pos_x = "Pos x"
    airbag_particle_pos_y = "Pos y"
    airbag_particle_pos_z = "Pos z"
    airbag_particle_vel_x = "Vel x"
    airbag_particle_vel_y = "Vel y"
    airbag_particle_vel_z = "Vel z"
    airbag_particle_mass = "Mass"
    airbag_particle_radius = "Radius"
    airbag_particle_spin_energy = 'Spin En'
    airbag_particle_tran_energy = 'Tran En'
    airbag_particle_neighbor_dist = 'NS dist'
    airbag_particle_gas_chamber_id = 'GasC ID'
    airbag_particle_chamber_id = 'Cham ID'
    airbag_particle_leakage = 'Leakage'

    stress_x = "Sigma-x"
    stress_y = "Sigma-y"
    stress_z = "Sigma-z"
    stress_xy = "Sigma-xy"
    stress_yz = "Sigma-yz"
    stress_xz = "Sigma-zx"
    eff_pstrain = "Effective plastic strain"
    history_vars = "extra_value_per_element"
    bending_moment_mx = "bending_moment Mx"
    bending_moment_my = "bending_moment My"
    bending_moment_mxy = "bending_moment Mxy"
    shear_force_x = "shear_resultant Qx"
    shear_force_y = "shear_resultant Qy"
    normal_force_x = "normal_resultant Nx"
    normal_force_y = "normal_resultant Ny"
    normal_force_xy = "normal_resultant Nxy"
    thickness = "thickness"
    unknown_1 = "element_dependent_variable_1"
    unknown_2 = "element_dependent_variable_2"
    strain_inner_x = "Epsilon-x  (inner)"
    strain_inner_y = "Epsilon-y  (inner)"
    strain_inner_z = "Epsilon-z  (inner)"
    strain_inner_xy = "Epsilon-xy (inner)"
    strain_inner_yz = "Epsilon-yz (inner)"
    strain_inner_xz = "Epsilon-zx (inner)"
    strain_outer_x = "Epsilon-x (outer)"
    strain_outer_y = "Epsilon-y (outer)"
    strain_outer_z = "Epsilon-z (outer)"
    strain_outer_xy = "Epsilon-xy (outer)"
    strain_outer_yz = "Epsilon-yz (outer)"
    strain_outer_xz = "Epsilon-zx (outer)"
    internal_energy = "internal_energy"

    strain_x = "Epsilon-x (IP    1)"
    strain_y = "Epsilon-y (IP    1)"
    strain_z = "Epsilon-z (IP    1)"
    strain_xy = "Epsilon-xy (IP    1)"
    strain_yz = "Epsilon-yz (IP    1)"
    strain_xz = "Epsilon-zx (IP    1)"

    @staticmethod
    def from_string(femzip_name: str) -> 'FemzipArrayType':
        """ Converts a variable name to a array type string

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
