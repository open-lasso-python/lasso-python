import logging
import re
import traceback
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from lasso.dyna.array_type import ArrayType
from lasso.femzip.femzip_api import FemzipAPI, FemzipFileMetadata, VariableInfo
from lasso.femzip.fz_config import FemzipArrayType, FemzipVariableCategory, get_last_int_of_line


TRANSL_FEMZIP_ARRATYPE_TO_D3PLOT_ARRAYTYPE: Dict[
    Tuple[FemzipArrayType, FemzipVariableCategory], Set[str]
] = {
    # GLOBAL
    (FemzipArrayType.GLOBAL_DATA, FemzipVariableCategory.GLOBAL): {
        # ArrayType.global_timesteps,
        ArrayType.global_internal_energy,
        ArrayType.global_kinetic_energy,
        ArrayType.global_total_energy,
        ArrayType.global_velocity,
    },
    # PART
    (FemzipArrayType.PART_RESULTS, FemzipVariableCategory.PART): {
        ArrayType.part_hourglass_energy,
        ArrayType.part_internal_energy,
        ArrayType.part_kinetic_energy,
        ArrayType.part_mass,
        ArrayType.part_velocity,
    },
    # NODE
    (FemzipArrayType.NODE_DISPLACEMENT, FemzipVariableCategory.NODE): {ArrayType.node_displacement},
    (FemzipArrayType.NODE_ACCELERATIONS, FemzipVariableCategory.NODE): {
        ArrayType.node_acceleration
    },
    (FemzipArrayType.NODE_VELOCITIES, FemzipVariableCategory.NODE): {ArrayType.node_velocity},
    (FemzipArrayType.NODE_TEMPERATURES, FemzipVariableCategory.NODE): {ArrayType.node_temperature},
    (FemzipArrayType.NODE_HEAT_FLUX, FemzipVariableCategory.NODE): {ArrayType.node_heat_flux},
    (FemzipArrayType.NODE_MASS_SCALING, FemzipVariableCategory.NODE): {ArrayType.node_mass_scaling},
    (FemzipArrayType.NODE_TEMPERATURE_GRADIENT, FemzipVariableCategory.NODE): {
        ArrayType.node_temperature_gradient
    },
    # BEAM
    (FemzipArrayType.BEAM_AXIAL_FORCE, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_axial_force
    },
    (FemzipArrayType.BEAM_S_BENDING_MOMENT, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_bending_moment
    },
    (FemzipArrayType.BEAM_T_BENDING_MOMENT, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_bending_moment
    },
    (FemzipArrayType.BEAM_S_SHEAR_RESULTANT, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_shear_force
    },
    (FemzipArrayType.BEAM_T_SHEAR_RESULTANT, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_shear_force
    },
    (FemzipArrayType.BEAM_TORSIONAL_MOMENT, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_torsion_moment
    },
    (FemzipArrayType.BEAM_AXIAL_STRESS, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_axial_stress
    },
    (FemzipArrayType.BEAM_SHEAR_STRESS_RS, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_shear_stress
    },
    (FemzipArrayType.BEAM_SHEAR_STRESS_TR, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_shear_stress
    },
    (FemzipArrayType.BEAM_PLASTIC_STRAIN, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_plastic_strain
    },
    (FemzipArrayType.BEAM_AXIAL_STRAIN, FemzipVariableCategory.BEAM): {
        ArrayType.element_beam_axial_strain
    },
    # SHELL
    (FemzipArrayType.STRESS_X, FemzipVariableCategory.SHELL): {ArrayType.element_shell_stress},
    (FemzipArrayType.STRESS_Y, FemzipVariableCategory.SHELL): {ArrayType.element_shell_stress},
    (FemzipArrayType.STRESS_Z, FemzipVariableCategory.SHELL): {ArrayType.element_shell_stress},
    (FemzipArrayType.STRESS_XY, FemzipVariableCategory.SHELL): {ArrayType.element_shell_stress},
    (FemzipArrayType.STRESS_YZ, FemzipVariableCategory.SHELL): {ArrayType.element_shell_stress},
    (FemzipArrayType.STRESS_XZ, FemzipVariableCategory.SHELL): {ArrayType.element_shell_stress},
    (FemzipArrayType.EFF_PSTRAIN, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_effective_plastic_strain
    },
    (FemzipArrayType.HISTORY_VARS, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_history_vars
    },
    (FemzipArrayType.BENDING_MOMENT_MX, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_bending_moment
    },
    (FemzipArrayType.BENDING_MOMENT_MY, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_bending_moment
    },
    (FemzipArrayType.BENDING_MOMENT_MXY, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_bending_moment
    },
    (FemzipArrayType.SHEAR_FORCE_X, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_shear_force
    },
    (FemzipArrayType.SHEAR_FORCE_Y, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_shear_force
    },
    (FemzipArrayType.NORMAL_FORCE_X, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_normal_force
    },
    (FemzipArrayType.NORMAL_FORCE_Y, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_normal_force
    },
    (FemzipArrayType.NORMAL_FORCE_XY, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_normal_force
    },
    (FemzipArrayType.THICKNESS, FemzipVariableCategory.SHELL): {ArrayType.element_shell_thickness},
    (FemzipArrayType.UNKNOWN_1, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_unknown_variables
    },
    (FemzipArrayType.UNKNOWN_2, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_unknown_variables
    },
    (FemzipArrayType.STRAIN_INNER_X, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_INNER_Y, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_INNER_Z, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_INNER_XY, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_INNER_YZ, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_INNER_XZ, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_X, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_Y, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_Z, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_XY, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_YZ, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_XZ, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_strain
    },
    (FemzipArrayType.INTERNAL_ENERGY, FemzipVariableCategory.SHELL): {
        ArrayType.element_shell_internal_energy
    },
    # THICK SHELL
    (FemzipArrayType.STRESS_X, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_stress
    },
    (FemzipArrayType.STRESS_Y, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_stress
    },
    (FemzipArrayType.STRESS_Z, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_stress
    },
    (FemzipArrayType.STRESS_XY, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_stress
    },
    (FemzipArrayType.STRESS_YZ, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_stress
    },
    (FemzipArrayType.STRESS_XZ, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_stress
    },
    (FemzipArrayType.EFF_PSTRAIN, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_effective_plastic_strain
    },
    (FemzipArrayType.STRAIN_OUTER_X, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_Y, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_Z, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_XY, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_YZ, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_OUTER_XZ, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_INNER_X, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_INNER_Y, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_INNER_Z, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_INNER_XY, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_INNER_YZ, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    (FemzipArrayType.STRAIN_INNER_XZ, FemzipVariableCategory.THICK_SHELL): {
        ArrayType.element_tshell_strain
    },
    # SOLID
    (FemzipArrayType.STRESS_X, FemzipVariableCategory.SOLID): {ArrayType.element_solid_stress},
    (FemzipArrayType.STRESS_Y, FemzipVariableCategory.SOLID): {ArrayType.element_solid_stress},
    (FemzipArrayType.STRESS_Z, FemzipVariableCategory.SOLID): {ArrayType.element_solid_stress},
    (FemzipArrayType.STRESS_XY, FemzipVariableCategory.SOLID): {ArrayType.element_solid_stress},
    (FemzipArrayType.STRESS_YZ, FemzipVariableCategory.SOLID): {ArrayType.element_solid_stress},
    (FemzipArrayType.STRESS_XZ, FemzipVariableCategory.SOLID): {ArrayType.element_solid_stress},
    (FemzipArrayType.EFF_PSTRAIN, FemzipVariableCategory.SOLID): {
        ArrayType.element_solid_effective_plastic_strain
    },
    (FemzipArrayType.HISTORY_VARS, FemzipVariableCategory.SOLID): {
        ArrayType.element_solid_history_variables
    },
    (FemzipArrayType.STRAIN_X, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_Y, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_Z, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_XY, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_YZ, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_XZ, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_X, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_Y, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_Z, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_XY, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_YZ, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    (FemzipArrayType.STRAIN_XZ, FemzipVariableCategory.SOLID): {ArrayType.element_solid_strain},
    # AIRBAG
    (FemzipArrayType.AIRBAG_STATE_GEOM, FemzipVariableCategory.CPM_AIRBAG): {
        ArrayType.airbag_n_active_particles,
        ArrayType.airbag_bag_volume,
    },
    # AIRBAG PARTICLES
    (FemzipArrayType.AIRBAG_PARTICLE_GAS_CHAMBER_ID, FemzipVariableCategory.CPM_INT_VAR): {
        ArrayType.airbag_particle_gas_id
    },
    (FemzipArrayType.AIRBAG_PARTICLE_CHAMBER_ID, FemzipVariableCategory.CPM_INT_VAR): {
        ArrayType.airbag_particle_chamber_id
    },
    (FemzipArrayType.AIRBAG_PARTICLE_LEAKAGE, FemzipVariableCategory.CPM_INT_VAR): {
        ArrayType.airbag_particle_leakage
    },
    (FemzipArrayType.AIRBAG_PARTICLE_MASS, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_mass
    },
    (FemzipArrayType.AIRBAG_PARTICLE_POS_X, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_position
    },
    (FemzipArrayType.AIRBAG_PARTICLE_POS_Y, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_position
    },
    (FemzipArrayType.AIRBAG_PARTICLE_POS_Z, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_position
    },
    (FemzipArrayType.AIRBAG_PARTICLE_VEL_X, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_velocity
    },
    (FemzipArrayType.AIRBAG_PARTICLE_VEL_Y, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_velocity
    },
    (FemzipArrayType.AIRBAG_PARTICLE_VEL_Z, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_velocity
    },
    (FemzipArrayType.AIRBAG_PARTICLE_RADIUS, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_radius
    },
    (FemzipArrayType.AIRBAG_PARTICLE_SPIN_ENERGY, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_spin_energy
    },
    (FemzipArrayType.AIRBAG_PARTICLE_TRAN_ENERGY, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_translation_energy
    },
    (FemzipArrayType.AIRBAG_PARTICLE_NEIGHBOR_DIST, FemzipVariableCategory.CPM_FLOAT_VAR): {
        ArrayType.airbag_particle_nearest_segment_distance
    },
}

# indexes for various femzip arrays
stress_index = {
    FemzipArrayType.STRESS_X.value: 0,
    FemzipArrayType.STRESS_Y.value: 1,
    FemzipArrayType.STRESS_Z.value: 2,
    FemzipArrayType.STRESS_XY.value: 3,
    FemzipArrayType.STRESS_YZ.value: 4,
    FemzipArrayType.STRESS_XZ.value: 5,
    FemzipArrayType.NORMAL_FORCE_X.value: 0,
    FemzipArrayType.NORMAL_FORCE_Y.value: 1,
    FemzipArrayType.NORMAL_FORCE_XY.value: 2,
    FemzipArrayType.SHEAR_FORCE_X.value: 0,
    FemzipArrayType.SHEAR_FORCE_Y.value: 1,
    FemzipArrayType.STRAIN_INNER_X.value: 0,
    FemzipArrayType.STRAIN_INNER_Y.value: 1,
    FemzipArrayType.STRAIN_INNER_Z.value: 2,
    FemzipArrayType.STRAIN_INNER_XY.value: 3,
    FemzipArrayType.STRAIN_INNER_YZ.value: 4,
    FemzipArrayType.STRAIN_INNER_XZ.value: 5,
    FemzipArrayType.STRAIN_OUTER_X.value: 0,
    FemzipArrayType.STRAIN_OUTER_Y.value: 1,
    FemzipArrayType.STRAIN_OUTER_Z.value: 2,
    FemzipArrayType.STRAIN_OUTER_XY.value: 3,
    FemzipArrayType.STRAIN_OUTER_YZ.value: 4,
    FemzipArrayType.STRAIN_OUTER_XZ.value: 5,
    FemzipArrayType.BEAM_S_SHEAR_RESULTANT.value: 0,
    FemzipArrayType.BEAM_T_SHEAR_RESULTANT.value: 1,
    FemzipArrayType.BEAM_S_BENDING_MOMENT.value: 0,
    FemzipArrayType.BEAM_T_BENDING_MOMENT.value: 1,
    FemzipArrayType.STRAIN_X.value: 0,
    FemzipArrayType.STRAIN_Y.value: 1,
    FemzipArrayType.STRAIN_Z.value: 2,
    FemzipArrayType.STRAIN_XY.value: 3,
    FemzipArrayType.STRAIN_YZ.value: 4,
    FemzipArrayType.STRAIN_XZ.value: 5,
    FemzipArrayType.BEAM_SHEAR_STRESS_RS.value: 0,
    FemzipArrayType.BEAM_SHEAR_STRESS_TR.value: 1,
    FemzipArrayType.AIRBAG_PARTICLE_POS_X.value: 0,
    FemzipArrayType.AIRBAG_PARTICLE_POS_Y.value: 1,
    FemzipArrayType.AIRBAG_PARTICLE_POS_Z.value: 2,
    FemzipArrayType.AIRBAG_PARTICLE_VEL_X.value: 0,
    FemzipArrayType.AIRBAG_PARTICLE_VEL_Y.value: 1,
    FemzipArrayType.AIRBAG_PARTICLE_VEL_Z.value: 2,
    FemzipArrayType.BENDING_MOMENT_MX.value: 0,
    FemzipArrayType.BENDING_MOMENT_MY.value: 1,
    FemzipArrayType.BENDING_MOMENT_MXY.value: 2,
    FemzipArrayType.UNKNOWN_1.value: 0,
    FemzipArrayType.UNKNOWN_2.value: 1,
}


def femzip_to_d3plot(
    result_arrays: Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray]
) -> Dict[str, np.ndarray]:
    """Map femzip arrays to d3plot arrays

    Parameters
    ----------
    result_arrays:
        femzip arrays
    """
    mapper = FemzipMapper()
    mapper.map(result_arrays)

    return mapper.d3plot_arrays


class ArrayShapeInfo:
    """ArrayShapeInfo describes the shape of arrays"""

    n_layers: Union[int, None] = None
    n_vars: Union[int, None] = None
    n_entries: Union[int, None] = None
    n_timesteps: Union[int, None] = None

    def _set_attr(self, attr_name: str, value: Union[int, None]) -> None:
        self_attr_value = getattr(self, attr_name)
        if value is not None:
            if self_attr_value is None:
                setattr(self, attr_name, value)
            else:
                setattr(self, attr_name, max(self_attr_value, value))

    def set_n_layers(self, n_layers: Union[int, None]) -> None:
        """Set the number of layers

        Parameters
        ----------
        n_layers : Union[int, None]
            number of layers
        """
        self._set_attr("n_layers", n_layers)

    def set_n_vars(self, n_vars: Union[int, None]) -> None:
        """Set the number of variables

        Parameters
        ----------
        n_vars : Union[int, None]
            number of variables
        """
        self._set_attr("n_vars", n_vars)

    def set_n_entries(self, n_entries: Union[int, None]) -> None:
        """Set the number of entries

        Parameters
        ----------
        n_vars : Union[int, None]
            number of entries
        """
        self._set_attr("n_entries", n_entries)

    def set_n_timesteps(self, n_timesteps: Union[int, None]) -> None:
        """Set the number of timesteps

        Parameters
        ----------
        n_vars : Union[int, None]
            number of timesteps
        """
        self._set_attr("n_timesteps", n_timesteps)

    def to_shape(self) -> Tuple[int, ...]:
        """Set the number of variables

        Returns
        ----------
        shape : Tuple[int, ...]
            total shape
        """

        shape = [self.n_timesteps, self.n_entries]
        fortran_offset = 1
        if self.n_layers is not None:
            shape.append(self.n_layers + fortran_offset)
        if self.n_vars is not None:
            shape.append(self.n_vars + fortran_offset)
        return tuple(shape)


class D3plotArrayMapping:
    """D3plotArrayMapping maps femzip arrays to d3plot arrays"""

    d3plot_array_type: str
    d3_layer_slice: Union[slice, int, None] = None
    d3_var_slice: Union[slice, int, None] = None

    fz_layer_slice: Union[slice, int, None] = None
    fz_var_slice: Union[slice, int, None] = None

    just_assign: bool = False

    def to_slice(self) -> Tuple[Union[int, slice], ...]:
        """Get the slices mapping a femzip array to a d3plot array

        Returns
        -------
        slices: Tuple[Union[int, slice], ...]
        """
        slices: List[Union[slice, int]] = [slice(None), slice(None)]
        if self.d3_layer_slice is not None:
            slices.append(self.d3_layer_slice)
        if self.d3_var_slice is not None:
            slices.append(self.d3_var_slice)
        return tuple(slices)


class FemzipArrayInfo:
    """FemzipArrayInfo contains information about the femzip array"""

    # pylint: disable = too-many-instance-attributes

    full_name: str = ""
    short_name: str = ""
    index: int = -1
    category: FemzipVariableCategory
    array_type: FemzipArrayType
    array: np.ndarray

    i_layer: Union[int, None] = None
    i_var: Union[int, None] = None

    mappings: List[D3plotArrayMapping]

    def __init__(self):
        self.mappings = []

    def __str__(self) -> str:
        return f"""FemzipArrayInfo:
    full_name  = {self.full_name}
    short_name = {self.short_name}
    index    = {self.index}
    category = {self.category}
    array_type = {self.array_type}>
    i_layer = {self.i_layer}
    i_var   = {self.i_var}"""


class FemzipMapper:
    """Class for mapping femzip variable data to d3plots."""

    # regex pattern for reading variables
    name_separation_pattern = re.compile(r"(^[^(\n]+)(\([^\)]+\))*")

    FORTRAN_OFFSET: int = 1

    _d3plot_arrays: Dict[str, np.ndarray] = {}
    _fz_array_slices = {}

    def __init__(self):
        pass

    def map(self, result_arrays: Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray]):
        """Map femzip data to d3plot arrays.

        Parameters
        ----------
        result_arrays:
            femzip variable data
        """
        self._d3plot_arrays = {}
        self._fz_array_slices = {}

        # convert to internal datastructure
        array_infos = self._convert(result_arrays)

        # build the array shapes
        d3plot_array_shapes = self._build(array_infos)

        # init the numpy arrays
        self._d3plot_arrays = self._allocate_d3plot_arrays(d3plot_array_shapes)

        # add all the data to its right place
        self._map_arrays(array_infos, self._d3plot_arrays)

    def _convert(
        self, result_arrays: Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray]
    ) -> List[FemzipArrayInfo]:
        """Convert femzip result arrays into array infos

        Parameters
        ----------
        result_arrays: Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray]
            result arrays from femzip

        Returns
        -------
        array_infos: List[FemzipArrayInfo]
            infos about femzip arrays
        """

        array_infos = []

        # convert
        for (fz_index, fz_name, fz_cat), array in result_arrays.items():
            femzip_array_info = FemzipArrayInfo()
            femzip_array_info.index = fz_index
            femzip_array_info.full_name = fz_name
            femzip_array_info.category = fz_cat
            femzip_array_info.array = array
            femzip_array_info.array_type = FemzipArrayType.from_string(fz_name)

            var_name, i_layer, i_stress, i_history = self._parse_femzip_name(fz_name, fz_cat)

            femzip_array_info.short_name = var_name
            femzip_array_info.i_layer = i_layer
            femzip_array_info.i_var = i_stress if i_stress is not None else i_history

            array_infos.append(femzip_array_info)

        return array_infos

    @staticmethod
    def _build(fz_arrays: List[FemzipArrayInfo]) -> Dict[str, Tuple[int, ...]]:
        """Counts the occurrence of all variables in the result array such as the
        number of layers and stresses.

        Parameters
        ---------
        fz_arrays: List[FemzipArrayInfo]
            infos about femzip arrays

        Returns
        -------
        d3plot_array_shapes:
            shapes of the d3plot arrays required to be allocated

        Notes
        -----
        Some variables only have partial stress results written for Sigma-x and Sigma-y
        and layers one to three for example.
        """
        shape_infos: Dict[str, ArrayShapeInfo] = {}
        name_count: Dict[Tuple[str, FemzipVariableCategory], int] = {}

        for arr_info in fz_arrays:
            # print(arr_info)

            d3_array_types = TRANSL_FEMZIP_ARRATYPE_TO_D3PLOT_ARRAYTYPE[
                (arr_info.array_type, arr_info.category)
            ]

            # var_name = var_name.strip()
            for array_type in d3_array_types:
                # print(array_type)
                array_shape_info = shape_infos.get(array_type) or ArrayShapeInfo()

                # beam layer vars always have same name but
                # must be counted up as layers
                if (arr_info.full_name, arr_info.category) in name_count:
                    count = name_count[(arr_info.full_name, arr_info.category)]
                    i_layer = count + 1
                    name_count[(arr_info.full_name, arr_info.category)] = i_layer
                else:
                    name_count[(arr_info.full_name, arr_info.category)] = 0

                # update shape
                array_shape_info.set_n_timesteps(arr_info.array.shape[0])
                array_shape_info.set_n_entries(arr_info.array.shape[1])
                array_shape_info.set_n_layers(arr_info.i_layer)
                array_shape_info.set_n_vars(arr_info.i_var)

                shape_infos[array_type] = array_shape_info

                # where to put it
                mapping = D3plotArrayMapping()
                mapping.d3plot_array_type = array_type
                if arr_info.i_layer is not None:
                    mapping.d3_layer_slice = arr_info.i_layer
                if arr_info.i_var is not None:
                    mapping.d3_var_slice = arr_info.i_var
                # arrays to copy:
                # - node displacement, velocity, acceleration
                # - airbag integer vars (so we don't need to cast)
                if (
                    arr_info.array.ndim == 3
                    or arr_info.category == FemzipVariableCategory.CPM_INT_VAR
                ):
                    mapping.just_assign = True

                arr_info.mappings.append(mapping)

        # correct layers
        # if a field has the same name for multiple
        # layers such as beam axial stress, we needed
        # to count in order to determine if it had layers
        # now we need to correct i_layers from None to 0 for them
        name_count2 = {}
        for arr_info in fz_arrays:
            count = name_count[(arr_info.full_name, arr_info.category)]

            if count != 0 and arr_info.i_layer is None:
                count2 = name_count2.get((arr_info.full_name, arr_info.category), -1)
                count2 += 1
                arr_info.i_layer = count2
                name_count2[(arr_info.full_name, arr_info.category)] = count2

                for mapping in arr_info.mappings:
                    shape_info = shape_infos[mapping.d3plot_array_type]
                    shape_info.set_n_layers(count)
                    mapping.d3_layer_slice = count2

            # all arrays which are simply copied (slice has len 2 and only one target)
            # get a just assign flag
            if len(arr_info.mappings) == 2 and len(arr_info.mappings[0].to_slice()) == 2:
                arr_info.mappings[0].just_assign = True

                d3_array_types = TRANSL_FEMZIP_ARRATYPE_TO_D3PLOT_ARRAYTYPE[
                    (arr_info.array_type, arr_info.category)
                ]

                for array_type in d3_array_types:
                    del shape_infos[array_type]

        return {name: info.to_shape() for name, info in shape_infos.items()}

    def _map_arrays(self, array_infos: List[FemzipArrayInfo], d3plot_arrays: Dict[str, np.ndarray]):
        """Allocate a femzip variable to its correct position in
        the d3plot array dictionary.

        Parameters
        ---------
        array_infos: List[FemzipArrayInfo]
            femzip variables stored in a dictionary
        d3plot_arrays: Dict[str, np.ndarray]
            d3plot arrays pre-allocated

        Notes
        -----
            The keys are the femzip array name (un-parsed)
            and the category of the variable as an enum.
        """
        for arr_info in array_infos:
            if arr_info.category == FemzipVariableCategory.CPM_AIRBAG:
                d3plot_arrays[ArrayType.airbag_n_active_particles] = arr_info.array[:, :, 0].view(
                    np.int32
                )
                d3plot_arrays[ArrayType.airbag_bag_volume] = arr_info.array[:, :, 1]
            else:
                for mapping in arr_info.mappings:
                    if mapping.just_assign:
                        d3plot_arrays[mapping.d3plot_array_type] = arr_info.array
                        continue

                    slices = mapping.to_slice()
                    d3plot_array = d3plot_arrays[mapping.d3plot_array_type]

                    # for femzip arrays with same name first var_index is missing
                    if d3plot_array.ndim == 3 and len(slices) == 2 and arr_info.array.ndim == 2:
                        slices = (*slices, 0)

                    d3plot_array[slices] = arr_info.array

    def _allocate_d3plot_arrays(
        self, array_shapes: Dict[str, Tuple[int, ...]]
    ) -> Dict[str, np.ndarray]:
        """Initialize all the d3plot arrays.

        Parameters
        ----------
        array_shapes: array_shapes: Dict[str, Tuple[int, ...]]
            array shapes required to be allocated

        Returns
        -------
        d3plot_arrays: Dict[str, np.ndarray]
            d3plot arrays pre-allocated
        """
        d3plot_arrays = {}
        for key, shape in array_shapes.items():
            d3plot_arrays[key] = np.empty(shape, dtype=np.float32)
        return d3plot_arrays

    @property
    def d3plot_arrays(self):
        """Returns the mapped d3plot arrays."""
        return self._d3plot_arrays

    def _parse_femzip_name(
        self, fz_name: str, var_type: FemzipVariableCategory
    ) -> Tuple[str, Union[int, None], Union[int, None], Union[int, None]]:
        """Parses the femzip variable names.

        Parameters
        ----------
        fz_name:
            cryptic femzip variable name we need to parse
        var_type:
            the category of this variable e.g. shells, parts, global etc.

        Returns
        -------
        var_name:
            femzip variable name without integration and layer info
        i_layer:
            layer index
        i_stress:
            stress index
        i_history:
            history variable index
        """
        matches = self.name_separation_pattern.findall(fz_name)
        if not len(matches) == 1:
            err_msg = "Could not match femzip array name: {0}"
            raise ValueError(err_msg.format(fz_name))
        if not len(matches[0]) == 2:
            err_msg = "Could not match femzip array name: {0}"
            raise ValueError(err_msg.format(fz_name))

        (first_grp, second_grp) = matches[0]
        var_name, extra_value = get_last_int_of_line(first_grp)
        var_name = var_name.strip()

        # the slice 1:-1 leaves out the brackets '(' and ')'
        _, i_layer = get_last_int_of_line(second_grp[1:-1])

        if i_layer is not None:
            i_layer -= self.FORTRAN_OFFSET

        i_history: Union[int, None] = None

        if var_type != FemzipVariableCategory.PART or var_type != FemzipVariableCategory.GLOBAL:
            i_history = extra_value

        if i_history:
            i_history -= self.FORTRAN_OFFSET

        # set var name to the un-formatted femzip array type name
        if "Epsilon" in var_name:
            var_name = fz_name.strip()
            if "inner" in var_name:
                i_layer = 0
            elif "outer" in var_name:
                i_layer = 1
            else:
                # solid strain
                i_layer = 0

        i_stress: Union[int, None] = stress_index.get(var_name, None)

        return var_name, i_layer, i_stress, i_history


def filter_femzip_variables(
    file_metadata: FemzipFileMetadata, d3plot_array_filter: Union[Set[str], None]
) -> FemzipFileMetadata:
    """Filters variable infos regarding d3plot array types

    Parameters
    ----------
    file_metadata: FemzipFileMetadata
        metadata of femzip file including contained variables
    d3plot_array_filter: Union[Set[str], None]
        array types to filter for if wanted

    Returns
    -------
    file_metadata: FemzipFileMetadata
        filtered array according to array types
    """

    # pylint: disable = too-many-locals

    # find out which arrays we need and
    vars_to_copy: List[int] = []

    for i_var in range(file_metadata.number_of_variables):
        try:
            var_info: VariableInfo = file_metadata.variable_infos[i_var]
            var_type: int = var_info.var_type
            var_index: int = var_info.var_index
            var_name: str = var_info.name.decode("utf-8")

            logging.debug("%d, %d, %s", var_type, var_index, var_name.strip())

            if var_type == FemzipVariableCategory.GEOMETRY.value:
                continue

            # find out which array from name
            try:
                fz_array_type = FemzipArrayType.from_string(var_name)
            except ValueError:
                warn_msg = (
                    "Warning: lasso-python does not support femzip result"
                    f" field '{var_name.strip()}' category type '{var_type}'."
                )
                logging.warning(warn_msg)
                continue

            # check if we asked for the array
            matching_array_types = TRANSL_FEMZIP_ARRATYPE_TO_D3PLOT_ARRAYTYPE[
                (fz_array_type, FemzipVariableCategory(var_type))
            ]

            if d3plot_array_filter is not None:
                if not matching_array_types.intersection(d3plot_array_filter):
                    continue
            vars_to_copy.append(i_var)
        except Exception:
            trb_msg = traceback.format_exc()
            err_msg = "An error occurred while preprocessing femzip variable information: %s"
            logging.warning(err_msg, trb_msg)

    # copy filtered data
    filtered_file_metadata = FemzipFileMetadata()
    FemzipAPI.copy_struct(file_metadata, filtered_file_metadata)
    filtered_file_metadata.number_of_variables = len(vars_to_copy)

    # pylint: disable = invalid-name
    FilteredVariableInfoArrayType = len(vars_to_copy) * VariableInfo
    filtered_info_array_data = FilteredVariableInfoArrayType()

    for i_var, src_i_var in enumerate(vars_to_copy):
        FemzipAPI.copy_struct(
            file_metadata.variable_infos[src_i_var], filtered_info_array_data[i_var]
        )
    filtered_file_metadata.variable_infos = filtered_info_array_data

    return filtered_file_metadata
