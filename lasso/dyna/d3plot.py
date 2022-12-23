import ctypes
from dataclasses import dataclass
import logging
import mmap
import os
import pprint
import re
import struct
import tempfile
import traceback
import typing
import webbrowser
from typing import Any, BinaryIO, Dict, Iterable, List, Set, Tuple, Union

import numpy as np

from ..femzip.femzip_api import FemzipAPI, FemzipBufferInfo, FemzipVariableCategory
from ..io.binary_buffer import BinaryBuffer
from ..io.files import open_file_or_filepath
from ..logging import get_logger
from ..plotting import plot_shell_mesh
from .array_type import ArrayType
from .d3plot_header import D3plotFiletype, D3plotHeader
from .femzip_mapper import FemzipMapper, filter_femzip_variables
from .filter_type import FilterType

# pylint: disable = too-many-lines

FORTRAN_OFFSET = 1
LOGGER = get_logger(__name__)


def _check_ndim(d3plot, array_dim_names: Dict[str, List[str]]):
    """Checks if the specified array is fine in terms of ndim

    Parameters
    ----------
    d3plot: D3plot
        d3plot holding arrays
    array_dim_names: Dict[str, List[str]]
    """

    for type_name, dim_names in array_dim_names.items():
        if type_name in d3plot.arrays:
            array = d3plot.arrays[type_name]
            if array.ndim != len(dim_names):
                msg = "Array {0} must have {1} instead of {2} dimensions: ({3})"
                dim_names_text = ", ".join(dim_names)
                raise ValueError(msg.format(type_name, len(dim_names), array.ndim, dim_names_text))


def _check_array_occurrence(
    d3plot, array_names: List[str], required_array_names: List[str]
) -> bool:
    """Check if an array exists, if all depending on it exist too

    Parameters
    ----------
    array_names: List[str]
        list of base arrays
    required_array_names: List[str]
        list of array names which would be required

    Returns
    -------
    exists: bool
        if the arrays exist or not

    Raises
    ------
    ValueError
        If a required array is not present
    """

    if any(name in d3plot.arrays for name in array_names):
        if not all(name in d3plot.arrays for name in required_array_names):
            msg = "The arrays '{0}' require setting also the arrays '{1}'"
            raise ValueError(msg.format(", ".join(array_names), ", ".join(required_array_names)))
        return True
    return False


def _negative_to_positive_state_indexes(indexes: Set[int], n_entries) -> Set[int]:
    """Convert negative indexes of an iterable to positive ones

    Parameters
    ----------
    indexes: Set[int]
        indexes to check and convert
    n_entries: int
        total number of entries

    Returns
    -------
    new_entries: Set[int]
        the positive indexes
    """

    new_entries: Set[int] = set()
    for _, index in enumerate(indexes):
        new_index = index + n_entries if index < 0 else index
        if new_index >= n_entries:
            err_msg = "State '{0}' exceeds the maximum number of states of '{1}'"
            raise ValueError(err_msg.format(index, n_entries))
        new_entries.add(new_index)
    return new_entries


# pylint: disable = too-many-instance-attributes
class D3plotWriterSettings:
    """Settings class for d3plot writing"""

    def __init__(self, d3plot: Any, block_size_bytes: int, single_file: bool):

        # check the writing types
        if d3plot.header.itype == np.int32:
            self.itype = "<i"
        elif d3plot.header.itype == np.int64:
            self.itype = "<q"
        else:
            msg = "Invalid type for integers: {0}. np.int32 or np.int64 is required."
            raise RuntimeError(msg.format(d3plot.itype))

        if d3plot.header.ftype == np.float32:
            self.ftype = "<f"
        elif d3plot.header.ftype == np.float64:
            self.ftype = "<d"
        else:
            msg = "Invalid type for floats: {0}. np.float32 or np.float64 is required."
            raise RuntimeError(msg.format(d3plot.ftype))

        assert isinstance(d3plot, D3plot)
        self.d3plot = d3plot
        self._header = {}
        self.block_size_bytes = block_size_bytes
        self.mattyp = 0
        self.single_file = single_file
        self.mdlopt = 0
        self.n_shell_layers = 0
        self.n_rigid_shells = 0
        self.unique_beam_part_indexes = np.empty(0, dtype=self.itype)
        self.unique_shell_part_indexes = np.empty(0, dtype=self.itype)
        self.unique_solid_part_indexes = np.empty(0, dtype=self.itype)
        self.unique_tshell_part_indexes = np.empty(0, dtype=self.itype)
        self._str_codec = "utf-8"
        self.has_node_temperature_gradient = False
        self.has_node_residual_forces = False
        self.has_node_residual_moments = False
        self.has_plastic_strain_tensor = False
        self.has_thermal_strain_tensor = False
        self.n_solid_layers = 1

        self._allowed_int_types = (np.int8, np.int16, np.int32, np.int64, int)
        self._allowed_float_types = (np.float32, np.float64, float)

    @property
    def wordsize(self):
        """Get the wordsize to use for the d3plot

        Returns
        -------
            worsize : int
                D3plot wordsize
        """
        return self.d3plot.header.wordsize

    @property
    def header(self):
        """Dictionary holding all d3plot header information

        Notes
        -----
            The header is being build from the data stored in the d3plot.
        """
        return self._header

    @header.setter
    def set_header(self, new_header: dict):
        assert isinstance(new_header, dict)
        self._header = new_header

    # pylint: disable = too-many-branches, too-many-statements, too-many-locals
    def build_header(self):
        """Build the new d3plot header"""

        new_header = {}

        # TITLE
        new_header["title"] = self.d3plot.header.title
        # RUNTIME
        new_header["runtime"] = self.d3plot.header.runtime
        # FILETYPE
        new_header["filetype"] = self.d3plot.header.filetype.value
        # SOURCE VERSION
        new_header["source_version"] = self.d3plot.header.source_version
        # RELEASE VERSION
        new_header["release_version"] = self.d3plot.header.release_version
        # SOURCE VERSION
        new_header["version"] = self.d3plot.header.version

        # NDIM

        # check for rigid body data
        has_rigid_body_data = False
        has_reduced_rigid_body_data = False
        if (
            ArrayType.rigid_body_coordinates in self.d3plot.arrays
            or ArrayType.rigid_body_rotation_matrix in self.d3plot.arrays
        ):
            has_rigid_body_data = True
            has_reduced_rigid_body_data = True
        if (
            ArrayType.rigid_body_velocity in self.d3plot.arrays
            or ArrayType.rigid_body_rot_velocity in self.d3plot.arrays
            or ArrayType.rigid_body_acceleration in self.d3plot.arrays
            or ArrayType.rigid_body_rot_acceleration in self.d3plot.arrays
        ):
            has_reduced_rigid_body_data = False

        # check for rigid road
        required_arrays = [
            ArrayType.rigid_road_node_ids,
            ArrayType.rigid_road_node_coordinates,
            ArrayType.rigid_road_ids,
            ArrayType.rigid_road_segment_node_ids,
            ArrayType.rigid_road_segment_road_id,
        ]
        _check_array_occurrence(
            self.d3plot, array_names=required_arrays, required_array_names=required_arrays
        )
        has_rigid_road = ArrayType.rigid_road_node_ids in self.d3plot.arrays

        # check for mattyp shit
        # self.mattyp = 0
        # if not is_d3part and ArrayType.part_material_type in self.d3plot.arrays:
        #     self.mattyp = 1
        # elif is_d3part and ArrayType.part_material_type in self.d3plot.arrays:
        #     #
        #     self.mattyp = 0

        # check for mattyp
        is_d3part = self.d3plot.header.filetype == D3plotFiletype.D3PART

        self.mattyp = 0
        if not is_d3part and ArrayType.part_material_type in self.d3plot.arrays:
            self.mattyp = 1

            # rigid shells
            if ArrayType.element_shell_part_indexes in self.d3plot.arrays:
                part_mattyp = self.d3plot.arrays[ArrayType.part_material_type]
                shell_part_indexes = self.d3plot.arrays[ArrayType.element_shell_part_indexes]
                self.n_rigid_shells = (part_mattyp[shell_part_indexes] == 20).sum()
        elif is_d3part:
            self.mattyp = 0

        # set ndim finally
        #
        # This also confuses me from the manual  ...
        # It doesn't specify ndim clearly and only gives ranges.
        #
        # - has rigid body: rigid body data (movement etc.)
        # - rigid road: rigid road data
        # - mattyp: array with material types for each part
        #
        # Table:
        # |----------------|--------------------|------------|---------|----------|
        # | has_rigid_body | reduced rigid body | rigid road | mattyp  |   ndim   |
        # |----------------|--------------------|------------|---------|----------|
        # |     False      |        False       |    False   |    0    |     4    |
        # |     False      |        False       |    False   |    1    |     5    |
        # |     False (?)  |        False       |    True    |    0    |     6    |
        # |     False      |        False       |    True    |    1    |     7    |
        # |     True       |        False       |    False   |    0    |     8    |
        # |     True       |        True        |    True    |    0    |     9    |
        # |----------------|--------------------|------------|---------|----------|
        #
        # uncertainties: mattyp 0 or 1 ?!?!?
        if (
            not has_rigid_body_data
            and not has_reduced_rigid_body_data
            and not has_rigid_road
            and self.mattyp == 0
        ):
            new_header["ndim"] = 4
        elif (
            not has_rigid_body_data
            and not has_reduced_rigid_body_data
            and not has_rigid_road
            and self.mattyp == 1
        ):
            new_header["ndim"] = 5
        elif (
            not has_rigid_body_data
            and not has_reduced_rigid_body_data
            and has_rigid_road
            and self.mattyp == 0
        ):
            new_header["ndim"] = 6
        elif (
            not has_rigid_body_data
            and not has_reduced_rigid_body_data
            and has_rigid_road
            and self.mattyp == 1
        ):
            new_header["ndim"] = 7
        elif (
            has_rigid_body_data
            and not has_reduced_rigid_body_data
            and not has_rigid_road
            and self.mattyp == 0
        ):
            new_header["ndim"] = 8
        elif (
            has_rigid_body_data
            and has_reduced_rigid_body_data
            and has_rigid_road
            and self.mattyp == 0
        ):
            new_header["ndim"] = 9
        else:
            raise RuntimeError("Cannot determine haeder variable ndim.")

        # NUMNP
        new_header["numnp"] = (
            self.d3plot.arrays[ArrayType.node_coordinates].shape[0]
            if ArrayType.node_coordinates in self.d3plot.arrays
            else 0
        )

        # ICODE
        new_header["icode"] = self.d3plot.header.legacy_code_type

        # IT aka temperatures
        _check_array_occurrence(
            self.d3plot,
            array_names=[ArrayType.node_heat_flux],
            required_array_names=[ArrayType.node_temperature],
        )

        it_temp = 0
        if ArrayType.node_mass_scaling in self.d3plot.arrays:
            it_temp += 10

        if (
            ArrayType.node_temperature in self.d3plot.arrays
            and ArrayType.node_heat_flux not in self.d3plot.arrays
        ):
            it_temp += 1
        elif (
            ArrayType.node_temperature in self.d3plot.arrays
            and ArrayType.node_heat_flux in self.d3plot.arrays
        ):

            node_temp_shape = self.d3plot.arrays[ArrayType.node_temperature].shape
            if node_temp_shape.ndim == 2:
                it_temp += 2
            elif node_temp_shape.ndim == 3:
                it_temp += 3
            else:
                msg = "{1} is supposed to have either 2 or 3 dims and not '{0}'"
                raise RuntimeError(msg.format(node_temp_shape.ndim, ArrayType.node_temperature))
        else:
            # caught by _check_array_occurrence
            pass
        new_header["it"] = it_temp

        # IU - disp field indicator
        new_header["iu"] = 1 if ArrayType.node_displacement in self.d3plot.arrays else 0

        # IV - velicoty field indicator
        new_header["iv"] = 1 if ArrayType.node_velocity in self.d3plot.arrays else 0

        # IA - velicoty field indicator
        new_header["ia"] = 1 if ArrayType.node_acceleration in self.d3plot.arrays else 0

        # NEL8 - solid count
        n_solids = (
            self.d3plot.arrays[ArrayType.element_solid_node_indexes].shape[0]
            if ArrayType.element_solid_node_indexes in self.d3plot.arrays
            else 0
        )
        new_header["nel8"] = n_solids

        # helper var to track max material index across all element types
        # this is required to allocate the part array later
        # new_header["nmmat"] = 0

        # NUMMAT8 - solid material count
        required_arrays = [
            ArrayType.element_solid_node_indexes,
            ArrayType.element_solid_part_indexes,
        ]
        _check_array_occurrence(
            self.d3plot,
            array_names=required_arrays,
            required_array_names=required_arrays,
        )
        if ArrayType.element_solid_part_indexes in self.d3plot.arrays:
            part_indexes = self.d3plot.arrays[ArrayType.element_solid_part_indexes]
            unique_part_indexes = np.unique(part_indexes)
            self.unique_solid_part_indexes = unique_part_indexes
            new_header["nummat8"] = len(unique_part_indexes)

            # max_index = unique_part_indexes.max() + 1 \
            #     if len(part_indexes) else 0
            # new_header["nmmat"] = max(new_header["nmmat"],
            #                           max_index)
        else:
            new_header["nummat8"] = 0

        # NUMDS
        new_header["numds"] = self.d3plot.header.has_shell_four_inplane_gauss_points

        # NUMST
        new_header["numst"] = self.d3plot.header.unused_numst

        # NV3D - number of solid vars
        # NEIPH - number of solid history vars
        n_solid_layers = self.d3plot.check_array_dims(
            {
                ArrayType.element_solid_stress: 2,
                ArrayType.element_solid_effective_plastic_strain: 2,
                ArrayType.element_solid_history_variables: 2,
                ArrayType.element_solid_plastic_strain_tensor: 2,
                ArrayType.element_solid_thermal_strain_tensor: 2,
            },
            "n_solid_layers",
        )
        n_solid_layers = 1 if n_solid_layers < 1 else n_solid_layers
        self.n_solid_layers = n_solid_layers
        if n_solid_layers not in (1, 8):
            err_msg = "Solids must have either 1 or 8 integration layers not {0}."
            raise ValueError(err_msg.format(self.n_solid_layers))

        n_solid_hist_vars, _ = self.count_array_state_var(
            array_type=ArrayType.element_solid_history_variables,
            dimension_names=["n_timesteps", "n_solids", "n_solid_layers", "n_history_vars"],
            has_layers=True,
            n_layers=n_solid_layers,
        )
        n_solid_hist_vars = n_solid_hist_vars // n_solid_layers

        if ArrayType.element_solid_strain in self.d3plot.arrays:
            n_solid_hist_vars += 6
        # It is uncertain if this is counted as history var
        if ArrayType.element_solid_plastic_strain_tensor in self.d3plot.arrays:
            n_solid_hist_vars += 6
        # It is uncertain if this is counted as history var
        if ArrayType.element_solid_thermal_strain_tensor in self.d3plot.arrays:
            n_solid_hist_vars += 6
        n_solid_vars = (7 + n_solid_hist_vars) * n_solid_layers
        new_header["neiph"] = (
            n_solid_hist_vars if n_solids != 0 else self.d3plot.header.n_solid_history_vars
        )
        new_header["nv3d"] = n_solid_vars if n_solids != 0 else self.d3plot.header.n_solid_vars

        # NEL2 - beam count
        new_header["nel2"] = (
            self.d3plot.arrays[ArrayType.element_beam_node_indexes].shape[0]
            if ArrayType.element_beam_node_indexes in self.d3plot.arrays
            else 0
        )

        # NUMMAT2 - beam material count
        required_arrays = [
            ArrayType.element_beam_node_indexes,
            ArrayType.element_beam_part_indexes,
        ]
        _check_array_occurrence(
            self.d3plot,
            array_names=required_arrays,
            required_array_names=required_arrays,
        )
        if ArrayType.element_beam_part_indexes in self.d3plot.arrays:
            part_indexes = self.d3plot.arrays[ArrayType.element_beam_part_indexes]
            unique_part_indexes = np.unique(part_indexes)
            new_header["nummat2"] = len(unique_part_indexes)

            self.unique_beam_part_indexes = unique_part_indexes

            # max_index = unique_part_indexes.max() + 1 \
            #     if len(unique_part_indexes) else 0
            # new_header["nmmat"] = max(new_header["nmmat"],
            #                           max_index)
        else:
            new_header["nummat2"] = 0

        # NEIPB - beam history vars per integration point
        array_dims = {
            ArrayType.element_beam_shear_stress: 2,
            ArrayType.element_beam_axial_stress: 2,
            ArrayType.element_beam_plastic_strain: 2,
            ArrayType.element_beam_axial_strain: 2,
            ArrayType.element_beam_history_vars: 2,
        }
        n_beam_layers = self.d3plot.check_array_dims(array_dims, "n_beam_layers")
        new_header["beamip"] = n_beam_layers

        new_header["neipb"] = 0
        if ArrayType.element_beam_history_vars in self.d3plot.arrays:
            array = self.d3plot.arrays[ArrayType.element_beam_history_vars]
            if array.ndim != 4:
                msg = (
                    "Array '{0}' was expected to have 4 dimensions "
                    "(n_timesteps, n_beams, n_modes (3+n_beam_layers), "
                    "n_beam_history_vars)."
                )
                raise ValueError(msg.format(ArrayType.element_beam_history_vars))
            if array.shape[3] < 3:
                msg = (
                    "Array '{0}' dimension 3 must have have at least three"
                    " entries (beam layers: average, min, max)"
                )
                raise ValueError(msg.format(ArrayType.element_beam_history_vars))
            if array.shape[3] != 3 + n_beam_layers:
                msg = "Array '{0}' dimension 3 must have size (3+n_beam_layers). {1} != (3+{2})"
                raise ValueError(msg.format(ArrayType.element_beam_history_vars))
            new_header["neipb"] = array.shape[3]

        # NV1D - beam variable count
        new_header["nv1d"] = (
            6 + 5 * new_header["beamip"] + new_header["neipb"] * (3 + new_header["beamip"])
        )

        # NEL4 - number of shells
        n_shells = (
            self.d3plot.arrays[ArrayType.element_shell_node_indexes].shape[0]
            if ArrayType.element_shell_node_indexes in self.d3plot.arrays
            else 0
        )
        new_header["nel4"] = n_shells

        # NUMMAT4 - shell material count
        required_arrays = [
            ArrayType.element_shell_node_indexes,
            ArrayType.element_shell_part_indexes,
        ]
        _check_array_occurrence(
            self.d3plot,
            array_names=required_arrays,
            required_array_names=required_arrays,
        )
        if ArrayType.element_shell_part_indexes in self.d3plot.arrays:
            part_indexes = self.d3plot.arrays[ArrayType.element_shell_part_indexes]
            unique_part_indexes = np.unique(part_indexes)
            new_header["nummat4"] = len(unique_part_indexes)

            self.unique_shell_part_indexes = unique_part_indexes

            # max_index = unique_part_indexes.max() + 1 \
            #     if len(unique_part_indexes) else 0
            # new_header["nmmat"] = max(new_header["nmmat"],
            #                           max_index)
        else:
            new_header["nummat4"] = 0

        # NEIPS -shell history variable count
        n_shell_layers = 0
        if (
            ArrayType.element_shell_history_vars in self.d3plot.arrays
            or ArrayType.element_tshell_history_variables in self.d3plot.arrays
        ):

            n_shell_history_vars, n_shell_layers = self.count_array_state_var(
                array_type=ArrayType.element_shell_history_vars,
                dimension_names=["n_timesteps", "n_shells", "n_shell_layers", "n_history_vars"],
                has_layers=True,
                n_layers=n_shell_layers,
            )
            n_tshell_history_vars, n_tshell_layers = self.count_array_state_var(
                array_type=ArrayType.element_tshell_history_variables,
                dimension_names=["n_timesteps", "n_tshells", "n_shell_layers", "n_history_vars"],
                has_layers=True,
                n_layers=n_shell_layers,
            )

            if n_shell_layers != n_tshell_layers:
                msg = (
                    "Shells and thick shells must have the same amount "
                    "of integration layers: {0} != {1}"
                )
                raise RuntimeError(msg.format(n_shell_layers, n_tshell_layers))

            # we are tolerant here and simply add zero padding for the other
            # field later on
            new_header["neips"] = max(
                n_tshell_history_vars // n_tshell_layers, n_shell_history_vars // n_shell_layers
            )
        else:
            new_header["neips"] = 0

        array_dims = {
            ArrayType.element_shell_stress: 2,
            ArrayType.element_shell_effective_plastic_strain: 2,
            ArrayType.element_shell_history_vars: 2,
            ArrayType.element_tshell_stress: 2,
            ArrayType.element_tshell_effective_plastic_strain: 2,
            ArrayType.element_tshell_history_variables: 2,
        }
        n_shell_layers = self.d3plot.check_array_dims(array_dims, "n_shell_layers")
        self.n_shell_layers = n_shell_layers

        # NELTH - number of thick shell elements
        n_thick_shells = (
            self.d3plot.arrays[ArrayType.element_tshell_node_indexes].shape[0]
            if ArrayType.element_tshell_node_indexes in self.d3plot.arrays
            else 0
        )
        new_header["nelth"] = n_thick_shells

        # IOSHL1 - shell & solid stress flag
        if (
            ArrayType.element_shell_stress in self.d3plot.arrays
            or ArrayType.element_tshell_stress in self.d3plot.arrays
        ):
            new_header["ioshl1"] = 1000
        else:
            # if either stress or pstrain is written for solids
            # the whole block of 7 basic variables is always written
            # to the file
            if (
                ArrayType.element_solid_stress in self.d3plot.arrays
                or ArrayType.element_solid_effective_plastic_strain in self.d3plot.arrays
            ):
                new_header["ioshl1"] = 999
            else:
                new_header["ioshl1"] = 0

        if n_shells == 0 and n_thick_shells == 0 and n_solids == 0:
            new_header["ioshl1"] = (
                self.d3plot.header.raw_header["ioshl1"]
                if "ioshl1" in self.d3plot.header.raw_header
                else 0
            )

        if n_shells == 0 and n_thick_shells == 0 and n_solids != 0:
            if (
                "ioshl1" in self.d3plot.header.raw_header
                and self.d3plot.header.raw_header["ioshl1"] == 1000
            ):
                new_header["ioshl1"] = 1000

        # IOSHL2 - shell & solid pstrain flag
        if (
            ArrayType.element_shell_effective_plastic_strain in self.d3plot.arrays
            or ArrayType.element_tshell_effective_plastic_strain in self.d3plot.arrays
        ):
            new_header["ioshl2"] = 1000
        else:
            if ArrayType.element_solid_effective_plastic_strain in self.d3plot.arrays:
                new_header["ioshl2"] = 999
            else:
                new_header["ioshl2"] = 0

        if n_shells == 0 and n_thick_shells == 0 and n_solids == 0:
            new_header["ioshl2"] = (
                self.d3plot.header.raw_header["ioshl2"]
                if "ioshl2" in self.d3plot.header.raw_header
                else 0
            )

        if n_shells == 0 and n_thick_shells == 0 and n_solids != 0:
            if (
                "ioshl2" in self.d3plot.header.raw_header
                and self.d3plot.header.raw_header["ioshl2"] == 1000
            ):
                new_header["ioshl2"] = 1000

        # IOSHL3 - shell forces flag
        if (
            ArrayType.element_shell_shear_force in self.d3plot.arrays
            or ArrayType.element_shell_bending_moment in self.d3plot.arrays
            or ArrayType.element_shell_normal_force in self.d3plot.arrays
        ):
            new_header["ioshl3"] = 1000
        else:
            # new_header["ioshl3"] = 999
            new_header["ioshl3"] = 0

        if n_shells == 0:
            new_header["ioshl3"] = (
                self.d3plot.header.raw_header["ioshl3"]
                if "ioshl3" in self.d3plot.header.raw_header
                else 0
            )

        # IOSHL4 - shell energy+2 unknown+thickness flag
        if (
            ArrayType.element_shell_thickness in self.d3plot.arrays
            or ArrayType.element_shell_unknown_variables in self.d3plot.arrays
            or ArrayType.element_shell_internal_energy in self.d3plot.arrays
        ):
            new_header["ioshl4"] = 1000
        else:
            # new_header["ioshl4"] = 999
            new_header["ioshl4"] = 0

        if n_shells == 0:
            new_header["ioshl4"] = (
                self.d3plot.header.raw_header["ioshl4"]
                if "ioshl4" in self.d3plot.header.raw_header
                else 0
            )

        # IDTDT - Flags for various data in the database
        new_header["idtdt"] = 0
        istrn = 0
        if (
            ArrayType.element_shell_strain in self.d3plot.arrays
            or ArrayType.element_solid_strain in self.d3plot.arrays
            or ArrayType.element_tshell_strain in self.d3plot.arrays
        ):
            # new_header["idtdt"] = 10000
            istrn = 1
        new_header["istrn"] = istrn

        if ArrayType.node_temperature_gradient in self.d3plot.arrays:
            new_header["idtdt"] += 1
            self.has_node_temperature_gradient = True
        if (
            ArrayType.node_residual_forces in self.d3plot.arrays
            or ArrayType.node_residual_moments in self.d3plot.arrays
        ):
            new_header["idtdt"] += 10
            self.has_node_residual_forces = True
            self.has_node_residual_moments = True
        if (
            ArrayType.element_shell_plastic_strain_tensor in self.d3plot.arrays
            or ArrayType.element_solid_plastic_strain_tensor in self.d3plot.arrays
        ):
            new_header["idtdt"] += 100
            self.has_plastic_strain_tensor = True
        if (
            ArrayType.element_shell_thermal_strain_tensor in self.d3plot.arrays
            or ArrayType.element_solid_thermal_strain_tensor in self.d3plot.arrays
        ):
            new_header["idtdt"] += 1000
            self.has_thermal_strain_tensor = True
        if new_header["idtdt"] > 100 and new_header["istrn"]:
            new_header["idtdt"] += 10000

        # info of element deletion is encoded into maxint ...
        element_deletion_arrays = [
            ArrayType.element_beam_is_alive,
            ArrayType.element_shell_is_alive,
            ArrayType.element_tshell_is_alive,
            ArrayType.element_solid_is_alive,
        ]
        mdlopt = 0
        if any(name in self.d3plot.arrays for name in element_deletion_arrays):
            mdlopt = 2
        elif ArrayType.node_is_alive in self.d3plot.arrays:
            mdlopt = 1
        self.mdlopt = mdlopt

        # MAXINT - shell integration layer count
        array_dims = {
            ArrayType.element_shell_stress: 2,
            ArrayType.element_shell_effective_plastic_strain: 2,
            ArrayType.element_shell_history_vars: 2,
            ArrayType.element_tshell_stress: 2,
            ArrayType.element_tshell_effective_plastic_strain: 2,
        }
        n_shell_layers = self.d3plot.check_array_dims(array_dims, "n_layers")

        # beauty fix: take old shell layers if none exist
        if n_shell_layers == 0:
            n_shell_layers = self.d3plot.header.n_shell_tshell_layers

        if mdlopt == 0:
            new_header["maxint"] = n_shell_layers
        elif mdlopt == 1:
            new_header["maxint"] = -n_shell_layers
        elif mdlopt == 2:
            new_header["maxint"] = -(n_shell_layers + 10000)

        # NV2D - shell variable count
        has_shell_stress = new_header["ioshl1"] == 1000
        has_shell_pstrain = new_header["ioshl2"] == 1000
        has_shell_forces = new_header["ioshl3"] == 1000
        has_shell_other = new_header["ioshl4"] == 1000
        new_header["nv2d"] = (
            n_shell_layers * (6 * has_shell_stress + has_shell_pstrain + new_header["neips"])
            + 8 * has_shell_forces
            + 4 * has_shell_other
            + 12 * istrn
            + n_shell_layers * self.has_plastic_strain_tensor * 6
            + self.has_thermal_strain_tensor * 6
        )

        # NMSPH - number of sph nodes
        new_header["nmsph"] = (
            len(self.d3plot.arrays[ArrayType.sph_node_indexes])
            if ArrayType.sph_node_indexes in self.d3plot.arrays
            else 0
        )

        # NGPSPH - number of sph materials
        new_header["ngpsph"] = (
            len(np.unique(self.d3plot.arrays[ArrayType.sph_node_material_index]))
            if ArrayType.sph_node_material_index in self.d3plot.arrays
            else 0
        )

        # NUMMATT - thick shell material count
        required_arrays = [
            ArrayType.element_tshell_node_indexes,
            ArrayType.element_tshell_part_indexes,
        ]
        _check_array_occurrence(
            self.d3plot,
            array_names=required_arrays,
            required_array_names=required_arrays,
        )
        if ArrayType.element_tshell_part_indexes in self.d3plot.arrays:
            part_indexes = self.d3plot.arrays[ArrayType.element_tshell_part_indexes]
            unique_part_indexes = np.unique(part_indexes)
            new_header["nummatt"] = len(unique_part_indexes)

            self.unique_tshell_part_indexes = unique_part_indexes

            # max_index = unique_part_indexes.max() + 1 \
            #     if len(part_indexes) else 0
            # new_header["nmmat"] = max(new_header["nmmat"],
            #                           max_index)
        else:
            new_header["nummatt"] = 0

        # NV3DT
        new_header["nv3dt"] = (
            n_shell_layers * (6 * has_shell_stress + has_shell_pstrain + new_header["neips"])
            + 12 * istrn
        )

        # IALEMAT - number of ALE materials
        new_header["ialemat"] = (
            len(self.d3plot.arrays[ArrayType.ale_material_ids])
            if ArrayType.ale_material_ids in self.d3plot.arrays
            else 0
        )
        # NCFDV1
        new_header["ncfdv1"] = 0

        # NCFDV2
        new_header["ncfdv2"] = 0

        # NADAPT - number of adapted element to parent pairs ?!?
        new_header["ncfdv2"] = 0

        # NUMRBS (written to numbering header)
        if ArrayType.rigid_body_coordinates in self.d3plot.arrays:
            array = self.d3plot.arrays[ArrayType.rigid_body_coordinates]
            if array.ndim != 3:
                msg = "Array '{0}' was expected to have {1} dimensions ({2})."
                raise ValueError(
                    msg.format(
                        ArrayType.rigid_wall_force,
                        3,
                        ",".join(["n_timesteps", "n_rigid_bodies", "x_y_z"]),
                    )
                )
            new_header["numrbs"] = array.shape[1]
        else:
            new_header["numrbs"] = 0

        # NMMAT - material count (very complicated stuff ...)
        tmp_nmmat = (
            new_header["nummat2"]
            + new_header["nummat4"]
            + new_header["nummat8"]
            + new_header["nummatt"]
            + new_header["numrbs"]
        )
        if (
            ArrayType.part_ids in self.d3plot.arrays
            or ArrayType.part_internal_energy in self.d3plot.arrays
            or ArrayType.part_kinetic_energy in self.d3plot.arrays
            or ArrayType.part_mass in self.d3plot.arrays
            or ArrayType.part_velocity in self.d3plot.arrays
        ):

            tmp_nmmat2 = self.d3plot.check_array_dims(
                {
                    ArrayType.part_ids: 0,
                    ArrayType.part_internal_energy: 1,
                    ArrayType.part_kinetic_energy: 1,
                    ArrayType.part_mass: 1,
                    ArrayType.part_velocity: 1,
                },
                "n_parts",
            )

            new_header["nmmat"] = tmp_nmmat2

            # FIX
            # ...
            if new_header["nmmat"] > tmp_nmmat:
                new_header["numrbs"] = (
                    new_header["nmmat"]
                    - new_header["nummat2"]
                    - new_header["nummat4"]
                    - new_header["nummat8"]
                    - new_header["nummatt"]
                )
        else:
            new_header["nmmat"] = tmp_nmmat

        # NARBS - words for arbitrary numbering of everything
        # requires nmmat thus it was placed here
        new_header["narbs"] = (
            new_header["numnp"]
            + new_header["nel8"]
            + new_header["nel2"]
            + new_header["nel4"]
            + new_header["nelth"]
            + 3 * new_header["nmmat"]
        )
        # narbs header data
        if ArrayType.part_ids in self.d3plot.arrays:
            new_header["narbs"] += 16
        else:
            new_header["narbs"] += 10

        # NGLBV - number of global variables
        n_rigid_wall_vars = 0
        n_rigid_walls = 0
        if ArrayType.rigid_wall_force in self.d3plot.arrays:
            n_rigid_wall_vars = 1
            array = self.d3plot.arrays[ArrayType.rigid_wall_force]
            if array.ndim != 2:
                msg = "Array '{0}' was expected to have {1} dimensions ({2})."
                raise ValueError(
                    msg.format(
                        ArrayType.rigid_wall_force, 2, ",".join(["n_timesteps", "n_rigid_walls"])
                    )
                )
            n_rigid_walls = array.shape[1]
        if ArrayType.rigid_wall_position in self.d3plot.arrays:
            n_rigid_wall_vars = 4
            array = self.d3plot.arrays[ArrayType.rigid_wall_position]
            if array.ndim != 3:
                msg = "Array '{0}' was expected to have {1} dimensions ({2})."
                raise ValueError(
                    msg.format(
                        ArrayType.rigid_wall_position,
                        3,
                        ",".join(["n_timesteps", "n_rigid_walls", "x_y_z"]),
                    )
                )
            n_rigid_walls = array.shape[1]

        new_header["n_rigid_walls"] = n_rigid_walls
        new_header["n_rigid_wall_vars"] = n_rigid_wall_vars
        n_global_variables = 0
        if ArrayType.global_kinetic_energy in self.d3plot.arrays:
            n_global_variables = 1
        if ArrayType.global_internal_energy in self.d3plot.arrays:
            n_global_variables = 2
        if ArrayType.global_total_energy in self.d3plot.arrays:
            n_global_variables = 3
        if ArrayType.global_velocity in self.d3plot.arrays:
            n_global_variables = 6
        if ArrayType.part_internal_energy in self.d3plot.arrays:
            n_global_variables = 6 + 1 * new_header["nmmat"]
        if ArrayType.part_kinetic_energy in self.d3plot.arrays:
            n_global_variables = 6 + 2 * new_header["nmmat"]
        if ArrayType.part_velocity in self.d3plot.arrays:
            n_global_variables = 6 + 5 * new_header["nmmat"]
        if ArrayType.part_mass in self.d3plot.arrays:
            n_global_variables = 6 + 6 * new_header["nmmat"]
        if ArrayType.part_hourglass_energy in self.d3plot.arrays:
            n_global_variables = 6 + 7 * new_header["nmmat"]
        if n_rigid_wall_vars * n_rigid_walls != 0:
            n_global_variables = 6 + 7 * new_header["nmmat"] + n_rigid_wall_vars * n_rigid_walls
        new_header["nglbv"] = n_global_variables

        # NUMFLUID - total number of ALE fluid groups
        new_header["numfluid"] = 0

        # INN - Invariant node numbering fore shell and solid elements
        if self.d3plot.header.has_invariant_numbering:
            if "inn" in self.d3plot.header.raw_header and self.d3plot.header.raw_header["inn"] != 0:
                new_header["inn"] = self.d3plot.header.raw_header["inn"]
            else:
                new_header["inn"] = int(self.d3plot.header.has_invariant_numbering)
        else:
            new_header["inn"] = 0

        # NPEFG
        airbag_arrays = [
            ArrayType.airbags_first_particle_id,
            ArrayType.airbags_n_particles,
            ArrayType.airbags_ids,
            ArrayType.airbags_n_gas_mixtures,
            ArrayType.airbags_n_chambers,
            ArrayType.airbag_n_active_particles,
            ArrayType.airbag_bag_volume,
            ArrayType.airbag_particle_gas_id,
            ArrayType.airbag_particle_chamber_id,
            ArrayType.airbag_particle_leakage,
            ArrayType.airbag_particle_mass,
            ArrayType.airbag_particle_radius,
            ArrayType.airbag_particle_spin_energy,
            ArrayType.airbag_particle_translation_energy,
            ArrayType.airbag_particle_nearest_segment_distance,
            ArrayType.airbag_particle_position,
            ArrayType.airbag_particle_velocity,
        ]
        subver = 3 if any(name in self.d3plot.arrays for name in airbag_arrays) else 0

        # subver overwrite
        if self.d3plot.header.n_airbags:
            # pylint: disable = protected-access
            subver = self.d3plot._airbag_info.subver

        n_partgas = (
            len(self.d3plot.arrays[ArrayType.airbags_ids])
            if ArrayType.airbags_ids in self.d3plot.arrays
            else 0
        )

        new_header["npefg"] = 1000 * subver + n_partgas

        # NEL48 - extra nodes for 8 node shell elements
        required_arrays = [
            ArrayType.element_shell_node8_element_index,
            ArrayType.element_shell_node8_extra_node_indexes,
        ]
        _check_array_occurrence(
            self.d3plot,
            array_names=required_arrays,
            required_array_names=required_arrays,
        )
        new_header["nel48"] = (
            len(self.d3plot.arrays[ArrayType.element_shell_node8_element_index])
            if ArrayType.element_shell_node8_element_index in self.d3plot.arrays
            else 0
        )

        # NEL20 - 20 nodes solid elements
        required_arrays = [
            ArrayType.element_solid_node20_element_index,
            ArrayType.element_solid_node20_extra_node_indexes,
        ]
        _check_array_occurrence(
            self.d3plot,
            array_names=required_arrays,
            required_array_names=required_arrays,
        )
        if ArrayType.element_solid_node20_element_index in self.d3plot.arrays:
            new_header["nel20"] = len(
                self.d3plot.arrays[ArrayType.element_solid_node20_element_index]
            )
        else:
            new_header["nel20"] = 0

        # NT3D - thermal solid data
        if ArrayType.element_solid_thermal_data in self.d3plot.arrays:
            new_header["nt3d"] = len(self.d3plot.arrays[ArrayType.element_solid_thermal_data])
        else:
            new_header["nt3d"] = 0

        # NEL27 - 27 node solid elements
        required_arrays = [
            ArrayType.element_solid_node27_element_index,
            ArrayType.element_solid_node27_extra_node_indexes,
        ]
        _check_array_occurrence(
            self.d3plot,
            array_names=required_arrays,
            required_array_names=required_arrays,
        )
        if ArrayType.element_solid_node27_element_index in self.d3plot.arrays:
            new_header["nel27"] = len(
                self.d3plot.arrays[ArrayType.element_solid_node27_element_index]
            )
        else:
            new_header["nel27"] = 0

        # EXTRA - extra header variables
        # set only if any value is non-zero
        extra_hdr_variables = ["nel20", "nt3d", "nel27", "neipb"]
        if any(new_header[name] for name in extra_hdr_variables):
            new_header["extra"] = 64
        else:
            new_header["extra"] = 0

        # CHECKS

        # unique part indexes all ok
        for part_index in self.unique_beam_part_indexes:
            if part_index >= new_header["nmmat"]:
                msg = "{0} part index {1} is larger than number of materials {2}"
                raise ValueError(msg.format("beam", part_index, new_header["nmmat"]))
        for part_index in self.unique_shell_part_indexes:
            if part_index >= new_header["nmmat"]:
                msg = "{0} part index {1} is larger than number of materials {2}"
                raise ValueError(msg.format("shell", part_index, new_header["nmmat"]))
        for part_index in self.unique_solid_part_indexes:
            if part_index >= new_header["nmmat"]:
                msg = "{0} part index {1} is larger than number of materials {2}"
                raise ValueError(msg.format("solid", part_index, new_header["nmmat"]))
        for part_index in self.unique_tshell_part_indexes:
            if part_index >= new_header["nmmat"]:
                msg = "{0} part index {1} is larger than number of materials {2}"
                raise ValueError(msg.format("tshell", part_index, new_header["nmmat"]))

        # new header
        self._header = new_header

    # pylint: disable = too-many-return-statements
    def pack(self, value: Any, size=None, dtype_hint=None) -> bytes:
        """Pack a python value according to its settings

        Parameters
        ----------
        value: Any
            integer, float or string type value
        size: int
            size in bytes
        dtype_hint: `np.integer` or `np.floating` (default: None)
            dtype hint for numpy arrays (prevens wrong casting)

        Returns
        -------
        bytes: bytes
            value packed in bytes

        Raises
        ------
        RuntimeError
            If the type cannot be deserialized for being unknown.
        """

        assert dtype_hint in (None, np.integer, np.floating)

        # INT
        if isinstance(value, self._allowed_int_types):
            return struct.pack(self.itype, value)
        # FLOAT
        if isinstance(value, self._allowed_float_types):
            return struct.pack(self.ftype, value)
        # BYTES
        if isinstance(value, bytes):
            if size and len(value) > size:
                return value[:size]
            return value
        # BYTEARRAY
        if isinstance(value, bytearray):
            if size and len(value) > size:
                return bytes(value[:size])
            return bytes(value)
        # STRING
        if isinstance(value, str):
            if size:
                fmt = "{0:" + str(size) + "}"
                return fmt.format(value).encode(self._str_codec)

            return value.encode(self._str_codec)
        # ARRAY
        if isinstance(value, np.ndarray):

            if (value.dtype != self.ftype and dtype_hint == np.floating) or (
                value.dtype != self.itype and dtype_hint == np.integer
            ):

                # we need typehint
                if dtype_hint is None:
                    msg = "Please specify a dtype_hint (np.floating, np.integer)."
                    raise ValueError(msg)

                # determine new dtype
                new_dtype = self.itype if dtype_hint == np.integer else self.ftype

                # log conversion
                msg = "Converting array from %s to %s"
                LOGGER.info(msg, value.dtype, new_dtype)

                # warn if convert between int and float (possible bugs)
                if not np.issubdtype(value.dtype, dtype_hint):
                    LOGGER.warning(msg, value.dtype, new_dtype)

                value = value.astype(new_dtype)

            return value.tobytes()

        msg = "Cannot deserialize type '%s' of value '%s' for writing."
        raise RuntimeError(msg, type(value), value)

    def count_array_state_var(
        self, array_type: str, dimension_names: List[str], has_layers: bool, n_layers: int = 0
    ) -> Tuple[int, int]:
        """This functions checks and updates the variable count for certain types of arrays

        Parameters
        ----------
        array_type: str
            name of the shell layer array
        dimension_names: List[str]
            names of the array dimensions
        has_layers: bool
            if the array has integration layers
        n_layers: int
            number of (previous) shell layers, if unknown set to 0

        Returns
        -------
        n_vars: int
            variable count
        n_layers: int
            number of layers

        Raises
        ------
        ValueError
            If the dimensions of the array were invalid or an inconsistent
            number of integration layers was detected.
        """

        n_vars = 0

        if array_type in self.d3plot.arrays:
            array = self.d3plot.arrays[array_type]

            if array.ndim != len(dimension_names):
                msg = "Array '{0}' was expected to have {1} dimensions ({2})."
                raise ValueError(
                    msg.format(array_type, len(dimension_names), ", ".join(dimension_names))
                )

            if has_layers:
                if n_layers == 0:
                    n_layers = array.shape[2]
                else:
                    if n_layers != array.shape[2]:
                        msg = (
                            "Array '{0}' has '{1}' integration layers"
                            " but another array used '{2}'."
                        )
                        raise ValueError(msg.format(array_type, array.shape[2], n_layers))

                # last dimension is collapsed
                if array.ndim == 3:
                    n_vars = 1 * n_layers
                else:
                    n_vars = array.shape[3] * n_layers

            # no layers
            else:

                # last dimension is collapsed
                if array.ndim == 2:
                    n_vars = 1
                else:
                    n_vars = array.shape[2]

        return n_vars, n_layers


@dataclass
class MemoryInfo:
    """MemoryInfo contains info about memory regions in files"""

    start: int = 0
    length: int = 0
    filepath: str = ""
    n_states: int = 0
    filesize: int = 0
    use_mmap: bool = False


class FemzipInfo:
    """FemzipInfo contains information and wrappers for the femzip api"""

    api: FemzipAPI
    n_states: int = 0
    buffer_info: FemzipBufferInfo
    use_femzip: bool = False

    def __init__(self, filepath: str = ""):
        self.api = FemzipAPI()
        self.buffer_info = FemzipBufferInfo()

        if filepath:
            tmp_header = D3plotHeader().load_file(filepath)
            self.use_femzip = tmp_header.has_femzip_indicator

            if self.use_femzip:
                # there is a lot to go wrong
                try:
                    self.buffer_info = self.api.get_buffer_info(filepath)
                # loading femzip api failed
                except Exception as err:
                    raise RuntimeError(f"Failed to use Femzip: {err}") from err


class MaterialSectionInfo:
    """MaterialSectionInfo contains vars from the material section"""

    n_rigid_shells: int = 0


class SphSectionInfo:
    """SphSectionInfo contains vars from the sph geometry section"""

    n_sph_array_length: int = 11
    n_sph_vars: int = 0
    has_influence_radius: bool = False
    has_particle_pressure: bool = False
    has_stresses: bool = False
    has_plastic_strain: bool = False
    has_material_density: bool = False
    has_internal_energy: bool = False
    has_n_affecting_neighbors: bool = False
    has_strain_and_strainrate: bool = False
    has_true_strains: bool = False
    has_mass: bool = False
    n_sph_history_vars: int = 0


class AirbagInfo:
    """AirbagInfo contains vars used to describe the sph geometry section"""

    n_geometric_variables: int = 0
    n_airbag_state_variables: int = 0
    n_particle_state_variables: int = 0
    n_particles: int = 0
    n_airbags: int = 0
    # ?
    subver: int = 0
    n_chambers: int = 0

    def get_n_variables(self) -> int:
        """Get the number of airbag variables

        Returns
        -------
        n_airbag_vars: int
            number of airbag vars
        """
        return (
            self.n_geometric_variables
            + self.n_particle_state_variables
            + self.n_airbag_state_variables
        )


class NumberingInfo:
    """NumberingInfo contains vars from the part numbering section (ids)"""

    # the value(s) of ptr is initialized
    # as 1 since we need to make it
    # negative if part_ids are written
    # to file and 0 cannot do that ...
    # This is ok for self-made D3plots
    # since these fields are unused anyway
    ptr_node_ids: int = 1
    has_material_ids: bool = False
    ptr_solid_ids: int = 1
    ptr_beam_ids: int = 1
    ptr_shell_ids: int = 1
    ptr_thick_shell_ids: int = 1
    n_nodes: int = 0
    n_solids: int = 0
    n_beams: int = 0
    n_shells: int = 0
    n_thick_shells: int = 0
    ptr_material_ids: int = 1
    ptr_material_ids_defined_order: int = 1
    ptr_material_ids_crossref: int = 1
    n_parts: int = 0
    n_parts2: int = 0
    n_rigid_bodies: int = 0


@dataclass
class RigidBodyMetadata:
    """RigidBodyMetadata contains vars from the rigid body metadata section.
    This section comes before the individual rigid body data.
    """

    internal_number: int
    n_nodes: int
    node_indexes: np.ndarray
    n_active_nodes: int
    active_node_indexes: np.ndarray


class RigidBodyInfo:
    """RigidBodyMetadata contains vars for the individual rigid bodies"""

    rigid_body_metadata_list: Iterable[RigidBodyMetadata]
    n_rigid_bodies: int = 0

    def __init__(
        self, rigid_body_metadata_list: Iterable[RigidBodyMetadata], n_rigid_bodies: int = 0
    ):
        self.rigid_body_metadata_list = rigid_body_metadata_list
        self.n_rigid_bodies = n_rigid_bodies


class RigidRoadInfo:
    """RigidRoadInfo contains metadata for the description of rigid roads"""

    n_nodes: int = 0
    n_road_segments: int = 0
    n_roads: int = 0
    # ?
    motion: int = 0

    def __init__(
        self, n_nodes: int = 0, n_road_segments: int = 0, n_roads: int = 0, motion: int = 0
    ):
        self.n_nodes = n_nodes
        self.n_road_segments = n_road_segments
        self.n_roads = n_roads
        self.motion = motion


class StateInfo:
    """StateInfo holds metadata for states which is currently solely the timestep.
    We all had bigger plans in life ...
    """

    n_timesteps: int = 0

    def __init__(self, n_timesteps: int = 0):
        self.n_timesteps = n_timesteps


class D3plot:
    """Class used to read LS-Dyna d3plots"""

    _header: D3plotHeader
    _femzip_info: FemzipInfo
    _material_section_info: MaterialSectionInfo
    _sph_info: SphSectionInfo
    _airbag_info: AirbagInfo
    _numbering_info: NumberingInfo
    _rigid_body_info: RigidBodyInfo
    _rigid_road_info: RigidRoadInfo
    _buffer: Union[BinaryBuffer, None] = None

    # we all love secret settings
    use_advanced_femzip_api: bool = False

    # This amount of args is needed
    # pylint: disable = too-many-arguments, too-many-statements, unused-argument
    def __init__(
        self,
        filepath: str = None,
        use_femzip: Union[bool, None] = None,
        n_files_to_load_at_once: Union[int, None] = None,
        state_array_filter: Union[List[str], None] = None,
        state_filter: Union[None, Set[int]] = None,
        buffered_reading: bool = False,
    ):
        """Constructor for a D3plot

        Parameters
        ----------
        filepath: str
            path to a d3plot file
        use_femzip: bool
            Not used anymore.
        n_files_to_load_at_once: int
            *DEPRECATED* not used anymore, use `buffered_reading`
        state_array_filter: Union[List[str], None]
            names of arrays which will be the only ones loaded from state data
        state_filter: Union[None, Set[int]]
            which states to load. Negative indexes count backwards.
        buffered_reading: bool
            whether to pull only a single state into memory during reading

        Examples
        --------
            >>> from lasso.dyna import D3plot, ArrayType
            >>> # open and read everything
            >>> d3plot = D3plot("path/to/d3plot")

            >>> # only read node displacement
            >>> d3plot = D3plot("path/to/d3plot", state_array_filter=["node_displacement"])
            >>> # or with nicer syntax
            >>> d3plot = D3plot("path/to/d3plot", state_array_filter=[ArrayType.node_displacement])

            >>> # only load first and last state
            >>> d3plot = D3plot("path/to/d3plot", state_filter={0, -1})

            >>> # our computer lacks RAM so lets extract a specific array
            >>> # but only keep one state at a time in memory
            >>> d3plot = D3plot("path/to/d3plot",
            >>>                 state_array_filter=[ArrayType.node_displacement],
            >>>                 buffered_reading=True)

        Notes
        -----
            If dyna wrote multiple files for several states,
            only give the path to the first file.
        """
        super().__init__()

        LOGGER.debug("-------- D 3 P L O T --------")

        self._arrays = {}
        self._header = D3plotHeader()
        self._femzip_info = FemzipInfo(filepath=filepath if filepath is not None else "")
        self._material_section_info = MaterialSectionInfo()
        self._sph_info = SphSectionInfo()
        self._airbag_info = AirbagInfo()
        self._numbering_info = NumberingInfo()
        self._rigid_body_info = RigidBodyInfo(rigid_body_metadata_list=tuple())
        self._rigid_road_info = RigidRoadInfo()
        self._state_info = StateInfo()

        # which states to load
        self.state_filter = state_filter

        # how many files to load into memory at once
        if n_files_to_load_at_once is not None:
            warn_msg = "D3plot argument '{0}' is deprecated. Please use '{1}=True'."
            raise DeprecationWarning(warn_msg.format("n_files_to_load_at_once", "buffered_reading"))
        self.buffered_reading = buffered_reading or (state_filter is not None and any(state_filter))

        # arrays to filter out
        self.state_array_filter = state_array_filter

        # load memory accordingly
        # no femzip
        if filepath and not self._femzip_info.use_femzip:
            self._buffer = BinaryBuffer(filepath)
            self.bb_states = None
        # femzip
        elif filepath and self._femzip_info.use_femzip:
            self._buffer = self._read_femzip_geometry(filepath)
            # we need to reload the header
            self._header = D3plotHeader().load_file(self._buffer)
            self.bb_states = None
        # no data to load basically
        else:
            self._buffer = None
            self.bb_states = None

        self.geometry_section_size = 0

        # read header
        self._read_header()

        # read geometry
        self._parse_geometry()

        # read state data

        # try advanced femzip api
        if (
            filepath
            and self._femzip_info.use_femzip
            and self.use_advanced_femzip_api
            and self._femzip_info.api.has_femunziplib_license()
        ):

            LOGGER.debug("Advanced FEMZIP-API used")
            try:
                self._read_states_femzip_advanced(
                    filepath,
                )
            except Exception:
                trace = traceback.format_exc()
                warn_msg = (
                    "Error when using advanced Femzip API, "
                    "falling back to normal but slower Femzip API.\n%s"
                )
                LOGGER.warning(warn_msg, trace)

                # since we had a crash, we need to reload the file
                # to be sure we don't crash again
                self._femzip_info.api.close_current_file()
                self._femzip_info.api.read_geometry(filepath, self._femzip_info.buffer_info, False)
                # try normal femzip api
                self._read_states(filepath)
            finally:
                self._femzip_info.api.close_current_file()

        # normal state reading (femzip and non-femzip)
        elif filepath:
            self._read_states(filepath)
            if self._femzip_info.use_femzip:
                self._femzip_info.api.close_current_file()
        else:
            # no filepath = nothing to do
            pass

    def _read_femzip_geometry(self, filepath: str) -> BinaryBuffer:
        """Read the geometry from femzip

        Parameters
        ----------
        filepath: str
            path to the femzpi file

        Returns
        -------
        bb: BinaryBuffer
            memory of the geometry section
        """

        buffer_geo = self._femzip_info.api.read_geometry(
            filepath, buffer_info=self._femzip_info.buffer_info, close_file=False
        )

        # save
        buffer = BinaryBuffer()
        buffer.filepath_ = filepath
        buffer.memoryview = buffer_geo.cast("B")

        return buffer

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps loaded"""
        return self._state_info.n_timesteps

    @property
    def arrays(self) -> dict:
        """Dictionary holding all d3plot arrays

        Notes
        -----
            The corresponding keys of the dictionary can
            also be found in `lasso.dyna.ArrayTypes`, which
            helps with IDE integration and code safety.

        Examples
        --------
            >>> d3plot = D3plot("some/path/to/d3plot")
            >>> d3plot.arrays.keys()
            dict_keys(['irbtyp', 'node_coordinates', ...])
            >>> # The following is good coding practice
            >>> import lasso.dyna.ArrayTypes.ArrayTypes as atypes
            >>> d3plot.arrays[atypes.node_displacmeent].shape
        """
        return self._arrays

    @arrays.setter
    def arrays(self, array_dict: dict):
        assert isinstance(array_dict, dict)
        self._arrays = array_dict

    @property
    def header(self) -> D3plotHeader:
        """Instance holding all d3plot header information

        Returns
        -------
        header: D3plotHeader
            header of the d3plot

        Notes
        -----
            The header contains a lot of information such as number
            of elements, etc.

        Examples
        --------
            >>> d3plot = D3plot("some/path/to/d3plot")
            >>> # number of shells
            >>> d3plot.header.n_shells
            85624
        """
        return self._header

    @staticmethod
    def _is_end_of_file_marker(
        buffer: BinaryBuffer, position: int, ftype: Union[np.float32, np.float64]
    ) -> bool:
        """Check for the dyna eof marker at a certain position

        Parameters
        ----------
        bb: BinaryBuffer
            buffer holding memory
        position: int
            position in the buffer
        ftype: Union[np.float32, np.float64]
            floating point type

        Returns
        -------
        is_end_marker: bool
            if at the position is an end marker

        Notes
        -----
            The end of file marker is represented by a floating point
            number with the value -999999 (single precision hex: F02374C9,
            double precision hex: 000000007E842EC1).
        """

        if ftype not in (np.float32, np.float64):
            err_msg = "Floating point type '{0}' is not a floating point type."
            raise ValueError(err_msg.format(ftype))

        return buffer.read_number(position, ftype) == ftype(-999999)

    def _correct_file_offset(self):
        """Correct the position in the bytes

        Notes
        -----
            LS-Dyna writes its files zero padded at a size of
            512 words in block size. There might be a lot of
            unused trailing data in the rear we need to skip
            in order to get to the next useful data block.
        """

        if not self._buffer:
            return

        block_count = len(self._buffer) // (512 * self.header.wordsize)

        # Warning!
        # Resets the block count!
        self.geometry_section_size = (block_count + 1) * 512 * self.header.wordsize

    @property
    def _n_parts(self) -> int:
        """Get the number of parts contained in the d3plot

        Returns
        -------
        n_parts: int
            number of total parts
        """

        n_parts = (
            self.header.n_solid_materials
            + self.header.n_beam_materials
            + self.header.n_shell_materials
            + self.header.n_thick_shell_materials
            + self._numbering_info.n_rigid_bodies
        )

        return n_parts

    @property
    def _n_rigid_walls(self) -> int:
        """Get the number of rigid walls in the d3plot

        Returns
        -------
        n_rigid_walls: int
            number of rigid walls
        """

        # there have been cases that there are less than in the specs
        # indicated global vars. That breaks this computation, thus we
        # use max at the end.
        previous_global_vars = 6 + 7 * self._n_parts
        n_rigid_wall_vars = self.header.n_rigid_wall_vars
        n_rigid_walls = (self.header.n_global_vars - previous_global_vars) // n_rigid_wall_vars

        # if n_rigid_walls < 0:
        #     err_msg = "The computed number of rigid walls is negative ('{0}')."
        #     raise RuntimeError(err_msg.format(n_rigid_walls))

        return max(n_rigid_walls, 0)

    # pylint: disable = unused-argument, too-many-locals
    def _read_d3plot_file_generator(
        self, buffered_reading: bool, state_filter: Union[None, Set[int]]
    ) -> typing.Any:
        """Generator function for reading bare d3plot files

        Parameters
        ----------
        buffered_reading: bool
            whether to read one state at a time
        state_filter: Union[None, Set[int]]
            which states to filter out

        Yields
        ------
        buffer: BinaryBuffer
            buffer for each file
        n_states: int
            number of states from second yield on
        """

        # (1) STATES
        # This is dangerous. The following routine requires data from
        # several sections in the geometry part calling this too early crashes
        bytes_per_state = self._compute_n_bytes_per_state()
        file_infos = self._collect_file_infos(bytes_per_state)

        # some status
        n_files = len(file_infos)
        n_states = sum(map(lambda file_info: file_info.n_states, file_infos))
        LOGGER.debug("n_files found: %d", n_files)
        LOGGER.debug("n_states estimated: %d", n_states)

        # convert negative state indexes into positive ones
        if state_filter is not None:
            state_filter = _negative_to_positive_state_indexes(state_filter, n_states)

        # if using buffered reading, we load one state at a time
        # into memory
        if buffered_reading:
            file_infos_tmp: List[MemoryInfo] = []
            n_previous_states = 0
            for minfo in file_infos:
                for i_file_state in range(minfo.n_states):
                    i_global_state = n_previous_states + i_file_state

                    # do we need to skip this one
                    if state_filter and i_global_state not in state_filter:
                        continue

                    file_infos_tmp.append(
                        MemoryInfo(
                            start=minfo.start + i_file_state * bytes_per_state,
                            length=bytes_per_state,
                            filepath=minfo.filepath,
                            n_states=1,
                            filesize=minfo.filesize,
                            use_mmap=minfo.n_states != 1,
                        )
                    )

                n_previous_states += minfo.n_states
            file_infos = file_infos_tmp

        LOGGER.debug("buffers: %s", pprint.pformat([info.__dict__ for info in file_infos]))

        # number of states and if buffered reading is used
        n_states_selected = sum(map(lambda file_info: file_info.n_states, file_infos))
        yield n_states_selected

        sub_file_infos = [file_infos] if not buffered_reading else [[info] for info in file_infos]
        for sub_file_info_list in sub_file_infos:
            buffer, n_states = D3plot._read_file_from_memory_info(sub_file_info_list)
            yield buffer, n_states

    def _read_femzip_file_generator(
        self, buffered_reading: bool, state_filter: Union[None, Set[int]]
    ) -> typing.Any:
        """Generator function for reading femzipped d3plot files

        Parameters
        ----------
        buffered_reading: bool
            load state by state
        state_filter: Union[None, Set[int]]
            which states to filter out

        Yields
        ------
        buffer: BinaryBuffer
            binary buffer of a file
        n_states: int
            from second yield on, number of states for buffers
        """

        femzip_api = self._femzip_info.api

        # (1) STATES
        # number of states and if buffered reading is used
        buffer_info = self._femzip_info.buffer_info
        n_timesteps: int = buffer_info.n_timesteps

        # convert negative filter indexes
        state_filter_parsed: Set[int] = set()
        if state_filter is not None:
            state_filter_parsed = _negative_to_positive_state_indexes(state_filter, n_timesteps)
            n_states_to_load = len(state_filter)
        else:
            n_states_to_load = n_timesteps
            state_filter_parsed = set(range(n_timesteps))

        yield n_states_to_load

        n_files_to_load_at_once = n_timesteps if not buffered_reading else 1
        # pylint: disable = invalid-name
        BufferStateType = ctypes.c_float * (buffer_info.size_state * n_files_to_load_at_once)
        buffer_state = BufferStateType()

        buffer = BinaryBuffer()
        buffer.memoryview = memoryview(buffer_state)

        # do the thing
        i_timesteps_read = 0
        max_timestep = max(state_filter_parsed) if state_filter_parsed else 0
        for i_timestep in range(n_timesteps):

            # buffer offset
            buffer_current_state = buffer.memoryview[i_timesteps_read * buffer_info.size_state :]

            # read state
            femzip_api.read_single_state(i_timestep, buffer_info, state_buffer=buffer_current_state)

            if i_timestep in state_filter_parsed:
                i_timesteps_read += 1

            # Note:
            # the buffer is re-used here! This saves memory BUT
            # if memory is not copied we overwrite the same again and again
            # This is ok for buffered reading thus indirectly safe
            # since elsewhere the arrays get copied but keep it in mind!
            if i_timesteps_read != 0 and i_timesteps_read % n_files_to_load_at_once == 0:
                yield buffer, i_timesteps_read
                i_timesteps_read = 0

            # stop in case we have everything we needed
            if i_timestep >= max_timestep:
                if i_timesteps_read != 0:
                    yield buffer, i_timesteps_read
                break

        # do the thing
        femzip_api.close_current_file()

    def _read_states_femzip_advanced(self, filepath: str) -> None:
        """Read d3plot variables with advanced femzip API

        Parameters
        ----------
        filepath: str
            path to the femzipped d3plot
        """

        # convert filter
        d3plot_array_filter = set(self.state_array_filter) if self.state_array_filter else None

        # what vars are inside?
        api = self._femzip_info.api
        file_metadata = api.get_file_metadata(filepath)

        if file_metadata.number_of_timesteps <= 0:
            return

        # filter femzip vars according to requested d3plot vars
        file_metadata_filtered = filter_femzip_variables(
            file_metadata,
            d3plot_array_filter,
        )

        # read femzip arrays
        result_arrays = api.read_variables(
            file_metadata=file_metadata_filtered,
            n_parts=self.header.n_parts,
            n_rigid_walls=self._n_rigid_walls,
            n_rigid_wall_vars=self.header.n_rigid_wall_vars,
            n_airbag_particles=self._airbag_info.n_particles,
            n_airbags=self._airbag_info.n_airbags,
            state_filter=self.state_filter,
        )

        # special case arrays which need extra parsing
        keys_to_remove = []
        for (fz_index, fz_name, fz_cat), array in result_arrays.items():

            # global vars
            if fz_cat == FemzipVariableCategory.GLOBAL:
                keys_to_remove.append((fz_index, fz_name, fz_cat))
                self._read_states_globals(
                    state_data=array,
                    var_index=0,
                    array_dict=self.arrays,
                )

            # parts and rigid walls
            elif fz_cat == FemzipVariableCategory.PART:
                keys_to_remove.append((fz_index, fz_name, fz_cat))

                var_index = self._read_states_parts(
                    state_data=array, var_index=0, array_dict=self.arrays
                )

                self._read_states_rigid_walls(
                    state_data=array, var_index=var_index, array_dict=self.arrays
                )

        for key in keys_to_remove:
            del result_arrays[key]

        # transfer arrays
        mapper = FemzipMapper()
        mapper.map(result_arrays)

        # save arrays
        for plt_name, arr in mapper.d3plot_arrays.items():

            # femzip sometimes stores strain in solid history vars
            # but also sometimes separately
            if (
                plt_name == ArrayType.element_solid_history_variables
                and self.header.has_element_strain
                and ArrayType.element_solid_strain not in mapper.d3plot_arrays
            ):
                self.arrays[ArrayType.element_solid_strain] = arr[:, :, :, :6]
                tmp_array = arr[:, :, :, 6:]
                if all(tmp_array.shape):
                    self.arrays[plt_name] = tmp_array
            else:
                self.arrays[plt_name] = arr

        # ELEMENT DELETION
        #
        # somehow element deletion info is extra ...
        # buffer_info
        buffer_info = self._femzip_info.buffer_info
        deletion_array = api.read_state_deletion_info(
            buffer_info=buffer_info, state_filter=self.state_filter
        )
        self._read_states_is_alive(state_data=deletion_array, var_index=0, array_dict=self.arrays)

        # TIMESTEPS
        timestep_array = np.array(
            [buffer_info.timesteps[i_timestep] for i_timestep in range(buffer_info.n_timesteps)],
            dtype=self.header.ftype,
        )
        self.arrays[ArrayType.global_timesteps] = timestep_array

    def _read_header(self):
        """Read the d3plot header"""

        LOGGER.debug("-------- H E A D E R --------")

        if self._buffer:
            self._header.load_file(self._buffer)

        self.geometry_section_size = self._header.n_header_bytes

    def _parse_geometry(self):
        """Read the d3plot geometry"""

        LOGGER.debug("------ G E O M E T R Y ------")

        # read material section
        self._read_material_section()

        # read fluid material data
        self._read_fluid_material_data()

        # SPH element data flags
        self._read_sph_element_data_flags()

        # Particle Data
        self._read_particle_data()

        # Geometry Data
        self._read_geometry_data()

        # User Material, Node, Blabla IDs
        self._read_user_ids()

        # Rigid Body Description
        self._read_rigid_body_description()

        # Adapted Element Parent List
        # manual says not implemented

        # Smooth Particle Hydrodynamcis Node and Material list
        self._read_sph_node_and_material_list()

        # Particle Geometry Data
        self._read_particle_geometry_data()

        # Rigid Road Surface Data
        self._read_rigid_road_surface()

        # Connectivity for weirdo elements
        # 10 Node Tetra
        # 8 Node Shell
        # 20 Node Solid
        # 27 Node Solid
        self._read_extra_node_connectivity()

        # Header Part & Contact Interface Titles
        # this is a class method since it is also needed elsewhere
        self.geometry_section_size = self._read_header_part_contact_interface_titles(
            self.header,
            self._buffer,
            self.geometry_section_size,  # type: ignore
            self.arrays,
        )

        # Extra Data Types (for multi solver output)
        # ... not supported

    def _read_material_section(self):
        """This function reads the material type section"""

        if not self._buffer:
            return

        if not self.header.has_material_type_section:
            return

        LOGGER.debug("_read_material_section start at byte %d", self.geometry_section_size)

        position = self.geometry_section_size

        # failsafe
        original_position = self.geometry_section_size
        blocksize = (2 + self.header.n_parts) * self.header.wordsize

        try:

            # Material Type Data
            #
            # "This data is required because those shell elements
            # that are in a rigid body have no element data output
            # in the state data section."
            #
            # "The normal length of the shell element state data is:
            # NEL4 * NV2D, when the MATTYP flag is set the length is:
            # (NEL4 – NUMRBE) * NV2D. When reading the shell element data,
            # the material number must be checked against IRBRTYP list to
            # find the element’s material type. If the type = 20, then
            # all the values for the element to zero." (Manual 03.2016)

            self._material_section_info.n_rigid_shells = int(
                self._buffer.read_number(position, self._header.itype)
            )  # type: ignore
            position += self.header.wordsize

            test_nummat = self._buffer.read_number(position, self._header.itype)
            position += self.header.wordsize

            if test_nummat != self.header.n_parts:
                raise RuntimeError(
                    "nmmat (header) != nmmat (material type data): "
                    f"{self.header.n_parts} != {test_nummat}",
                )

            self.arrays[ArrayType.part_material_type] = self._buffer.read_ndarray(
                position, self.header.n_parts * self.header.wordsize, 1, self.header.itype
            )
            position += self.header.n_parts * self.header.wordsize

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            LOGGER.warning("A failure in %s was caught:\n%s", "_read_material_section", trb_msg)

            # fix position
            position = original_position + blocksize

        self.geometry_section_size = position
        LOGGER.debug("_read_material_section end at byte %d", self.geometry_section_size)

    def _read_fluid_material_data(self):
        """Read the fluid material data"""

        if not self._buffer:
            return

        if self.header.n_ale_materials == 0:
            return

        LOGGER.debug("_read_fluid_material_data start at byte %d", self.geometry_section_size)

        position = self.geometry_section_size

        # safety
        original_position = position
        blocksize = self.header.n_ale_materials * self.header.wordsize

        try:
            # Fluid Material Data
            array_length = self.header.n_ale_materials * self.header.wordsize
            self.arrays[ArrayType.ale_material_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self.header.itype
            )  # type: ignore
            position += array_length

        except Exception:

            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_fluid_material_data", trb_msg)

            # fix position
            position = original_position + blocksize

        # remember position
        self.geometry_section_size = position
        LOGGER.debug("_read_fluid_material_data end at byte %d", self.geometry_section_size)

    def _read_sph_element_data_flags(self):
        """Read the sph element data flags"""

        if not self._buffer:
            return

        if not self.header.n_sph_nodes:
            return

        LOGGER.debug("_read_sph_element_data_flags start at byte %d", self.geometry_section_size)

        position = self.geometry_section_size

        sph_element_data_words = {
            "isphfg1": (position, self._header.itype),
            "isphfg2": (position + 1 * self.header.wordsize, self._header.itype),
            "isphfg3": (position + 2 * self.header.wordsize, self._header.itype),
            "isphfg4": (position + 3 * self.header.wordsize, self._header.itype),
            "isphfg5": (position + 4 * self.header.wordsize, self._header.itype),
            "isphfg6": (position + 5 * self.header.wordsize, self._header.itype),
            "isphfg7": (position + 6 * self.header.wordsize, self._header.itype),
            "isphfg8": (position + 7 * self.header.wordsize, self._header.itype),
            "isphfg9": (position + 8 * self.header.wordsize, self._header.itype),
            "isphfg10": (position + 9 * self.header.wordsize, self._header.itype),
            "isphfg11": (position + 10 * self.header.wordsize, self._header.itype),
        }

        sph_header_data = self.header.read_words(self._buffer, sph_element_data_words)

        self._sph_info.n_sph_array_length = sph_header_data["isphfg1"]
        self._sph_info.has_influence_radius = sph_header_data["isphfg2"] != 0
        self._sph_info.has_particle_pressure = sph_header_data["isphfg3"] != 0
        self._sph_info.has_stresses = sph_header_data["isphfg4"] != 0
        self._sph_info.has_plastic_strain = sph_header_data["isphfg5"] != 0
        self._sph_info.has_material_density = sph_header_data["isphfg6"] != 0
        self._sph_info.has_internal_energy = sph_header_data["isphfg7"] != 0
        self._sph_info.has_n_affecting_neighbors = sph_header_data["isphfg8"] != 0
        self._sph_info.has_strain_and_strainrate = sph_header_data["isphfg9"] != 0
        self._sph_info.has_true_strains = sph_header_data["isphfg9"] < 0
        self._sph_info.has_mass = sph_header_data["isphfg10"] != 0
        self._sph_info.n_sph_history_vars = sph_header_data["isphfg11"]

        if self._sph_info.n_sph_array_length != 11:
            msg = (
                "Detected inconsistency: "
                f"isphfg = {self._sph_info.n_sph_array_length} but must be 11."
            )
            raise RuntimeError(msg)

        self._sph_info.n_sph_vars = (
            sph_header_data["isphfg2"]
            + sph_header_data["isphfg3"]
            + sph_header_data["isphfg4"]
            + sph_header_data["isphfg5"]
            + sph_header_data["isphfg6"]
            + sph_header_data["isphfg7"]
            + sph_header_data["isphfg8"]
            + abs(sph_header_data["isphfg9"])
            + sph_header_data["isphfg10"]
            + sph_header_data["isphfg11"]
            + 1
        )  # material number

        self.geometry_section_size += sph_header_data["isphfg1"] * self.header.wordsize
        LOGGER.debug("_read_sph_element_data_flags end at byte %d", self.geometry_section_size)

    def _read_particle_data(self):
        """Read the geometry section for particle data (airbags)"""

        if not self._buffer:
            return

        if "npefg" not in self.header.raw_header:
            return
        npefg = self.header.raw_header["npefg"]

        # let's stick to the manual, too lazy to decypther this test
        if npefg <= 0 or npefg > 10000000:
            return

        LOGGER.debug("_read_particle_data start at byte %d", self.geometry_section_size)

        position = self.geometry_section_size

        airbag_header = {
            # number of airbags
            "npartgas": npefg % 1000,
            # ?
            "subver": npefg // 1000,
        }

        particle_geometry_data_words = {
            # number of geometry variables
            "ngeom": (position, self._header.itype),
            # number of state variables
            "nvar": (position + 1 * self.header.wordsize, self._header.itype),
            # number of particles
            "npart": (position + 2 * self.header.wordsize, self._header.itype),
            # number of state geometry variables
            "nstgeom": (position + 3 * self.header.wordsize, self._header.itype),
        }

        self.header.read_words(self._buffer, particle_geometry_data_words, airbag_header)
        position += 4 * self.header.wordsize

        # transfer to info object
        self._airbag_info.n_airbags = npefg % 1000
        self._airbag_info.subver = npefg // 1000
        self._airbag_info.n_geometric_variables = airbag_header["ngeom"]
        self._airbag_info.n_particle_state_variables = airbag_header["nvar"]
        self._airbag_info.n_particles = airbag_header["npart"]
        self._airbag_info.n_airbag_state_variables = airbag_header["nstgeom"]

        if self._airbag_info.subver == 4:
            # number of chambers
            self._airbag_info.n_chambers = self._buffer.read_number(position, self._header.itype)
            position += self.header.wordsize

        n_airbag_variables = self._airbag_info.get_n_variables()

        # safety
        # from here on the code may fail
        original_position = position
        blocksize = 9 * n_airbag_variables * self.header.wordsize

        try:
            # variable typecodes
            self.arrays[ArrayType.airbag_variable_types] = self._buffer.read_ndarray(
                position, n_airbag_variables * self.header.wordsize, 1, self._header.itype
            )
            position += n_airbag_variables * self.header.wordsize

            # airbag variable names
            # every word is an ascii char
            airbag_variable_names = []
            var_width = 8

            for i_variable in range(n_airbag_variables):
                name = self._buffer.read_text(
                    position + (i_variable * var_width) * self.header.wordsize,
                    var_width * self.header.wordsize,
                )
                airbag_variable_names.append(name[:: self.header.wordsize])

            self.arrays[ArrayType.airbag_variable_names] = airbag_variable_names
            position += n_airbag_variables * var_width * self.header.wordsize

        except Exception:

            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_particle_data", trb_msg)

            # fix position
            position = original_position + blocksize

        # update position marker
        self.geometry_section_size = position
        LOGGER.debug("_read_particle_data start at byte %d", self.geometry_section_size)

    # pylint: disable = too-many-branches
    def _read_geometry_data(self):
        """Read the data from the geometry section"""

        if not self._buffer:
            return

        LOGGER.debug("_read_geometry_data start at byte %d", self.geometry_section_size)

        # not sure but I think never used by LS-Dyna
        # anyway needs to be detected in the header and not here,
        # though it is mentioned in this section of the database manual
        #
        # is_packed = True if self.header['ndim'] == 3 else False
        # if is_packed:
        #     raise RuntimeError("Can not deal with packed "\
        #                        "geometry data (ndim == {}).".format(self.header['ndim']))

        position = self.geometry_section_size

        # node coords
        n_nodes = self.header.n_nodes
        n_dimensions = self.header.n_dimensions
        section_word_length = n_dimensions * n_nodes
        try:
            node_coordinates = self._buffer.read_ndarray(
                position, section_word_length * self.header.wordsize, 1, self.header.ftype
            ).reshape((n_nodes, n_dimensions))
            self.arrays[ArrayType.node_coordinates] = node_coordinates
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %d was caught:\n%s"
            LOGGER.warning(msg, "_read_geometry_data, node_coordinates", trb_msg)
        finally:
            position += section_word_length * self.header.wordsize

        # solid data
        n_solids = self.header.n_solids
        section_word_length = 9 * n_solids
        try:
            elem_solid_data = self._buffer.read_ndarray(
                position, section_word_length * self.header.wordsize, 1, self._header.itype
            ).reshape((n_solids, 9))
            solid_connectivity = elem_solid_data[:, :8]
            solid_part_indexes = elem_solid_data[:, 8]
            self.arrays[ArrayType.element_solid_node_indexes] = solid_connectivity - FORTRAN_OFFSET
            self.arrays[ArrayType.element_solid_part_indexes] = solid_part_indexes - FORTRAN_OFFSET
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_geometry_data, solids_geometry", trb_msg)
        finally:
            position += section_word_length * self.header.wordsize

        # ten node solids extra nodes
        if self.header.has_solid_2_extra_nodes:
            section_word_length = 2 * n_solids
            try:
                self.arrays[
                    ArrayType.element_solid_extra_nodes
                ] = elem_solid_data = self._buffer.read_ndarray(
                    position, section_word_length * self.header.wordsize, 1, self._header.itype
                ).reshape(
                    (n_solids, 2)
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_geometry_data, ten_node_solids", trb_msg)
            finally:
                position += section_word_length * self.header.wordsize

        # 8 node thick shells
        n_thick_shells = self.header.n_thick_shells
        section_word_length = 9 * n_thick_shells
        try:
            elem_tshell_data = self._buffer.read_ndarray(
                position, section_word_length * self.header.wordsize, 1, self._header.itype
            ).reshape((self.header.n_thick_shells, 9))
            self.arrays[ArrayType.element_tshell_node_indexes] = (
                elem_tshell_data[:, :8] - FORTRAN_OFFSET
            )
            self.arrays[ArrayType.element_tshell_part_indexes] = (
                elem_tshell_data[:, 8] - FORTRAN_OFFSET
            )
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_geometry_data, tshells_geometry", trb_msg)
        finally:
            position += section_word_length * self.header.wordsize

        # beams
        n_beams = self.header.n_beams
        section_word_length = 6 * n_beams
        try:
            elem_beam_data = self._buffer.read_ndarray(
                position, section_word_length * self.header.wordsize, 1, self._header.itype
            ).reshape((n_beams, 6))
            self.arrays[ArrayType.element_beam_part_indexes] = elem_beam_data[:, 5] - FORTRAN_OFFSET
            self.arrays[ArrayType.element_beam_node_indexes] = (
                elem_beam_data[:, :5] - FORTRAN_OFFSET
            )
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_geometry_data, beams_geometry", trb_msg)
        finally:
            position += section_word_length * self.header.wordsize

        # shells
        n_shells = self.header.n_shells
        section_word_length = 5 * n_shells
        try:
            elem_shell_data = self._buffer.read_ndarray(
                position, section_word_length * self.header.wordsize, 1, self._header.itype
            ).reshape((self.header.n_shells, 5))
            self.arrays[ArrayType.element_shell_node_indexes] = (
                elem_shell_data[:, :4] - FORTRAN_OFFSET
            )
            self.arrays[ArrayType.element_shell_part_indexes] = (
                elem_shell_data[:, 4] - FORTRAN_OFFSET
            )
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_geometry_data, shells_geometry", trb_msg)
        finally:
            position += section_word_length * self.header.wordsize

        # update word position
        self.geometry_section_size = position

        LOGGER.debug("_read_geometry_data end at byte %d", self.geometry_section_size)

    def _read_user_ids(self):

        if not self._buffer:
            return

        if not self.header.has_numbering_section:
            self.arrays[ArrayType.node_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_nodes + FORTRAN_OFFSET, dtype=self.header.itype
            )
            self.arrays[ArrayType.element_solid_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_solids + FORTRAN_OFFSET, dtype=self.header.itype
            )
            self.arrays[ArrayType.element_beam_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_beams + FORTRAN_OFFSET, dtype=self.header.itype
            )
            self.arrays[ArrayType.element_shell_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_shells + FORTRAN_OFFSET, dtype=self.header.itype
            )
            self.arrays[ArrayType.element_tshell_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_thick_shells + FORTRAN_OFFSET, dtype=self.header.itype
            )
            self.arrays[ArrayType.part_ids] = np.arange(
                FORTRAN_OFFSET, self.header.n_parts + FORTRAN_OFFSET, dtype=self.header.itype
            )
            return

        LOGGER.debug("_read_user_ids start at byte %d", self.geometry_section_size)

        position = self.geometry_section_size

        # safety
        original_position = position
        blocksize = self.header.raw_header["narbs"] * self.header.wordsize

        try:
            numbering_words = {
                "nsort": (position, self._header.itype),
                "nsrh": (position + 1 * self.header.wordsize, self._header.itype),
                "nsrb": (position + 2 * self.header.wordsize, self._header.itype),
                "nsrs": (position + 3 * self.header.wordsize, self._header.itype),
                "nsrt": (position + 4 * self.header.wordsize, self._header.itype),
                "nsortd": (position + 5 * self.header.wordsize, self._header.itype),
                "nsrhd": (position + 6 * self.header.wordsize, self._header.itype),
                "nsrbd": (position + 7 * self.header.wordsize, self._header.itype),
                "nsrsd": (position + 8 * self.header.wordsize, self._header.itype),
                "nsrtd": (position + 9 * self.header.wordsize, self._header.itype),
            }

            extra_numbering_words = {
                "nsrma": (position + 10 * self.header.wordsize, self._header.itype),
                "nsrmu": (position + 11 * self.header.wordsize, self._header.itype),
                "nsrmp": (position + 12 * self.header.wordsize, self._header.itype),
                "nsrtm": (position + 13 * self.header.wordsize, self._header.itype),
                "numrbs": (position + 14 * self.header.wordsize, self._header.itype),
                "nmmat": (position + 15 * self.header.wordsize, self._header.itype),
            }

            numbering_header = self.header.read_words(self._buffer, numbering_words)
            position += len(numbering_words) * self.header.wordsize

            # let's make life easier
            info = self._numbering_info

            # transfer first bunch
            info.ptr_node_ids = abs(numbering_header["nsort"])
            info.has_material_ids = numbering_header["nsort"] < 0
            info.ptr_solid_ids = numbering_header["nsrh"]
            info.ptr_beam_ids = numbering_header["nsrb"]
            info.ptr_shell_ids = numbering_header["nsrs"]
            info.ptr_thick_shell_ids = numbering_header["nsrt"]
            info.n_nodes = numbering_header["nsortd"]
            info.n_solids = numbering_header["nsrhd"]
            info.n_beams = numbering_header["nsrbd"]
            info.n_shells = numbering_header["nsrsd"]
            info.n_thick_shells = numbering_header["nsrtd"]

            if info.has_material_ids:

                # read extra header
                self.header.read_words(self._buffer, extra_numbering_words, numbering_header)
                position += len(extra_numbering_words) * self.header.wordsize

                # transfer more
                info.ptr_material_ids = numbering_header["nsrma"]
                info.ptr_material_ids_defined_order = numbering_header["nsrmu"]
                info.ptr_material_ids_crossref = numbering_header["nsrmp"]
                info.n_parts = numbering_header["nsrtm"]
                info.n_rigid_bodies = numbering_header["numrbs"]
                info.n_parts2 = numbering_header["nmmat"]
            else:
                info.n_parts = self.header.n_parts

            # let's do a quick check
            n_words_computed = (
                len(numbering_header)
                + info.n_nodes
                + info.n_shells
                + info.n_beams
                + info.n_solids
                + info.n_thick_shells
                + info.n_parts * 3
            )
            if n_words_computed != self.header.n_numbering_section_words:
                warn_msg = (
                    "ID section: The computed word count does "
                    "not match the header word count: %d != %d."
                    " The ID arrays might contain errors."
                )
                LOGGER.warning(warn_msg, n_words_computed, self.header.n_numbering_section_words)
            # node ids
            array_length = info.n_nodes * self.header.wordsize
            self.arrays[ArrayType.node_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length
            # solid ids
            array_length = info.n_solids * self.header.wordsize
            self.arrays[ArrayType.element_solid_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length
            # beam ids
            array_length = info.n_beams * self.header.wordsize
            self.arrays[ArrayType.element_beam_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length
            # shell ids
            array_length = info.n_shells * self.header.wordsize
            self.arrays[ArrayType.element_shell_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length
            # tshell ids
            array_length = info.n_thick_shells * self.header.wordsize
            self.arrays[ArrayType.element_tshell_ids] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length

            # part ids
            #
            # this makes no sense but materials are output three times at this section
            # but the length of the array (nmmat) is only output if nsort < 0. In
            # the other case the length is unknown ...
            #
            # Bugfix:
            # The material arrays (three times) are always output, even if nsort < 0
            # which means they are not used. Quite confusing, especially since nmmat
            # is output in the main header and numbering header.
            #
            if "nmmat" in numbering_header:

                if info.n_parts != self.header.n_parts:
                    err_msg = (
                        "nmmat in the file header (%d) and in the "
                        "numbering header (%d) are inconsistent."
                    )
                    raise RuntimeError(err_msg, self.header.n_parts, info.n_parts)

                array_length = info.n_parts * self.header.wordsize

                self.arrays[ArrayType.part_ids] = self._buffer.read_ndarray(
                    position, info.n_parts * self.header.wordsize, 1, self._header.itype
                )
                position += info.n_parts * self.header.wordsize

                self.arrays[ArrayType.part_ids_unordered] = self._buffer.read_ndarray(
                    position, info.n_parts * self.header.wordsize, 1, self._header.itype
                )
                position += info.n_parts * self.header.wordsize

                self.arrays[ArrayType.part_ids_cross_references] = self._buffer.read_ndarray(
                    position, info.n_parts * self.header.wordsize, 1, self._header.itype
                )
                position += info.n_parts * self.header.wordsize

            else:
                position += 3 * self.header.n_parts * self.header.wordsize

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_user_ids", trb_msg)

            # fix position
            position = original_position + blocksize

        # update position
        self.geometry_section_size = position
        LOGGER.debug("_read_user_ids end at byte %d", self.geometry_section_size)

    def _read_rigid_body_description(self):
        """Read the rigid body description section"""

        if not self._buffer:
            return

        if not self.header.has_rigid_body_data:
            return

        LOGGER.debug("_read_rigid_body_description start at byte %d", self.geometry_section_size)

        position = self.geometry_section_size

        rigid_body_description_header = {
            "nrigid": self._buffer.read_number(position, self._header.itype)
        }
        position += self.header.wordsize

        info = self._rigid_body_info
        info.n_rigid_bodies = rigid_body_description_header["nrigid"]

        rigid_bodies: List[RigidBodyMetadata] = []
        for _ in range(info.n_rigid_bodies):
            rigid_body_info = {
                # rigid body part internal number
                "mrigid": self._buffer.read_number(position, self._header.itype),
                # number of nodes in rigid body
                "numnodr": self._buffer.read_number(
                    position + self.header.wordsize, self._header.itype
                ),
            }
            position += 2 * self.header.wordsize

            # internal node number of rigid body
            array_length = rigid_body_info["numnodr"] * self.header.wordsize
            rigid_body_info["noder"] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length

            # number of active (non-rigid) nodes
            rigid_body_info["numnoda"] = self._buffer.read_number(position, self._header.itype)
            position += self.header.wordsize

            # internal node numbers of active nodes
            array_length = rigid_body_info["numnoda"] * self.header.wordsize
            rigid_body_info["nodea"] = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            )
            position += array_length

            # transfer props
            body_metadata = RigidBodyMetadata(
                internal_number=rigid_body_info["mrigid"],
                n_nodes=rigid_body_info["numnodr"],
                node_indexes=rigid_body_info["noder"],
                n_active_nodes=rigid_body_info["numnoda"],
                active_node_indexes=rigid_body_info["nodea"],
            )

            # append to list
            rigid_bodies.append(body_metadata)

        # save rigid body info to header
        info.rigid_body_metadata_list = rigid_bodies

        # save arrays
        rigid_body_n_nodes = []
        rigid_body_part_indexes = []
        rigid_body_n_active_nodes = []
        rigid_body_node_indexes_list = []
        rigid_body_active_node_indexes_list = []
        for rigid_body_info in rigid_bodies:
            rigid_body_part_indexes.append(rigid_body_info.internal_number)
            rigid_body_n_nodes.append(rigid_body_info.n_nodes)
            rigid_body_node_indexes_list.append(rigid_body_info.node_indexes - FORTRAN_OFFSET)
            rigid_body_n_active_nodes.append(rigid_body_info.n_active_nodes)
            rigid_body_active_node_indexes_list.append(
                rigid_body_info.active_node_indexes - FORTRAN_OFFSET
            )

        self.arrays[ArrayType.rigid_body_part_indexes] = (
            np.array(rigid_body_part_indexes, dtype=self._header.itype) - FORTRAN_OFFSET
        )
        self.arrays[ArrayType.rigid_body_n_nodes] = np.array(
            rigid_body_n_nodes, dtype=self._header.itype
        )
        self.arrays[ArrayType.rigid_body_n_active_nodes] = np.array(
            rigid_body_n_active_nodes, dtype=self._header.itype
        )
        self.arrays[ArrayType.rigid_body_node_indexes_list] = rigid_body_node_indexes_list
        self.arrays[
            ArrayType.rigid_body_active_node_indexes_list
        ] = rigid_body_active_node_indexes_list

        # update position
        self.geometry_section_size = position
        LOGGER.debug("_read_rigid_body_description end at byte %d", self.geometry_section_size)

    def _read_sph_node_and_material_list(self):
        """Read SPH node and material list"""

        if not self._buffer:
            return

        if self.header.n_sph_nodes <= 0:
            return

        LOGGER.debug(
            "_read_sph_node_and_material_list start at byte %d", self.geometry_section_size
        )

        position = self.geometry_section_size

        array_length = self.header.n_sph_nodes * self.header.wordsize * 2
        try:
            # read info array
            sph_node_matlist = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            ).reshape((self.header.n_sph_nodes, 2))

            # save array
            self.arrays[ArrayType.sph_node_indexes] = sph_node_matlist[:, 0] - FORTRAN_OFFSET
            self.arrays[ArrayType.sph_node_material_index] = sph_node_matlist[:, 1] - FORTRAN_OFFSET

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_sph_node_and_material_list", trb_msg)

        finally:
            # update position
            self.geometry_section_size += array_length

        LOGGER.debug("_read_sph_node_and_material_list end at byte %d", self.geometry_section_size)

    def _read_particle_geometry_data(self):
        """Read the particle geometry data"""

        if not self._buffer:
            return

        if "npefg" not in self.header.raw_header:
            return

        if self.header.raw_header["npefg"] <= 0:
            return

        LOGGER.debug("_read_particle_geometry_data start at byte %d", self.geometry_section_size)

        info = self._airbag_info

        position = self.geometry_section_size

        # size of geometry section checking
        ngeom = info.n_geometric_variables
        if ngeom not in [4, 5]:
            raise RuntimeError("variable ngeom in the airbag header must be 4 or 5.")

        original_position = position
        blocksize = info.n_airbags * ngeom * self.header.wordsize
        try:

            # extract geometry as a single array
            array_length = blocksize
            particle_geom_data = self._buffer.read_ndarray(
                position, array_length, 1, self._header.itype
            ).reshape((info.n_airbags, ngeom))
            position += array_length

            # store arrays
            self.arrays[ArrayType.airbags_first_particle_id] = particle_geom_data[:, 0]
            self.arrays[ArrayType.airbags_n_particles] = particle_geom_data[:, 1]
            self.arrays[ArrayType.airbags_ids] = particle_geom_data[:, 2]
            self.arrays[ArrayType.airbags_n_gas_mixtures] = particle_geom_data[:, 3]
            if ngeom == 5:
                self.arrays[ArrayType.airbags_n_chambers] = particle_geom_data[:, 4]

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in %d was caught:\n%s"
            LOGGER.warning(msg, "_read_particle_geometry_data", trb_msg)

            # fix position
            position = original_position + blocksize

        # update position
        self.geometry_section_size = position

        LOGGER.debug("_read_particle_geometry_data end at byte %d", self.geometry_section_size)

    def _read_rigid_road_surface(self):
        """Read rigid road surface data"""

        if not self._buffer:
            return

        if not self.header.has_rigid_road_surface:
            return

        LOGGER.debug("_read_rigid_road_surface start at byte %d", self.geometry_section_size)

        position = self.geometry_section_size

        # read header
        rigid_road_surface_words = {
            "nnode": (position, self._header.itype),
            "nseg": (position + 1 * self.header.wordsize, self._header.itype),
            "nsurf": (position + 2 * self.header.wordsize, self._header.itype),
            "motion": (position + 3 * self.header.wordsize, self._header.itype),
        }

        rigid_road_header = self.header.read_words(self._buffer, rigid_road_surface_words)
        position += 4 * self.header.wordsize

        self._rigid_road_info = RigidRoadInfo(
            n_nodes=rigid_road_header["nnode"],
            n_roads=rigid_road_header["nsurf"],
            n_road_segments=rigid_road_header["nseg"],
            motion=rigid_road_header["motion"],
        )
        info = self._rigid_road_info

        # node ids
        array_length = info.n_nodes * self.header.wordsize
        rigid_road_node_ids = self._buffer.read_ndarray(
            position, array_length, 1, self._header.itype
        )
        self.arrays[ArrayType.rigid_road_node_ids] = rigid_road_node_ids
        position += array_length

        # node xyz
        array_length = info.n_nodes * 3 * self.header.wordsize
        rigid_road_node_coords = self._buffer.read_ndarray(
            position, array_length, 1, self.header.ftype
        ).reshape((info.n_nodes, 3))
        self.arrays[ArrayType.rigid_road_node_coordinates] = rigid_road_node_coords
        position += array_length

        # read road segments
        # Warning: must be copied
        rigid_road_ids = np.empty(info.n_roads, dtype=self._header.itype)
        rigid_road_nsegments = np.empty(info.n_roads, dtype=self._header.itype)
        rigid_road_segment_node_ids = []

        # this array is created since the array database requires
        # constant sized arrays, and we dump all segments into one
        # array. In order to distinguish which segment
        # belongs to which road, this new array keeps track of it
        rigid_road_segment_road_id = []

        # n_total_segments = 0
        for i_surf in range(info.n_roads):
            # surface id
            surf_id = self._buffer.read_number(position, self._header.itype)  # type: ignore
            position += self.header.wordsize
            rigid_road_ids[i_surf] = surf_id

            # number of segments of surface
            surf_nseg = self._buffer.read_number(
                position + 1 * self.header.wordsize, self._header.itype
            )  # type: ignore
            position += self.header.wordsize
            rigid_road_nsegments[i_surf] = surf_nseg

            # count total segments
            # n_total_segments += surf_nseg

            # node ids of surface segments
            array_length = 4 * surf_nseg * self.header.wordsize
            surf_segm_node_ids = self._buffer.read_ndarray(
                position,  # type: ignore
                array_length,  # type: ignore
                1,
                self._header.itype,
            ).reshape((surf_nseg, 4))
            position += array_length
            rigid_road_segment_node_ids.append(surf_segm_node_ids)

            # remember road id for segments
            rigid_road_segment_road_id += [surf_id] * surf_nseg

        # save arrays
        self.arrays[ArrayType.rigid_road_ids] = rigid_road_ids
        self.arrays[ArrayType.rigid_road_n_segments] = rigid_road_nsegments
        self.arrays[ArrayType.rigid_road_segment_node_ids] = np.concatenate(
            rigid_road_segment_node_ids
        )
        self.arrays[ArrayType.rigid_road_segment_road_id] = np.asarray(rigid_road_segment_road_id)

        # update position
        self.geometry_section_size = position
        LOGGER.debug("_read_rigid_road_surface end at byte %d", self.geometry_section_size)

    # pylint: disable = too-many-branches
    def _read_extra_node_connectivity(self):
        """Read the extra node data for creepy elements"""

        if not self._buffer:
            return

        LOGGER.debug("_read_extra_node_connectivity start at byte %d", self.geometry_section_size)

        position = self.geometry_section_size

        # extra 2 node connectivity for 10 node tetrahedron elements
        if self.header.has_solid_2_extra_nodes:
            array_length = 2 * self.header.n_solids * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids, 2))
                self.arrays[ArrayType.element_solid_node10_extra_node_indexes] = (
                    array - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_extra_node_connectivity, solid10", trb_msg)
            finally:
                position += array_length

        # 8 node shell elements
        if self.header.n_shells_8_nodes > 0:
            array_length = 5 * self.header.n_shells_8_nodes * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_shells_8_nodes, 5))
                self.arrays[ArrayType.element_shell_node8_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_shell_node8_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_extra_node_connectivity, shell8", trb_msg)
            finally:
                position += array_length

        # 20 node solid elements
        if self.header.n_solids_20_node_hexas > 0:
            array_length = 13 * self.header.n_solids_20_node_hexas * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_20_node_hexas, 13))
                self.arrays[ArrayType.element_solid_node20_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node20_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_extra_node_connectivity, solid20", trb_msg)
            finally:
                position += array_length

        # 27 node solid hexas
        if (
            self.header.n_solids_27_node_hexas > 0
            and self.header.quadratic_elems_has_full_connectivity
        ):
            array_length = 28 * self.header.n_solids_27_node_hexas * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_27_node_hexas, 28))
                self.arrays[ArrayType.element_solid_node27_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node27_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_extra_node_connectivity, solid27", trb_msg)
            finally:
                position += array_length

        # 21 node solid pentas
        if (
            self.header.n_solids_21_node_pentas > 0
            and self.header.quadratic_elems_has_full_connectivity
        ):
            array_length = 22 * self.header.n_solids_21_node_pentas * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_21_node_pentas, 22))
                self.arrays[ArrayType.element_solid_node21_penta_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node21_penta_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_extra_node_connectivity, solid21p", trb_msg)
            finally:
                position += array_length

        # 15 node solid tetras
        if (
            self.header.n_solids_15_node_tetras > 0
            and self.header.quadratic_elems_has_full_connectivity
        ):
            # manual says 8 but this seems odd
            array_length = 8 * self.header.n_solids_15_node_tetras * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_15_node_tetras, 8))
                self.arrays[ArrayType.element_solid_node15_tetras_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node15_tetras_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_extra_node_connectivity, solid15t", trb_msg)
            finally:
                position += array_length

        # 20 node solid tetras
        if self.header.n_solids_20_node_tetras > 0 and self.header.has_cubic_solids:
            array_length = 21 * self.header.n_solids_20_node_tetras * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_20_node_tetras, 21))
                self.arrays[ArrayType.element_solid_node20_tetras_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node20_tetras_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_extra_node_connectivity, solid20t", trb_msg)
            finally:
                position += array_length

        # 40 node solid tetras
        if self.header.n_solids_40_node_pentas > 0 and self.header.has_cubic_solids:
            array_length = 41 * self.header.n_solids_40_node_pentas * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_40_node_pentas, 41))
                self.arrays[ArrayType.element_solid_node40_pentas_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node40_pentas_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_extra_node_connectivity, solid40t", trb_msg)
            finally:
                position += array_length

        # 64 node solid tetras
        if self.header.n_solids_64_node_hexas > 0 and self.header.has_cubic_solids:
            array_length = 65 * self.header.n_solids_64_node_hexas * self.header.wordsize
            try:
                array = self._buffer.read_ndarray(
                    position, array_length, 1, self._header.itype
                ).reshape((self.header.n_solids_64_node_hexas, 65))
                self.arrays[ArrayType.element_solid_node64_hexas_element_index] = (
                    array[:, 0] - FORTRAN_OFFSET
                )
                self.arrays[ArrayType.element_solid_node64_hexas_extra_node_indexes] = (
                    array[:, 1:] - FORTRAN_OFFSET
                )
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_extra_node_connectivity, solid64t", trb_msg)
            finally:
                position += array_length

        # update position
        self.geometry_section_size = position

        LOGGER.debug("_read_extra_node_connectivity end at byte %d", self.geometry_section_size)

    # pylint: disable = too-many-branches
    @classmethod
    def _read_header_part_contact_interface_titles(
        cls,
        header: D3plotHeader,
        buffer: Union[BinaryBuffer, None],
        geometry_section_size: int,
        arrays: dict,
    ) -> int:
        """Read the header for the parts, contacts and interfaces

        Parameters
        ----------
        header: D3plotHeader
            d3plot header
        bb: BinaryBuffer
            buffer holding geometry
        geometry_section_size: int
            size of the geometry section until now
        arrays: dict
            dictionary holding arrays and where arrays will be saved into

        Returns
        -------
        geometry_section_size: int
            new size of the geometry section
        """

        if not buffer:
            return geometry_section_size

        if header.filetype not in (
            D3plotFiletype.D3PLOT,
            D3plotFiletype.D3PART,
            D3plotFiletype.INTFOR,
        ):
            return geometry_section_size

        LOGGER.debug(
            "_read_header_part_contact_interface_titles start at byte %d", geometry_section_size
        )

        position = geometry_section_size

        # Security
        #
        # we try to read the titles ahead. If dyna writes multiple files
        # then the first file is geometry only thus failing here has no
        # impact on further state reading.
        # If though states are compressed into the first file then we are
        # in trouble here even when catching here.
        try:
            # there is only output if there is an eof marker
            # at least I think I fixed such a bug in the past
            if not cls._is_end_of_file_marker(buffer, position, header.ftype):
                return geometry_section_size

            position += header.wordsize

            # section have types here according to what is inside
            ntypes = []

            # read first ntype
            current_ntype = buffer.read_number(position, header.itype)

            while current_ntype in [90000, 90001, 90002, 90020]:

                # title output
                if current_ntype == 90000:

                    ntypes.append(current_ntype)
                    position += header.wordsize

                    # Bugfix:
                    # the titles are always 18*4 bytes, even if the wordsize
                    # is 8 bytes for the entire file.
                    titles_wordsize = 4

                    array_length = 18 * titles_wordsize
                    header.title2 = buffer.read_text(position, array_length)
                    position += array_length

                # some title output
                elif current_ntype in [90001, 90002, 90020]:

                    ntypes.append(current_ntype)
                    position += header.wordsize

                    # number of parts
                    entry_count = buffer.read_number(position, header.itype)
                    position += header.wordsize

                    # Bugfix:
                    # the titles are always 18*4 bytes, even if the wordsize
                    # is 8 bytes for the entire file.
                    titles_wordsize = 4

                    # part ids and corresponding titles
                    array_type = np.dtype(
                        [("ids", header.itype), ("titles", "S" + str(18 * titles_wordsize))]
                    )
                    array_length = (header.wordsize + 18 * titles_wordsize) * int(entry_count)
                    tmp_arrays = buffer.read_ndarray(position, array_length, 1, array_type)
                    position += array_length

                    # save stuff
                    if current_ntype == 90001:
                        arrays[ArrayType.part_titles_ids] = tmp_arrays["ids"]
                        arrays[ArrayType.part_titles] = tmp_arrays["titles"]
                    elif current_ntype == 90002:
                        arrays[ArrayType.contact_title_ids] = tmp_arrays["ids"]
                        arrays[ArrayType.contact_titles] = tmp_arrays["titles"]
                    elif current_ntype == 90020:
                        arrays["icfd_part_title_ids"] = tmp_arrays["ids"]
                        arrays["icfd_part_titles"] = tmp_arrays["titles"]

                # d3prop
                elif current_ntype == 90100:

                    ntypes.append(current_ntype)
                    position += header.wordsize

                    # number of keywords
                    nline = buffer.read_number(position, header.itype)
                    position += header.wordsize

                    # Bugfix:
                    # the titles are always 18*4 bytes, even if the wordsize
                    # is 8 bytes for the entire file.
                    titles_wordsize = 4

                    # keywords
                    array_length = 20 * titles_wordsize * int(nline)
                    d3prop_keywords = buffer.read_ndarray(
                        position, array_length, 1, np.dtype("S" + str(titles_wordsize * 20))
                    )
                    position += array_length

                    # save
                    arrays["d3prop_keywords"] = d3prop_keywords

                # not sure whether there is an eof file here
                # do not have a test file to check ...
                if cls._is_end_of_file_marker(buffer, position, header.ftype):
                    position += header.wordsize

                # next one
                if buffer.size <= position:
                    break
                current_ntype = buffer.read_number(position, header.itype)

            header.n_types = tuple(ntypes)

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_header_part_contact_interface_titles", trb_msg)

        # remember position
        geometry_section_size = position
        LOGGER.debug(
            "_read_header_part_contact_interface_titles end at byte %d", geometry_section_size
        )

        return geometry_section_size

    @staticmethod
    def _read_states_allocate_arrays(
        header: D3plotHeader,
        material_section_info: MaterialSectionInfo,
        airbag_info: AirbagInfo,
        rigid_road_info: RigidRoadInfo,
        rigid_body_info: RigidBodyInfo,
        n_states: int,
        n_rigid_walls: int,
        n_parts: int,
        array_names: Union[Iterable[str], None],
        array_dict: dict,
    ) -> None:
        """Allocate the state arrays

        Parameters
        ----------
        header: D3plotHeader
            header of the d3plot
        material_section_info: MaterialSectionInfo
            info about the material section data
        airbag_info: AirbagInfo
            info for airbags
        rigid_road_info: RigidRoadInfo
            info for rigid roads
        rigid_body_info: RigidBodyInfo
            info for rigid bodies
        n_states: int
            number of states to allocate memory for
        n_rigid_walls: int
            number of rigid walls
        n_parts: int
            number of parts
        array_names: Union[Iterable[str], None]
            names of state arrays to allocate (all if None)
        array_dict: dict
            dictionary to allocate arrays into
        """

        # (1) ARRAY SHAPES
        # general
        n_dim = header.n_dimensions
        # nodes
        n_nodes = header.n_nodes
        # solids
        n_solids = header.n_solids
        n_solids_thermal_vars = header.n_solid_thermal_vars
        n_solids_strain_vars = 6 * header.has_element_strain * (header.n_solid_history_vars >= 6)
        n_solid_thermal_strain_vars = 6 * header.has_solid_shell_thermal_strain_tensor
        n_solid_plastic_strain_vars = 6 * header.has_solid_shell_plastic_strain_tensor
        n_solid_layers = header.n_solid_layers
        n_solids_history_vars = (
            header.n_solid_history_vars
            - n_solids_strain_vars
            - n_solid_thermal_strain_vars
            - n_solid_plastic_strain_vars
        )
        # thick shells
        n_tshells = header.n_thick_shells
        n_tshells_history_vars = header.n_shell_tshell_history_vars
        n_tshells_layers = header.n_shell_tshell_layers
        # beams
        n_beams = header.n_beams
        n_beams_history_vars = header.n_beam_history_vars
        n_beam_vars = header.n_beam_vars
        n_beams_layers = max(
            int((-3 * n_beams_history_vars + n_beam_vars - 6) / (n_beams_history_vars + 5)), 0
        )
        # shells
        n_shells = header.n_shells
        n_shells_reduced = header.n_shells - material_section_info.n_rigid_shells
        n_shell_layers = header.n_shell_tshell_layers
        n_shell_history_vars = header.n_shell_tshell_history_vars
        # sph
        allocate_sph = header.n_sph_nodes != 0
        n_sph_particles = header.n_sph_nodes if allocate_sph else 0
        # airbags
        allocate_airbags = header.n_airbags != 0
        n_airbags = header.n_airbags if allocate_airbags else 0
        n_airbag_particles = airbag_info.n_particles if allocate_airbags else 0
        # rigid roads
        allocate_rigid_roads = rigid_road_info.n_roads != 0
        n_roads = rigid_road_info.n_roads if allocate_rigid_roads else 0
        # rigid bodies
        n_rigid_bodies = rigid_body_info.n_rigid_bodies

        # dictionary to lookup array types
        state_array_shapes = {
            # global
            ArrayType.global_timesteps: [n_states],
            ArrayType.global_kinetic_energy: [n_states],
            ArrayType.global_internal_energy: [n_states],
            ArrayType.global_total_energy: [n_states],
            ArrayType.global_velocity: [n_states, 3],
            # parts
            ArrayType.part_internal_energy: [n_states, n_parts],
            ArrayType.part_kinetic_energy: [n_states, n_parts],
            ArrayType.part_velocity: [n_states, n_parts, 3],
            ArrayType.part_mass: [n_states, n_parts],
            ArrayType.part_hourglass_energy: [n_states, n_parts],
            # rigid wall
            ArrayType.rigid_wall_force: [n_states, n_rigid_walls],
            ArrayType.rigid_wall_position: [n_states, n_rigid_walls, 3],
            # nodes
            ArrayType.node_temperature: [n_states, n_nodes, 3]
            if header.has_node_temperature_layers
            else [n_states, n_nodes],
            ArrayType.node_heat_flux: [n_states, n_nodes, 3],
            ArrayType.node_mass_scaling: [n_states, n_nodes],
            ArrayType.node_displacement: [n_states, n_nodes, n_dim],
            ArrayType.node_velocity: [n_states, n_nodes, n_dim],
            ArrayType.node_acceleration: [n_states, n_nodes, n_dim],
            ArrayType.node_temperature_gradient: [n_states, n_nodes],
            ArrayType.node_residual_forces: [n_states, n_nodes, 3],
            ArrayType.node_residual_moments: [n_states, n_nodes, 3],
            # solids
            ArrayType.element_solid_thermal_data: [n_states, n_solids, n_solids_thermal_vars],
            ArrayType.element_solid_stress: [n_states, n_solids, n_solid_layers, 6],
            ArrayType.element_solid_effective_plastic_strain: [n_states, n_solids, n_solid_layers],
            ArrayType.element_solid_history_variables: [
                n_states,
                n_solids,
                n_solid_layers,
                n_solids_history_vars,
            ],
            ArrayType.element_solid_strain: [n_states, n_solids, n_solid_layers, 6],
            ArrayType.element_solid_is_alive: [n_states, n_solids],
            ArrayType.element_solid_plastic_strain_tensor: [n_states, n_solids, n_solid_layers, 6],
            ArrayType.element_solid_thermal_strain_tensor: [n_states, n_solids, n_solid_layers, 6],
            # thick shells
            ArrayType.element_tshell_stress: [n_states, n_tshells, n_tshells_layers, 6],
            ArrayType.element_tshell_effective_plastic_strain: [
                n_states,
                n_tshells,
                n_tshells_layers,
            ],
            ArrayType.element_tshell_history_variables: [
                n_states,
                n_tshells,
                n_tshells_layers,
                n_tshells_history_vars,
            ],
            ArrayType.element_tshell_strain: [n_states, n_tshells, 2, 6],
            ArrayType.element_tshell_is_alive: [n_states, n_tshells],
            # beams
            ArrayType.element_beam_axial_force: [n_states, n_beams],
            ArrayType.element_beam_shear_force: [n_states, n_beams, 2],
            ArrayType.element_beam_bending_moment: [n_states, n_beams, 2],
            ArrayType.element_beam_torsion_moment: [n_states, n_beams],
            ArrayType.element_beam_shear_stress: [n_states, n_beams, n_beams_layers, 2],
            ArrayType.element_beam_axial_stress: [n_states, n_beams, n_beams_layers],
            ArrayType.element_beam_plastic_strain: [n_states, n_beams, n_beams_layers],
            ArrayType.element_beam_axial_strain: [n_states, n_beams, n_beams_layers],
            ArrayType.element_beam_history_vars: [
                n_states,
                n_beams,
                n_beams_layers + 3,
                n_beams_history_vars,
            ],
            ArrayType.element_beam_is_alive: [n_states, n_beams],
            # shells
            ArrayType.element_shell_stress: [n_states, n_shells_reduced, n_shell_layers, 6],
            ArrayType.element_shell_effective_plastic_strain: [
                n_states,
                n_shells_reduced,
                n_shell_layers,
            ],
            ArrayType.element_shell_history_vars: [
                n_states,
                n_shells_reduced,
                n_shell_layers,
                n_shell_history_vars,
            ],
            ArrayType.element_shell_bending_moment: [n_states, n_shells_reduced, 3],
            ArrayType.element_shell_shear_force: [n_states, n_shells_reduced, 2],
            ArrayType.element_shell_normal_force: [n_states, n_shells_reduced, 3],
            ArrayType.element_shell_thickness: [n_states, n_shells_reduced],
            ArrayType.element_shell_unknown_variables: [n_states, n_shells_reduced, 2],
            ArrayType.element_shell_internal_energy: [n_states, n_shells_reduced],
            ArrayType.element_shell_strain: [n_states, n_shells_reduced, 2, 6],
            ArrayType.element_shell_thermal_strain_tensor: [n_states, n_shells_reduced, 6],
            ArrayType.element_shell_plastic_strain_tensor: [
                n_states,
                n_shells_reduced,
                n_shell_layers,
                6,
            ],
            ArrayType.element_shell_is_alive: [n_states, n_shells],
            # sph
            ArrayType.sph_deletion: [n_states, n_sph_particles],
            ArrayType.sph_radius: [n_states, n_sph_particles],
            ArrayType.sph_pressure: [n_states, n_sph_particles],
            ArrayType.sph_stress: [n_states, n_sph_particles, 6],
            ArrayType.sph_effective_plastic_strain: [n_states, n_sph_particles],
            ArrayType.sph_density: [n_states, n_sph_particles],
            ArrayType.sph_internal_energy: [n_states, n_sph_particles],
            ArrayType.sph_n_neighbors: [n_states, n_sph_particles],
            ArrayType.sph_strain: [n_states, n_sph_particles, 6],
            ArrayType.sph_mass: [n_states, n_sph_particles],
            # airbag
            ArrayType.airbag_n_active_particles: [n_states, n_airbags],
            ArrayType.airbag_bag_volume: [n_states, n_airbags],
            ArrayType.airbag_particle_gas_id: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_chamber_id: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_leakage: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_mass: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_radius: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_spin_energy: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_translation_energy: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_nearest_segment_distance: [n_states, n_airbag_particles],
            ArrayType.airbag_particle_position: [n_states, n_airbag_particles, 3],
            ArrayType.airbag_particle_velocity: [n_states, n_airbag_particles, 3],
            # rigid road
            ArrayType.rigid_road_displacement: [n_states, n_roads, 3],
            ArrayType.rigid_road_velocity: [n_states, n_roads, 3],
            # rigid body
            ArrayType.rigid_body_coordinates: [n_states, n_rigid_bodies, 3],
            ArrayType.rigid_body_rotation_matrix: [n_states, n_rigid_bodies, 9],
            ArrayType.rigid_body_velocity: [n_states, n_rigid_bodies, 3],
            ArrayType.rigid_body_rot_velocity: [n_states, n_rigid_bodies, 3],
            ArrayType.rigid_body_acceleration: [n_states, n_rigid_bodies, 3],
            ArrayType.rigid_body_rot_acceleration: [n_states, n_rigid_bodies, 3],
        }

        # only allocate available arrays
        if array_names is None:
            array_names = ArrayType.get_state_array_names()

        # BUGFIX
        # These arrays are actually integer types, all other state arrays
        # are floats
        int_state_arrays = [
            ArrayType.airbag_n_active_particles,
            ArrayType.airbag_particle_gas_id,
            ArrayType.airbag_particle_chamber_id,
            ArrayType.airbag_particle_leakage,
        ]

        # (2) ALLOCATE ARRAYS
        # this looper allocates the arrays specified by the user.
        for array_name in array_names:

            array_dtype = header.ftype if array_name not in int_state_arrays else header.itype

            if array_name in state_array_shapes:
                array_dict[array_name] = np.empty(state_array_shapes[array_name], dtype=array_dtype)
            else:
                raise ValueError(
                    f"Array '{array_name}' is not a state array. "
                    f"Please try one of: {list(state_array_shapes.keys())}",
                )

    @staticmethod
    def _read_states_transfer_memory(
        i_state: int, buffer_array_dict: dict, master_array_dict: dict
    ):
        """Transfers the memory from smaller buffer arrays with only a few
        timesteps into the major one

        Parameters
        ----------
        i_state: int
            current state index
        buffer_array_dict: dict
            dict with arrays of only a few timesteps
        master_array_dict: dict
            dict with the parent master arrays

        Notes
        -----
            If an array in the master dict is not found in the buffer dict
            then this array is set to `None`.
        """

        state_array_names = ArrayType.get_state_array_names()

        arrays_to_delete = []
        for array_name, array in master_array_dict.items():

            # copy memory to big array
            if array_name in buffer_array_dict:
                buffer_array = buffer_array_dict[array_name]
                n_states_buffer_array = buffer_array.shape[0]
                array[i_state : i_state + n_states_buffer_array] = buffer_array
            else:
                # remove unnecesary state arrays (not geometry arrays!)
                # we "could" deal with this in the allocate function
                # by not allocating them but this would replicate code
                # in the reading functions
                if array_name in state_array_names:
                    arrays_to_delete.append(array_name)

        for array_name in arrays_to_delete:
            del master_array_dict[array_name]

    def _compute_n_bytes_per_state(self) -> int:
        """Computes the number of bytes for every state

        Returns
        -------
        n_bytes_per_state: int
            number of bytes of every state
        """

        if not self.header:
            return 0

        # timestep
        timestep_offset = 1 * self.header.wordsize
        # global vars
        global_vars_offset = self.header.n_global_vars * self.header.wordsize
        # node vars
        n_node_vars = (
            self.header.has_node_displacement
            + self.header.has_node_velocity
            + self.header.has_node_acceleration
        ) * self.header.n_dimensions

        if self.header.has_node_temperatures:
            n_node_vars += 1
        if self.header.has_node_temperature_layers:
            n_node_vars += 2
        if self.header.has_node_heat_flux:
            n_node_vars += 3
        if self.header.has_node_mass_scaling:
            n_node_vars += 1
        if self.header.has_node_temperature_gradient:
            n_node_vars += 1
        if self.header.has_node_residual_forces:
            n_node_vars += 3
        if self.header.has_node_residual_moments:
            n_node_vars += 3

        node_data_offset = n_node_vars * self.header.n_nodes * self.header.wordsize
        # thermal shit
        therm_data_offset = (
            self.header.n_solid_thermal_vars * self.header.n_solids * self.header.wordsize
        )
        # solids
        solid_offset = self.header.n_solids * self.header.n_solid_vars * self.header.wordsize
        # tshells
        tshell_offset = (
            self.header.n_thick_shells * self.header.n_thick_shell_vars * self.header.wordsize
        )
        # beams
        beam_offset = self.header.n_beams * self.header.n_beam_vars * self.header.wordsize
        # shells
        shell_offset = (
            (self.header.n_shells - self._material_section_info.n_rigid_shells)
            * self.header.n_shell_vars
            * self.header.wordsize
        )
        # Manual
        # "NOTE: This CFDDATA is no longer output by ls-dyna."
        cfd_data_offset = 0
        # sph
        sph_offset = self.header.n_sph_nodes * self._sph_info.n_sph_vars * self.header.wordsize
        # deleted nodes and elems ... or nothing
        elem_deletion_offset = 0
        if self.header.has_node_deletion_data:
            elem_deletion_offset = self.header.n_nodes * self.header.wordsize
        elif self.header.has_element_deletion_data:
            elem_deletion_offset = (
                self.header.n_beams
                + self.header.n_shells
                + self.header.n_solids
                + self.header.n_thick_shells
            ) * self.header.wordsize
        # airbag particle offset
        if self._airbag_info.n_airbags:
            particle_state_offset = (
                self._airbag_info.n_airbags * self._airbag_info.n_airbag_state_variables
                + self._airbag_info.n_particles * self._airbag_info.n_particle_state_variables
            ) * self.header.wordsize
        else:
            particle_state_offset = 0
        # rigid road stuff whoever uses this
        road_surface_offset = self._rigid_road_info.n_roads * 6 * self.header.wordsize
        # rigid body motion data
        if self.header.has_rigid_body_data:
            n_rigids = self._rigid_body_info.n_rigid_bodies
            n_rigid_vars = 12 if self.header.has_reduced_rigid_body_data else 24
            rigid_body_motion_offset = n_rigids * n_rigid_vars * self.header.wordsize
        else:
            rigid_body_motion_offset = 0
        # ... not supported
        extra_data_offset = 0

        n_bytes_per_state = (
            timestep_offset
            + global_vars_offset
            + node_data_offset
            + therm_data_offset
            + solid_offset
            + tshell_offset
            + beam_offset
            + shell_offset
            + cfd_data_offset
            + sph_offset
            + elem_deletion_offset
            + particle_state_offset
            + road_surface_offset
            + rigid_body_motion_offset
            + extra_data_offset
        )
        return n_bytes_per_state

    def _read_states(self, filepath: str):
        """Read the states from the d3plot

        Parameters
        ----------
        filepath: str
            path to the d3plot
        """

        if not self._buffer or not filepath:
            self._state_info.n_timesteps = 0
            return

        LOGGER.debug("-------- S T A T E S --------")
        LOGGER.debug("_read_states with geom offset %d", self.geometry_section_size)

        # (0) OFFSETS
        bytes_per_state = self._compute_n_bytes_per_state()
        LOGGER.debug("bytes_per_state: %d", bytes_per_state)

        # load the memory from the files
        if self._femzip_info.use_femzip:
            bytes_per_state += 1 * self.header.wordsize
            self.bb_generator = self._read_femzip_file_generator(
                self.buffered_reading, self.state_filter
            )
        else:
            self.bb_generator = self._read_d3plot_file_generator(
                self.buffered_reading, self.state_filter
            )

        # (1) READ STATE DATA
        n_states = next(self.bb_generator)

        # determine whether to transfer arrays
        if not self.buffered_reading:
            transfer_arrays = False
        else:
            transfer_arrays = True
        if self.state_filter is not None and any(self.state_filter):
            transfer_arrays = True
        if self.state_array_filter:
            transfer_arrays = True

        # arrays need to be preallocated if we transfer them
        if transfer_arrays:
            self._read_states_allocate_arrays(
                self.header,
                self._material_section_info,
                self._airbag_info,
                self._rigid_road_info,
                self._rigid_body_info,
                n_states,
                self._n_rigid_walls,
                self._n_parts,
                self.state_array_filter,
                self.arrays,
            )

        i_state = 0
        for bb_states, n_states in self.bb_generator:

            # dictionary to store the temporary, partial arrays
            # if we do not transfer any arrays we store them directly
            # in the classes main dict
            array_dict = {} if transfer_arrays else self.arrays

            # sometimes there is just a geometry in the file
            if n_states == 0:
                continue

            # state data as array
            array_length = int(n_states) * int(bytes_per_state)
            state_data = bb_states.read_ndarray(0, array_length, 1, self.header.ftype)
            state_data = state_data.reshape((n_states, -1))

            var_index = 0

            # global state header
            var_index = self._read_states_global_section(state_data, var_index, array_dict)

            # node data
            var_index = self._read_states_nodes(state_data, var_index, array_dict)

            # thermal solid data
            var_index = self._read_states_solids_thermal(state_data, var_index, array_dict)

            # cfddata was originally here

            # solids
            var_index = self._read_states_solids(state_data, var_index, array_dict)

            # tshells
            var_index = self._read_states_tshell(state_data, var_index, array_dict)

            # beams
            var_index = self._read_states_beams(state_data, var_index, array_dict)

            # shells
            var_index = self._read_states_shell(state_data, var_index, array_dict)

            # element and node deletion info
            var_index = self._read_states_is_alive(state_data, var_index, array_dict)

            # sph
            var_index = self._read_states_sph(state_data, var_index, array_dict)

            # airbag particle data
            var_index = self._read_states_airbags(state_data, var_index, array_dict)

            # road surface data
            var_index = self._read_states_road_surfaces(state_data, var_index, array_dict)

            # rigid body motion
            var_index = self._read_states_rigid_body_motion(state_data, var_index, array_dict)

            # transfer memory
            if transfer_arrays:
                self._read_states_transfer_memory(i_state, array_dict, self.arrays)

            # increment state counter
            i_state += n_states
            self._state_info.n_timesteps = i_state

        if transfer_arrays:
            self._buffer = None
            self.bb_states = None

    def _read_states_global_section(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read the global vars for the state

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        LOGGER.debug("_read_states_global_section start at var_index %d", var_index)

        # we wrap globals, parts and rigid walls into a single try
        # catch block since in the header the global section is
        # defined by those three. If we fail in any of those we can
        # only heal by skipping all together and jumping forward
        original_var_index = var_index
        try:
            # timestep
            array_dict[ArrayType.global_timesteps] = state_data[:, var_index]
            var_index += 1

            # global stuff
            var_index = self._read_states_globals(state_data, var_index, array_dict)

            # parts
            var_index = self._read_states_parts(state_data, var_index, array_dict)

            # rigid walls
            var_index = self._read_states_rigid_walls(state_data, var_index, array_dict)

        except Exception:
            # print
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_states_global_section", trb_msg)
        finally:
            timestep_var_size = 1
            var_index = original_var_index + self.header.n_global_vars + timestep_var_size

        LOGGER.debug("_read_states_global_section end at var_index %d", var_index)

        return var_index

    def _read_states_globals(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the part data in the state section

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        n_global_vars = self.header.n_global_vars

        # global stuff
        i_global_var = 0
        if i_global_var < n_global_vars:
            array_dict[ArrayType.global_kinetic_energy] = state_data[:, var_index + i_global_var]
            i_global_var += 1
        if i_global_var < n_global_vars:
            array_dict[ArrayType.global_internal_energy] = state_data[:, var_index + i_global_var]
            i_global_var += 1
        if i_global_var < n_global_vars:
            array_dict[ArrayType.global_total_energy] = state_data[:, var_index + i_global_var]
            i_global_var += 1
        if i_global_var + 3 <= n_global_vars:
            array_dict[ArrayType.global_velocity] = state_data[
                :, var_index + i_global_var : var_index + i_global_var + 3
            ]
            i_global_var += 3

        return var_index + i_global_var

    def _read_states_parts(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the part data in the state section

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        n_states = state_data.shape[0]
        timestep_word = 1
        n_global_vars = self.header.n_global_vars + timestep_word

        # part infos
        # n_parts = self._n_parts
        n_parts = self.header.n_parts

        # part internal energy
        if var_index + n_parts <= n_global_vars:
            array_dict[ArrayType.part_internal_energy] = state_data[
                :, var_index : var_index + n_parts
            ]
            var_index += n_parts

        # part kinetic energy
        if var_index + n_parts <= n_global_vars:
            array_dict[ArrayType.part_kinetic_energy] = state_data[
                :, var_index : var_index + n_parts
            ]
            var_index += n_parts

        # part velocity
        if var_index + 3 * n_parts <= n_global_vars:
            array_dict[ArrayType.part_velocity] = state_data[
                :, var_index : var_index + 3 * n_parts
            ].reshape((n_states, n_parts, 3))
            var_index += 3 * n_parts

        # part mass
        if var_index + n_parts <= n_global_vars:
            array_dict[ArrayType.part_mass] = state_data[:, var_index : var_index + n_parts]
            var_index += n_parts

        # part hourglass energy
        if var_index + n_parts <= n_global_vars:
            array_dict[ArrayType.part_hourglass_energy] = state_data[
                :, var_index : var_index + n_parts
            ]
            var_index += n_parts

        return var_index

    def _read_states_rigid_walls(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read the rigid wall data in the state section

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        n_states = state_data.shape[0]

        i_global_var = 6 + 7 * self.header.n_parts
        n_global_vars = self.header.n_global_vars

        # rigid walls
        previous_global_vars = i_global_var
        n_rigid_wall_vars = 4 if self.header.version >= 971 else 1
        # +1 is timestep which is not considered a global var ... seriously
        n_rigid_walls = self._n_rigid_walls
        if n_global_vars >= previous_global_vars + n_rigid_walls * n_rigid_wall_vars:
            if (
                previous_global_vars + n_rigid_walls * n_rigid_wall_vars
                != self.header.n_global_vars
            ):
                LOGGER.warning("Bug while reading global data for rigid walls. Skipping this data.")
                var_index += self.header.n_global_vars - previous_global_vars
            else:

                # rigid wall force
                if n_rigid_walls * n_rigid_wall_vars != 0:
                    array_dict[ArrayType.rigid_wall_force] = state_data[
                        :, var_index : var_index + n_rigid_walls
                    ]
                    var_index += n_rigid_walls

                    # rigid wall position
                    if n_rigid_wall_vars > 1:
                        array_dict[ArrayType.rigid_wall_position] = state_data[
                            :, var_index : var_index + 3 * n_rigid_walls
                        ].reshape(n_states, n_rigid_walls, 3)
                        var_index += 3 * n_rigid_walls

        return var_index

    def _read_states_nodes(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the node data in the state section

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_nodes <= 0:
            return var_index

        LOGGER.debug("_read_states_nodes start at var_index %d", var_index)

        n_dim = self.header.n_dimensions
        n_states = state_data.shape[0]
        n_nodes = self.header.n_nodes

        # displacement
        if self.header.has_node_displacement:
            try:
                tmp_array = state_data[:, var_index : var_index + n_dim * n_nodes].reshape(
                    (n_states, n_nodes, n_dim)
                )
                array_dict[ArrayType.node_displacement] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_nodes, node_displacement", trb_msg)
            finally:
                var_index += n_dim * n_nodes

        # temperatures
        if self.header.has_node_temperatures:

            # only node temperatures
            if not self.header.has_node_temperature_layers:
                try:
                    array_dict[ArrayType.node_temperature] = state_data[
                        :, var_index : var_index + n_nodes
                    ]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_nodes, node_temperatures", trb_msg)
                finally:
                    var_index += n_nodes
            # node temperature layers
            else:
                try:
                    tmp_array = state_data[:, var_index : var_index + 3 * n_nodes].reshape(
                        (n_states, n_nodes, 3)
                    )
                    array_dict[ArrayType.node_temperature] = tmp_array
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_nodes, node_temperatures_layers", trb_msg)
                finally:
                    var_index += 3 * n_nodes

        # node heat flux
        if self.header.has_node_heat_flux:
            try:
                tmp_array = state_data[:, var_index : var_index + 3 * n_nodes].reshape(
                    (n_states, n_nodes, 3)
                )
                array_dict[ArrayType.node_heat_flux] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_nodes, node_heat_flux", trb_msg)
            finally:
                var_index += 3 * n_nodes

        # mass scaling
        if self.header.has_node_mass_scaling:
            try:
                array_dict[ArrayType.node_mass_scaling] = state_data[
                    :, var_index : var_index + n_nodes
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_nodes, node_mass_scaling", trb_msg)
            finally:
                var_index += n_nodes

        # node temperature gradient
        # Unclear: verify (could also be between temperature and node heat flux)
        if self.header.has_node_temperature_gradient:
            try:
                array_dict[ArrayType.node_temperature_gradient] = state_data[
                    :, var_index : var_index + n_nodes
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_nodes, node_temperature_gradient", trb_msg)
            finally:
                var_index += n_nodes

        # node residual forces and moments
        # Unclear: verify (see before, according to docs this is after previous)
        if self.header.has_node_residual_forces:
            try:
                array_dict[ArrayType.node_residual_forces] = state_data[
                    :, var_index : var_index + 3 * n_nodes
                ].reshape((n_states, n_nodes, 3))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_nodes, node_residual_forces", trb_msg)
            finally:
                var_index += n_nodes * 3

        if self.header.has_node_residual_moments:
            try:
                array_dict[ArrayType.node_residual_moments] = state_data[
                    :, var_index : var_index + 3 * n_nodes
                ].reshape((n_states, n_nodes, 3))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_nodes, node_residual_moments", trb_msg)
            finally:
                var_index += n_nodes * 3

        # velocity
        if self.header.has_node_velocity:
            try:
                tmp_array = state_data[:, var_index : var_index + n_dim * n_nodes].reshape(
                    (n_states, n_nodes, n_dim)
                )
                array_dict[ArrayType.node_velocity] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_nodes, node_velocity", trb_msg)
            finally:
                var_index += n_dim * n_nodes

        # acceleration
        if self.header.has_node_acceleration:
            try:
                tmp_array = state_data[:, var_index : var_index + n_dim * n_nodes].reshape(
                    (n_states, n_nodes, n_dim)
                )
                array_dict[ArrayType.node_acceleration] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_nodes, node_acceleration", trb_msg)
            finally:
                var_index += n_dim * n_nodes

        LOGGER.debug("_read_states_nodes end at var_index %d", var_index)

        return var_index

    def _read_states_solids_thermal(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read the thermal data for solids

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_solid_thermal_vars <= 0:
            return var_index

        LOGGER.debug("_read_states_solids_thermal start at var_index %d", var_index)

        n_states = state_data.shape[0]
        n_solids = self.header.n_solids
        n_thermal_vars = self.header.n_solid_thermal_vars

        try:
            tmp_array = state_data[:, var_index : var_index + n_solids * n_thermal_vars]
            array_dict[ArrayType.element_solid_thermal_data] = tmp_array.reshape(
                (n_states, n_solids, n_thermal_vars)
            )
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_states_solids_thermal", trb_msg)
        finally:
            var_index += n_thermal_vars * n_solids

        LOGGER.debug("_read_states_solids_thermal end at var_index %d", var_index)

        return var_index

    def _read_states_solids(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the state data of the solid elements

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_solids <= 0 or self.header.n_solid_vars <= 0:
            return var_index

        LOGGER.debug("_read_states_solids start at var_index %d", var_index)

        n_solid_vars = self.header.n_solid_vars
        n_solids = self.header.n_solids
        n_states = state_data.shape[0]
        n_strain_vars = 6 * self.header.has_element_strain
        n_history_vars = self.header.n_solid_history_vars
        n_solid_layers = self.header.n_solid_layers

        # double safety here, if either the formatting of the solid state data
        # or individual arrays fails then we catch it
        try:
            # this is a sanity check if the manual was understood correctly
            #
            # NOTE due to plotcompress we disable this check, it can delete
            # variables so that stress or pstrain might be missing despite
            # being always present in the file spec
            #
            # n_solid_vars2 = (7 +
            #                  n_history_vars)

            # if n_solid_vars2 != n_solid_vars:
            #     msg = "n_solid_vars != n_solid_vars_computed: {} != {}."\
            #           + " Solid variables might be wrong."
            #     LOGGER.warning(msg.format(n_solid_vars, n_solid_vars2))

            solid_state_data = state_data[
                :, var_index : var_index + n_solid_vars * n_solids
            ].reshape((n_states, n_solids, n_solid_layers, n_solid_vars // n_solid_layers))

            i_solid_var = 0

            # stress
            try:
                if self.header.has_solid_stress:
                    array_dict[ArrayType.element_solid_stress] = solid_state_data[:, :, :, :6]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_solids, stress", trb_msg)
            finally:
                i_solid_var += 6 * self.header.has_solid_stress

            # effective plastic strain
            try:
                # in case plotcompress deleted stresses but pstrain exists
                if self.header.has_solid_pstrain:
                    array_dict[ArrayType.element_solid_effective_plastic_strain] = solid_state_data[
                        :, :, :, i_solid_var
                    ].reshape((n_states, n_solids, n_solid_layers))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_solids, eff_plastic_strain", trb_msg)
            finally:
                i_solid_var += 1 * self.header.has_solid_pstrain

            # history vars
            if n_history_vars:
                try:
                    array_dict[ArrayType.element_solid_history_variables] = solid_state_data[
                        :, :, :, i_solid_var : i_solid_var + n_history_vars
                    ]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_solids, history_variables", trb_msg)
                finally:
                    i_solid_var += n_history_vars

            # strain
            # they are the last 6 entries of the history vars
            if n_strain_vars:
                try:
                    array_dict[ArrayType.element_solid_strain] = array_dict[
                        ArrayType.element_solid_history_variables
                    ][:, :, :, -n_strain_vars:]

                    array_dict[ArrayType.element_solid_history_variables] = array_dict[
                        ArrayType.element_solid_history_variables
                    ][:, :, :, :-n_strain_vars]

                    if not all(array_dict[ArrayType.element_solid_history_variables].shape):
                        del array_dict[ArrayType.element_solid_history_variables]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_solids, strain", trb_msg)

            # plastic strain tensor
            if self.header.has_solid_shell_plastic_strain_tensor:
                try:
                    array_dict[ArrayType.element_solid_plastic_strain_tensor] = solid_state_data[
                        :, :, :, i_solid_var : i_solid_var + 6
                    ]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(
                        msg, "_read_states_solids, element_solid_plastic_strain_tensor", trb_msg
                    )
                finally:
                    i_solid_var += 6

            # thermal strain tensor
            if self.header.has_solid_shell_thermal_strain_tensor:
                try:
                    array_dict[ArrayType.element_solid_thermal_strain_tensor] = solid_state_data[
                        :, :, i_solid_var : i_solid_var + 6
                    ]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(
                        msg, "_read_states_solids, element_solid_thermal_strain_tensor", trb_msg
                    )
                finally:
                    i_solid_var += 6

        # catch formatting in solid_state_datra
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_states_solids, solid_state_data", trb_msg)
        # always increment variable count
        finally:
            var_index += n_solids * n_solid_vars

        LOGGER.debug("_read_states_solids end at var_index %d", var_index)

        return var_index

    def _read_states_tshell(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the state data for thick shell elements

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_thick_shells <= 0 or self.header.n_thick_shell_vars <= 0:
            return var_index

        LOGGER.debug("_read_states_tshell start at var_index %d", var_index)

        n_states = state_data.shape[0]
        n_tshells = self.header.n_thick_shells
        n_history_vars = self.header.n_shell_tshell_history_vars
        n_layers = self.header.n_shell_tshell_layers
        n_layer_vars = n_layers * (
            6 * self.header.has_shell_tshell_stress
            + self.header.has_shell_tshell_pstrain
            + n_history_vars
        )
        n_strain_vars = 12 * self.header.has_element_strain
        n_thsell_vars = self.header.n_thick_shell_vars
        has_stress = self.header.has_shell_tshell_stress
        has_pstrain = self.header.has_shell_tshell_pstrain

        try:
            # this is a sanity check if the manual was understood correctly
            n_tshell_vars2 = n_layer_vars + n_strain_vars

            if n_tshell_vars2 != n_thsell_vars:
                msg = (
                    "n_tshell_vars != n_tshell_vars_computed: %d != %d."
                    " Thick shell variables might be wrong."
                )
                LOGGER.warning(msg, n_thsell_vars, n_tshell_vars2)

            # thick shell element data
            tshell_data = state_data[:, var_index : var_index + n_thsell_vars * n_tshells]
            tshell_data = tshell_data.reshape((n_states, n_tshells, n_thsell_vars))

            # extract layer data
            tshell_layer_data = tshell_data[:, :, slice(0, n_layer_vars)]
            tshell_layer_data = tshell_layer_data.reshape((n_states, n_tshells, n_layers, -1))
            tshell_nonlayer_data = tshell_data[:, :, n_layer_vars:]

            # STRESS
            i_tshell_layer_var = 0
            if has_stress:
                try:
                    array_dict[ArrayType.element_tshell_stress] = tshell_layer_data[
                        :, :, :, i_tshell_layer_var : i_tshell_layer_var + 6
                    ].reshape((n_states, n_tshells, n_layers, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %d was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_tshell, stress", trb_msg)
                finally:
                    i_tshell_layer_var += 6

            # PSTRAIN
            if has_pstrain:
                try:
                    array_dict[
                        ArrayType.element_tshell_effective_plastic_strain
                    ] = tshell_layer_data[:, :, :, i_tshell_layer_var].reshape(
                        (n_states, n_tshells, n_layers)
                    )
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_tshell, eff_plastic_strain", trb_msg)
                finally:
                    i_tshell_layer_var += 1

            # HISTORY VARS
            if n_history_vars:
                try:
                    array_dict[ArrayType.element_tshell_history_variables] = tshell_layer_data[
                        :, :, :, i_tshell_layer_var : i_tshell_layer_var + n_history_vars
                    ].reshape((n_states, n_tshells, n_layers, n_history_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_tshell, history_variables", trb_msg)

            # STRAIN (only non layer data for tshells)
            if n_strain_vars:
                try:
                    tshell_nonlayer_data = tshell_nonlayer_data[:, :, :n_strain_vars]
                    array_dict[ArrayType.element_tshell_strain] = tshell_nonlayer_data.reshape(
                        (n_states, n_tshells, 2, 6)
                    )
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_tshell, strain", trb_msg)

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_states_tshell, tshell_data", trb_msg)
        finally:
            var_index += n_thsell_vars * n_tshells

        LOGGER.debug("_read_states_tshell end at var_index %d", var_index)

        return var_index

    def _read_states_beams(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the state data for beams

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_beams <= 0 or self.header.n_beam_vars <= 0:
            return var_index

        LOGGER.debug("_read_states_beams start at var_index %d", var_index)

        # usual beam vars
        # pylint: disable = invalid-name
        N_BEAM_BASIC_VARS = 6
        # beam intergration point vars
        # pylint: disable = invalid-name
        N_BEAM_IP_VARS = 5

        n_states = state_data.shape[0]
        n_beams = self.header.n_beams
        n_history_vars = self.header.n_beam_history_vars
        n_beam_vars = self.header.n_beam_vars
        n_layers = int(
            (-3 * n_history_vars + n_beam_vars - N_BEAM_BASIC_VARS)
            / (n_history_vars + N_BEAM_IP_VARS)
        )
        # n_layer_vars = 6 + N_BEAM_IP_VARS * n_layers
        n_layer_vars = N_BEAM_IP_VARS * n_layers

        try:
            # beam element data
            beam_data = state_data[:, var_index : var_index + n_beam_vars * n_beams]
            beam_data = beam_data.reshape((n_states, n_beams, n_beam_vars))

            # extract layer data
            beam_nonlayer_data = beam_data[:, :, :N_BEAM_BASIC_VARS]
            beam_layer_data = beam_data[:, :, N_BEAM_BASIC_VARS : N_BEAM_BASIC_VARS + n_layer_vars]
            beam_layer_data = beam_layer_data.reshape((n_states, n_beams, n_layers, N_BEAM_IP_VARS))

            # axial force
            try:
                array_dict[ArrayType.element_beam_axial_force] = beam_nonlayer_data[
                    :, :, 0
                ].reshape((n_states, n_beams))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_beams, axial_force", trb_msg)

            # shear force
            try:
                array_dict[ArrayType.element_beam_shear_force] = beam_nonlayer_data[
                    :, :, 1:3
                ].reshape((n_states, n_beams, 2))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_beams, shear_force", trb_msg)

            # bending moment
            try:
                array_dict[ArrayType.element_beam_bending_moment] = beam_nonlayer_data[
                    :, :, 3:5
                ].reshape((n_states, n_beams, 2))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_beams, bending_moment", trb_msg)

            # torsion moment
            try:
                array_dict[ArrayType.element_beam_torsion_moment] = beam_nonlayer_data[
                    :, :, 5
                ].reshape((n_states, n_beams))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_beams, torsion_moment", trb_msg)

            if n_layers:

                # BUGFIX?
                # According to the database manual the first
                # two layer vars are the shear stress and then
                # axial stress. Tests with FEMZIP and META though
                # suggests that axial stress comes first.

                # axial stress
                try:
                    array_dict[ArrayType.element_beam_axial_stress] = beam_layer_data[:, :, :, 0]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_beams, axial_stress", trb_msg)

                # shear stress
                try:
                    array_dict[ArrayType.element_beam_shear_stress] = beam_layer_data[:, :, :, 1:3]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_beams, shear_stress", trb_msg)

                # eff. plastic strain
                try:
                    array_dict[ArrayType.element_beam_plastic_strain] = beam_layer_data[:, :, :, 3]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_beams, eff_plastic_strain", trb_msg)

                # axial strain
                try:
                    array_dict[ArrayType.element_beam_axial_strain] = beam_layer_data[:, :, :, 4]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_beams, axial_strain", trb_msg)

            # history vars
            if n_history_vars:
                try:
                    array_dict[ArrayType.element_beam_history_vars] = beam_data[
                        :, :, 6 + n_layer_vars :
                    ].reshape((n_states, n_beams, 3 + n_layers, n_history_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_beams, history_variables", trb_msg)

        # failure of formatting beam state data
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_states_beams, beam_state_data", trb_msg)
        # always increment variable index
        finally:
            var_index += n_beams * n_beam_vars

        LOGGER.debug("_read_states_beams end at var_index %d", var_index)

        return var_index

    def _read_states_shell(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the state data for shell elements

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        # bugfix
        #
        # Interestingly, dyna seems to write result values for rigid shells in
        # the d3part file, but not in the d3plot. Of course this is not
        # documented ...
        n_reduced_shells = (
            self.header.n_shells
            if self.header.filetype == D3plotFiletype.D3PART
            else self.header.n_shells - self._material_section_info.n_rigid_shells
        )

        if self.header.n_shell_vars <= 0 or n_reduced_shells <= 0:
            return var_index

        LOGGER.debug("_read_states_shell start at var_index %d", var_index)

        n_states = state_data.shape[0]
        n_shells = n_reduced_shells
        n_shell_vars = self.header.n_shell_vars

        # what is in the file?
        n_layers = self.header.n_shell_tshell_layers
        n_history_vars = self.header.n_shell_tshell_history_vars
        n_stress_vars = 6 * self.header.has_shell_tshell_stress
        n_pstrain_vars = 1 * self.header.has_shell_tshell_pstrain
        n_force_variables = 8 * self.header.has_shell_forces
        n_extra_variables = 4 * self.header.has_shell_extra_variables
        n_strain_vars = 12 * self.header.has_element_strain
        n_plastic_strain_tensor = 6 * n_layers * self.header.has_solid_shell_plastic_strain_tensor
        n_thermal_strain_tensor = 6 * self.header.has_solid_shell_thermal_strain_tensor

        try:
            # this is a sanity check if the manual was understood correctly
            n_shell_vars2 = (
                n_layers * (n_stress_vars + n_pstrain_vars + n_history_vars)
                + n_force_variables
                + n_extra_variables
                + n_strain_vars
                + n_plastic_strain_tensor
                + n_thermal_strain_tensor
            )

            if n_shell_vars != n_shell_vars2:
                msg = (
                    "n_shell_vars != n_shell_vars_computed: %d != %d."
                    " Shell variables might be wrong."
                )
                LOGGER.warning(msg, n_shell_vars, n_shell_vars2)

            n_layer_vars = n_layers * (n_stress_vars + n_pstrain_vars + n_history_vars)

            # shell element data
            shell_data = state_data[:, var_index : var_index + n_shell_vars * n_shells]
            shell_data = shell_data.reshape((n_states, n_shells, n_shell_vars))

            # extract layer data
            shell_layer_data = shell_data[:, :, :n_layer_vars]
            shell_layer_data = shell_layer_data.reshape((n_states, n_shells, n_layers, -1))
            shell_nonlayer_data = shell_data[:, :, n_layer_vars:]

            # save layer stuff
            # STRESS
            layer_var_index = 0
            if n_stress_vars:
                try:
                    array_dict[ArrayType.element_shell_stress] = shell_layer_data[
                        :, :, :, :n_stress_vars
                    ].reshape((n_states, n_shells, n_layers, n_stress_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_shells, stress", trb_msg)
                finally:
                    layer_var_index += n_stress_vars

            # PSTRAIN
            if n_pstrain_vars:
                try:
                    array_dict[ArrayType.element_shell_effective_plastic_strain] = shell_layer_data[
                        :, :, :, layer_var_index
                    ].reshape((n_states, n_shells, n_layers))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_shells, stress", trb_msg)
                finally:
                    layer_var_index += 1

            # HISTORY VARIABLES
            if n_history_vars:
                try:
                    array_dict[ArrayType.element_shell_history_vars] = shell_layer_data[
                        :, :, :, layer_var_index : layer_var_index + n_history_vars
                    ].reshape((n_states, n_shells, n_layers, n_history_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_shells, history_variables", trb_msg)
                finally:
                    layer_var_index += n_history_vars

            # save nonlayer stuff
            # forces
            nonlayer_var_index = 0
            if n_force_variables:
                try:
                    array_dict[ArrayType.element_shell_bending_moment] = shell_nonlayer_data[
                        :, :, 0:3
                    ].reshape((n_states, n_shells, 3))
                    array_dict[ArrayType.element_shell_shear_force] = shell_nonlayer_data[
                        :, :, 3:5
                    ].reshape((n_states, n_shells, 2))
                    array_dict[ArrayType.element_shell_normal_force] = shell_nonlayer_data[
                        :, :, 5:8
                    ].reshape((n_states, n_shells, 3))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_shells, forces", trb_msg)
                finally:
                    nonlayer_var_index += n_force_variables

            # weird stuff
            if n_extra_variables:
                try:
                    array_dict[ArrayType.element_shell_thickness] = shell_nonlayer_data[
                        :, :, nonlayer_var_index
                    ].reshape((n_states, n_shells))
                    array_dict[ArrayType.element_shell_unknown_variables] = shell_nonlayer_data[
                        :, :, nonlayer_var_index + 1 : nonlayer_var_index + 3
                    ].reshape((n_states, n_shells, 2))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_shells, history_variables", trb_msg)
                finally:
                    nonlayer_var_index += 3

            # strain present
            if n_strain_vars:
                try:
                    shell_strain = shell_nonlayer_data[
                        :, :, nonlayer_var_index : nonlayer_var_index + n_strain_vars
                    ]
                    array_dict[ArrayType.element_shell_strain] = shell_strain.reshape(
                        (n_states, n_shells, 2, 6)
                    )
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_shells, strain", trb_msg)
                finally:
                    nonlayer_var_index += n_strain_vars

            # internal energy is behind strain if strain is written
            if self.header.has_shell_extra_variables:
                try:
                    array_dict[ArrayType.element_shell_internal_energy] = shell_nonlayer_data[
                        :, :, nonlayer_var_index
                    ].reshape((n_states, n_shells))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_shells, internal_energy", trb_msg)

            # PLASTIC STRAIN TENSOR
            if n_plastic_strain_tensor:
                try:
                    pstrain_tensor = shell_nonlayer_data[
                        :, :, nonlayer_var_index : nonlayer_var_index + n_plastic_strain_tensor
                    ]
                    array_dict[
                        ArrayType.element_shell_plastic_strain_tensor
                    ] = pstrain_tensor.reshape((n_states, n_shells, n_layers, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(
                        msg, "_read_states_shells, element_shell_plastic_strain_tensor", trb_msg
                    )
                finally:
                    nonlayer_var_index += n_plastic_strain_tensor

            # THERMAL STRAIN TENSOR
            if n_thermal_strain_tensor:
                try:
                    thermal_tensor = shell_nonlayer_data[
                        :, :, nonlayer_var_index : nonlayer_var_index + n_thermal_strain_tensor
                    ]
                    array_dict[
                        ArrayType.element_shell_thermal_strain_tensor
                    ] = thermal_tensor.reshape((n_states, n_shells, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(
                        msg, "_read_states_shells, element_shell_thermal_strain_tensor", trb_msg
                    )
                finally:
                    nonlayer_var_index += n_thermal_strain_tensor

        # error in formatting shell state data
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_states_shell, shell_state_data", trb_msg)

        # always increment variable index
        finally:
            var_index += n_shell_vars * n_shells

        LOGGER.debug("_read_states_shell end at var_index %d", var_index)

        return var_index

    def _read_states_is_alive(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read deletion info for nodes, elements, etc

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if not self.header.has_node_deletion_data and not self.header.has_element_deletion_data:
            return var_index

        LOGGER.debug("_read_states_is_alive start at var_index %s", var_index)

        n_states = state_data.shape[0]

        # NODES
        if self.header.has_node_deletion_data:
            n_nodes = self.header.n_nodes

            if n_nodes > 0:
                try:
                    array_dict[ArrayType.node_is_alive] = state_data[
                        :, var_index : var_index + n_nodes
                    ]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_is_alive, nodes", trb_msg)
                finally:
                    var_index += n_nodes

        # element deletion info
        elif self.header.has_element_deletion_data:
            n_solids = self.header.n_solids
            n_tshells = self.header.n_thick_shells
            n_shells = self.header.n_shells
            n_beams = self.header.n_beams
            # n_elems = n_solids + n_tshells + n_shells + n_beams

            # SOLIDS
            if n_solids > 0:
                try:
                    array_dict[ArrayType.element_solid_is_alive] = state_data[
                        :, var_index : var_index + n_solids
                    ].reshape((n_states, n_solids))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_is_alive, solids", trb_msg)
                finally:
                    var_index += n_solids

            # TSHELLS
            if n_tshells > 0:
                try:
                    array_dict[ArrayType.element_tshell_is_alive] = state_data[
                        :, var_index : var_index + n_tshells
                    ].reshape((n_states, n_tshells))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_is_alive, solids", trb_msg)
                finally:
                    var_index += n_tshells

            # SHELLS
            if n_shells > 0:
                try:
                    array_dict[ArrayType.element_shell_is_alive] = state_data[
                        :, var_index : var_index + n_shells
                    ].reshape((n_states, n_shells))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_is_alive, shells", trb_msg)
                finally:
                    var_index += n_shells

            # BEAMS
            if n_beams > 0:
                try:
                    array_dict[ArrayType.element_beam_is_alive] = state_data[
                        :, var_index : var_index + n_beams
                    ].reshape((n_states, n_beams))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_is_alive, beams", trb_msg)
                finally:
                    var_index += n_beams

        LOGGER.debug("_read_states_is_alive end at var_index %d", var_index)

        return var_index

    def _read_states_sph(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the sph state data

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_sph_nodes <= 0:
            return var_index

        LOGGER.debug("_read_states_sph start at var_index %d", var_index)

        info = self._sph_info
        n_states = state_data.shape[0]
        n_particles = self.header.n_sph_nodes
        n_variables = info.n_sph_vars

        # extract data
        try:
            sph_data = state_data[:, var_index : var_index + n_particles * n_variables]

            i_var = 1

            # deletion
            try:
                array_dict[ArrayType.sph_deletion] = sph_data[:, 0] < 0
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_sph, deletion", trb_msg)

            # particle radius
            if info.has_influence_radius:
                try:
                    array_dict[ArrayType.sph_radius] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_sph, radius", trb_msg)
                finally:
                    i_var += 1

            # pressure
            if info.has_particle_pressure:
                try:
                    array_dict[ArrayType.sph_pressure] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_sph, pressure", trb_msg)
                finally:
                    i_var += 1

            # stress
            if info.has_stresses:
                try:
                    array_dict[ArrayType.sph_stress] = sph_data[
                        :, i_var : i_var + n_particles * 6
                    ].reshape((n_states, n_particles, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_sph, pressure", trb_msg)
                finally:
                    i_var += 6 * n_particles

            # eff. plastic strain
            if info.has_plastic_strain:
                try:
                    array_dict[ArrayType.sph_effective_plastic_strain] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_sph, eff_plastic_strain", trb_msg)
                finally:
                    i_var += 1

            # density
            if info.has_material_density:
                try:
                    array_dict[ArrayType.sph_density] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_sph, density", trb_msg)
                finally:
                    i_var += 1

            # internal energy
            if info.has_internal_energy:
                try:
                    array_dict[ArrayType.sph_internal_energy] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_sph, internal_energy", trb_msg)
                finally:
                    i_var += 1

            # number of neighbors
            if info.has_n_affecting_neighbors:
                try:
                    array_dict[ArrayType.sph_n_neighbors] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_sph, n_neighbors", trb_msg)
                finally:
                    i_var += 1

            # strain and strainrate
            if info.has_strain_and_strainrate:

                try:
                    array_dict[ArrayType.sph_strain] = sph_data[
                        :, i_var : i_var + n_particles * 6
                    ].reshape((n_states, n_particles, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_sph, strain", trb_msg)
                finally:
                    i_var += 6 * n_particles

                try:
                    array_dict[ArrayType.sph_strainrate] = sph_data[
                        :, i_var : i_var + n_particles * 6
                    ].reshape((n_states, n_particles, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_sph, strainrate", trb_msg)
                finally:
                    i_var += 6 * n_particles

            # mass
            if info.has_mass:
                try:
                    array_dict[ArrayType.sph_mass] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in %s was caught:\n%s"
                    LOGGER.warning(msg, "_read_states_sph, pressure", trb_msg)
                finally:
                    i_var += 1

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_states_sph, sph_data", trb_msg)
        finally:
            var_index += n_particles * n_variables

        LOGGER.debug("_read_states_sph end at var_index %d", var_index)

        return var_index

    def _read_states_airbags(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        """Read the airbag state data

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if self.header.n_airbags <= 0:
            return var_index

        LOGGER.debug("_read_states_airbags start at var_index %d", var_index)

        n_states = state_data.shape[0]
        info = self._airbag_info
        n_airbag_geom_vars = info.n_geometric_variables
        n_airbags = info.n_airbags
        n_state_airbag_vars = info.n_airbag_state_variables
        n_particles = info.n_particles
        n_particle_vars = info.n_particle_state_variables

        # Warning
        # I am not sure if this is right ...
        n_total_vars = n_airbags * n_state_airbag_vars + n_particles * n_particle_vars

        try:
            # types
            # nlist = ngeom + nvar + nstgeom
            airbag_var_types = self.arrays[ArrayType.airbag_variable_types]
            airbag_var_names = self.arrays[ArrayType.airbag_variable_names]
            # geom_var_types = airbag_var_types[:n_airbag_geom_vars]
            particle_var_types = airbag_var_types[
                n_airbag_geom_vars : n_airbag_geom_vars + n_particle_vars
            ]
            particle_var_names = airbag_var_names[
                n_airbag_geom_vars : n_airbag_geom_vars + n_particle_vars
            ]

            airbag_state_var_types = airbag_var_types[n_airbag_geom_vars + n_particle_vars :]
            airbag_state_var_names = airbag_var_names[n_airbag_geom_vars + n_particle_vars :]

            # required for dynamic reading
            def get_dtype(type_flag):
                return self._header.itype if type_flag == 1 else self.header.ftype

            # extract airbag data
            airbag_state_data = state_data[:, var_index : var_index + n_total_vars]

            # airbag data
            airbag_data = airbag_state_data[:, : n_airbags * n_state_airbag_vars].reshape(
                (n_states, n_airbags, n_state_airbag_vars)
            )
            airbag_state_offset = n_airbags * n_state_airbag_vars

            # particle data
            particle_data = airbag_state_data[
                :, airbag_state_offset : airbag_state_offset + n_particles * n_particle_vars
            ].reshape((n_states, n_particles, n_particle_vars))

            # save sh...

            # airbag state vars
            for i_airbag_state_var in range(n_state_airbag_vars):
                var_name = airbag_state_var_names[i_airbag_state_var].strip()
                var_type = airbag_state_var_types[i_airbag_state_var]

                if var_name.startswith("Act Gas"):
                    try:
                        array_dict[ArrayType.airbag_n_active_particles] = airbag_data[
                            :, :, i_airbag_state_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, airbag_n_active_particles", trb_msg
                        )
                elif var_name.startswith("Bag Vol"):
                    try:
                        array_dict[ArrayType.airbag_bag_volume] = airbag_data[
                            :, :, i_airbag_state_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s was caught:\n%s"
                        LOGGER.warning(msg, "_read_states_airbags, airbag_volume", trb_msg)
                else:
                    warn_msg = "Unknown airbag state var: '%s'. Skipping it."
                    LOGGER.warning(warn_msg, var_name)

            # particles yay
            for i_particle_var in range(n_particle_vars):
                var_type = particle_var_types[i_particle_var]
                var_name = particle_var_names[i_particle_var].strip()

                # particle gas id
                if var_name.startswith("GasC ID"):
                    try:
                        array_dict[ArrayType.airbag_particle_gas_id] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, particle_gas_id", var_name, trb_msg
                        )
                # particle chamber id
                elif var_name.startswith("Cham ID"):
                    try:
                        array_dict[ArrayType.airbag_particle_chamber_id] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, particle_gas_id", var_name, trb_msg
                        )
                # particle leakage
                elif var_name.startswith("Leakage"):
                    try:
                        array_dict[ArrayType.airbag_particle_leakage] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, particle_gas_id", var_name, trb_msg
                        )
                # particle mass
                elif var_name.startswith("Mass"):
                    try:
                        array_dict[ArrayType.airbag_particle_mass] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, particle_gas_id", var_name, trb_msg
                        )
                    # particle radius
                    try:
                        array_dict[ArrayType.airbag_particle_radius] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, particle_gas_id", var_name, trb_msg
                        )
                # particle spin energy
                elif var_name.startswith("Spin En"):
                    try:
                        array_dict[ArrayType.airbag_particle_spin_energy] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, particle_gas_id", var_name, trb_msg
                        )
                # particle translational energy
                elif var_name.startswith("Tran En"):
                    try:
                        array_dict[ArrayType.airbag_particle_translation_energy] = particle_data[
                            :, :, i_particle_var
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, particle_gas_id", var_name, trb_msg
                        )
                # particle segment distance
                elif var_name.startswith("NS dist"):
                    try:
                        array_dict[
                            ArrayType.airbag_particle_nearest_segment_distance
                        ] = particle_data[:, :, i_particle_var].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, particle_gas_id", var_name, trb_msg
                        )
                # particle position
                elif var_name.startswith("Pos x"):
                    try:
                        particle_var_names_stripped = [
                            entry.strip() for entry in particle_var_names
                        ]
                        i_particle_var_x = i_particle_var
                        i_particle_var_y = particle_var_names_stripped.index("Pos y")
                        i_particle_var_z = particle_var_names_stripped.index("Pos z")

                        array_dict[ArrayType.airbag_particle_position] = particle_data[
                            :, :, (i_particle_var_x, i_particle_var_y, i_particle_var_z)
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, particle_gas_id", var_name, trb_msg
                        )
                elif var_name.startswith("Pos y"):
                    # handled in Pos x
                    pass
                elif var_name.startswith("Pos z"):
                    # handled in Pos x
                    pass
                # particle velocity
                elif var_name.startswith("Vel x"):
                    try:
                        particle_var_names_stripped = [
                            entry.strip() for entry in particle_var_names
                        ]
                        i_particle_var_x = i_particle_var
                        i_particle_var_y = particle_var_names_stripped.index("Vel y")
                        i_particle_var_z = particle_var_names_stripped.index("Vel z")

                        array_dict[ArrayType.airbag_particle_velocity] = particle_data[
                            :, :, (i_particle_var_x, i_particle_var_y, i_particle_var_z)
                        ].view(get_dtype(var_type))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in %s %s was caught:\n%s"
                        LOGGER.warning(
                            msg, "_read_states_airbags, particle_gas_id", var_name, trb_msg
                        )

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_states_airbags, particle_data", trb_msg)
        finally:
            var_index += n_total_vars

        LOGGER.debug("_read_states_airbags end at var_index %d", var_index)

        return var_index

    def _read_states_road_surfaces(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read the road surfaces state data for whoever wants this ...

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if not self.header.has_rigid_road_surface:
            return var_index

        LOGGER.debug("_read_states_road_surfaces start at var_index %s", var_index)

        n_states = state_data.shape[0]
        info = self._rigid_road_info
        n_roads = info.n_roads

        try:
            # read road data
            road_data = state_data[:, var_index : var_index + 6 * n_roads].reshape(
                (n_states, n_roads, 2, 3)
            )

            # DISPLACEMENT
            try:
                array_dict[ArrayType.rigid_road_displacement] = road_data[:, :, 0, :]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_road_surfaces, road_displacement", trb_msg)

            # VELOCITY
            try:
                array_dict[ArrayType.rigid_road_velocity] = road_data[:, :, 1, :]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_road_surfaces, road_velocity", trb_msg)

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_states_road_surfaces, road_data", trb_msg)
        finally:
            var_index += 6 * n_roads

        LOGGER.debug("_read_states_road_surfaces end at var_index %d", var_index)

        return var_index

    def _read_states_rigid_body_motion(
        self, state_data: np.ndarray, var_index: int, array_dict: dict
    ) -> int:
        """Read the road surfaces state data for whoever want this ...

        Parameters
        ----------
        state_data: np.ndarray
            array with entire state data
        var_index: int
            variable index in the state data array
        array_dict: dict
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index: int
            updated variable index after reading the section
        """

        if not self.header.has_rigid_body_data:
            return var_index

        LOGGER.debug("_read_states_rigid_body_motion start at var_index %d", var_index)

        info = self._rigid_body_info
        n_states = state_data.shape[0]
        n_rigids = info.n_rigid_bodies
        n_rigid_vars = 12 if self.header.has_reduced_rigid_body_data else 24

        try:
            # do the thing
            rigid_body_data = state_data[
                :, var_index : var_index + n_rigids * n_rigid_vars
            ].reshape((n_states, n_rigids, n_rigid_vars))

            # let the party begin
            # rigid coordinates
            try:
                array_dict[ArrayType.rigid_body_coordinates] = rigid_body_data[:, :, :3]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_rigid_body_motion, coordinates", trb_msg)
            finally:
                i_var = 3

            # rotation matrix
            try:
                array_dict[ArrayType.rigid_body_rotation_matrix] = rigid_body_data[
                    :, :, i_var : i_var + 9
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_rigid_body_motion, rot_matrix", trb_msg)
            finally:
                i_var += 9

            if self.header.has_reduced_rigid_body_data:
                return var_index

            # velocity pewpew
            try:
                array_dict[ArrayType.rigid_body_velocity] = rigid_body_data[:, :, i_var : i_var + 3]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_rigid_body_motion, velocity", trb_msg)
            finally:
                i_var += 3

            # rotational velocity
            try:
                array_dict[ArrayType.rigid_body_rot_velocity] = rigid_body_data[
                    :, :, i_var : i_var + 3
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_rigid_body_motion, rot_velocity", trb_msg)
            finally:
                i_var += 3

            # acceleration
            try:
                array_dict[ArrayType.rigid_body_acceleration] = rigid_body_data[
                    :, :, i_var : i_var + 3
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_rigid_body_motion, acceleration", trb_msg)
            finally:
                i_var += 3

            # rotational acceleration
            try:
                array_dict[ArrayType.rigid_body_rot_acceleration] = rigid_body_data[
                    :, :, i_var : i_var + 3
                ]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_read_states_rigid_body_motion, rot_acceleration", trb_msg)
            finally:
                i_var += 3

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in %s was caught:\n%s"
            LOGGER.warning(msg, "_read_states_rigid_body_motion, rigid_body_data", trb_msg)

        finally:
            var_index += n_rigids * n_rigid_vars

        LOGGER.debug("_read_states_rigid_body_motion end at var_index %d", var_index)

        return var_index

    def _collect_file_infos(self, size_per_state: int) -> List[MemoryInfo]:
        """This routine collects the memory and file info for the d3plot files

        Parameters
        ----------
        size_per_state: int
            size of every state to be read

        Returns
        -------
        memory_infos: List[MemoryInfo]
            memory infos about the states

        Notes
        -----
            State data is expected directly behind geometry data
            Unfortunately data is spread across multiple files.
            One file could contain geometry and state data but states
            may also be littered accross several files. This would
            not be an issue, if dyna would not always write in blocks
            of 512 words of memory, leaving zero byte padding blocks
            at the end of files. These need to be removed and/or taken
            care of.
        """

        if not self._buffer:
            return []

        base_filepath = self.header.filepath

        # bugfix
        # If you encounter these int casts more often here this is why:
        # Some ints around here are numpy.int32 which can overflow
        # (sometimes there is a warning ... sometimes not ...)
        # we cast to python ints in order to prevent overflow.
        size_per_state = int(size_per_state)

        # Info:
        #
        # We need to determine here how many states are in every file
        # without really loading the file itself. For big files this is
        # simply filesize // state_size.
        # For files though with a smaller filesize this may cause issues
        # e.g.
        # filesize 2048 bytes (minimum filesize from dyna)
        # geom_size 200 bytes
        # state_size 200 bytes
        # File contains:
        # -> 1 state * state_size + geom_size = 400 bytes
        # Wrong State Estimation:
        # -> (filesize - geom_size) // state_size = 9 states != 1 state
        #
        # To avoid this wrong number of states when reading small files
        # we need to search the end mark (here nonzero byte) from the rear
        # of the file.
        # This though needs the file to be loaded into memory. To make this
        # very light, we simply memorymap a small fraction of the file starting
        # from the rear until we have our nonzero byte. Since the end mark
        # is usually in the first block loaded, there should not be any performance
        # concerns, even with bigger files.

        # query for state files
        filepaths = D3plot._find_dyna_result_files(base_filepath)

        # compute state data in first file
        # search therefore the first non-zero byte from the rear
        last_nonzero_byte_index = self._buffer.size
        mview_inv_arr = np.asarray(self._buffer.memoryview[::-1])
        # pylint: disable = invalid-name
        BLOCK_SIZE = 2048
        for start in range(0, self._buffer.size, BLOCK_SIZE):
            (nz_indexes,) = np.nonzero(mview_inv_arr[start : start + BLOCK_SIZE])
            if len(nz_indexes):
                last_nonzero_byte_index = self._buffer.size - (start + nz_indexes[0])
                break
        n_states_beyond_geom = (
            last_nonzero_byte_index - self.geometry_section_size
        ) // size_per_state

        # bugfix: if states are too big we can get a negative estimation
        n_states_beyond_geom = max(0, n_states_beyond_geom)

        # memory required later
        memory_infos = [
            MemoryInfo(
                start=self.geometry_section_size,  # type: ignore
                length=n_states_beyond_geom * size_per_state,  # type: ignore
                filepath=base_filepath,
                n_states=n_states_beyond_geom,  # type: ignore
                filesize=self._buffer.size,
                use_mmap=True,
            )
        ]

        # compute amount of state data in every further file
        for filepath in filepaths:
            filesize = os.path.getsize(filepath)
            last_nonzero_byte_index = -1

            n_blocks = filesize // mmap.ALLOCATIONGRANULARITY
            rest_size = filesize % mmap.ALLOCATIONGRANULARITY
            block_length = mmap.ALLOCATIONGRANULARITY
            with open(filepath, "rb") as fp:

                # search last rest block (page-aligned)
                # page-aligned means the start must be
                # a multiple of mmap.ALLOCATIONGRANULARITY
                # otherwise we get an error on linux
                if rest_size:
                    start = n_blocks * block_length
                    mview = memoryview(
                        mmap.mmap(
                            fp.fileno(), offset=start, length=rest_size, access=mmap.ACCESS_READ
                        ).read()
                    )
                    (nz_indexes,) = np.nonzero(mview[::-1])
                    if len(nz_indexes):
                        last_nonzero_byte_index = start + rest_size - nz_indexes[0]

                # search in blocks from the reair
                if last_nonzero_byte_index == -1:
                    for i_block in range(n_blocks - 1, -1, -1):
                        start = block_length * i_block
                        mview = memoryview(
                            mmap.mmap(
                                fp.fileno(),
                                offset=start,
                                length=block_length,
                                access=mmap.ACCESS_READ,
                            ).read()
                        )
                        (nz_indexes,) = np.nonzero(mview[::-1])
                        if len(nz_indexes):
                            index = block_length - nz_indexes[0]
                            last_nonzero_byte_index = start + index
                            break

            if last_nonzero_byte_index == -1:
                msg = "The file {0} seems to be missing it's endmark."
                raise RuntimeError(msg.format(filepath))

            # BUGFIX
            # In d3eigv it could be observed that there is not necessarily an end mark.
            # As a consequence the last byte can indeed be zero. We control this by
            # checking if the last nonzero byte was smaller than the state size which
            # makes no sense.
            if (
                self.header.filetype == D3plotFiletype.D3EIGV
                and last_nonzero_byte_index < size_per_state <= filesize
            ):
                last_nonzero_byte_index = size_per_state

            n_states_in_file = last_nonzero_byte_index // size_per_state
            memory_infos.append(
                MemoryInfo(
                    start=0,
                    length=size_per_state * (n_states_in_file),
                    filepath=filepath,
                    n_states=n_states_in_file,
                    filesize=filesize,
                    use_mmap=False,
                )
            )

        return memory_infos

    @staticmethod
    def _read_file_from_memory_info(
        memory_infos: Union[MemoryInfo, List[MemoryInfo]]
    ) -> Tuple[BinaryBuffer, int]:
        """Read files from a single or multiple memory infos

        Parameters
        ----------
        memory_infos: MemoryInfo or List[MemoryInfo]
            memory infos for loading a file (see `D3plot._collect_file_infos`)

        Returns
        -------
        bb_states: BinaryBuffer
            New binary buffer with all states perfectly linear in memory
        n_states: int
            Number of states to be expected

        Notes
        -----
            This routine in contrast to `D3plot._read_state_bytebuffer` is used
            to load only a fraction of files into memory.
        """

        # single file case
        if isinstance(memory_infos, MemoryInfo):
            memory_infos = [memory_infos]

        # allocate memory
        # bugfix: casting to int prevents int32 overflow for large files
        memory_required = 0
        for mem in memory_infos:
            memory_required += int(mem.length)
        mview = memoryview(bytearray(memory_required))

        # transfer memory for other files
        n_states = 0
        total_offset = 0
        for minfo in memory_infos:
            LOGGER.debug("opening: %s", minfo.filepath)

            with open(minfo.filepath, "br") as fp:
                # NOTE
                # mmap is too slow but maybe there are faster
                # ways to use mmap correctly
                # if minfo.use_mmap:

                #     # memory mapping can only be done page aligned
                #     mmap_start = (minfo.start // mmap.ALLOCATIONGRANULARITY) * \
                #         mmap.ALLOCATIONGRANULARITY
                #     mview_start = minfo.start - mmap_start

                #     end = minfo.start + minfo.length
                #     n_end_pages = (end // mmap.ALLOCATIONGRANULARITY +
                #                    (end % mmap.ALLOCATIONGRANULARITY != 0))
                #     mmap_length = n_end_pages * mmap.ALLOCATIONGRANULARITY - mmap_start
                #     if mmap_start + mmap_length > minfo.filesize:
                #         mmap_length = minfo.filesize - mmap_start

                #     with mmap.mmap(fp.fileno(),
                #                    length=mmap_length,
                #                    offset=mmap_start,
                #                    access=mmap.ACCESS_READ) as mp:
                #         # mp.seek(mview_start)
                #         # mview[total_offset:total_offset +
                #         #       minfo.length] = mp.read(minfo.length)

                #         mview[total_offset:total_offset +
                #               minfo.length] = mp[mview_start:mview_start + minfo.length]

                # else:
                fp.seek(minfo.start)
                fp.readinto(mview[total_offset : total_offset + minfo.length])  # type: ignore

            total_offset += minfo.length
            n_states += minfo.n_states

        # save
        bb_states = BinaryBuffer()
        bb_states.memoryview = mview

        return bb_states, n_states

    def _read_state_bytebuffer(self, size_per_state: int):
        """This routine reads the data for state information

        Parameters
        ----------
        size_per_state: int
            size of every state to be read

        Returns
        -------
        bb_states: BinaryBuffer
            New binary buffer with all states perfectly linear in memory
        n_states: int
            Number of states to be expected

        Notes
        -----
            State data is expected directly behind geometry data
            Unfortunately data is spread across multiple files.
            One file could contain geometry and state data but states
            may also be littered accross several files. This would
            not be an issue, if dyna would not always write in blocks
            of 512 words of memory, leaving zero byte padding blocks
            at the end of files. These need to be removed and/or taken
            care of.
        """

        if not self._buffer:
            return BinaryBuffer(), 0

        memory_infos = self._collect_file_infos(size_per_state)

        # allocate memory
        # bugfix: casting to int prevents int32 overflow for large files
        memory_required = 0
        for mem in memory_infos:
            memory_required += int(mem.length)
        mview = memoryview(bytearray(memory_required))

        # transfer memory from first file
        n_states = memory_infos[0].n_states
        start = memory_infos[0].start
        length = memory_infos[0].length
        end = start + length
        mview[:length] = self._buffer.memoryview[start:end]

        # transfer memory for other files
        total_offset = length
        for minfo in memory_infos[1:]:
            with open(minfo.filepath, "br") as fp:
                fp.seek(minfo.start)
                fp.readinto(mview[total_offset : total_offset + length])  # type: ignore

            total_offset += length
            n_states += minfo.n_states

        # save
        bb_states = BinaryBuffer()
        bb_states.memoryview = mview
        return bb_states, n_states

    @staticmethod
    def _find_dyna_result_files(filepath: str):
        """Searches all dyna result files

        Parameters
        ----------
        filepath: str
            path to the first basic d3plot file

        Returns
        -------
        filepaths: list of str
            path to all dyna files

        Notes
        -----
            The dyna files usually follow a scheme to
            simply have the base name and numbers appended
            e.g. (d3plot, d3plot0001, d3plot0002, etc.)
        """

        file_dir = os.path.dirname(filepath)
        file_dir = file_dir if len(file_dir) != 0 else "."
        file_basename = os.path.basename(filepath)

        pattern = f"({file_basename})[0-9]+$"
        reg = re.compile(pattern)

        filepaths = [
            os.path.join(file_dir, path)
            for path in os.listdir(file_dir)
            if os.path.isfile(os.path.join(file_dir, path)) and reg.match(path)
        ]

        # alphasort files to handle d3plots with more than 100 files
        # e.g. d3plot01, d3plot02, ..., d3plot100
        def convert(text):
            return int(text) if text.isdigit() else text.lower()

        number_pattern = "([0-9]+)"

        def alphanum_key(key):
            return [convert(c) for c in re.split(number_pattern, key)]

        return sorted(filepaths, key=alphanum_key)

    def _determine_wordsize(self):
        """Determine the precision of the file

        Returns
        -------
        wordsize: int
            size of each word in bytes
        """

        if not self._buffer:
            return 4, np.int32, np.float32

        # test file type flag (1=d3plot, 5=d3part, 11=d3eigv)

        # single precision
        value = self._buffer.read_number(44, np.int32)
        if value > 1000:
            value -= 1000
        if value in (1, 5, 11):
            return 4, np.int32, np.float32

        # double precision
        value = self._buffer.read_number(88, np.int64)
        if value > 1000:
            value -= 1000
        if value in (1, 5, 11):
            return 8, np.int64, np.float64

        raise RuntimeError(f"Unknown file type '{value}'.")

    def plot(
        self,
        i_timestep: int = 0,
        field: Union[np.ndarray, None] = None,
        is_element_field: bool = True,
        fringe_limits: Union[Tuple[float, float], None] = None,
        export_filepath: str = "",
    ):
        """Plot the d3plot geometry

        Parameters
        ----------
        i_timestep: int
            timestep index to plot
        field: Union[np.ndarray, None]
            Array containing a field value for every element or node
        is_element_field: bool
            if the specified field is for elements or nodes
        fringe_limits: Union[Tuple[float, float], None]
            limits for the fringe bar. Set by default to min and max.
        export_filepath: str
            filepath to export the html to

        Notes
        -----
            Currently only shell elements can be plotted, since for
            solids the surface needs extraction.

        Examples
        --------
            Plot deformation of last timestep.

            >>> d3plot = D3plot("path/to/d3plot")
            >>> d3plot.plot(-1)
            >>> # get eff. plastic strain
            >>> pstrain = d3plot.arrays[ArrayType.element_shell_effective_plastic_strain]
            >>> pstrain.shape
            (1, 4696, 3)
            >>> # mean across all 3 integration points
            >>> pstrain = pstrain.mean(axis=2)
            >>> pstrain.shape
            (1, 4696)
            >>> # we only have 1 timestep here but let's take last one in general
            >>> last_timestep = -1
            >>> d3plot.plot(0, field=pstrain[last_timestep])
            >>> # we don't like the fringe, let's adjust
            >>> d3plot.plot(0, field=pstrain[last_timestep], fringe_limits=(0, 0.3))
        """

        assert i_timestep < self._state_info.n_timesteps
        assert ArrayType.node_displacement in self.arrays
        if fringe_limits:
            assert len(fringe_limits) == 2

        # shell nodes
        shell_node_indexes = self.arrays[ArrayType.element_shell_node_indexes]

        # get node displacement
        node_xyz = self.arrays[ArrayType.node_displacement][i_timestep, :, :]

        # check for correct field size
        if isinstance(field, np.ndarray):
            assert field.ndim == 1
            if is_element_field and len(shell_node_indexes) != len(field):  # type: ignore
                msg = "Element indexes and field have different len: {} != {}"
                raise ValueError(msg.format(shell_node_indexes.shape, field.shape))
            if not is_element_field and len(node_xyz) != len(field):  # type: ignore
                msg = "Node field and coords have different len: {} != {}"
                raise ValueError(msg.format(node_xyz.shape, field.shape))

        # create plot
        _html = plot_shell_mesh(
            node_coordinates=node_xyz,
            shell_node_indexes=shell_node_indexes,
            field=field,
            is_element_field=is_element_field,
            fringe_limits=fringe_limits,
        )

        # store in a temporary file
        tempdir = tempfile.gettempdir()
        tempdir = os.path.join(tempdir, "lasso")
        if not os.path.isdir(tempdir):
            os.mkdir(tempdir)

        for tmpfile in os.listdir(tempdir):
            tmpfile = os.path.join(tempdir, tmpfile)
            if os.path.isfile(tmpfile):
                os.remove(tmpfile)

        if export_filepath:
            with open(export_filepath, "w", encoding="utf-8") as fp:
                fp.write(_html)
        else:
            # create new temp file
            with tempfile.NamedTemporaryFile(
                dir=tempdir, suffix=".html", mode="w", delete=False
            ) as fp:
                fp.write(_html)
                webbrowser.open(fp.name)

    def write_d3plot(
        self, filepath: Union[str, BinaryIO], block_size_bytes: int = 2048, single_file: bool = True
    ):
        """Write a d3plot file again **(pro version only)**

        Parameters
        ----------
        filepath: Union[str, BinaryIO]
            filepath of the new d3plot file or an opened file handle
        block_size_bytes: int
            D3plots are originally written in byte-blocks causing zero-padding at the end of
            files. This can be controlled by this parameter. Set to 0 for no padding.
        single_file: bool
            whether to write all states into a single file

        Examples
        --------
            Modify an existing d3plot:

            >>> d3plot = D3plot("path/to/d3plot")
            >>> hvars = d3plot.array[ArrayType.element_shell_history_vars]
            >>> hvars.shape
            (1, 4696, 3, 19)
            >>> new_history_var = np.random.random((1, 4696, 3, 1))
            >>> new_hvars = np.concatenate([hvars, new_history_var], axis=3)
            >>> d3plot.array[ArrayType.element_shell_history_vars] = new_hvars
            >>> d3plot.write_d3plot("path/to/new/d3plot")

            Write a new d3plot from scratch:

            >>> d3plot = D3plot()
            >>> d3plot.arrays[ArrayType.node_coordinates] = np.array([[0, 0, 0],
            ...                                                       [1, 0, 0],
            ...                                                       [0, 1, 0]])
            >>> d3plot.arrays[ArrayType.element_shell_node_indexes] = np.array([[0, 2, 1, 1]])
            >>> d3plot.arrays[ArrayType.element_shell_part_indexes] = np.array([0])
            >>> d3plot.arrays[ArrayType.node_displacement] = np.array([[[0, 0, 0],
            ...                                                         [1, 0, 0],
            ...                                                         [0, 1, 0]]])
            >>> d3plot.write_d3plot("yay.d3plot")

        Notes
        -----
            This function is not available in the public version please contact
            LASSO directly in case of further interest.
        """

        # if there is a single buffer, write all in
        if not isinstance(filepath, str):
            single_file = True

        # determine write settings
        write_settings = D3plotWriterSettings(self, block_size_bytes, single_file)
        write_settings.build_header()

        # remove old files
        if isinstance(filepath, str):
            filepaths = D3plot._find_dyna_result_files(filepath)
            for path in filepaths:
                if os.path.isfile(path):
                    os.remove(path)

        # write geometry file
        with open_file_or_filepath(filepath, "wb") as fp:

            n_bytes_written = 0
            msg = "wrote {0} after {1}."

            # header
            n_bytes_written += self._write_header(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_header")

            # material section
            n_bytes_written += self._write_geom_material_section(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geom_material_section")

            # fluid material data
            n_bytes_written += self._write_geom_fluid_material_header(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geom_fluid_material_header")

            # SPH element data flags
            n_bytes_written += self._write_geom_sph_element_data_flags(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geom_sph_element_data_flags")

            # Particle Data
            n_bytes_written += self._write_geom_particle_flags(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geom_particle_flags")

            # Geometry Data
            n_bytes_written += self._write_geometry(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geometry")

            # User Material, Node, Blabla IDs
            n_bytes_written += self._write_geom_user_ids(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geom_user_ids")

            # Rigid Body Description
            n_bytes_written += self._write_geom_rigid_body_description(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geom_rigid_body_description")

            # Adapted Element Parent List
            # not supported

            # Smooth Particle Hydrodynamcis Node and Material list
            n_bytes_written += self._write_geom_sph_node_and_materials(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geom_sph_node_and_materials")

            # Particle Geometry Data
            n_bytes_written += self._write_geom_particle_geometry_data(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geom_particle_geometry_data")

            # Rigid Road Surface Data
            n_bytes_written += self._write_geom_rigid_road_surface(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geom_rigid_road_surface")

            # Connectivity for weirdo elements
            # 10 Node Tetra
            # 8 Node Shell
            # 20 Node Solid
            # 27 Node Solid
            n_bytes_written += self._write_geom_extra_node_data(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_geom_extra_node_data")

            # end mark
            n_bytes_written += fp.write(write_settings.pack(-999999.0))
            LOGGER.debug(msg, n_bytes_written, "_end_mark")

            # Header Part & Contact Interface Titles
            n_bytes_written_before_titles = n_bytes_written
            n_bytes_written += self._write_header_part_contact_interface_titles(fp, write_settings)
            LOGGER.debug(msg, n_bytes_written, "_write_header_part_contact_interface_titles")

            if n_bytes_written_before_titles != n_bytes_written:

                # we seal the file here with an endmark
                n_bytes_written += fp.write(write_settings.pack(-999999.0))
                LOGGER.debug(msg, n_bytes_written, "_end_mark")
            else:
                pass
                # we already set an end-mark before
                # that is perfectly fine

            # correct zero padding at the end
            if block_size_bytes > 0:
                zero_bytes = self._get_zero_byte_padding(n_bytes_written, block_size_bytes)
                n_bytes_written += fp.write(zero_bytes)
                LOGGER.debug(msg, n_bytes_written, "_zero_byte_padding")

            msg = "Wrote {0} bytes to geometry file."
            LOGGER.debug(msg, n_bytes_written)

            # Extra Data Types (for multi solver output)
            # not supported

        # write states
        self._write_states(filepath, write_settings)

    def _write_header(self, fp: typing.IO[Any], settings: D3plotWriterSettings) -> int:

        wordsize = settings.wordsize

        header_words = {
            "title": (0 * wordsize, 10 * wordsize),
            "runtime": (10 * wordsize, wordsize),
            "filetype": (11 * wordsize, wordsize),
            "source_version": (12 * wordsize, wordsize),
            "release_version": (13 * wordsize, wordsize),
            "version": (14 * wordsize, wordsize),
            "ndim": (15 * wordsize, wordsize),
            "numnp": (16 * wordsize, wordsize),
            "icode": (17 * wordsize, wordsize),
            "nglbv": (18 * wordsize, wordsize),
            "it": (19 * wordsize, wordsize),
            "iu": (20 * wordsize, wordsize),
            "iv": (21 * wordsize, wordsize),
            "ia": (22 * wordsize, wordsize),
            "nel8": (23 * wordsize, wordsize),
            "nummat8": (24 * wordsize, wordsize),
            "numds": (25 * wordsize, wordsize),
            "numst": (26 * wordsize, wordsize),
            "nv3d": (27 * wordsize, wordsize),
            "nel2": (28 * wordsize, wordsize),
            "nummat2": (29 * wordsize, wordsize),
            "nv1d": (30 * wordsize, wordsize),
            "nel4": (31 * wordsize, wordsize),
            "nummat4": (32 * wordsize, wordsize),
            "nv2d": (33 * wordsize, wordsize),
            "neiph": (34 * wordsize, wordsize),
            "neips": (35 * wordsize, wordsize),
            "maxint": (36 * wordsize, wordsize),
            "nmsph": (37 * wordsize, wordsize),
            "ngpsph": (38 * wordsize, wordsize),
            "narbs": (39 * wordsize, wordsize),
            "nelth": (40 * wordsize, wordsize),
            "nummatt": (41 * wordsize, wordsize),
            "nv3dt": (42 * wordsize, wordsize),
            "ioshl1": (43 * wordsize, wordsize),
            "ioshl2": (44 * wordsize, wordsize),
            "ioshl3": (45 * wordsize, wordsize),
            "ioshl4": (46 * wordsize, wordsize),
            "ialemat": (47 * wordsize, wordsize),
            "ncfdv1": (48 * wordsize, wordsize),
            "ncfdv2": (49 * wordsize, wordsize),
            # "nadapt": (50*wordsize, wordsize),
            "nmmat": (51 * wordsize, wordsize),
            "numfluid": (52 * wordsize, wordsize),
            "inn": (53 * wordsize, wordsize),
            "npefg": (54 * wordsize, wordsize),
            "nel48": (55 * wordsize, wordsize),
            "idtdt": (56 * wordsize, wordsize),
            "extra": (57 * wordsize, wordsize),
        }

        header_extra_words = {
            "nel20": (64 * wordsize, wordsize),
            "nt3d": (65 * wordsize, wordsize),
            "nel27": (66 * wordsize, wordsize),
            "neipb": (67 * wordsize, wordsize),
        }

        new_header = settings.header

        barray = bytearray((64 + new_header["extra"]) * wordsize)

        for name, (position, size) in header_words.items():
            barray[position : position + size] = settings.pack(new_header[name], size)

        if new_header["extra"] > 0:
            for name, (position, size) in header_extra_words.items():
                barray[position : position + size] = settings.pack(new_header[name], size)

        n_bytes_written = fp.write(barray)

        # check
        n_bytes_expected = (64 + new_header["extra"]) * settings.wordsize
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_geom_material_section(
        self, fp: typing.IO[Any], settings: D3plotWriterSettings
    ) -> int:

        if settings.mattyp <= 0:
            return 0

        _check_ndim(self, {ArrayType.part_material_type: ["n_parts"]})

        part_material_type_original = self.arrays[ArrayType.part_material_type]
        # part_material_type = np.full(settings.header["nmmat"], -1,
        #                               dtype=settings.itype)

        # if ArrayType.element_solid_part_indexes in self.arrays:
        #     unique_part_indexes = settings.unique_solid_part_indexes
        #     part_material_type[unique_part_indexes] = \
        #         part_material_type_original[unique_part_indexes]
        # if ArrayType.element_beam_part_indexes in self.arrays:
        #     unique_part_indexes = settings.unique_beam_part_indexes
        #     part_material_type[unique_part_indexes] = \
        #         part_material_type_original[unique_part_indexes]
        # if ArrayType.element_shell_part_indexes in self.arrays:
        #     unique_part_indexes = settings.unique_shell_part_indexes
        #     part_material_type[unique_part_indexes] = \
        #         part_material_type_original[unique_part_indexes]
        # if ArrayType.element_tshell_part_indexes in self.arrays:
        #     unique_part_indexes = settings.unique_tshell_part_indexes
        #     part_material_type[unique_part_indexes] = \
        #         part_material_type_original[unique_part_indexes]

        numrbe = settings.n_rigid_shells

        n_bytes_written = 0
        n_bytes_written += fp.write(settings.pack(numrbe))
        n_bytes_written += fp.write(settings.pack(len(part_material_type_original)))
        n_bytes_written += fp.write(settings.pack(part_material_type_original))

        # check
        n_bytes_expected = (len(part_material_type_original) + 2) * settings.wordsize
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_geom_fluid_material_header(
        self, fp: typing.IO[Any], settings: D3plotWriterSettings
    ) -> int:

        if settings.header["ialemat"] == 0:
            return 0

        _check_ndim(self, {ArrayType.ale_material_ids: ["n_ale_parts"]})

        array = self.arrays[ArrayType.ale_material_ids]
        n_bytes_written = fp.write(settings.pack(array, dtype_hint=np.integer))

        # check
        n_bytes_expected = settings.header["ialemat"] * settings.wordsize
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_geom_sph_element_data_flags(
        self, fp: typing.IO[Any], settings: D3plotWriterSettings
    ) -> int:

        if settings.header["nmsph"] <= 0:
            return 0

        n_sph_var_count = 0

        # radius
        n_sph_radius_vars = 1 if ArrayType.sph_radius in self.arrays else 0
        n_sph_var_count += n_sph_radius_vars

        # pressure
        n_sph_pressure_vars = 1 if ArrayType.sph_pressure in self.arrays else 0
        n_sph_var_count += n_sph_pressure_vars

        # stress
        n_sph_stress_vars = 6 if ArrayType.sph_stress in self.arrays else 0
        n_sph_var_count += n_sph_stress_vars

        # eff pstrain
        n_sph_eff_pstrain_vars = 1 if ArrayType.sph_effective_plastic_strain in self.arrays else 0
        n_sph_var_count += n_sph_eff_pstrain_vars

        # density
        n_sph_density_vars = 1 if ArrayType.sph_density in self.arrays else 0
        n_sph_var_count += n_sph_density_vars

        # internal energy
        n_sph_internal_energy_vars = 1 if ArrayType.sph_internal_energy in self.arrays else 0
        n_sph_var_count += n_sph_internal_energy_vars

        # n neighbors
        n_sph_n_neighbors_vars = 1 if ArrayType.sph_n_neighbors in self.arrays else 0
        n_sph_var_count += n_sph_n_neighbors_vars

        # strains
        n_sph_strain_vars = 6 if ArrayType.sph_strain in self.arrays else 0
        n_sph_var_count += n_sph_strain_vars

        # mass
        n_sph_mass_vars = 1 if ArrayType.sph_mass in self.arrays else 0
        n_sph_var_count += n_sph_mass_vars

        # history vars
        n_sph_history_vars = 0
        if ArrayType.sph_history_vars in self.arrays:
            n_sph_history_vars, _ = settings.count_array_state_var(
                ArrayType.sph_history_vars,
                ["n_timesteps", "n_sph_particles", "n_sph_history_vars"],
                False,
            )
        n_sph_var_count += n_sph_history_vars

        # write
        n_bytes_written = 0
        n_bytes_written += fp.write(settings.pack(n_sph_var_count))
        n_bytes_written += fp.write(settings.pack(n_sph_radius_vars))
        n_bytes_written += fp.write(settings.pack(n_sph_pressure_vars))
        n_bytes_written += fp.write(settings.pack(n_sph_stress_vars))
        n_bytes_written += fp.write(settings.pack(n_sph_eff_pstrain_vars))
        n_bytes_written += fp.write(settings.pack(n_sph_density_vars))
        n_bytes_written += fp.write(settings.pack(n_sph_internal_energy_vars))
        n_bytes_written += fp.write(settings.pack(n_sph_n_neighbors_vars))
        n_bytes_written += fp.write(settings.pack(n_sph_strain_vars))
        n_bytes_written += fp.write(settings.pack(n_sph_mass_vars))
        n_bytes_written += fp.write(settings.pack(n_sph_history_vars))

        # check
        n_bytes_expected = 11 * settings.wordsize
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_geom_particle_flags(self, fp: typing.IO[Any], settings: D3plotWriterSettings) -> int:

        npefg = settings.header["npefg"]

        if npefg <= 0 or npefg > 10000000:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.airbags_n_particles: ["n_airbags"],
                ArrayType.airbags_n_chambers: ["n_airbags"],
            },
        )

        # n_airbags = npefg % 1000
        subver = npefg // 1000

        # airbag geometry var count
        ngeom = 5 if ArrayType.airbags_n_chambers in self.arrays else 4

        # state variable count
        # see later
        nvar = 14

        # n particles
        n_particles = 0
        if ArrayType.airbags_n_particles in self.arrays:
            n_particles = np.sum(self.arrays[ArrayType.airbags_n_particles])

        # airbag state var count
        nstgeom = 2

        # write
        n_bytes_written = 0
        n_bytes_written += fp.write(settings.pack(ngeom))
        n_bytes_written += fp.write(settings.pack(nvar))
        n_bytes_written += fp.write(settings.pack(n_particles))
        n_bytes_written += fp.write(settings.pack(nstgeom))
        if subver == 4:
            # This was never validated
            n_bytes_written += fp.write(
                settings.pack(self.arrays[ArrayType.airbags_n_chambers].sum())
            )

        # check
        n_bytes_expected = (5 if subver == 4 else 4) * settings.wordsize
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        # typecode for variables
        # pylint: disable = invalid-name
        INT_TC = 1
        # pylint: disable = invalid-name
        FLOAT_TC = 2
        nlist_names_typecodes = [
            # airbag geometry data (ngeom)
            ["Start N ", INT_TC],
            ["Npart   ", INT_TC],
            ["Bag  ID ", INT_TC],
            ["NGasC   ", INT_TC],
            ["NCham   ", INT_TC],
            # state particle data (nvar)
            ["GasC ID ", INT_TC],
            ["Cham ID ", INT_TC],
            ["Leakage ", INT_TC],
            ["Pos x   ", FLOAT_TC],
            ["Pos y   ", FLOAT_TC],
            ["Pos z   ", FLOAT_TC],
            ["Vel x   ", FLOAT_TC],
            ["Vel y   ", FLOAT_TC],
            ["Vel z   ", FLOAT_TC],
            ["Mass    ", FLOAT_TC],
            ["Radius  ", FLOAT_TC],
            ["Spin En ", FLOAT_TC],
            ["Tran En ", FLOAT_TC],
            ["NS dist ", FLOAT_TC],
            # airbag state vars (nstgeom)
            ["Act Gas ", INT_TC],
            ["Bag Vol ", FLOAT_TC],
        ]

        # airbag var typecodes
        for _, typecode in nlist_names_typecodes:
            n_bytes_written += fp.write(settings.pack(typecode))

        # airbag var names
        # every word is an ascii char. So, we need to set
        # only the first byte to the ascii char code
        fmt_string = "{0:" + str(settings.wordsize) + "}"
        for name, _ in nlist_names_typecodes:
            name_formatted = fmt_string.format(name).encode("ascii")
            for ch in name_formatted:
                barray = bytearray(settings.wordsize)
                barray[0] = ch

                n_bytes_written += fp.write(settings.pack(barray, settings.wordsize))

        # check
        n_bytes_expected += len(nlist_names_typecodes) * 9 * settings.wordsize
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_geometry(self, fp: typing.IO[Any], settings: D3plotWriterSettings) -> int:

        n_bytes_written = 0

        # pre-checks
        _check_ndim(
            self,
            {
                ArrayType.node_coordinates: ["n_nodes", "x_y_z"],
                ArrayType.element_solid_node_indexes: ["n_solids", "n_element_nodes"],
                ArrayType.element_solid_part_indexes: ["n_solids"],
                ArrayType.element_solid_extra_nodes: ["n_solids", "n_extra_nodes"],
                ArrayType.element_tshell_node_indexes: ["n_tshells", "n_element_nodes"],
                ArrayType.element_tshell_part_indexes: ["n_tshells"],
                ArrayType.element_beam_node_indexes: ["n_beams", "n_element_nodes"],
                ArrayType.element_beam_part_indexes: ["n_beams"],
                ArrayType.element_shell_node_indexes: ["n_shells", "n_element_nodes"],
                ArrayType.element_shell_part_indexes: ["n_shells"],
            },
        )
        self.check_array_dims({ArrayType.node_coordinates: 1}, "x_y_z", 3)

        array_dims = {
            ArrayType.element_solid_node_indexes: 0,
            ArrayType.element_solid_part_indexes: 0,
            ArrayType.element_solid_extra_nodes: 0,
        }
        n_solids = self.check_array_dims(array_dims, "n_solids")
        self.check_array_dims({ArrayType.element_solid_node_indexes: 1}, "n_element_nodes", 8)
        self.check_array_dims({ArrayType.element_solid_extra_nodes: 1}, "n_extra_nodes", 2)
        array_dims = {
            ArrayType.element_tshell_node_indexes: 0,
            ArrayType.element_tshell_part_indexes: 0,
        }
        self.check_array_dims(array_dims, "n_tshells")
        self.check_array_dims({ArrayType.element_tshell_node_indexes: 1}, "n_element_nodes", 8)
        array_dims = {
            ArrayType.element_beam_node_indexes: 0,
            ArrayType.element_beam_part_indexes: 0,
        }
        self.check_array_dims(array_dims, "n_beams")
        self.check_array_dims({ArrayType.element_beam_node_indexes: 1}, "n_element_nodes", 5)
        array_dims = {
            ArrayType.element_shell_node_indexes: 0,
            ArrayType.element_shell_part_indexes: 0,
        }
        self.check_array_dims(array_dims, "n_shells")
        self.check_array_dims({ArrayType.element_shell_node_indexes: 1}, "n_element_nodes", 4)

        # NODES
        node_coordinates = (
            self.arrays[ArrayType.node_coordinates]
            if ArrayType.node_coordinates in self.arrays
            else np.zeros((0, settings.header["ndim"]), dtype=self.header.ftype)
        )
        n_bytes_written += fp.write(settings.pack(node_coordinates, dtype_hint=np.floating))

        # SOLIDS
        solid_node_indexes = (
            self.arrays[ArrayType.element_solid_node_indexes] + FORTRAN_OFFSET
            if ArrayType.element_solid_node_indexes in self.arrays
            else np.zeros((0, 8), dtype=self._header.itype)
        )
        solid_part_indexes = (
            self.arrays[ArrayType.element_solid_part_indexes] + FORTRAN_OFFSET
            if ArrayType.element_solid_part_indexes in self.arrays
            else np.zeros(0, dtype=self._header.itype)
        )
        solid_geom_array = np.concatenate(
            (solid_node_indexes, solid_part_indexes.reshape(n_solids, 1)), axis=1
        )
        n_bytes_written += fp.write(settings.pack(solid_geom_array, dtype_hint=np.integer))

        # SOLID 10
        # the two extra nodes
        if ArrayType.element_solid_extra_nodes in self.arrays:
            array = self.arrays[ArrayType.element_solid_extra_nodes] + FORTRAN_OFFSET
            n_bytes_written += fp.write(settings.pack(array, dtype_hint=np.integer))

        # THICK SHELLS
        tshell_node_indexes = (
            self.arrays[ArrayType.element_tshell_node_indexes] + FORTRAN_OFFSET
            if ArrayType.element_tshell_node_indexes in self.arrays
            else np.zeros((0, 8), dtype=self._header.itype)
        )
        tshell_part_indexes = (
            self.arrays[ArrayType.element_tshell_part_indexes] + FORTRAN_OFFSET
            if ArrayType.element_tshell_part_indexes in self.arrays
            else np.zeros(0, dtype=self._header.itype)
        )
        tshell_geom_array = np.concatenate(
            (tshell_node_indexes, tshell_part_indexes.reshape(-1, 1)), axis=1
        )
        n_bytes_written += fp.write(settings.pack(tshell_geom_array, dtype_hint=np.integer))

        # BEAMS
        beam_node_indexes = (
            self.arrays[ArrayType.element_beam_node_indexes] + FORTRAN_OFFSET
            if ArrayType.element_beam_node_indexes in self.arrays
            else np.zeros((0, 5), dtype=self._header.itype)
        )
        beam_part_indexes = (
            self.arrays[ArrayType.element_beam_part_indexes] + FORTRAN_OFFSET
            if ArrayType.element_beam_part_indexes in self.arrays
            else np.zeros(0, dtype=self._header.itype)
        )
        beam_geom_array = np.concatenate(
            (beam_node_indexes, beam_part_indexes.reshape(-1, 1)), axis=1
        )
        n_bytes_written += fp.write(settings.pack(beam_geom_array, dtype_hint=np.integer))

        # SHELLS
        shell_node_indexes = (
            self.arrays[ArrayType.element_shell_node_indexes] + FORTRAN_OFFSET
            if ArrayType.element_shell_node_indexes in self.arrays
            else np.zeros((0, 4), dtype=self._header.itype)
        )
        shell_part_indexes = (
            self.arrays[ArrayType.element_shell_part_indexes] + FORTRAN_OFFSET
            if ArrayType.element_shell_part_indexes in self.arrays
            else np.zeros(0, dtype=self._header.itype)
        )
        shell_geom_array = np.concatenate(
            (shell_node_indexes, shell_part_indexes.reshape(-1, 1)), axis=1
        )
        n_bytes_written += fp.write(settings.pack(shell_geom_array, dtype_hint=np.integer))

        # check
        n_bytes_expected = (
            settings.header["numnp"] * 3
            + abs(settings.header["nel8"]) * 9
            + settings.header["nelth"] * 9
            + settings.header["nel2"] * 6
            + settings.header["nel4"] * 5
        ) * settings.wordsize
        if ArrayType.element_solid_extra_nodes in self.arrays:
            n_bytes_expected += 2 * abs(settings.header["nel8"])
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        # return the chunks
        return n_bytes_written

    def _write_geom_user_ids(self, fp: typing.IO[Any], settings: D3plotWriterSettings) -> int:

        narbs = settings.header["narbs"]
        if narbs == 0:
            return 0

        info = self._numbering_info

        _check_ndim(
            self,
            {
                ArrayType.node_ids: ["n_nodes"],
                ArrayType.element_solid_ids: ["n_solids"],
                ArrayType.element_beam_ids: ["n_beams"],
                ArrayType.element_shell_ids: ["n_shells"],
                ArrayType.element_tshell_ids: ["n_tshells"],
                ArrayType.part_ids: ["n_parts"],
                ArrayType.part_ids_unordered: ["n_parts"],
                ArrayType.part_ids_cross_references: ["n_parts"],
            },
        )

        n_bytes_written = 0

        # NUMBERING HEADER

        # nsort seems to be solver internal pointer
        # ... hopefully
        nsort = info.ptr_node_ids
        nsort *= -1 if ArrayType.part_ids in self.arrays else 1

        n_bytes_written += fp.write(settings.pack(nsort))

        nsrh = abs(nsort) + settings.header["numnp"]
        n_bytes_written += fp.write(settings.pack(nsrh))

        nsrb = nsrh + abs(settings.header["nel8"])
        n_bytes_written += fp.write(settings.pack(nsrb))

        nsrs = nsrb + settings.header["nel2"]
        n_bytes_written += fp.write(settings.pack(nsrs))

        nsrt = nsrs + settings.header["nel4"]
        n_bytes_written += fp.write(settings.pack(nsrt))

        nsortd = settings.header["numnp"]
        n_bytes_written += fp.write(settings.pack(nsortd))

        nsrhd = abs(settings.header["nel8"])
        n_bytes_written += fp.write(settings.pack(nsrhd))

        nsrbd = settings.header["nel2"]
        n_bytes_written += fp.write(settings.pack(nsrbd))

        nsrsd = settings.header["nel4"]
        n_bytes_written += fp.write(settings.pack(nsrsd))

        nsrtd = settings.header["nelth"]
        n_bytes_written += fp.write(settings.pack(nsrtd))

        if ArrayType.part_ids in self.arrays:
            # some lsdyna material pointer
            nsrma = info.ptr_material_ids
            n_bytes_written += fp.write(settings.pack(nsrma))

            # some lsdyna material pointer
            nsrmu = info.ptr_material_ids_defined_order
            n_bytes_written += fp.write(settings.pack(nsrmu))

            # some lsdyna material pointer
            nsrmp = info.ptr_material_ids_crossref
            n_bytes_written += fp.write(settings.pack(nsrmp))

            # "Total number of materials (parts)"
            nsrtm = settings.header["nmmat"]
            n_bytes_written += fp.write(settings.pack(nsrtm))

            # Total number of nodal rigid body constraint sets
            numrbs = settings.header["numrbs"]
            n_bytes_written += fp.write(settings.pack(numrbs))

            # Total number of materials
            # ... coz it's fun doing nice things twice
            nmmat = settings.header["nmmat"]
            n_bytes_written += fp.write(settings.pack(nmmat))

        # NODE IDS
        node_ids = (
            self.arrays[ArrayType.node_ids]
            if ArrayType.node_ids in self.arrays
            else np.arange(
                FORTRAN_OFFSET, settings.header["numnp"] + FORTRAN_OFFSET, dtype=settings.itype
            )
        )
        n_bytes_written += fp.write(settings.pack(node_ids, dtype_hint=np.integer))

        # SOLID IDS
        solid_ids = (
            self.arrays[ArrayType.element_solid_ids]
            if ArrayType.element_solid_ids in self.arrays
            else np.arange(
                FORTRAN_OFFSET, settings.header["nel8"] + FORTRAN_OFFSET, dtype=settings.itype
            )
        )
        n_bytes_written += fp.write(settings.pack(solid_ids, dtype_hint=np.integer))

        # BEAM IDS
        beam_ids = (
            self.arrays[ArrayType.element_beam_ids]
            if ArrayType.element_beam_ids in self.arrays
            else np.arange(
                FORTRAN_OFFSET, settings.header["nel2"] + FORTRAN_OFFSET, dtype=settings.itype
            )
        )
        n_bytes_written += fp.write(settings.pack(beam_ids, dtype_hint=np.integer))

        # SHELL IDS
        shell_ids = (
            self.arrays[ArrayType.element_shell_ids]
            if ArrayType.element_shell_ids in self.arrays
            else np.arange(
                FORTRAN_OFFSET, settings.header["nel4"] + FORTRAN_OFFSET, dtype=settings.itype
            )
        )
        n_bytes_written += fp.write(settings.pack(shell_ids, dtype_hint=np.integer))

        # TSHELL IDS
        tshell_ids = (
            self.arrays[ArrayType.element_tshell_ids]
            if ArrayType.element_tshell_ids in self.arrays
            else np.arange(
                FORTRAN_OFFSET, settings.header["nelth"] + FORTRAN_OFFSET, dtype=settings.itype
            )
        )
        n_bytes_written += fp.write(settings.pack(tshell_ids, dtype_hint=np.integer))

        # MATERIALS .... yay
        #
        # lsdyna generates duplicate materials originally
        # thus nmmat in header is larger than the materials used
        # by the elements. Some are related to rigid bodies
        # but some are also generated internally by material models
        # by the following procedure the material array is larger
        # than the actual amount of materials (there may be unused
        # material ids), but it ensures a relatively consistent writing

        material_ids = np.full(settings.header["nmmat"], -1, dtype=self._header.itype)
        if ArrayType.part_ids in self.arrays:
            part_ids = self.arrays[ArrayType.part_ids]
            material_ids = part_ids
        else:
            material_ids = np.arange(start=0, stop=settings.header["nmmat"], dtype=settings.itype)

        n_bytes_written += fp.write(settings.pack(material_ids, dtype_hint=np.integer))

        # unordered material ids can be ignored
        data_array = np.zeros(settings.header["nmmat"], dtype=settings.itype)
        if ArrayType.part_ids_unordered in self.arrays:
            array = self.arrays[ArrayType.part_ids_unordered]
            end_index = min(len(array), len(data_array))
            data_array[:end_index] = array[:end_index]
        n_bytes_written += fp.write(settings.pack(data_array, dtype_hint=np.integer))

        # also cross-reference array for ids
        data_array = np.zeros(settings.header["nmmat"], dtype=settings.itype)
        if ArrayType.part_ids_cross_references in self.arrays:
            array = self.arrays[ArrayType.part_ids_cross_references]
            end_index = min(len(array), len(data_array))
            data_array[:end_index] = array[:end_index]
        n_bytes_written += fp.write(settings.pack(data_array, dtype_hint=np.integer))

        # check
        n_bytes_expected = settings.header["narbs"] * settings.wordsize
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_geom_rigid_body_description(
        self, fp: typing.IO[Any], settings: D3plotWriterSettings
    ) -> int:

        # this type of rigid body descriptions are very rare
        # and thus badly tested

        if settings.header["ndim"] not in (8, 9):
            return 0

        _check_ndim(
            self,
            {
                ArrayType.rigid_body_part_indexes: ["n_rigid_bodies"],
            },
        )
        array_dims = {
            ArrayType.rigid_body_part_indexes: 0,
            ArrayType.rigid_body_node_indexes_list: 0,
            ArrayType.rigid_body_active_node_indexes_list: 0,
        }
        if not _check_array_occurrence(self, list(array_dims.keys()), list(array_dims.keys())):
            return 0

        # check length
        # cannot use self._check_array_dims due to some lists
        dim_size = -1
        for typename in array_dims:
            array = self.arrays[typename]
            if dim_size < 0:
                dim_size = len(array)
            else:
                if len(array) != dim_size:
                    dimension_size_dict = {
                        typename2: len(self.arrays[typename2]) for typename2 in array_dims
                    }
                    msg = "Inconsistency in array dim '{0}' detected:\n{1}"
                    size_list = [
                        f"   - name: {typename}, dim: {array_dims[typename]}, size: {size}"
                        for typename, size in dimension_size_dict.items()
                    ]
                    raise ValueError(msg.format("n_rigid_bodies", "\n".join(size_list)))

        rigid_body_part_indexes = self.arrays[ArrayType.rigid_body_part_indexes] + FORTRAN_OFFSET
        # rigid_body_n_nodes = self.arrays[ArrayType.rigid_body_n_nodes]
        rigid_body_node_indexes_list = self.arrays[ArrayType.rigid_body_node_indexes_list]
        # rigid_body_n_active_nodes = self.arrays[ArrayType.rigid_body_n_active_nodes]
        rigid_body_active_node_indexes_list = self.arrays[
            ArrayType.rigid_body_active_node_indexes_list
        ]

        n_bytes_written = 0
        n_bytes_expected = settings.wordsize

        # NRIGID
        n_rigid_bodies = len(rigid_body_part_indexes)
        n_bytes_written += fp.write(settings.pack(n_rigid_bodies))

        for i_rigid in range(n_rigid_bodies):
            # part index
            n_bytes_written += fp.write(settings.pack(rigid_body_part_indexes[i_rigid]))
            # node indexes
            array = rigid_body_node_indexes_list[i_rigid] + FORTRAN_OFFSET
            n_bytes_written += fp.write(settings.pack(len(array)))
            n_bytes_written += fp.write(settings.pack(array, dtype_hint=np.integer))
            # active node indexes
            array = rigid_body_active_node_indexes_list[i_rigid]
            n_bytes_written += fp.write(settings.pack(len(array)))
            n_bytes_written += fp.write(settings.pack(array, dtype_hint=np.integer))

            n_bytes_expected += settings.wordsize * (
                3
                + len(rigid_body_node_indexes_list[i_rigid])
                + len(rigid_body_active_node_indexes_list[i_rigid])
            )

        # check
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_geom_sph_node_and_materials(
        self, fp: typing.IO[Any], settings: D3plotWriterSettings
    ) -> int:

        nmsph = settings.header["nmsph"]

        if nmsph <= 0:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.sph_node_indexes: ["n_sph_nodes"],
                ArrayType.sph_node_material_index: ["n_sph_nodes"],
            },
        )
        array_dims = {
            ArrayType.sph_node_indexes: 0,
            ArrayType.sph_node_material_index: 0,
        }
        array_names = list(array_dims.keys())
        _check_array_occurrence(self, array_names, array_names)
        self.check_array_dims(array_dims, "n_sph_nodes", nmsph)

        sph_node_indexes = self.arrays[ArrayType.sph_node_indexes] + FORTRAN_OFFSET
        sph_node_material_index = self.arrays[ArrayType.sph_node_material_index] + FORTRAN_OFFSET
        sph_data = np.concatenate((sph_node_indexes, sph_node_material_index), axis=1)

        # write
        n_bytes_written = fp.write(settings.pack(sph_data, dtype_hint=np.integer))

        # check
        n_bytes_expected = nmsph * settings.wordsize * 2
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_geom_particle_geometry_data(
        self, fp: typing.IO[Any], settings: D3plotWriterSettings
    ) -> int:

        npefg = settings.header["npefg"]
        if npefg <= 0:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.airbags_first_particle_id: ["n_airbags"],
                ArrayType.airbags_n_particles: ["n_airbags"],
                ArrayType.airbags_ids: ["n_airbags"],
                ArrayType.airbags_n_gas_mixtures: ["n_airbags"],
                ArrayType.airbags_n_chambers: ["n_airbags"],
            },
        )
        array_dims = {
            ArrayType.airbags_first_particle_id: 0,
            ArrayType.airbags_n_particles: 0,
            ArrayType.airbags_ids: 0,
            ArrayType.airbags_n_gas_mixtures: 0,
            ArrayType.airbags_n_chambers: 0,
        }
        array_names = list(array_dims.keys())
        _check_array_occurrence(self, array_names, array_names)
        self.check_array_dims(array_dims, "n_airbags")

        # get the arrays
        array_list = [
            self.arrays[ArrayType.airbags_first_particle_id].reshape(-1, 1),
            self.arrays[ArrayType.airbags_n_particles].reshape(-1, 1),
            self.arrays[ArrayType.airbags_ids].reshape(-1, 1),
            self.arrays[ArrayType.airbags_n_gas_mixtures].reshape(-1, 1),
        ]
        if ArrayType.airbags_n_chambers in self.arrays:
            array_list.append(self.arrays[ArrayType.airbags_n_chambers].reshape(-1, 1))

        # write
        airbag_geometry_data = np.concatenate(array_list, axis=1)
        n_bytes_written = fp.write(settings.pack(airbag_geometry_data, dtype_hint=np.integer))

        # check
        n_airbags = npefg % 1000
        ngeom = 5 if ArrayType.airbags_n_chambers in self.arrays else 4
        n_bytes_expected = n_airbags * ngeom * settings.wordsize
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_geom_rigid_road_surface(
        self, fp: typing.IO[Any], settings: D3plotWriterSettings
    ) -> int:

        if settings.header["ndim"] <= 5:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.rigid_road_node_ids: ["rigid_road_n_nodes"],
                ArrayType.rigid_road_node_coordinates: ["rigid_road_n_nodes", "x_y_z"],
                ArrayType.rigid_road_segment_node_ids: ["n_segments", "n_nodes"],
                ArrayType.rigid_road_segment_road_id: ["n_segments"],
            },
        )
        array_dims = {
            ArrayType.rigid_road_node_ids: 0,
            ArrayType.rigid_road_node_coordinates: 0,
        }
        n_rigid_road_nodes = self.check_array_dims(array_dims, "rigid_road_n_nodes")
        self.check_array_dims({ArrayType.rigid_road_node_coordinates: 1}, "x_y_z", 3)
        array_dims = {
            ArrayType.rigid_road_n_segments: 0,
            ArrayType.rigid_road_segment_node_ids: 0,
            ArrayType.rigid_road_segment_road_id: 0,
        }
        n_rigid_roads = self.check_array_dims(array_dims, "n_rigid_roads")
        n_bytes_written = 0

        # NODE COUNT
        n_bytes_written += fp.write(settings.pack(n_rigid_road_nodes))

        # SEGMENT COUNT
        # This was never verified
        n_total_segments = np.sum(
            len(segment_ids) for segment_ids in self.arrays[ArrayType.rigid_road_segment_node_ids]
        )
        n_bytes_written += fp.write(settings.pack(n_total_segments))

        # SURFACE COUNT
        n_bytes_written += fp.write(settings.pack(n_rigid_roads))

        # MOTION FLAG - if motion data is output
        # by default let's just say ... yeah baby
        # This was never verified
        n_bytes_written += fp.write(settings.pack(1))

        # RIGID ROAD NODE IDS
        rigid_road_node_ids = self.arrays[ArrayType.rigid_road_node_ids]
        n_bytes_written += fp.write(settings.pack(rigid_road_node_ids, dtype_hint=np.integer))

        # RIGID ROAD NODE COORDS
        rigid_road_node_coordinates = self.arrays[ArrayType.rigid_road_node_coordinates]
        n_bytes_written += fp.write(
            settings.pack(rigid_road_node_coordinates, dtype_hint=np.floating)
        )

        # SURFACE ID
        # SURFACE N_SEGMENTS
        # SURFACE SEGMENTS
        rigid_road_segment_road_id = self.arrays[ArrayType.rigid_road_segment_road_id]
        rigid_road_segment_node_ids = self.arrays[ArrayType.rigid_road_segment_node_ids]

        for segment_id, node_ids in zip(rigid_road_segment_road_id, rigid_road_segment_node_ids):
            n_bytes_written += fp.write(settings.pack(segment_id))
            n_bytes_written += fp.write(settings.pack(len(node_ids)))
            n_bytes_written += fp.write(settings.pack(node_ids, dtype_hint=np.integer))

        # check
        n_bytes_expected = (
            4 + 4 * n_rigid_road_nodes + n_rigid_roads * (2 + 4 * n_total_segments)
        ) * settings.wordsize
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_geom_extra_node_data(
        self, fp: typing.IO[Any], settings: D3plotWriterSettings
    ) -> int:

        n_bytes_written = 0

        # 10 NODE SOLIDS
        if settings.header["nel8"] < 0:
            _check_ndim(
                self,
                {
                    ArrayType.element_solid_node10_extra_node_indexes: [
                        "n_solids",
                        "2_extra_node_ids",
                    ],
                },
            )
            array_dims = {
                ArrayType.element_solid_node_indexes: 0,
                ArrayType.element_solid_node10_extra_node_indexes: 0,
            }
            self.check_array_dims(array_dims, "n_solids")
            self.check_array_dims(
                {ArrayType.element_solid_node10_extra_node_indexes: 1}, "extra_node_ids", 2
            )

            extra_nodes = (
                self.arrays[ArrayType.element_solid_node10_extra_node_indexes] + FORTRAN_OFFSET
            )

            n_bytes_written += fp.write(settings.pack(extra_nodes, dtype_hint=np.integer))

        # 8 NODE SHELLS
        if settings.header["nel48"] > 0:
            _check_ndim(
                self,
                {
                    ArrayType.element_shell_node8_element_index: ["n_node8_shells"],
                    ArrayType.element_shell_node8_extra_node_indexes: [
                        "n_node8_shells",
                        "4_extra_node_ids",
                    ],
                },
            )
            array_dims = {
                ArrayType.element_shell_node8_element_index: 0,
                ArrayType.element_shell_node8_extra_node_indexes: 0,
            }
            self.check_array_dims(array_dims, "n_node8_shells")
            self.check_array_dims(
                {ArrayType.element_shell_node8_extra_node_indexes: 1}, "extra_node_ids", 4
            )

            element_indexes = (
                self.arrays[ArrayType.element_shell_node8_element_index] + FORTRAN_OFFSET
            )
            extra_nodes = (
                self.arrays[ArrayType.element_shell_node8_extra_node_indexes] + FORTRAN_OFFSET
            )

            geom_data = np.concatenate((element_indexes, extra_nodes), axis=1)

            n_bytes_written += fp.write(settings.pack(geom_data, dtype_hint=np.integer))

        # 20 NODE SOLIDS
        if settings.header["extra"] > 0 and settings.header["nel20"] > 0:
            _check_ndim(
                self,
                {
                    ArrayType.element_solid_node20_element_index: ["n_node20_solids"],
                    ArrayType.element_solid_node20_extra_node_indexes: [
                        "n_node20_solids",
                        "12_extra_node_ids",
                    ],
                },
            )
            array_dims = {
                ArrayType.element_solid_node20_element_index: 0,
                ArrayType.element_solid_node20_extra_node_indexes: 0,
            }
            self.check_array_dims(array_dims, "n_node20_solids")
            self.check_array_dims(
                {ArrayType.element_solid_node20_extra_node_indexes: 1}, "extra_node_ids", 12
            )

            element_indexes = (
                self.arrays[ArrayType.element_solid_node20_element_index] + FORTRAN_OFFSET
            )
            extra_nodes = (
                self.arrays[ArrayType.element_solid_node20_extra_node_indexes] + FORTRAN_OFFSET
            )

            geom_data = np.concatenate((element_indexes, extra_nodes), axis=1)

            n_bytes_written += fp.write(settings.pack(geom_data, dtype_hint=np.integer))

        # 27 NODE SOLIDS
        if settings.header["extra"] > 0 and settings.header["nel27"] > 0:
            _check_ndim(
                self,
                {
                    ArrayType.element_solid_node20_element_index: ["n_node27_solids"],
                    ArrayType.element_solid_node20_extra_node_indexes: [
                        "n_node27_solids",
                        "19_extra_node_ids",
                    ],
                },
            )
            array_dims = {
                ArrayType.element_solid_node27_element_index: 0,
                ArrayType.element_solid_node27_extra_node_indexes: 0,
            }
            self.check_array_dims(array_dims, "n_node27_solids")
            self.check_array_dims(
                {ArrayType.element_solid_node27_extra_node_indexes: 1}, "extra_node_ids", 19
            )

            element_indexes = (
                self.arrays[ArrayType.element_solid_node27_element_index] + FORTRAN_OFFSET
            )
            extra_nodes = (
                self.arrays[ArrayType.element_solid_node27_extra_node_indexes] + FORTRAN_OFFSET
            )

            geom_data = np.concatenate((element_indexes, extra_nodes), axis=1)

            n_bytes_written += fp.write(settings.pack(geom_data, dtype_hint=np.integer))

        # check
        has_nel10 = settings.header["nel8"] < 0
        n_bytes_expected = (
            has_nel10 * abs(settings.header["nel8"])
            + settings.header["nel48"] * 5
            + settings.header["nel20"] * 13
            + settings.header["nel27"] * 20
        ) * settings.wordsize
        D3plot._compare_n_bytes_checksum(n_bytes_written, n_bytes_expected)

        return n_bytes_written

    def _write_header_part_contact_interface_titles(
        self, fp: typing.IO[Any], settings: D3plotWriterSettings
    ) -> int:

        n_bytes_written = 0

        # PART TITLES
        _check_ndim(
            self,
            {
                # ArrayType.part_titles: ["n_parts", "n_chars"],
                ArrayType.part_titles_ids: ["n_parts"],
            },
        )
        array_dimensions = {
            ArrayType.part_titles: 0,
            ArrayType.part_titles_ids: 0,
        }
        if _check_array_occurrence(
            self, list(array_dimensions.keys()), list(array_dimensions.keys())
        ):
            self.check_array_dims(array_dimensions, "n_parts")

            ntype = 90001

            n_bytes_written += fp.write(settings.pack(ntype))

            part_titles_ids = self.arrays[ArrayType.part_titles_ids]
            part_titles = self.arrays[ArrayType.part_titles]

            n_entries = len(part_titles)
            n_bytes_written += fp.write(settings.pack(n_entries))

            # title words always have 4 byte size
            title_wordsize = 4
            max_len = 18 * title_wordsize
            fmt_name = "{0:" + str(max_len) + "}"
            for pid, title in zip(part_titles_ids, part_titles):
                title = title.decode("ascii")
                n_bytes_written += fp.write(settings.pack(pid))

                formatted_title = fmt_name.format(title[:max_len])
                n_bytes_written += fp.write(settings.pack(formatted_title, max_len))

        # TITLE2
        # yet another title, coz double is always more fun
        if "title2" in self.header.title2:
            ntype = 90000

            # title words always have 4 bytes
            title_wordsize = 4
            title_size_words = 18

            fmt_title2 = "{0:" + str(title_wordsize * title_size_words) + "}"
            title2 = fmt_title2.format(self.header.title2[: settings.wordsize * title_size_words])

            n_bytes_written += fp.write(settings.pack(ntype))
            n_bytes_written += fp.write(settings.pack(title2, settings.wordsize * title_size_words))

        # CONTACT TITLES
        array_dimensions = {
            ArrayType.contact_titles: 0,
            ArrayType.contact_title_ids: 0,
        }
        if _check_array_occurrence(
            self, list(array_dimensions.keys()), list(array_dimensions.keys())
        ):
            self.check_array_dims(array_dimensions, "n_parts")

            ntype = 90002
            n_bytes_written += fp.write(settings.pack(ntype))

            titles_ids = self.arrays[ArrayType.contact_title_ids]
            titles = self.arrays[ArrayType.contact_titles]

            n_entries = len(titles)
            n_bytes_written += fp.write(settings.pack(n_entries))

            max_len = 18 * self.header.wordsize
            fmt_name = "{0:" + str(max_len) + "}"
            for pid, title in zip(titles_ids, titles):
                n_bytes_written += fp.write(settings.pack(pid))

                formatted_title = fmt_name.format(title[:max_len])
                n_bytes_written += fp.write(settings.pack(formatted_title))

        return n_bytes_written

    def _write_states(
        self, filepath: Union[str, typing.BinaryIO], settings: D3plotWriterSettings
    ) -> int:

        # did we store any states?
        n_timesteps_written = 0

        # if timestep array is missing check for any state arrays
        if ArrayType.global_timesteps not in self.arrays:
            # if any state array is present simply make up a timestep array
            if any(array_name in self.arrays for array_name in ArrayType.get_state_array_names()):
                array_dims = {array_name: 0 for array_name in ArrayType.get_state_array_names()}
                n_timesteps = self.check_array_dims(
                    array_dimensions=array_dims, dimension_name="n_timesteps"
                )
                self._state_info.n_timesteps = n_timesteps
                self.arrays[ArrayType.global_timesteps] = np.arange(
                    0, n_timesteps, dtype=settings.ftype
                )
            # no state data so we call it a day
            else:
                return n_timesteps_written

        # formatter for state files
        timesteps = self.arrays[ArrayType.global_timesteps]
        n_timesteps = len(timesteps)
        fmt_state_file_counter = "{0:02d}"

        # single file or multiple file handling
        state_fp: Union[None, typing.BinaryIO] = None
        file_to_close: Union[None, typing.BinaryIO] = None
        if isinstance(filepath, str):
            if settings.single_file:
                # pylint: disable = consider-using-with
                state_fp = file_to_close = open(filepath + fmt_state_file_counter.format(1), "ab")
            else:
                # create a new file per timestep
                # see time loop
                pass
        else:
            state_fp = filepath

        try:
            # time looping ... wheeeeeeeee
            for i_timestep, _ in enumerate(timesteps):

                # open new state file ... or not
                state_filepath_or_file = (
                    filepath + fmt_state_file_counter.format(i_timestep + 1)
                    if isinstance(filepath, str) and state_fp is None
                    else state_fp
                )

                n_bytes_written = 0

                with open_file_or_filepath(state_filepath_or_file, "ab") as fp:

                    # GLOBALS
                    n_bytes_written += self._write_states_globals(fp, i_timestep, settings)

                    # NODE DATA
                    n_bytes_written += self._write_states_nodes(fp, i_timestep, settings)

                    # SOLID THERMAL DATA
                    n_bytes_written += self._write_states_solid_thermal_data(
                        fp, i_timestep, settings
                    )

                    # CFDDATA
                    # not supported

                    # SOLIDS
                    n_bytes_written += self._write_states_solids(fp, i_timestep, settings)

                    # THICK SHELLS
                    n_bytes_written += self._write_states_tshells(fp, i_timestep, settings)

                    # spocky ... BEAM me up
                    n_bytes_written += self._write_states_beams(fp, i_timestep, settings)

                    # SHELLS
                    n_bytes_written += self._write_states_shells(fp, i_timestep, settings)

                    # DELETION INFO
                    n_bytes_written += self._write_states_deletion_info(fp, i_timestep, settings)

                    # SPH
                    n_bytes_written += self._write_states_sph(fp, i_timestep, settings)

                    # AIRBAG
                    n_bytes_written += self._write_states_airbags(fp, i_timestep, settings)

                    # RIGID ROAD
                    n_bytes_written += self._write_states_rigid_road(fp, i_timestep, settings)

                    # RIGID BODY
                    n_bytes_written += self._write_states_rigid_bodies(fp, i_timestep, settings)

                    # EXTRA DATA
                    # not supported

                    # end mark
                    # at the end for single file buffer
                    # or behind each state file
                    if not settings.single_file or i_timestep == n_timesteps - 1:
                        n_bytes_written += fp.write(settings.pack(-999999.0))

                        if settings.block_size_bytes > 0:
                            zero_bytes = self._get_zero_byte_padding(
                                n_bytes_written, settings.block_size_bytes
                            )
                            n_bytes_written += fp.write(zero_bytes)

                    # log
                    msg = "_write_states wrote %d bytes"
                    LOGGER.debug(msg, n_bytes_written)
                    n_timesteps_written += 1

        finally:
            # close file if required
            if file_to_close is not None:
                file_to_close.close()

        return n_timesteps_written

    def _write_states_globals(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        _check_ndim(
            self,
            {
                ArrayType.global_kinetic_energy: ["n_timesteps"],
                ArrayType.global_internal_energy: ["n_timesteps"],
                ArrayType.global_total_energy: ["n_timesteps"],
                ArrayType.global_velocity: ["n_timesteps", "vx_vy_vz"],
            },
        )
        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.global_kinetic_energy: 0,
            ArrayType.global_internal_energy: 0,
            ArrayType.global_total_energy: 0,
            ArrayType.global_velocity: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")

        byte_checksum = 0

        n_global_vars = settings.header["nglbv"]

        # TIME
        timesteps = self.arrays[ArrayType.global_timesteps]
        byte_checksum += fp.write(settings.pack(timesteps[i_timestep]))

        # GLOBAL KINETIC ENERGY
        if n_global_vars >= 1:
            array_type = ArrayType.global_kinetic_energy
            value = (
                self.arrays[array_type][i_timestep]
                if array_type in self.arrays
                else self.header.ftype(0.0)
            )
            byte_checksum += fp.write(settings.pack(value, dtype_hint=np.floating))

        # GLOBAL INTERNAL ENERGY
        if n_global_vars >= 2:
            array_type = ArrayType.global_internal_energy
            value = (
                self.arrays[array_type][i_timestep]
                if array_type in self.arrays
                else self.header.ftype(0.0)
            )
            byte_checksum += fp.write(settings.pack(value, dtype_hint=np.floating))

        # GLOBAL TOTAL ENERGY
        if n_global_vars >= 3:
            array_type = ArrayType.global_total_energy
            value = (
                self.arrays[array_type][i_timestep]
                if array_type in self.arrays
                else self.header.ftype(0.0)
            )
            byte_checksum += fp.write(settings.pack(value, dtype_hint=np.floating))

        # GLOBAL VELOCITY
        if n_global_vars >= 6:
            self.check_array_dims({ArrayType.global_velocity: 1}, "vx_vy_vz", 3)
            array_type = ArrayType.global_velocity
            array = (
                self.arrays[array_type][i_timestep]
                if array_type in self.arrays
                else np.zeros(3, self.header.ftype)
            )
            byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

        # PARTS
        #
        # Parts always need special love since dyna
        # writes many dummy parts
        _check_ndim(
            self,
            {
                ArrayType.part_internal_energy: ["n_timesteps", "n_parts"],
                ArrayType.part_kinetic_energy: ["n_timesteps", "n_parts"],
                ArrayType.part_velocity: ["n_timesteps", "n_parts", "vx_vy_vz"],
                ArrayType.part_mass: ["n_timesteps", "n_parts"],
                ArrayType.part_hourglass_energy: ["n_timesteps", "n_parts"],
            },
        )
        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.part_internal_energy: 0,
            ArrayType.part_kinetic_energy: 0,
            ArrayType.part_velocity: 0,
            ArrayType.part_mass: 0,
            ArrayType.part_hourglass_energy: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")

        self.check_array_dims({ArrayType.part_velocity: 2}, "vx_vy_vz", 3)

        n_parts = settings.header["nmmat"]

        def _write_part_field(array_type: str, default_shape: Union[int, Tuple], dtype: np.dtype):
            array = (
                self.arrays[array_type][i_timestep]
                if array_type in self.arrays
                else np.zeros(default_shape, self.header.ftype)
            )

            if len(array):
                dummy_array = array
                return fp.write(settings.pack(dummy_array, dtype_hint=np.floating))

            return 0

        # PART INTERNAL ENERGY
        if n_global_vars >= 6 + n_parts:
            byte_checksum += _write_part_field(
                ArrayType.part_internal_energy, n_parts, settings.ftype
            )

        # PART KINETIC ENERGY
        if n_global_vars >= 6 + 2 * n_parts:
            byte_checksum += _write_part_field(
                ArrayType.part_kinetic_energy, n_parts, settings.ftype
            )

        # PART VELOCITY
        if n_global_vars >= 6 + 5 * n_parts:
            byte_checksum += _write_part_field(
                ArrayType.part_velocity, (n_parts, 3), settings.ftype
            )

        # PART MASS
        if n_global_vars >= 6 + 6 * n_parts:
            byte_checksum += _write_part_field(ArrayType.part_mass, n_parts, settings.ftype)

        # PART HOURGLASS ENERGY
        if n_global_vars >= 6 + 7 * n_parts:
            byte_checksum += _write_part_field(
                ArrayType.part_hourglass_energy, n_parts, settings.ftype
            )

        # RIGID WALL
        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.rigid_wall_force: 0,
            ArrayType.rigid_wall_position: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")
        array_dims = {
            ArrayType.rigid_wall_force: 1,
            ArrayType.rigid_wall_position: 1,
        }
        self.check_array_dims(array_dims, "n_rigid_walls")
        self.check_array_dims({ArrayType.rigid_wall_position: 2}, "x_y_z", 3)

        n_rigid_wall_vars = settings.header["n_rigid_wall_vars"]
        n_rigid_walls = settings.header["n_rigid_walls"]
        if n_global_vars >= 6 + 7 * n_parts + n_rigid_wall_vars * n_rigid_walls:
            if n_rigid_wall_vars >= 1:
                array = self.arrays[ArrayType.rigid_wall_force][i_timestep]
                byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))
            if n_rigid_wall_vars >= 4:
                array = self.arrays[ArrayType.rigid_wall_position][i_timestep]
                byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

        # check byte checksum
        # pylint: disable = invalid-name
        TIME_WORDSIZE = 1
        byte_checksum_target = (TIME_WORDSIZE + settings.header["nglbv"]) * settings.wordsize
        if byte_checksum != byte_checksum_target:
            msg = (
                "byte checksum wrong: "
                f"{byte_checksum_target} (header) != {byte_checksum} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_globals", byte_checksum)

        return byte_checksum

    def _write_states_nodes(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        n_nodes = settings.header["numnp"]
        if n_nodes <= 0:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.node_displacement: ["n_timesteps", "n_nodes", "x_y_z"],
                ArrayType.node_velocity: ["n_timesteps", "n_nodes", "vx_vy_vz"],
                ArrayType.node_acceleration: ["n_timesteps", "n_nodes", "ax_ay_az"],
                ArrayType.node_heat_flux: ["n_timesteps", "n_nodes", "hx_hy_hz"],
                # INFO: cannot check since it may have 1 or 3 values per node
                # ArrayType.node_temperature: ["n_timesteps","n_nodes"],
                ArrayType.node_mass_scaling: ["n_timesteps", "n_nodes"],
                ArrayType.node_temperature_gradient: ["n_timesteps", "n_nodes"],
                ArrayType.node_residual_forces: ["n_timesteps", "n_nodes", "fx_fy_fz"],
                ArrayType.node_residual_moments: ["n_timesteps", "n_nodes", "mx_my_mz"],
            },
        )
        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.node_displacement: 0,
            ArrayType.node_velocity: 0,
            ArrayType.node_acceleration: 0,
            ArrayType.node_heat_flux: 0,
            ArrayType.node_temperature: 0,
            ArrayType.node_mass_scaling: 0,
            ArrayType.node_temperature_gradient: 0,
            ArrayType.node_residual_forces: 0,
            ArrayType.node_residual_moments: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")
        array_dims = {
            ArrayType.node_coordinates: 0,
            ArrayType.node_displacement: 1,
            ArrayType.node_velocity: 1,
            ArrayType.node_acceleration: 1,
            ArrayType.node_heat_flux: 1,
            ArrayType.node_temperature: 1,
            ArrayType.node_mass_scaling: 1,
            ArrayType.node_temperature_gradient: 1,
            ArrayType.node_residual_forces: 1,
            ArrayType.node_residual_moments: 1,
        }
        self.check_array_dims(array_dims, "n_nodes")
        self.check_array_dims({ArrayType.node_heat_flux: 2}, "x_y_z", 3)
        self.check_array_dims({ArrayType.node_displacement: 2}, "dx_dy_dz", 3)
        self.check_array_dims({ArrayType.node_velocity: 2}, "vx_vy_vz", 3)
        self.check_array_dims({ArrayType.node_acceleration: 2}, "ax_ay_az", 3)
        self.check_array_dims({ArrayType.node_residual_forces: 2}, "fx_fy_fz", 3)
        self.check_array_dims({ArrayType.node_residual_moments: 2}, "mx_my_mz", 3)

        byte_checksum = 0

        it = settings.header["it"]
        has_mass_scaling = False
        if it >= 10:
            it -= 10
            has_mass_scaling = True

        n_nodes = settings.header["numnp"]

        # NODE DISPLACEMENT
        if settings.header["iu"]:
            array = self.arrays[ArrayType.node_displacement][i_timestep]
            byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

        if it != 0:

            # NODE TEMPERATURES
            array_type = ArrayType.node_temperature
            array = (
                self.arrays[array_type][i_timestep]
                if array_type in self.arrays
                else np.zeros(n_nodes, dtype=settings.ftype)
            )

            # just 1 temperature per node
            if it < 3:
                byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))
            # 3 temperatures per node
            else:
                self.check_array_dims({ArrayType.node_temperature: 2}, "node_layer", 3)
                byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

            # NODE HEAT FLUX
            if it >= 2:
                array = self.arrays[ArrayType.node_heat_flux][i_timestep]
                byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

        # NODE MASS SCALING
        if has_mass_scaling:
            array = self.arrays[ArrayType.node_mass_scaling][i_timestep]
            byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

        # NODE TEMPERATURE GRADIENT
        if settings.has_node_temperature_gradient:
            array = self.arrays[ArrayType.node_temperature_gradient][i_timestep]
            byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

        # NODE RESIDUAL FORCES
        if settings.has_node_residual_forces:
            array = (
                self.arrays[ArrayType.node_residual_forces][i_timestep]
                if ArrayType.node_residual_forces in self.arrays
                else np.zeros((n_nodes, 3), dtype=settings.ftype)
            )
            byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

        # NODE RESIDUAL MOMENTS
        if settings.has_node_residual_moments:
            array = (
                self.arrays[ArrayType.node_residual_moments][i_timestep]
                if ArrayType.node_residual_forces in self.arrays
                else np.zeros((n_nodes, 3), dtype=settings.ftype)
            )
            byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

        # NODE VELOCITY
        if settings.header["iv"]:
            array = self.arrays[ArrayType.node_velocity][i_timestep]
            byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

        # NODE ACCELERATION
        if settings.header["ia"]:
            array = self.arrays[ArrayType.node_acceleration][i_timestep]
            byte_checksum += fp.write(settings.pack(array, dtype_hint=np.floating))

        # check the checksum
        n_thermal_vars = 0
        if settings.header["it"] % 10 == 1:
            n_thermal_vars = 1
        elif settings.header["it"] % 10 == 2:
            n_thermal_vars = 4
        elif settings.header["it"] % 10 == 3:
            n_thermal_vars = 6

        if settings.header["it"] // 10 == 1:
            n_thermal_vars += 1

        n_temp_gradient_vars = settings.has_node_temperature_gradient
        n_residual_forces_vars = settings.has_node_residual_forces * 3
        n_residual_moments_vars = settings.has_node_residual_moments * 3

        # pylint: disable = invalid-name
        NDIM = 3
        byte_checksum_target = (
            (
                (settings.header["iu"] + settings.header["iv"] + settings.header["ia"]) * NDIM
                + n_thermal_vars
                + n_temp_gradient_vars
                + n_residual_forces_vars
                + n_residual_moments_vars
            )
            * settings.wordsize
            * settings.header["numnp"]
        )
        if byte_checksum != byte_checksum_target:
            msg = (
                "byte checksum wrong: "
                "{byte_checksum_target} (header) != {byte_checksum} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_nodes", byte_checksum)

        return byte_checksum

    def _write_states_solid_thermal_data(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        if settings.header["nt3d"] <= 0:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.element_solid_thermal_data: [
                    "n_timesteps",
                    "n_solids",
                    "n_solids_thermal_vars",
                ]
            },
        )

        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.element_solid_thermal_data: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")

        array_dims = {
            ArrayType.element_solid_node_indexes: 0,
            ArrayType.element_solid_thermal_data: 1,
        }
        self.check_array_dims(array_dims, "n_solids")

        array = self.arrays[ArrayType.element_solid_thermal_data][i_timestep]
        n_bytes_written = fp.write(settings.pack(array, dtype_hint=np.floating))

        # check bytes
        n_bytes_expected = (
            settings.header["nt3d"] * abs(settings.header["nel8"]) * settings.wordsize
        )
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_thermal_data", n_bytes_written)

        return n_bytes_written

    def _write_states_solids(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        n_solids = abs(settings.header["nel8"])
        n_solid_vars = settings.header["nv3d"]
        n_solid_layers = settings.n_solid_layers

        if n_solids == 0 or n_solid_vars <= 0:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.element_solid_stress: [
                    "n_timesteps",
                    "n_solids",
                    "n_solid_layers",
                    "σx_σy_σz_σxy_σyz_σxz",
                ],
                ArrayType.element_solid_effective_plastic_strain: [
                    "n_timesteps",
                    "n_solid_layers",
                    "n_solids",
                ],
                ArrayType.element_solid_history_variables: [
                    "n_timesteps",
                    "n_solids",
                    "n_solid_layers",
                    "n_solid_history_vars",
                ],
                ArrayType.element_solid_strain: [
                    "n_timesteps",
                    "n_solids",
                    "n_solid_layers",
                    "εx_εy_εz_εxy_εyz_εxz",
                ],
                ArrayType.element_solid_plastic_strain_tensor: [
                    "n_timesteps",
                    "n_solids",
                    "n_solid_layers",
                    "εx_εy_εz_εxy_εyz_εxz",
                ],
                ArrayType.element_solid_thermal_strain_tensor: [
                    "n_timesteps",
                    "n_solids",
                    "n_solid_layers",
                    "εx_εy_εz_εxy_εyz_εxz",
                ],
            },
        )

        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.element_solid_stress: 0,
            ArrayType.element_solid_effective_plastic_strain: 0,
            ArrayType.element_solid_history_variables: 0,
            ArrayType.element_solid_strain: 0,
            ArrayType.element_solid_plastic_strain_tensor: 0,
            ArrayType.element_solid_thermal_strain_tensor: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")

        array_dims = {
            ArrayType.element_solid_node_indexes: 0,
            ArrayType.element_solid_stress: 1,
            ArrayType.element_solid_effective_plastic_strain: 1,
            ArrayType.element_solid_history_variables: 1,
            ArrayType.element_solid_strain: 1,
            ArrayType.element_solid_plastic_strain_tensor: 1,
            ArrayType.element_solid_thermal_strain_tensor: 1,
        }
        self.check_array_dims(array_dims, "n_solids")

        array_dims = {
            ArrayType.element_solid_stress: 2,
            ArrayType.element_solid_effective_plastic_strain: 2,
            ArrayType.element_solid_history_variables: 2,
            ArrayType.element_solid_strain: 2,
            ArrayType.element_solid_plastic_strain_tensor: 2,
            ArrayType.element_solid_thermal_strain_tensor: 2,
        }
        self.check_array_dims(array_dims, "n_solid_layers")

        self.check_array_dims({ArrayType.element_solid_stress: 3}, "σx_σy_σz_σxy_σyz_σxz", 6)

        self.check_array_dims({ArrayType.element_solid_strain: 3}, "εx_εy_εz_εxy_εyz_εxz", 6)

        self.check_array_dims(
            {ArrayType.element_solid_plastic_strain_tensor: 3}, "εx_εy_εz_εxy_εyz_εxz", 6
        )

        self.check_array_dims(
            {ArrayType.element_solid_thermal_strain_tensor: 3}, "εx_εy_εz_εxy_εyz_εxz", 6
        )

        # allocate array
        solid_data = np.zeros(
            (n_solids, n_solid_layers, n_solid_vars // n_solid_layers), dtype=settings.ftype
        )

        # SOLID STRESS
        if ArrayType.element_solid_stress in self.arrays:
            try:
                array = self.arrays[ArrayType.element_solid_stress][i_timestep]
                solid_data[:, :, 0:6] = array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_write_states_solids, element_solid_stress", trb_msg)

        # SOLID EFFECTIVE PSTRAIN
        if ArrayType.element_solid_effective_plastic_strain in self.arrays:
            try:
                array = self.arrays[ArrayType.element_solid_effective_plastic_strain][i_timestep]
                solid_data[:, :, 6] = array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(
                    msg, "_write_states_solids, element_solid_effective_plastic_strain", trb_msg
                )

        # SOLID HISTORY VARIABLES
        # (strains, pstrain tensor and thermal tensor are excluded here)
        has_strain = settings.header["istrn"]
        n_solid_history_variables = (
            settings.header["neiph"]
            - 6 * has_strain
            - 6 * settings.has_plastic_strain_tensor
            - 6 * settings.has_thermal_strain_tensor
        )

        if n_solid_history_variables:
            try:
                array = self.arrays[ArrayType.element_solid_history_variables][i_timestep]
                solid_data[:, :, 7 : 7 + n_solid_history_variables] = array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(
                    msg, "_write_states_solids, element_solid_history_variables", trb_msg
                )

        # SOLID STRAIN
        if has_strain and ArrayType.element_solid_strain in self.arrays:
            try:
                array = self.arrays[ArrayType.element_solid_strain][i_timestep]
                offset = 7 + n_solid_history_variables
                solid_data[:, :, offset : offset + 6] = array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(msg, "_write_states_solids, element_solid_strain", trb_msg)

        # PLASTIC STRAIN TENSOR
        if (
            settings.has_plastic_strain_tensor
            and ArrayType.element_solid_plastic_strain_tensor in self.arrays
        ):
            try:
                array = self.arrays[ArrayType.element_solid_plastic_strain_tensor][i_timestep]
                offset = 7 + n_solid_history_variables + 6 * has_strain
                solid_data[:, :, offset : offset + 6] = array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(
                    msg, "_write_states_solids, element_solid_plastic_strain_tensor", trb_msg
                )

        # THERMAL STRAIN TENSOR
        if (
            settings.has_thermal_strain_tensor
            and ArrayType.element_solid_thermal_strain_tensor in self.arrays
        ):
            try:
                array = self.arrays[ArrayType.element_solid_thermal_strain_tensor][i_timestep]
                offset = (
                    7
                    + n_solid_history_variables
                    + 6 * has_strain
                    + 6 * settings.has_plastic_strain_tensor
                )
                solid_data[:, :, offset : offset + 6] = array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in %s was caught:\n%s"
                LOGGER.warning(
                    msg, "_write_states_solids, element_solid_thermal_strain_tensor", trb_msg
                )

        n_bytes_written = fp.write(settings.pack(solid_data, dtype_hint=np.floating))

        # check bytes
        n_bytes_expected = (
            settings.header["nv3d"] * abs(settings.header["nel8"]) * settings.wordsize
        )
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_solids", n_bytes_written)

        return n_bytes_written

    def _write_states_tshells(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        n_tshells = settings.header["nelth"]
        n_tshell_vars = settings.header["nv3dt"]
        if n_tshells <= 0 or n_tshell_vars <= 0:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.element_tshell_stress: [
                    "n_timesteps",
                    "n_tshells",
                    "n_shell_layers",
                    "σx_σy_σz_σxy_σyz_σxz",
                ],
                ArrayType.element_tshell_strain: [
                    "n_timesteps",
                    "n_tshells",
                    "upper_lower",
                    "εx_εy_εz_εxy_εyz_εxz",
                ],
                ArrayType.element_tshell_effective_plastic_strain: [
                    "n_timesteps",
                    "n_tshells",
                    "n_shell_layers",
                ],
                ArrayType.element_tshell_history_variables: [
                    "n_timesteps",
                    "n_tshells",
                    "n_shell_layers",
                    "n_tshell_history_vars",
                ],
            },
        )

        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.element_tshell_stress: 0,
            ArrayType.element_tshell_strain: 0,
            ArrayType.element_tshell_effective_plastic_strain: 0,
            ArrayType.element_tshell_history_variables: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")

        array_dims = {
            ArrayType.element_tshell_node_indexes: 0,
            ArrayType.element_tshell_stress: 1,
            ArrayType.element_tshell_strain: 1,
            ArrayType.element_tshell_effective_plastic_strain: 1,
            ArrayType.element_tshell_history_variables: 1,
        }
        self.check_array_dims(array_dims, "n_tshells")

        self.check_array_dims({ArrayType.element_tshell_stress: 3}, "σx_σy_σz_σxy_σyz_σxz", 6)

        self.check_array_dims({ArrayType.element_tshell_strain: 2}, "upper_lower", 2)

        self.check_array_dims({ArrayType.element_tshell_strain: 3}, "εx_εy_εz_εxy_εyz_εxz", 6)

        has_stress = settings.header["ioshl1"] == 1000
        has_pstrain = settings.header["ioshl2"] == 1000
        n_history_vars = settings.header["neips"]
        n_layer_vars = settings.n_shell_layers * (6 * has_stress + has_pstrain + n_history_vars)

        tshell_data = np.zeros((n_tshells, n_tshell_vars), settings.ftype)
        tshell_layer_data = tshell_data[:, :n_layer_vars].reshape(
            (n_tshells, settings.n_shell_layers, -1)
        )
        tshell_nonlayer_data = tshell_data[:, n_layer_vars:]

        # TSHELL STRESS
        if has_stress:
            if ArrayType.element_tshell_stress in self.arrays:
                array = self.arrays[ArrayType.element_tshell_stress][i_timestep]
                tshell_layer_data[:, :, 0:6] = array

        # TSHELL EFF. PLASTIC STRAIN
        if has_pstrain:
            if ArrayType.element_tshell_effective_plastic_strain in self.arrays:
                array = self.arrays[ArrayType.element_tshell_effective_plastic_strain][i_timestep]
                start_index = 6 * has_stress
                tshell_layer_data[:, :, start_index] = array

        # TSHELL HISTORY VARS
        if n_history_vars != 0:
            if ArrayType.element_tshell_history_variables in self.arrays:
                array = self.arrays[ArrayType.element_tshell_history_variables][i_timestep]
                start_index = 6 * has_stress + has_pstrain
                end_index = start_index + array.shape[2]
                tshell_layer_data[:, :, start_index:end_index] = array

        # TSHELL STRAIN
        if settings.header["istrn"]:
            if ArrayType.element_tshell_strain in self.arrays:
                array = self.arrays[ArrayType.element_tshell_strain][i_timestep]
                start_index = 6 * has_stress + has_pstrain + n_history_vars
                tshell_nonlayer_data[:, :] = array.reshape(n_tshells, 12)

        n_bytes_written = fp.write(settings.pack(tshell_data, dtype_hint=np.floating))

        # check bytes
        n_bytes_expected = (
            settings.header["nv3dt"] * abs(settings.header["nelth"]) * settings.wordsize
        )
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_tshells", n_bytes_written)

        return n_bytes_written

    def _write_states_beams(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        n_beams = settings.header["nel2"]
        n_beam_vars = settings.header["nv1d"]
        if n_beams <= 0 or n_beam_vars <= 0:
            return 0

        n_beam_layers = settings.header["beamip"]
        n_beam_history_vars = settings.header["neipb"]

        _check_ndim(
            self,
            {
                ArrayType.element_beam_axial_force: ["n_timesteps", "n_beams"],
                ArrayType.element_beam_shear_force: ["n_timesteps", "n_beams", "fs_ft"],
                ArrayType.element_beam_bending_moment: ["n_timesteps", "n_beams", "ms_mt"],
                ArrayType.element_beam_torsion_moment: ["n_timesteps", "n_beams"],
                ArrayType.element_beam_shear_stress: [
                    "n_timesteps",
                    "n_beams",
                    "n_beam_layers",
                    "σrs_σtr",
                ],
                ArrayType.element_beam_axial_stress: ["n_timesteps", "n_beams", "n_beam_layers"],
                ArrayType.element_beam_plastic_strain: ["n_timesteps", "n_beams", "n_beam_layers"],
                ArrayType.element_beam_axial_strain: ["n_timesteps", "n_beams", "n_beam_layers"],
                ArrayType.element_beam_history_vars: [
                    "n_timesteps",
                    "n_beams",
                    "n_beam_layers+3",
                    "n_beam_history_vars",
                ],
            },
        )

        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.element_beam_axial_force: 0,
            ArrayType.element_beam_shear_force: 0,
            ArrayType.element_beam_bending_moment: 0,
            ArrayType.element_beam_torsion_moment: 0,
            ArrayType.element_beam_shear_stress: 0,
            ArrayType.element_beam_axial_stress: 0,
            ArrayType.element_beam_plastic_strain: 0,
            ArrayType.element_beam_axial_strain: 0,
            ArrayType.element_beam_history_vars: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")
        array_dims = {
            ArrayType.element_beam_axial_force: 1,
            ArrayType.element_beam_shear_force: 1,
            ArrayType.element_beam_bending_moment: 1,
            ArrayType.element_beam_torsion_moment: 1,
            ArrayType.element_beam_shear_stress: 1,
            ArrayType.element_beam_axial_stress: 1,
            ArrayType.element_beam_plastic_strain: 1,
            ArrayType.element_beam_axial_strain: 1,
            ArrayType.element_beam_history_vars: 1,
        }
        self.check_array_dims(array_dims, "n_beams")
        self.check_array_dims({ArrayType.element_beam_shear_force: 2}, "fs_ft", 2)
        self.check_array_dims({ArrayType.element_beam_bending_moment: 2}, "ms_mt", 2)
        array_dims = {
            ArrayType.element_beam_shear_stress: 2,
            ArrayType.element_beam_axial_stress: 2,
            ArrayType.element_beam_plastic_strain: 2,
            ArrayType.element_beam_axial_strain: 2,
            ArrayType.element_beam_history_vars: 2,
        }
        n_beam_layers = self.check_array_dims(array_dims, "n_beam_layers")
        self.check_array_dims({ArrayType.element_beam_shear_stress: 3}, "σrs_σtr", 2)
        self.check_array_dims(
            {ArrayType.element_beam_history_vars: 2}, "n_modes", n_beam_layers + 3
        )

        # allocate buffer
        beam_data = np.zeros((n_beams, n_beam_vars), dtype=settings.ftype)
        n_layer_vars_total = 5 * n_beam_layers
        beam_layer_data = beam_data[:, 6 : 6 + n_layer_vars_total].reshape(
            (n_beams, n_beam_layers, 5)
        )
        beam_history_vars = beam_data[:, 6 + n_layer_vars_total :].reshape(
            (n_beams, 3 + n_beam_layers, n_beam_history_vars)
        )

        # BEAM AXIAL FORCE
        if ArrayType.element_beam_axial_force in self.arrays:
            array = self.arrays[ArrayType.element_beam_axial_force][i_timestep]
            beam_data[:, 0] = array

        # BEAM SHEAR FORCE
        if ArrayType.element_beam_shear_force in self.arrays:
            array = self.arrays[ArrayType.element_beam_shear_force][i_timestep]
            beam_data[:, 1:3] = array

        # BEAM BENDING MOMENTUM
        if ArrayType.element_beam_bending_moment in self.arrays:
            array = self.arrays[ArrayType.element_beam_bending_moment][i_timestep]
            beam_data[:, 3:5] = array

        # BEAM TORSION MOMENTUM
        if ArrayType.element_beam_torsion_moment in self.arrays:
            array = self.arrays[ArrayType.element_beam_torsion_moment][i_timestep]
            beam_data[:, 5] = array

        if n_beam_layers:
            array = (
                self.arrays[ArrayType.element_beam_axial_stress][i_timestep]
                if ArrayType.element_beam_axial_stress in self.arrays
                else np.zeros((n_beams, n_beam_layers), dtype=settings.ftype)
            )
            beam_layer_data[:, :, 0] = array

            array = (
                self.arrays[ArrayType.element_beam_shear_stress][i_timestep]
                if ArrayType.element_beam_shear_stress in self.arrays
                else np.zeros((n_beams, n_beam_layers, 2), dtype=settings.ftype)
            )
            beam_layer_data[:, :, 1:3] = array

            array = (
                self.arrays[ArrayType.element_beam_plastic_strain][i_timestep]
                if ArrayType.element_beam_plastic_strain in self.arrays
                else np.zeros((n_beams, n_beam_layers), dtype=settings.ftype)
            )
            beam_layer_data[:, :, 3] = array

            array = (
                self.arrays[ArrayType.element_beam_axial_strain][i_timestep]
                if ArrayType.element_beam_axial_strain in self.arrays
                else np.zeros((n_beams, n_beam_layers), dtype=settings.ftype)
            )
            beam_layer_data[:, :, 4] = array

        # BEAM HISTORY VARIABLES
        if n_beam_history_vars:
            array = (
                self.arrays[ArrayType.element_beam_history_vars][i_timestep]
                if ArrayType.element_beam_history_vars in self.arrays
                else np.zeros(
                    (n_beams, n_beam_layers + 3, n_beam_history_vars), dtype=settings.ftype
                )
            )
            beam_history_vars[:, :, :] = array

        n_bytes_written = fp.write(settings.pack(beam_data, dtype_hint=np.floating))

        # check bytes
        n_bytes_expected = settings.header["nv1d"] * settings.header["nel2"] * settings.wordsize
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_tshells", n_bytes_written)

        return n_bytes_written

    def _write_states_shells(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        n_shells = settings.header["nel4"]
        n_shell_vars = settings.header["nv2d"]
        n_rigid_shells = settings.n_rigid_shells
        is_d3part = self.header.filetype == D3plotFiletype.D3PART
        # d3part writes results also for rigid shells
        n_reduced_shells = n_shells if is_d3part else n_shells - n_rigid_shells

        if n_reduced_shells <= 0 or n_shell_vars <= 0:
            return 0

        has_stress = settings.header["ioshl1"] == 1000
        has_pstrain = settings.header["ioshl2"] == 1000
        has_forces = settings.header["ioshl3"] == 1000
        has_else = settings.header["ioshl4"] == 1000
        has_strain = settings.header["istrn"] != 0
        n_shell_history_vars = settings.header["neips"]

        _check_ndim(
            self,
            {
                ArrayType.element_shell_stress: [
                    "n_timesteps",
                    "n_shells",
                    "n_shell_layers",
                    "σx_σy_σz_σxy_σyz_σxz",
                ],
                ArrayType.element_shell_effective_plastic_strain: [
                    "n_timesteps",
                    "n_shells",
                    "n_shell_layers",
                ],
                ArrayType.element_shell_history_vars: [
                    "n_timesteps",
                    "n_shells",
                    "n_shell_layers",
                    "n_shell_history_vars",
                ],
                ArrayType.element_shell_bending_moment: ["n_timesteps", "n_shells", "mx_my_mxy"],
                ArrayType.element_shell_shear_force: ["n_timesteps", "n_shells", "qx_qy"],
                ArrayType.element_shell_normal_force: ["n_timesteps", "n_shells", "nx_ny_nxy"],
                ArrayType.element_shell_thickness: ["n_timesteps", "n_shells"],
                ArrayType.element_shell_unknown_variables: [
                    "n_timesteps",
                    "n_shells",
                    "n_extra_vars",
                ],
                ArrayType.element_shell_internal_energy: ["n_timesteps", "n_shells"],
                ArrayType.element_shell_strain: [
                    "n_timesteps",
                    "n_shells",
                    "upper_lower",
                    "εx_εy_εz_εxy_εyz_εxz",
                ],
            },
        )

        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.element_shell_stress: 0,
            ArrayType.element_shell_effective_plastic_strain: 0,
            ArrayType.element_shell_history_vars: 0,
            ArrayType.element_shell_bending_moment: 0,
            ArrayType.element_shell_shear_force: 0,
            ArrayType.element_shell_normal_force: 0,
            ArrayType.element_shell_thickness: 0,
            ArrayType.element_shell_unknown_variables: 0,
            ArrayType.element_shell_internal_energy: 0,
            ArrayType.element_shell_strain: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")

        array_dims = {
            ArrayType.element_shell_stress: 1,
            ArrayType.element_shell_effective_plastic_strain: 1,
            ArrayType.element_shell_history_vars: 1,
            ArrayType.element_shell_bending_moment: 1,
            ArrayType.element_shell_shear_force: 1,
            ArrayType.element_shell_normal_force: 1,
            ArrayType.element_shell_thickness: 1,
            ArrayType.element_shell_unknown_variables: 1,
            ArrayType.element_shell_internal_energy: 1,
            ArrayType.element_shell_strain: 1,
        }
        n_reduced_shells = self.check_array_dims(array_dims, "n_shells")
        if not is_d3part and n_reduced_shells != n_shells - n_rigid_shells:
            msg = (
                "Parts with mattyp 20 (rigid material) were specified."
                " For these parts no state data is output in dyna."
                " The state arrays are thus expected output data for only"
                f" {n_shells - n_rigid_shells} shells and not {n_reduced_shells}."
            )
            raise ValueError(msg)

        array_dims = {
            ArrayType.element_shell_stress: 2,
            ArrayType.element_shell_effective_plastic_strain: 2,
            ArrayType.element_shell_history_vars: 2,
        }
        n_shell_layers = self.check_array_dims(array_dims, "n_shell_layers")

        self.check_array_dims({ArrayType.element_shell_stress: 3}, "σx_σy_σz_σxy_σyz_σxz", 6)
        self.check_array_dims({ArrayType.element_shell_bending_moment: 2}, "mx_my_mxy", 3)
        self.check_array_dims({ArrayType.element_shell_shear_force: 2}, "qx_qy")
        self.check_array_dims({ArrayType.element_shell_strain: 2}, "upper_lower", 2)
        self.check_array_dims({ArrayType.element_shell_strain: 3}, "εx_εy_εz_εxy_εyz_εxz", 6)

        # allocate buffer
        shell_data = np.zeros((n_reduced_shells, n_shell_vars), dtype=settings.ftype)
        n_layer_vars = has_stress * 6 + has_pstrain + n_shell_history_vars
        n_layer_vars_total = n_layer_vars * n_shell_layers

        shell_layer_data = shell_data[:, :n_layer_vars_total].reshape(
            (n_reduced_shells, n_shell_layers, n_layer_vars)
        )
        shell_nonlayer_data = shell_data[:, n_layer_vars_total:]

        start_layer_index = 0
        end_layer_index = 0

        # SHELL STRESS
        if has_stress:
            start_layer_index = 0
            end_layer_index = 6
            if ArrayType.element_shell_stress in self.arrays:
                array = self.arrays[ArrayType.element_shell_stress][i_timestep]
                shell_layer_data[:, :, start_layer_index:end_layer_index] = array

        # EFF PSTRAIN
        if has_pstrain:
            start_layer_index = end_layer_index
            end_layer_index = start_layer_index + has_pstrain
            if ArrayType.element_shell_effective_plastic_strain in self.arrays:
                array = self.arrays[ArrayType.element_shell_effective_plastic_strain][i_timestep]
                shell_layer_data[:, :, start_layer_index:end_layer_index] = array.reshape(
                    (n_reduced_shells, n_shell_layers, 1)
                )

        # SHELL HISTORY VARS
        if n_shell_history_vars:
            start_layer_index = end_layer_index
            end_layer_index = start_layer_index + n_shell_history_vars
            if ArrayType.element_shell_history_vars in self.arrays:
                array = self.arrays[ArrayType.element_shell_history_vars][i_timestep]
                n_hist_vars_arr = array.shape[2]
                end_layer_index2 = start_layer_index + min(n_hist_vars_arr, n_shell_history_vars)
                shell_layer_data[:, :, start_layer_index:end_layer_index2] = array

        start_index = 0
        end_index = 0

        # SHELL FORCES
        if has_forces:
            start_index = end_index
            end_index = start_index + 8

            # MOMENTUM
            if ArrayType.element_shell_bending_moment in self.arrays:
                start_index2 = start_index
                end_index2 = start_index + 3
                array = self.arrays[ArrayType.element_shell_bending_moment][i_timestep]
                shell_nonlayer_data[:, start_index2:end_index2] = array

            # SHEAR
            if ArrayType.element_shell_shear_force in self.arrays:
                start_index2 = start_index + 3
                end_index2 = start_index + 5
                array = self.arrays[ArrayType.element_shell_shear_force][i_timestep]
                shell_nonlayer_data[:, start_index2:end_index2] = array

            # NORMAL
            if ArrayType.element_shell_normal_force in self.arrays:
                start_index2 = start_index + 5
                end_index2 = start_index + 8
                array = self.arrays[ArrayType.element_shell_normal_force][i_timestep]
                shell_nonlayer_data[:, start_index2:end_index2] = array

        if has_else:
            start_index = end_index
            end_index = start_index + 3

            # THICKNESS
            if ArrayType.element_shell_thickness in self.arrays:
                start_index2 = start_index
                end_index2 = start_index + 1
                array = self.arrays[ArrayType.element_shell_thickness][i_timestep]
                shell_nonlayer_data[:, start_index2:end_index2] = array.reshape(
                    (n_reduced_shells, 1)
                )

            # ELEMENT SPECIFIC VARS
            if ArrayType.element_shell_unknown_variables in self.arrays:
                start_index2 = start_index + 1
                end_index2 = start_index + 3
                array = self.arrays[ArrayType.element_shell_unknown_variables][i_timestep]
                shell_nonlayer_data[:, start_index2:end_index2] = array

        # SHELL STRAIN
        #
        # Strain is squeezed between the 3rd and 4th var of the else block
        if has_strain:
            start_index = end_index
            end_index = start_index + 12

            if ArrayType.element_shell_strain in self.arrays:
                array = self.arrays[ArrayType.element_shell_strain][i_timestep]
                shell_nonlayer_data[:, start_index:end_index] = array.reshape(
                    (n_reduced_shells, 12)
                )

        # INTERNAL ENERGY
        if has_else:
            start_index = end_index
            end_index = start_index + 1

            if ArrayType.element_shell_internal_energy in self.arrays:
                array = self.arrays[ArrayType.element_shell_internal_energy][i_timestep]
                shell_nonlayer_data[:, start_index:end_index] = array.reshape((n_reduced_shells, 1))

        # THERMAL STRAIN TENSOR
        if settings.has_plastic_strain_tensor:
            start_index = end_index
            end_index = start_index + n_shell_layers * 6

            if ArrayType.element_shell_plastic_strain_tensor in self.arrays:
                array = self.arrays[ArrayType.element_shell_plastic_strain_tensor][i_timestep]
                shell_nonlayer_data[:, start_index:end_index] = array.reshape(
                    (n_reduced_shells, n_shell_layers * 6)
                )

        # PLASTIC THERMAL TENSOR
        if settings.has_thermal_strain_tensor:
            start_index = end_index
            end_index = start_index + 6

            if ArrayType.element_shell_thermal_strain_tensor in self.arrays:
                array = self.arrays[ArrayType.element_shell_thermal_strain_tensor][i_timestep]
                shell_nonlayer_data[:, start_index:end_index] = array.reshape((n_reduced_shells, 6))

        n_bytes_written = fp.write(settings.pack(shell_data, dtype_hint=np.floating))

        # check bytes
        # *(settings.header["nel4"]-settings.n_rigid_shells)\
        n_bytes_expected = settings.header["nv2d"] * n_reduced_shells * settings.wordsize
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_shells", n_bytes_written)

        return n_bytes_written

    def _write_states_deletion_info(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        if settings.mdlopt <= 0:
            return 0

        n_bytes_written = 0
        n_bytes_expected = 0

        # NODE DELETION
        if settings.mdlopt == 1:

            _check_ndim(self, {ArrayType.node_is_alive: ["n_timesteps", "n_nodes"]})

            array_dims = {
                ArrayType.global_timesteps: 0,
                ArrayType.node_is_alive: 0,
            }
            self.check_array_dims(array_dims, "n_timesteps")

            array_dims = {
                ArrayType.node_coordinates: 0,
                ArrayType.node_is_alive: 1,
            }
            self.check_array_dims(array_dims, "n_nodes")

            n_nodes = settings.header["numnp"]

            array = (
                self.arrays[ArrayType.node_is_alive]
                if ArrayType.node_is_alive in self.arrays
                else np.zeros(n_nodes, dtype=settings.ftype)
            )

            n_bytes_written += fp.write(settings.pack(array, dtype_hint=np.floating))

            # check
            n_bytes_expected = settings.header["numnp"] * settings.wordsize

        # ELEMENT DELETION
        elif settings.mdlopt == 2:

            _check_ndim(
                self,
                {
                    ArrayType.element_solid_is_alive: ["n_timesteps", "n_solids"],
                    ArrayType.element_shell_is_alive: ["n_timesteps", "n_shells"],
                    ArrayType.element_beam_is_alive: ["n_timesteps", "n_beams"],
                    ArrayType.element_tshell_is_alive: ["n_timesteps", "n_tshells"],
                },
            )

            array_dims = {
                ArrayType.global_timesteps: 0,
                ArrayType.element_solid_is_alive: 0,
                ArrayType.element_shell_is_alive: 0,
                ArrayType.element_beam_is_alive: 0,
                ArrayType.element_tshell_is_alive: 0,
            }
            self.check_array_dims(array_dims, "n_timesteps")

            array_dims = {
                ArrayType.element_solid_node_indexes: 0,
                ArrayType.element_solid_is_alive: 1,
            }
            self.check_array_dims(array_dims, "n_solids")

            array_dims = {
                ArrayType.element_beam_node_indexes: 0,
                ArrayType.element_beam_is_alive: 1,
            }
            self.check_array_dims(array_dims, "n_beams")

            array_dims = {
                ArrayType.element_shell_node_indexes: 0,
                ArrayType.element_shell_is_alive: 1,
            }
            self.check_array_dims(array_dims, "n_shells")

            array_dims = {
                ArrayType.element_tshell_node_indexes: 0,
                ArrayType.element_tshell_is_alive: 1,
            }
            self.check_array_dims(array_dims, "n_tshells")

            n_solids = settings.header["nel8"]
            n_tshells = settings.header["nelth"]
            n_shells = settings.header["nel4"]
            n_beams = settings.header["nel2"]

            # SOLID DELETION
            array = (
                self.arrays[ArrayType.element_solid_is_alive][i_timestep]
                if ArrayType.element_solid_is_alive in self.arrays
                else np.ones(n_solids, dtype=settings.ftype)
            )
            n_bytes_written += fp.write(settings.pack(array, dtype_hint=np.floating))

            # THICK SHELL DELETION
            array = (
                self.arrays[ArrayType.element_tshell_is_alive][i_timestep]
                if ArrayType.element_tshell_is_alive in self.arrays
                else np.ones(n_tshells, dtype=settings.ftype)
            )
            n_bytes_written += fp.write(settings.pack(array, dtype_hint=np.floating))

            # SHELL DELETION
            array = (
                self.arrays[ArrayType.element_shell_is_alive][i_timestep]
                if ArrayType.element_shell_is_alive in self.arrays
                else np.ones(n_shells, dtype=settings.ftype)
            )
            n_bytes_written += fp.write(settings.pack(array, dtype_hint=np.floating))

            # BEAM DELETION
            array = (
                self.arrays[ArrayType.element_beam_is_alive][i_timestep]
                if ArrayType.element_beam_is_alive in self.arrays
                else np.ones(n_beams, dtype=settings.ftype)
            )
            n_bytes_written += fp.write(settings.pack(array, dtype_hint=np.floating))

            # check
            n_bytes_expected = (
                settings.header["nel2"]
                + settings.header["nel4"]
                + abs(settings.header["nel8"])
                + settings.header["nelth"]
            ) * settings.wordsize

        else:
            msg = f"Invalid mdlopt flag during write process: {settings.mdlopt}"
            raise RuntimeError(msg)

        # check bytes
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_deletion_info", n_bytes_written)

        return n_bytes_written

    def _write_states_sph(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        if settings.header["nmsph"] <= 0:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.sph_deletion: ["n_timesteps", "n_particles"],
                ArrayType.sph_radius: ["n_timesteps", "n_particles"],
                ArrayType.sph_pressure: ["n_timesteps", "n_particles"],
                ArrayType.sph_stress: ["n_timesteps", "n_particles", "σx_σy_σz_σxy_σyz_σxz"],
                ArrayType.sph_effective_plastic_strain: ["n_timesteps", "n_particles"],
                ArrayType.sph_density: ["n_timesteps", "n_particles"],
                ArrayType.sph_internal_energy: ["n_timesteps", "n_particles"],
                ArrayType.sph_n_neighbors: ["n_timesteps", "n_particles"],
                ArrayType.sph_strain: ["n_timesteps", "n_particles", "εx_εy_εz_εxy_εyz_εxz"],
                ArrayType.sph_mass: ["n_timesteps", "n_particles"],
            },
        )

        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.sph_deletion: 0,
            ArrayType.sph_radius: 0,
            ArrayType.sph_pressure: 0,
            ArrayType.sph_stress: 0,
            ArrayType.sph_effective_plastic_strain: 0,
            ArrayType.sph_density: 0,
            ArrayType.sph_internal_energy: 0,
            ArrayType.sph_n_neighbors: 0,
            ArrayType.sph_strain: 0,
            ArrayType.sph_mass: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")

        array_dims = {
            ArrayType.sph_node_indexes: 0,
            ArrayType.sph_deletion: 1,
            ArrayType.sph_radius: 1,
            ArrayType.sph_pressure: 1,
            ArrayType.sph_stress: 1,
            ArrayType.sph_effective_plastic_strain: 1,
            ArrayType.sph_density: 1,
            ArrayType.sph_internal_energy: 1,
            ArrayType.sph_n_neighbors: 1,
            ArrayType.sph_strain: 1,
            ArrayType.sph_mass: 1,
        }
        n_particles = self.check_array_dims(array_dims, "n_particles")
        self.check_array_dims({ArrayType.sph_stress: 2}, "σx_σy_σz_σxy_σyz_σxz", 6)
        self.check_array_dims({ArrayType.sph_strain: 2}, "εx_εy_εz_εxy_εyz_εxz", 6)

        n_sph_variables = settings.header["numsph"]

        sph_data = np.zeros((n_particles, n_sph_variables))

        start_index = 0
        end_index = 0

        # SPH MATERIAL AND DELETION
        start_index = 0
        end_index = 1
        array = (
            self.arrays[ArrayType.sph_deletion][i_timestep]
            if ArrayType.sph_deletion in self.arrays
            else np.ones(n_particles)
        )
        sph_data[:, start_index:end_index] = array

        # INFLUENCE RADIUS
        if settings.header["isphfg2"]:
            start_index = end_index
            end_index = start_index + n_particles
            if ArrayType.sph_radius in self.arrays:
                array = self.arrays[ArrayType.sph_radius][i_timestep]
                sph_data[:, start_index:end_index] = array

        # PRESSURE
        if settings.header["isphfg3"]:
            start_index = end_index
            end_index = start_index + n_particles
            if ArrayType.sph_pressure in self.arrays:
                array = self.arrays[ArrayType.sph_pressure][i_timestep]
                sph_data[:, start_index:end_index] = array

        # STRESS
        if settings.header["isphfg4"]:
            start_index = end_index
            end_index = start_index + 6 * n_particles
            if ArrayType.sph_stress in self.arrays:
                array = self.arrays[ArrayType.sph_stress][i_timestep]
                sph_data[:, start_index:end_index] = array

        # PSTRAIN
        if settings.header["isphfg5"]:
            start_index = end_index
            end_index = start_index + n_particles
            if ArrayType.sph_effective_plastic_strain in self.arrays:
                array = self.arrays[ArrayType.sph_effective_plastic_strain][i_timestep]
                sph_data[:, start_index:end_index] = array

        # DENSITY
        if settings.header["isphfg6"]:
            start_index = end_index
            end_index = start_index + n_particles
            if ArrayType.sph_density in self.arrays:
                array = self.arrays[ArrayType.sph_density][i_timestep]
                sph_data[:, start_index:end_index] = array

        # INTERNAL ENERGY
        if settings.header["isphfg7"]:
            start_index = end_index
            end_index = start_index + n_particles
            if ArrayType.sph_internal_energy in self.arrays:
                array = self.arrays[ArrayType.sph_internal_energy][i_timestep]
                sph_data[:, start_index:end_index] = array

        # INTERNAL ENERGY
        if settings.header["isphfg8"]:
            start_index = end_index
            end_index = start_index + n_particles
            if ArrayType.sph_n_neighbors in self.arrays:
                array = self.arrays[ArrayType.sph_n_neighbors][i_timestep]
                sph_data[:, start_index:end_index] = array

        # STRAIN
        if settings.header["isphfg9"]:
            start_index = end_index
            end_index = start_index + n_particles * 6
            if ArrayType.sph_strain in self.arrays:
                array = self.arrays[ArrayType.sph_strain][i_timestep]
                sph_data[:, start_index:end_index] = array

        # MASS
        if settings.header["isphfg10"]:
            start_index = end_index
            end_index = start_index + n_particles
            if ArrayType.sph_mass in self.arrays:
                array = self.arrays[ArrayType.sph_mass][i_timestep]
                sph_data[:, start_index:end_index] = array

        n_bytes_written = fp.write(settings.pack(sph_data, dtype_hint=np.floating))

        # check bytes
        n_bytes_expected = (
            settings.header["nv2d"]
            * (settings.header["nel4"] - settings.header["numrbe"])
            * settings.wordsize
        )
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_sph", n_bytes_written)

        return n_bytes_written

    def _write_states_airbags(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        if settings.header["npefg"] <= 0:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.airbag_n_active_particles: ["n_timesteps", "n_airbags"],
                ArrayType.airbag_bag_volume: ["n_timesteps", "n_airbags"],
                ArrayType.airbag_particle_gas_id: ["n_timesteps", "n_particles"],
                ArrayType.airbag_particle_chamber_id: ["n_timesteps", "n_particles"],
                ArrayType.airbag_particle_leakage: ["n_timesteps", "n_particles"],
                ArrayType.airbag_particle_mass: ["n_timesteps", "n_particles"],
                ArrayType.airbag_particle_radius: ["n_timesteps", "n_particles"],
                ArrayType.airbag_particle_spin_energy: ["n_timesteps", "n_particles"],
                ArrayType.airbag_particle_translation_energy: ["n_timesteps", "n_particles"],
                ArrayType.airbag_particle_nearest_segment_distance: ["n_timesteps", "n_particles"],
                ArrayType.airbag_particle_position: ["n_timesteps", "n_particles", "x_y_z"],
                ArrayType.airbag_particle_velocity: ["n_timesteps", "n_particles", "vx_vy_vz"],
            },
        )

        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.airbag_n_active_particles: 0,
            ArrayType.airbag_bag_volume: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")

        array_dims = {
            ArrayType.airbags_ids: 0,
            ArrayType.airbag_n_active_particles: 1,
            ArrayType.airbag_bag_volume: 1,
        }
        n_airbags = self.check_array_dims(array_dims, "n_airbags")
        assert n_airbags == settings.header["npefg"] % 1000

        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.airbag_particle_gas_id: 0,
            ArrayType.airbag_particle_chamber_id: 0,
            ArrayType.airbag_particle_leakage: 0,
            ArrayType.airbag_particle_mass: 0,
            ArrayType.airbag_particle_radius: 0,
            ArrayType.airbag_particle_spin_energy: 0,
            ArrayType.airbag_particle_translation_energy: 0,
            ArrayType.airbag_particle_nearest_segment_distance: 0,
            ArrayType.airbag_particle_position: 0,
            ArrayType.airbag_particle_velocity: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")

        array_dims = {
            ArrayType.airbag_particle_gas_id: 1,
            ArrayType.airbag_particle_chamber_id: 1,
            ArrayType.airbag_particle_leakage: 1,
            ArrayType.airbag_particle_mass: 1,
            ArrayType.airbag_particle_radius: 1,
            ArrayType.airbag_particle_spin_energy: 1,
            ArrayType.airbag_particle_translation_energy: 1,
            ArrayType.airbag_particle_nearest_segment_distance: 1,
            ArrayType.airbag_particle_position: 1,
            ArrayType.airbag_particle_velocity: 1,
        }
        n_particles = self.check_array_dims(array_dims, "n_particles")

        self.check_array_dims({ArrayType.airbag_particle_position: 2}, "x_y_z", 3)

        self.check_array_dims({ArrayType.airbag_particle_velocity: 2}, "vx_vy_vz", 3)

        # Info:
        # we cast integers to floats here (no conversion, just a cast)
        # to be able to concatenate the arrays while preserving the
        # bytes internally.

        # AIRBAG STATE DATA
        airbag_n_active_particles = (
            self.arrays[ArrayType.airbag_n_active_particles][i_timestep]
            if ArrayType.airbag_n_active_particles in self.arrays
            else np.zeros(n_airbags, dtype=settings.itype)
        )
        airbag_n_active_particles = airbag_n_active_particles.view(settings.ftype)

        airbag_bag_volume = (
            self.arrays[ArrayType.airbag_bag_volume][i_timestep]
            if ArrayType.airbag_bag_volume in self.arrays
            else np.zeros(n_airbags, dtype=settings.ftype)
        )

        airbag_data = np.concatenate(
            [
                airbag_n_active_particles.reshape(n_airbags, 1),
                airbag_bag_volume.reshape(n_airbags, 1),
            ],
            axis=1,
        )
        n_bytes_written = fp.write(settings.pack(airbag_data, dtype_hint=np.floating))

        # particle var names
        array_particle_list = []

        # PARTICLE GAS ID
        array = (
            self.arrays[ArrayType.airbag_particle_gas_id][i_timestep]
            if ArrayType.airbag_particle_gas_id in self.arrays
            else np.zeros(n_particles, dtype=settings.itype)
        )
        array = array.view(settings.ftype)
        array_particle_list.append(array.reshape(-1, 1))

        # PARTICLE CHAMBER ID
        array = (
            self.arrays[ArrayType.airbag_particle_chamber_id][i_timestep]
            if ArrayType.airbag_particle_chamber_id in self.arrays
            else np.zeros(n_particles, dtype=settings.itype)
        )
        array = array.view(settings.ftype)
        array_particle_list.append(array.reshape(-1, 1))

        # PARTICLE LEAKAGE
        array = (
            self.arrays[ArrayType.airbag_particle_leakage][i_timestep]
            if ArrayType.airbag_particle_leakage in self.arrays
            else np.zeros(n_particles, dtype=settings.itype)
        )
        array = array.view(settings.ftype)
        array_particle_list.append(array.reshape(-1, 1))

        # PARTICLE POSITION
        array = (
            self.arrays[ArrayType.airbag_particle_position][i_timestep]
            if ArrayType.airbag_particle_position in self.arrays
            else np.zeros((n_particles, 3), dtype=settings.ftype)
        )
        array_particle_list.append(array)

        # PARTICLE VELOCITY
        array = (
            self.arrays[ArrayType.airbag_particle_velocity][i_timestep]
            if ArrayType.airbag_particle_velocity in self.arrays
            else np.zeros((n_particles, 3), dtype=settings.ftype)
        )
        array_particle_list.append(array)

        # PARTICLE MASS
        array = (
            self.arrays[ArrayType.airbag_particle_mass][i_timestep]
            if ArrayType.airbag_particle_mass in self.arrays
            else np.zeros(n_particles, dtype=settings.ftype)
        )
        array_particle_list.append(array.reshape(-1, 1))

        # PARTICLE RADIUS
        array = (
            self.arrays[ArrayType.airbag_particle_radius][i_timestep]
            if ArrayType.airbag_particle_radius in self.arrays
            else np.zeros(n_particles, dtype=settings.ftype)
        )
        array_particle_list.append(array.reshape(-1, 1))

        # PARTICLE SPIN ENERGY
        array = (
            self.arrays[ArrayType.airbag_particle_spin_energy][i_timestep]
            if ArrayType.airbag_particle_spin_energy in self.arrays
            else np.zeros(n_particles, dtype=settings.ftype)
        )
        array_particle_list.append(array.reshape(-1, 1))

        # PARTICLE TRANSL ENERGY
        array = (
            self.arrays[ArrayType.airbag_particle_translation_energy][i_timestep]
            if ArrayType.airbag_particle_translation_energy in self.arrays
            else np.zeros(n_particles, dtype=settings.ftype)
        )
        array_particle_list.append(array.reshape(-1, 1))

        # PARTICLE NEAREST NEIGHBOR DISTANCE
        array = (
            self.arrays[ArrayType.airbag_particle_nearest_segment_distance][i_timestep]
            if ArrayType.airbag_particle_nearest_segment_distance in self.arrays
            else np.zeros(n_particles, dtype=settings.ftype)
        )
        array_particle_list.append(array.reshape(-1, 1))

        airbag_particle_data = np.concatenate(array_particle_list, axis=1)
        n_bytes_written += fp.write(settings.pack(airbag_particle_data, dtype_hint=np.floating))

        # check bytes
        n_bytes_expected = (2 * n_airbags + n_particles * 14) * settings.wordsize
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_airbags", n_bytes_written)

        return n_bytes_written

    def _write_states_rigid_road(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        if settings.header["ndim"] <= 5:
            return 0

        _check_ndim(
            self,
            {
                ArrayType.rigid_road_displacement: ["n_timesteps", "n_rigid_roads", "x_y_z"],
                ArrayType.rigid_road_velocity: ["n_timesteps", "n_rigid_roads", "vx_vy_vz"],
            },
        )

        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.rigid_road_displacement: 0,
            ArrayType.rigid_road_velocity: 0,
        }
        self.check_array_dims(array_dims, "n_rigid_roads")

        array_dims = {
            ArrayType.rigid_road_segment_road_id: 0,
            ArrayType.rigid_road_displacement: 1,
            ArrayType.rigid_road_velocity: 1,
        }
        n_rigid_roads = self.check_array_dims(array_dims, "n_rigid_roads")

        self.check_array_dims({ArrayType.rigid_road_displacement: 2}, "x_y_z", 3)

        self.check_array_dims({ArrayType.rigid_road_velocity: 2}, "vx_vy_vz", 3)

        rigid_road_data = np.zeros((n_rigid_roads, 2, 3), dtype=settings.ftype)

        # RIGID ROAD DISPLACEMENT
        if ArrayType.rigid_road_displacement in self.arrays:
            array = self.arrays[ArrayType.rigid_road_displacement][i_timestep]
            rigid_road_data[:, 0, :] = array

        # RIGID ROAD VELOCITY
        if ArrayType.rigid_road_velocity in self.arrays:
            array = self.arrays[ArrayType.rigid_road_velocity][i_timestep]
            rigid_road_data[:, 1, :] = array

        n_bytes_written = fp.write(settings.pack(rigid_road_data, dtype_hint=np.floating))

        # check bytes
        n_bytes_expected = settings.header["nv1d"] * settings.header["nel2"] * settings.wordsize
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_rigid_road", n_bytes_written)

        return n_bytes_written

    def _write_states_rigid_bodies(
        self, fp: typing.IO[Any], i_timestep: int, settings: D3plotWriterSettings
    ) -> int:

        if 8 <= settings.header["ndim"] <= 9:
            pass
        else:
            return 0

        has_reduced_data = settings.header["ndim"] == 9

        _check_ndim(
            self,
            {
                ArrayType.rigid_body_coordinates: ["n_timesteps", "n_rigid_bodies", "x_y_z"],
                ArrayType.rigid_body_rotation_matrix: ["n_timesteps", "n_rigid_bodies", "matrix"],
                ArrayType.rigid_body_velocity: ["n_timesteps", "n_rigid_bodies", "vx_vy_vz"],
                ArrayType.rigid_body_rot_velocity: ["n_timesteps", "n_rigid_bodies", "rvx_rvy_rvz"],
                ArrayType.rigid_body_acceleration: ["n_timesteps", "n_rigid_bodies", "ax_ay_az"],
                ArrayType.rigid_body_rot_acceleration: [
                    "n_timesteps",
                    "n_rigid_bodies",
                    "rax_ray_raz",
                ],
            },
        )

        array_dims = {
            ArrayType.global_timesteps: 0,
            ArrayType.rigid_body_coordinates: 0,
            ArrayType.rigid_body_rotation_matrix: 0,
            ArrayType.rigid_body_velocity: 0,
            ArrayType.rigid_body_rot_velocity: 0,
            ArrayType.rigid_body_acceleration: 0,
            ArrayType.rigid_body_rot_acceleration: 0,
        }
        self.check_array_dims(array_dims, "n_timesteps")

        array_dims = {
            ArrayType.rigid_body_part_indexes: 1,
            ArrayType.rigid_body_coordinates: 1,
            ArrayType.rigid_body_rotation_matrix: 1,
            ArrayType.rigid_body_velocity: 1,
            ArrayType.rigid_body_rot_velocity: 1,
            ArrayType.rigid_body_acceleration: 1,
            ArrayType.rigid_body_rot_acceleration: 1,
        }
        n_rigid_bodies = self.check_array_dims(array_dims, "n_rigid_bodies")

        self.check_array_dims({ArrayType.rigid_body_coordinates: 2}, "x_y_z", 3)

        self.check_array_dims({ArrayType.rigid_body_rotation_matrix: 2}, "matrix", 9)

        self.check_array_dims({ArrayType.rigid_body_velocity: 2}, "vx_vy_vz", 3)

        self.check_array_dims({ArrayType.rigid_body_rot_velocity: 2}, "rvx_rvy_rvz", 3)

        self.check_array_dims({ArrayType.rigid_body_acceleration: 2}, "ax_ay_az", 3)

        self.check_array_dims({ArrayType.rigid_body_rot_acceleration: 2}, "rax_ray_raz", 3)

        # allocate block
        rigid_body_data = (
            np.zeros((n_rigid_bodies, 12), dtype=settings.ftype)
            if has_reduced_data
            else np.zeros((n_rigid_bodies, 24), dtype=settings.ftype)
        )

        start_index = 0
        end_index = 0

        # COORDINATES
        if ArrayType.rigid_body_coordinates in self.arrays:
            start_index = end_index
            end_index = start_index + 3
            array = self.arrays[ArrayType.rigid_body_coordinates][i_timestep]
            rigid_body_data[:, start_index:end_index] = array

        # ROTATION MATRIX
        if ArrayType.rigid_body_rotation_matrix in self.arrays:
            start_index = end_index
            end_index = start_index + 9
            array = self.arrays[ArrayType.rigid_body_coordinates][i_timestep]
            rigid_body_data[:, start_index:end_index] = array

        if not has_reduced_data:

            # VELOCITY
            if ArrayType.rigid_body_velocity in self.arrays:
                start_index = end_index
                end_index = start_index + 3
                array = self.arrays[ArrayType.rigid_body_velocity][i_timestep]
                rigid_body_data[:, start_index:end_index] = array

            # ROTATION VELOCITY
            if ArrayType.rigid_body_rot_velocity in self.arrays:
                start_index = end_index
                end_index = start_index + 3
                array = self.arrays[ArrayType.rigid_body_rot_velocity][i_timestep]
                rigid_body_data[:, start_index:end_index] = array

            # ACCELERATION
            if ArrayType.rigid_body_acceleration in self.arrays:
                start_index = end_index
                end_index = start_index + 3
                array = self.arrays[ArrayType.rigid_body_acceleration][i_timestep]
                rigid_body_data[:, start_index:end_index] = array

            # ROTATION ACCELERATION
            if ArrayType.rigid_body_rot_acceleration in self.arrays:
                start_index = end_index
                end_index = start_index + 3
                array = self.arrays[ArrayType.rigid_body_rot_acceleration][i_timestep]
                rigid_body_data[:, start_index:end_index] = array

        n_bytes_written = fp.write(settings.pack(rigid_body_data, dtype_hint=np.floating))

        # check bytes
        n_bytes_expected = settings.header["nv1d"] * settings.header["nel2"] * settings.wordsize
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

        # log
        msg = "%s wrote %d bytes."
        LOGGER.debug(msg, "_write_states_rigid_bodies", n_bytes_written)

        return n_bytes_written

    def check_array_dims(
        self, array_dimensions: Dict[str, int], dimension_name: str, dimension_size: int = -1
    ):
        """This function checks if multiple arrays share an array dimensions
        with the same size.

        Parameters
        ----------
        array_dimensions: Dict[str, int]
            Array name and expected number of dimensions as dict
        dimension_name: str
            Name of the array dimension for error messages
        dimension_size: int
            Optional expected size. If not set then all entries must equal
            the first value collected.

        Raises
        ------
        ValueError
            If dimensions do not match in any kind of way.
        """

        dimension_size_dict = {}

        # collect all dimensions
        for typename, dimension_index in array_dimensions.items():
            if typename not in self.arrays:
                continue

            array = self.arrays[typename]

            if dimension_index >= array.ndim:
                msg = (
                    f"Array '{typename}' requires at least "
                    f"{dimension_index} dimensions ({dimension_name})"
                )
                raise ValueError(msg)

            dimension_size_dict[typename] = array.shape[dimension_index]

        # static dimension
        if dimension_size >= 0:
            arrays_with_wrong_dims = {
                typename: size
                for typename, size in dimension_size_dict.items()
                if size != dimension_size
            }

            if arrays_with_wrong_dims:
                msg = "The dimension %s of the following arrays is expected to have size %d:\n%s"
                msg_arrays = [
                    f" - name: {typename} dim: {array_dimensions[typename]} size: {size}"
                    for typename, size in arrays_with_wrong_dims.items()
                ]
                raise ValueError(msg, dimension_name, dimension_size, "\n".join(msg_arrays))

        # dynamic dimensions
        else:
            if dimension_size_dict:
                unique_sizes = np.unique(list(dimension_size_dict.values()))
                if len(unique_sizes) > 1:
                    msg = "Inconsistency in array dim '%d' detected:\n%s"
                    size_list = [
                        f"   - name: {typename}, dim: {array_dimensions[typename]}, size: {size}"
                        for typename, size in dimension_size_dict.items()
                    ]
                    raise ValueError(msg, dimension_name, "\n".join(size_list))
                if len(unique_sizes) == 1:
                    dimension_size = unique_sizes[0]

        if dimension_size < 0:
            return 0

        return dimension_size

    @staticmethod
    def _compare_n_bytes_checksum(n_bytes_written: int, n_bytes_expected: int):
        """Throw if the byte checksum was not ok

        Parameters
        ----------
        n_bytes_written: int
            bytes written to the file
        n_bytes_expected: int
            bytes expected from the header computation

        Raises
        ------
        RuntimeError
            If the byte count doesn't match.
        """
        if n_bytes_expected != n_bytes_written:
            msg = (
                "byte checksum wrong: "
                f"{n_bytes_expected} (header) != {n_bytes_written} (checksum)"
            )
            raise RuntimeError(msg)

    def _get_zero_byte_padding(self, n_bytes_written: int, block_size_bytes: int):
        """Compute the zero byte-padding at the end of files

        Parameters
        ----------
        n_bytes_written: int
            number of bytes already written to file
        block_size_bytes: int
            byte block size of the file

        Returns
        -------
        zero_bytes: bytes
            zero-byte padding ready to be written to the file
        """

        if block_size_bytes > 0:
            remaining_bytes = n_bytes_written % block_size_bytes
            n_bytes_to_fill = block_size_bytes - remaining_bytes if remaining_bytes != 0 else 0
            return b"\x00" * n_bytes_to_fill

        return b""

    def compare(self, d3plot2, array_eps: Union[float, None] = None):
        """Compare two d3plots and print the info

        Parameters
        ----------
        d3plot2: D3plot
            second d3plot
        array_eps: float or None
            tolerance for arrays

        Returns
        -------
        hdr_differences: dict
            differences in the header
        array_differences: dict
            difference between arrays as message

        Examples
        --------
            Comparison of a femzipped file and an uncompressed file. Femzip
            is a lossy compression, thus precision is traded for memory.

            >>> d3plot1 = D3plot("path/to/d3plot")
            >>> d3plot2 = D3plot("path/to/d3plot.fz")
            >>> hdr_diff, array_diff = d3plot1.compare(d3plot2)
            >>> for arraytype, msg in array_diff.items():
            >>>     print(name, msg)
            node_coordinates Δmax = 0.050048828125
            node_displacement Δmax = 0.050048828125
            node_velocity Δmax = 0.050048828125
            node_acceleration Δmax = 49998984.0
            element_beam_axial_force Δmax = 6.103515625e-05
            element_shell_stress Δmax = 0.0005035400390625
            element_shell_thickness Δmax = 9.999999717180685e-10
            element_shell_unknown_variables Δmax = 0.0005000010132789612
            element_shell_internal_energy Δmax = 188.41957092285156

        """

        # pylint: disable = too-many-nested-blocks

        assert isinstance(d3plot2, D3plot)
        d3plot1 = self

        hdr_differences = d3plot1.header.compare(d3plot2.header)

        # ARRAY COMPARISON
        array_differences = {}

        array_names = list(d3plot1.arrays.keys()) + list(d3plot2.arrays.keys())

        for name in array_names:

            array1 = (
                d3plot1.arrays[name] if name in d3plot1.arrays else "Array is missing in original"
            )

            array2 = d3plot2.arrays[name] if name in d3plot2.arrays else "Array is missing in other"

            # d3parts write results for rigid shells.
            # when rewriting as d3plot we simply
            # don't write the part_material_types
            # array which is the same as having no
            # rigid shells.
            d3plot1_is_d3part = d3plot1.header.filetype == D3plotFiletype.D3PART
            d3plot2_is_d3part = d3plot2.header.filetype == D3plotFiletype.D3PART
            if name == "part_material_type" and (d3plot1_is_d3part or d3plot2_is_d3part):
                continue

            # we have an array to compare
            if isinstance(array1, str):
                array_differences[name] = array1
            elif isinstance(array2, str):
                array_differences[name] = array2
            elif isinstance(array2, np.ndarray):
                comparison = False

                # compare arrays
                if isinstance(array1, np.ndarray):
                    if array1.shape != array2.shape:
                        comparison = f"shape mismatch {array1.shape} != {array2.shape}"
                    else:
                        if np.issubdtype(array1.dtype, np.number) and np.issubdtype(
                            array2.dtype, np.number
                        ):
                            diff = np.abs(array1 - array2)
                            if diff.size:
                                if array_eps is not None:
                                    diff2 = diff[diff > array_eps]
                                    if diff2.size:
                                        diff2_max = diff2.max()
                                        if diff2_max:
                                            comparison = f"Δmax = {diff2_max}"
                                else:
                                    diff_max = diff.max()
                                    if diff_max:
                                        comparison = f"Δmax = {diff_max}"
                        else:
                            n_mismatches = (array1 != array2).sum()
                            if n_mismatches:
                                comparison = f"Mismatches: {n_mismatches}"

                else:
                    comparison = "Arrays don't match"

                # print
                if comparison:
                    array_differences[name] = comparison

        return hdr_differences, array_differences

    def get_part_filter(
        self, filter_type: FilterType, part_ids: Iterable[int], for_state_array: bool = True
    ) -> np.ndarray:
        """Get a part filter for different entities

        Parameters
        ----------
        filter_type: lasso.dyna.FilterType
            the array type to filter for (beam, shell, solid, tshell, node)
        part_ids: Iterable[int]
            part ids to filter out
        for_state_array: bool
            if the filter is meant for a state array. Makes a difference
            for shells if rigid bodies are in the model (mattyp == 20)

        Returns
        -------
        mask: np.ndarray
            mask usable on arrays to filter results

        Examples
        --------
            >>> from lasso.dyna import D3plot, ArrayType, FilterType
            >>> d3plot = D3plot("path/to/d3plot")
            >>> part_ids = [13, 14]
            >>> mask = d3plot.get_part_filter(FilterType.shell)
            >>> shell_stress = d3plot.arrays[ArrayType.element_shell_stress]
            >>> shell_stress.shape
            (34, 7463, 3, 6)
            >>> # select only parts from part_ids
            >>> shell_stress_parts = shell_stress[:, mask]
        """

        # nodes are treated separately
        if filter_type == FilterType.NODE:
            node_index_arrays = []

            if ArrayType.element_shell_node_indexes in self.arrays:
                shell_filter = self.get_part_filter(
                    FilterType.SHELL, part_ids, for_state_array=False
                )
                shell_node_indexes = self.arrays[ArrayType.element_shell_node_indexes]
                node_index_arrays.append(shell_node_indexes[shell_filter].flatten())

            if ArrayType.element_solid_node_indexes in self.arrays:
                solid_filter = self.get_part_filter(
                    FilterType.SOLID, part_ids, for_state_array=False
                )
                solid_node_indexes = self.arrays[ArrayType.element_solid_node_indexes]
                node_index_arrays.append(solid_node_indexes[solid_filter].flatten())

            if ArrayType.element_tshell_node_indexes in self.arrays:
                tshell_filter = self.get_part_filter(
                    FilterType.TSHELL, part_ids, for_state_array=False
                )
                tshell_node_indexes = self.arrays[ArrayType.element_tshell_node_indexes]
                node_index_arrays.append(tshell_node_indexes[tshell_filter].flatten())

            return np.unique(np.concatenate(node_index_arrays))

        # we need part ids first
        if ArrayType.part_ids in self.arrays:
            d3plot_part_ids = self.arrays[ArrayType.part_ids]
        elif ArrayType.part_titles_ids in self.arrays:
            d3plot_part_ids = self.arrays[ArrayType.part_titles_ids]
        else:
            msg = "D3plot does neither contain '%s' nor '%s'"
            raise RuntimeError(msg, ArrayType.part_ids, ArrayType.part_titles_ids)

        # if we filter parts we can stop here
        if filter_type == FilterType.PART:
            return np.isin(d3plot_part_ids, part_ids)

        # get part indexes from part ids
        part_indexes = np.argwhere(np.isin(d3plot_part_ids, part_ids)).flatten()

        # associate part indexes with entities
        if filter_type == FilterType.BEAM:
            entity_part_indexes = self.arrays[ArrayType.element_beam_part_indexes]
        elif filter_type == FilterType.SHELL:
            entity_part_indexes = self.arrays[ArrayType.element_shell_part_indexes]

            # shells may contain "rigid body shell elements"
            # for these shells no state data is output and thus
            # the state arrays have a reduced element count
            if for_state_array and self._material_section_info.n_rigid_shells != 0:
                mat_types = self.arrays[ArrayType.part_material_type]
                mat_type_filter = mat_types[entity_part_indexes] != 20
                entity_part_indexes = entity_part_indexes[mat_type_filter]

        elif filter_type == FilterType.TSHELL:
            entity_part_indexes = self.arrays[ArrayType.element_tshell_part_indexes]
        elif filter_type == FilterType.SOLID:
            entity_part_indexes = self.arrays[ArrayType.element_solid_part_indexes]
        else:
            msg = "Invalid filter_type '%s'. Use lasso.dyna.FilterType."
            raise ValueError(msg, filter_type)

        mask = np.isin(entity_part_indexes, part_indexes)
        return mask

    @staticmethod
    def enable_logger(enable: bool):
        """Enable the logger for this class

        Parameters
        ----------
        enable: bool
            whether to enable logging for this class
        """

        if enable:
            LOGGER.setLevel(logging.DEBUG)
        else:
            LOGGER.setLevel(logging.NOTSET)
