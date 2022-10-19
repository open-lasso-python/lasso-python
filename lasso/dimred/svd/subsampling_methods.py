import os
import random
import time
from typing import List, Sequence, Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ...dyna import ArrayType, D3plot


def _mark_dead_eles(node_indexes: np.ndarray, alive_shells: np.ndarray) -> np.ndarray:
    """
    Returns a mask to filter out elements mark as 'no alive'

    Parameters
    ----------
    node_indexes: ndarray
        Array containing node indexes
    alive_nodes: ndarray
        Array containing float value representing if element is alive.
        Expected for D3plot.arrays[ArrayType.element_shell_is_alive] or equivalent for beams etc

    Returns
    -------
    node_coordinate_mask: np.ndarray
        Array containing indizes of alive shells.
        Use node_coordinates[node_coordinate_mask] to get all nodes alive.

    See Also
    --------
    bury_the_dead(), also removes dead beam nodes
    """

    dead_eles_shell = np.unique(np.where(alive_shells == 0)[1])

    ele_filter = np.zeros(node_indexes.shape[0])
    ele_filter[dead_eles_shell] = 1
    ele_filter_bool = ele_filter == 1

    dead_nodes = np.unique(node_indexes[ele_filter_bool])

    return dead_nodes


def _extract_shell_parts(
    part_list: Sequence[int], d3plot: D3plot
) -> Union[Tuple[np.ndarray, np.ndarray], str]:
    """
    Extracts a shell part defined by its part ID out of the given d3plot.
    Returns a new node index, relevant coordinates and displacement

    Parameters
    ----------
    part_list: list
        List of part IDs of the parts that shall be extracted
    d3plot: D3plot
        D3plot the part shall be extracted from

    Returns
    -------
    node_coordinates: ndarray
        Numpy array containing the node coordinates of the extracted part
    node_displacement: ndarray
        Numpy array containing the node displacement of the extracted part
    err_msg: str
        If an error occurs, a string containing the error msg is returned instead
    """

    # pylint: disable = too-many-locals, too-many-statements

    # convert into list
    part_list = list(part_list)

    shell_node_indexes = d3plot.arrays[ArrayType.element_shell_node_indexes]
    shell_part_indexes = d3plot.arrays[ArrayType.element_shell_part_indexes]
    beam_node_indexes = d3plot.arrays[ArrayType.element_beam_node_indexes]
    beam_part_indexes = d3plot.arrays[ArrayType.element_beam_part_indexes]
    solid_node_indexes = d3plot.arrays[ArrayType.element_solid_node_indexes]
    solid_part_indexes = d3plot.arrays[ArrayType.element_solid_part_indexes]
    tshell_node_indexes = d3plot.arrays[ArrayType.element_tshell_node_indexes]
    tshell_part_indexes = d3plot.arrays[ArrayType.element_tshell_part_indexes]

    node_coordinates = d3plot.arrays[ArrayType.node_coordinates]
    node_displacement = d3plot.arrays[ArrayType.node_displacement]

    alive_mask = np.full((node_coordinates.shape[0]), True)

    if ArrayType.element_shell_is_alive in d3plot.arrays:
        dead_shell_mask = _mark_dead_eles(
            shell_node_indexes, d3plot.arrays[ArrayType.element_shell_is_alive]
        )
        alive_mask[dead_shell_mask] = False
    if ArrayType.element_beam_is_alive in d3plot.arrays:
        dead_beam_mask = _mark_dead_eles(
            beam_node_indexes, d3plot.arrays[ArrayType.element_beam_is_alive]
        )
        alive_mask[dead_beam_mask] = False
    if ArrayType.element_solid_is_alive in d3plot.arrays:
        dead_solid_mask = _mark_dead_eles(
            solid_node_indexes, d3plot.arrays[ArrayType.element_solid_is_alive]
        )
        alive_mask[dead_solid_mask] = False
    if ArrayType.element_tshell_is_alive in d3plot.arrays:
        dead_tshell_mask = _mark_dead_eles(
            tshell_node_indexes, d3plot.arrays[ArrayType.element_tshell_is_alive]
        )
        alive_mask[dead_tshell_mask] = False

    if len(part_list) > 0:
        try:
            part_ids = d3plot.arrays[ArrayType.part_ids]
        except KeyError:
            err_msg = "KeyError: Loaded plot has no parts"
            return err_msg
        part_ids_as_list = part_ids.tolist()
        # check if parts exist
        for part in part_list:
            try:
                part_ids_as_list.index(int(part))
            except ValueError:
                err_msg = "ValueError: Could not find part: {0}"
                return err_msg.format(part)

        def mask_parts(
            part_list2: List[int], element_part_index: np.ndarray, element_node_index: np.ndarray
        ) -> np.ndarray:

            element_part_filter = np.full(element_part_index.shape, False)
            proc_parts = []

            for pid in part_list2:
                part_index = part_ids_as_list.index(int(pid))
                locs = np.where(element_part_index == part_index)[0]
                if not locs.shape == (0,):
                    proc_parts.append(pid)
                element_part_filter[locs] = True

            for part in proc_parts:
                part_list2.pop(part_list2.index(part))

            unique_element_node_indexes = np.unique(element_node_index[element_part_filter])

            return unique_element_node_indexes

        # shells:
        unique_shell_node_indexes = mask_parts(part_list, shell_part_indexes, shell_node_indexes)

        # beams
        unique_beam_node_indexes = mask_parts(part_list, beam_part_indexes, beam_node_indexes)

        # solids:
        unique_solide_node_indexes = mask_parts(part_list, solid_part_indexes, solid_node_indexes)

        # tshells
        unique_tshell_node_indexes = mask_parts(part_list, tshell_part_indexes, tshell_node_indexes)

        # this check may seem redundant, but also verifies that our masking of parts works
        if not len(part_list) == 0:
            err_msg = "Value Error: Could not find parts: " + str(part_list)
            return err_msg

        # New coordinate mask
        coord_mask = np.full((node_coordinates.shape[0]), False)
        coord_mask[unique_shell_node_indexes] = True
        coord_mask[unique_solide_node_indexes] = True
        coord_mask[unique_beam_node_indexes] = True
        coord_mask[unique_tshell_node_indexes] = True

        inv_alive_mask = np.logical_not(alive_mask)
        coord_mask[inv_alive_mask] = False

        node_coordinates = node_coordinates[coord_mask]
        node_displacement = node_displacement[:, coord_mask]
    else:
        node_coordinates = node_coordinates[alive_mask]
        node_displacement = node_displacement[:, alive_mask]

    return node_coordinates, node_displacement


def create_reference_subsample(
    load_path: str, parts: Sequence[int], nr_samples=2000
) -> Union[Tuple[np.ndarray, float, float], str]:
    """
    Loads the D3plot at load_path, extracts the node coordinates of part 13, returns
    a random subsample of these nodes

    Parameters
    ----------
    load_path: str
        Filepath of the D3plot
    parts: Sequence[int]
        List of parts to be extracted
    nr_samples: int
        How many nodes are subsampled

    Returns
    -------
    reference_sample: np.array
        Numpy array containing the reference sample
    t_total: float
        Total time required for subsampling
    t_load: float
        Time required to load plot
    err_msg: str
        If an error occurs, a string containing the error is returned instead
    """
    t_null = time.time()
    try:
        plot = D3plot(
            load_path,
            state_array_filter=[ArrayType.node_displacement, ArrayType.element_shell_is_alive],
        )
    except Exception:
        err_msg = (
            f"Failed to load {load_path}! Please make sure it is a D3plot file. "
            f"This might be due to {os.path.split(load_path)[1]} being a timestep of a plot"
        )
        return err_msg

    t_load = time.time() - t_null
    result = _extract_shell_parts(parts, plot)
    if isinstance(result, str):
        return result

    coordinates = result[0]
    if coordinates.shape[0] < nr_samples:
        err_msg = "Number of nodes is lower than desired samplesize"
        return err_msg

    random.seed("seed")
    samples = random.sample(range(len(coordinates)), nr_samples)

    reference_sample = coordinates[samples]
    t_total = time.time() - t_null
    return reference_sample, t_total, t_load


def remap_random_subsample(
    load_path: str, parts: list, reference_subsample: np.ndarray
) -> Union[Tuple[np.ndarray, float, float], str]:
    """
    Remaps the specified sample onto a new mesh provided by reference subsampl, using knn matching

    Parameters
    ----------
    load_path: str
        Filepath of the desired D3plot
    parts: list of int
        Which parts shall be extracted from the D3plot
    reference_subsample: np.array
        Numpy array containing the reference nodes

    Returns
    -------
    subsampled_displacement: np.ndarray
        Subsampled displacement of provided sample
    t_total: float
        Total time required to perform subsampling
    t_load: float
        Time required to load D3plot
    err_msg: str
        If an error occured, a string is returned instead containing the error
    """
    t_null = time.time()
    try:
        plot = D3plot(
            load_path,
            state_array_filter=[ArrayType.node_displacement, ArrayType.element_shell_is_alive],
        )
    except Exception:
        err_msg = (
            f"Failed to load {load_path}! Please make sure it is a D3plot file. "
            f"This might be due to {os.path.split(load_path)[1]} being a timestep of a plot"
        )
        return err_msg

    t_load = time.time() - t_null
    result = _extract_shell_parts(parts, plot)
    if isinstance(result, str):
        return result

    coordinates, displacement = result[0], result[1]

    quarantine_zone = NearestNeighbors(n_neighbors=1, n_jobs=4).fit(coordinates)
    _, quarantined_index = quarantine_zone.kneighbors(reference_subsample)

    subsampled_displacement = displacement[:, quarantined_index[:, 0]]

    return subsampled_displacement, time.time() - t_null, t_load
