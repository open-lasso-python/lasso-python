import os
import typing
import warnings

import h5py
import numpy as np

# scipy is C-code which causes invalid linter warning about ConvexHull not
# being around.
# pylint: disable = no-name-in-module
from scipy.spatial import ConvexHull
from scipy.stats import binned_statistic_2d
from sklearn.preprocessing import normalize

warnings.simplefilter(action="ignore", category=FutureWarning)


def _create_sphere_mesh(diameter: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compute the alpha and beta increments for a
        meshed sphere for binning the projected values

    Parameters
    ----------
    diameter : np.ndarray
        sphere diameter

    Returns
    -------
    bin_alpha : np.ndarray
        alpha bin boundaries
    bin_beta : np.ndarray
        beta bin boundaries
    """

    assert diameter.dtype == np.float

    # partition latitude
    n_alpha = 145

    # sphere radius
    r = diameter / 2

    # sphere area
    a_sphere = 4 * np.pi * r**2

    # number of elements
    n_ele = 144**2

    # area of one element
    a_ele = a_sphere / n_ele

    # bin values for alpha and the increment
    bin_alpha, delt_alpha = np.linspace(0, 2 * np.pi, n_alpha, retstep=True)

    # for beta axis binning
    count = np.linspace(0.0, 144.0, 145)
    # compute required bin boundaries to ensure area of each element is the same
    tmp = count * a_ele
    tmp /= r**2 * delt_alpha
    bin_beta = 1 - tmp

    # In case of trailing floats (-1.00000004 for example)
    if bin_beta[-1] < -1:
        bin_beta[-1] = -1

    bin_beta = np.arccos(bin_beta)

    return bin_alpha, bin_beta


def _project_to_sphere(
    points: np.ndarray, centroid: np.ndarray, axis: str = "Z"
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """compute the projection vectors of centroid to each point in terms of spherical coordinates

    Parameters
    ----------
    points : np.ndarray
        hashes of first model
    centroid : np.ndarray
        hashes of first model
    AXIS : str
        global axis position

    Returns
    -------
    proj_alpha : np.ndarray
        alpha angles of all points
    proj_beta : np.ndarray
        beta angle of all points
    """
    # standard global axis
    indexes = [0, 1, 2]

    # correct the indexes based on user input
    if axis == "Z":
        indexes = [0, 1, 2]  # z axis aligned with global z-axis
    elif axis == "Y":
        indexes = [0, 2, 1]  # z axis aligned with global y-axis
    elif axis == "X":
        indexes = [2, 1, 0]  # z axis aligned with global x-axis

    # projection
    vec = points - centroid

    # normalize
    vec = normalize(vec, axis=1, norm="l2")

    # alpha based on sphere axis alignment
    ang = np.arctan2(vec[:, indexes[1]], vec[:, indexes[0]])

    # atan2 returns neg angles for values greater than 180
    neg_indexes = np.where(ang < 0)
    ang[neg_indexes] += 2 * np.pi

    proj_alpha = ang
    proj_beta = np.arccos(vec[:, indexes[2]])

    return proj_alpha, proj_beta


def sphere_hashing(
    bin_numbers: np.ndarray, bin_counts: np.ndarray, field: np.ndarray
) -> np.ndarray:
    """Compute average field values for all the binned values

    Parameters
    ----------
    bin_numbers : np.ndarray
        bin numbers for the respective index for the x and y-axis
    bin_counts : np.ndarray
        number of points that fall into each bin
    field : np.ndarray
        a fields value (p_strain,velocity etc..)

    Returns
    -------
    binned_field : np.ndarray
        the averaged field values for each field
    """
    # bin_numbers holds the bin_number for its respective index and must have
    # same length as the number of points
    assert len(bin_numbers[0] == len(field))
    # check data types
    assert bin_numbers.dtype == np.int
    assert bin_counts.dtype == np.float

    n_rows = bin_counts.shape[0]
    n_cols = bin_counts.shape[1]

    # bin x and y indexes for each point in field
    binx = np.asarray(bin_numbers[0]) - 1
    biny = np.asarray(bin_numbers[1]) - 1

    # bincout for averaging
    bin_count = np.zeros((n_rows, n_cols))

    # averaged result to return
    binned_field = np.zeros((n_rows, n_cols))

    # bin the field values
    binned_field[binx[:], biny[:]] += field[:]
    # count
    bin_count[binx[:], biny[:]] += 1

    binned_field = binned_field.flatten()
    bin_count = bin_count.flatten()

    # exclude all zero entries
    nonzero_inds = np.where(bin_count != 0)
    # average the fields
    binned_field[nonzero_inds] /= bin_count[nonzero_inds]

    return binned_field


def compute_hashes(
    source_path: str, target_path: str = None, n_files: int = None, ret_vals: bool = False
):
    """Compute the hashes using spherical projection of the field values

    Parameters
    ----------
    source_path : str
        path to source directory from which the displacements/strains are
        loaded, this directory should contain HDF5 files of the data
    target_path : str (optional)
        directory in which the hashes are to be written to
    n_files : int (optional)
        number of files to process, useful for verification and quick visualization
    ret_vals : bool (optional)
        return the hashes, setting this to true, be aware that the hash list can
        take up a lot of ram

    Returns
    -------
    hashes : np.ndarray
        hashed field values

    Notes
    -----
        Key for node_displacements for all timesteps: 'xyz'
        Key for field values for all timesteps: 'fields'
    """

    # pylint: disable = too-many-locals

    node_displacement_key = "xyz"
    fields_key = "fields"
    file_name = "run_"
    counter = 0

    hashed_data = []
    # if n_files is none get total number of files in directory
    if n_files is None:
        n_files = len(os.listdir(source_path))

    # load the displacements and compute the hashes for each run and consider
    # the last time step only
    for ii in range(n_files):
        with h5py.File(source_path + file_name + str(ii) + ".h5", "r") as hf:
            node_displacements = hf[node_displacement_key]
            fields = hf[fields_key]

        xyz = node_displacements[:, 0, :]

        # centorid of point cloud
        centroid = np.mean(xyz, axis=0)

        # convex hull of point cloud
        hull = ConvexHull(xyz)
        dist = np.linalg.norm(hull.max_bound - hull.min_bound)

        # compute the bin intervals for alpha and beta split into 144 elements
        bins_a, bins_b = _create_sphere_mesh(dist)

        # compute the point projections
        proj_alpha, proj_beta = _project_to_sphere(xyz, centroid, axis="Y")

        # bin the spherical coordinates in terms of alpha and beta
        histo = binned_statistic_2d(
            proj_alpha, proj_beta, None, "count", bins=[bins_a, bins_b], expand_binnumbers=True
        )
        # get the field value
        p_strains = fields[:, -1]

        # compute hashes
        hashes = sphere_hashing(histo.binnumber, histo.statistic, p_strains)

        if target_path:
            # write the hashes for each timestep to file
            with h5py.File(target_path + "hashes_sphere_" + str(counter) + ".h5", "w") as hf:
                hf.create_dataset("hashes", data=hashes)

        if ret_vals:
            hashed_data.append(hashes)

    return np.asarray(hashed_data)
