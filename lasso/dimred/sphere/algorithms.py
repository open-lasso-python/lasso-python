import numpy as np

from sklearn.preprocessing import normalize

# scipy is C-code which causes invalid linter warning about ConvexHull not
# being around.
# pylint: disable = no-name-in-module
from scipy.spatial import ConvexHull
from scipy.stats import binned_statistic_2d
from scipy.stats._binned_statistic import BinnedStatistic2dResult


def to_spherical_coordinates(points: np.ndarray, centroid: np.ndarray, axis: str = "Z"):
    """Converts the points to spherical coordinates.

    Parameters
    ----------
    points: np.ndarray
        The point cloud to be sphered.
    centroid: np.ndarray
        Centroid of the point cloud.
    axis: str
        Sphere axis in the global coordinate system.

    Returns
    -------
    az : np.ndarray
        Azimuthal angle vector.
    po: np.ndarray
        Polar angle vector.

    Notes
    -----
    The local x-axis is set as the zero marker for azimuthal angles.
    """
    indexes = [0, 1, 2]
    # set the correct indexes for swapping if the sphere
    # axis is not aligned with the global z axis
    if axis == "Y":
        indexes = [0, 2, 1]  # sphere z axis aligned with global y-axis
    elif axis == "X":
        indexes = [2, 1, 0]  # sphere z axis aligned with global x-axis

    # vectors from centroid to points
    vec = points - centroid
    vec = normalize(vec, axis=1, norm="l2")

    # azimuthal angles on the local xy plane
    # x-axis is the zero marker, and we correct
    # all negative angles
    az = np.arctan2(vec[:, indexes[1]], vec[:, indexes[0]])
    neg_indexes = np.where(az < 0)
    az[neg_indexes] += 2 * np.pi

    # polar angles
    po = np.arccos(vec[:, indexes[2]])

    return az, po


def sphere_hashing(histo: BinnedStatistic2dResult, field: np.ndarray):
    """Compute the hash of each bucket in the histogram by mapping
    the bin numbers to the field values and scaling the field values
    by the number of counts in each bin.

    Parameters
    ----------
    histo: BinnedStatistic2dResult
        3D histogram containing the indexes of all points of a simulation
        mapped to their projected bins.
    field: ndarray

    Returns
    -------
    hashes: np.ndarray
        The hashing result of all points mapped to an embedding space.

    """
    bin_n = histo.binnumber

    assert len(bin_n[0] == len(field))

    # get dims of the embedding space
    n_rows = histo.statistic.shape[0]
    n_cols = histo.statistic.shape[1]

    # bin stores the indexes of the points
    # index 0 stores the azimuthal angles
    # index 1 stores the polar angles
    # we want zero indexing
    binx = np.asarray(bin_n[0]) - 1
    biny = np.asarray(bin_n[1]) - 1

    # allocate arrays
    bin_count = np.zeros((n_rows, n_cols))
    hashes = np.zeros((n_rows, n_cols))

    # sum all the field values to each bin
    hashes[binx[:], biny[:]] += field[:]
    bin_count[binx[:], biny[:]] += 1

    hashes = hashes.flatten()
    bin_count = bin_count.flatten()

    # exclude all zero entries
    nonzero_inds = np.where(bin_count != 0)

    # average the fields
    hashes[nonzero_inds] /= bin_count[nonzero_inds]

    return hashes


def create_sphere(diameter: float):
    """Creates two vectors along the alpha and beta axis of a sphere. Alpha represents
     the angle from the sphere axis to the equator. Beta between vectors from the
     center of the sphere to one of the poles and the equator.

     Parameters
     ----------
     diameter:
        Diameter of the sphere.

    Returns
    -------
    bin_beta: np.ndarray
        Bin bounds for the beta angles.

    bin_alpha: np.ndarray
        Bin bounds for the alpha angles.

    """
    # number of partitions for equator
    n_alpha = 145
    # number of partitions for longitude
    n_beta = 144

    r = diameter / 2.0

    # area of sphere
    a_sphere = 4 * np.pi * r**2
    n_ele = n_beta**2
    a_ele = a_sphere / n_ele

    # alpha angles around the equator and the size of one step
    bin_alpha, delt_alpha = np.linspace(0, 2 * np.pi, n_alpha, retstep=True)

    # bins for beta axis in terms of axis coorindates between -1 and 1
    count = np.linspace(0.0, float(n_beta), 145)
    tmp = count * a_ele
    tmp /= r**2 * delt_alpha
    bin_beta = 1 - tmp
    if bin_beta[-1] < -1:
        bin_beta[-1] = -1

    bin_beta = np.arccos(bin_beta)
    return bin_alpha, bin_beta


def compute_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Computes the similarity of each embedding.

    Parameters
    ----------
    embeddings: np.ndarray
        Model embeddings.

    Returns
    -------
    smatrix: np.ndarray
        Similarity matrix.
    """

    n_runs = len(embeddings)
    smatrix = np.empty((n_runs, n_runs), dtype=np.float32)
    for ii in range(n_runs):
        for jj in range(n_runs):
            smatrix[ii, jj] = np.dot(embeddings[ii], embeddings[jj]) / np.sqrt(
                np.dot(embeddings[ii], embeddings[ii]) * np.dot(embeddings[jj], embeddings[jj])
            )

    return smatrix


def create_historgram(
    cloud: np.ndarray, sphere_axis: str = "Z", planar: bool = False
) -> BinnedStatistic2dResult:
    """Builds a histogram using the blocks of a sphered globe and returns a
    binned statistics result for two dimensions.

    Parameters
    ----------
    cloud: np.ndarray
        Point cloud around which we create an embedding.
    sphere_axis: str
        Axis of the sphere. This is aligned with the global axis system.
    planar: bool
        Set to true for planar point clouds and false for higher dimensions.

    Returns
    -------
    stats: BinnedStatistic2dResult
        Returns a 2D histogram of the sphere with bin numbers and bin statistics.
    """
    # casting to array because of typing
    centroid = np.array(np.mean(cloud, axis=0))

    qhull_options = ""
    if planar:
        qhull_options = "QJ"

    hull = ConvexHull(cloud, qhull_options=qhull_options)

    # we need to determine the largest distance in this point
    # cloud, so we can give the sphere a dimension
    # we can also create a sphere of random size but this could
    # skew the results
    dist = np.linalg.norm(hull.max_bound - hull.min_bound)

    bins_a, bins_b = create_sphere(dist)

    cloud_alpha, cloud_beta = to_spherical_coordinates(cloud, centroid, axis=sphere_axis)

    return binned_statistic_2d(
        cloud_alpha, cloud_beta, None, "count", bins=[bins_a, bins_b], expand_binnumbers=True
    )
