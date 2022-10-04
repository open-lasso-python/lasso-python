import random
from typing import Union
import numpy as np
from sklearn.neighbors import KDTree


def unique_subsamples(start: int, end: int, n_samples: int, seed=None) -> np.ndarray:
    """Retrieve unique subsample indexes

    Parameters
    ----------
    start: int
        starting index of population
    end: int
        ending index of population (end <= start)
    n_samples: int
        number of samples to draw
    seed: int
        seed for random number generator

    Returns
    -------
    indexes: np.ndarray
        unique sample indexes
    """
    assert start <= end

    if end - start < n_samples:
        n_samples = end - start

    random.seed(seed)
    indexes = np.array(random.sample(range(start, end), n_samples), dtype=np.int64)
    random.seed()
    return indexes


def homogenize_density(
    points: np.ndarray,
    dim: int = 2,
    target_distance: Union[float, None] = None,
    n_neighbors: int = 18,
    seed=None,
) -> np.ndarray:
    """homogenize a cloud density by probabilities

    Parameters
    ----------
    points: np.ndarray
        point cloud
    dim: int
        intrinsic dimension of the data
    target_distance: float
        target distance to aim for
    n_neighbors: int
        neighbors used for computation of average neighborhood distance
    seed: int
        seed for random number generator

    Returns
    -------
    is_selected: np.ndarray
        boolean array indicating which subsamples were selected
    """
    n_neighbors = min(n_neighbors, len(points))

    random.seed(seed)
    d, _ = KDTree(points).query(points, k=n_neighbors + 1)
    d_average = np.average(d[:, 1:], axis=1)
    if target_distance is None:
        target_distance = np.median(d_average)
    is_selected = np.array(
        [
            dist >= target_distance or random.random() < (dist / target_distance) ** dim
            for i, dist in enumerate(d_average)
        ]
    )
    random.seed()
    return is_selected
