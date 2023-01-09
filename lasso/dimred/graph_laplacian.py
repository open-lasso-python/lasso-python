from typing import Union

import numpy as np
from scipy.sparse import csgraph, dok_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import KDTree


def run_graph_laplacian(
    points: np.ndarray,
    n_eigenmodes: int = 5,
    min_neighbors: Union[int, None] = None,
    sigma: Union[float, None] = None,
    search_radius: Union[float, None] = None,
):
    """
    Compute a graph laplacian.

    Parameters
    ----------
    points : np.ndarray
        points with features
    n_eigenmodes : int
        number of eigenmodes to compute
    min_neighbors : int
        The minimum number of neighbors of a point to be considered for the laplacian
        weights. Can be used to avoid unconnected points.
    sigma : float
        The standard deviation of the gaussian normal distribution function used to
        transform the distances for the inverse distance based weighting.
    search_radius:


    Returns
    -------
    eigenvalues : np.ndarray
        eigenvalues from the graph
    eigenvectors : np.ndarray
        eigenvectors with shape (n_points x n_eigenvectors)
    """
    with np.warnings.catch_warnings():
        regex_string = (
            r"the matrix subclass is not the recommended way to represent"
            + r"matrices or deal with linear algebra"
        )
        np.warnings.filterwarnings("ignore", regex_string)
        lapl = _laplacian_gauss_idw(points, min_neighbors, sigma, search_radius)
        return _laplacian(lapl, n_eigenmodes)


def _laplacian_gauss_idw(
    points: np.ndarray,
    min_neighbors: Union[int, None] = None,
    sigma: Union[float, None] = None,
    search_radius: Union[float, None] = None,
):
    """
    Calculates the laplacian matrix for the sample points of a manifold. The inverse
    of the gauss-transformed distance is used as weighting of the neighbors.

    Parameters
    ----------
    points: array-like, shape (n_points, n_components) :
      The sampling points of a manifold.
    min_neighbors: int
      The minimum number of neighbors of a point to be considered for the laplacian
      weights. Can be used to avoid unconnected points.
    sigma: float
      The standard deviation of the gaussian normal distribution function used to
      transform the distances for the inverse distance based weighting.
    search_radius : float
        radius search parameter for nearest neighbors

    Returns
    -------
    L: array-like, shape (n_points, n_points)
      The laplacian matrix for manifold given by its sampling `points`.
    """
    assert 2 == points.ndim

    if min_neighbors is None:
        min_neighbors = points.shape[1]

    tree = KDTree(points)

    if sigma is None:
        d, _ = tree.query(points, 2 + 2 * points.shape[1], return_distance=True)
        sigma = np.sum(d[:, -2:])
        sigma /= 3 * len(points)

    if search_radius is None:
        search_radius = 3 * sigma

    graph = dok_matrix((len(points), len(points)), dtype=np.double)

    for i, (j, d, e, k) in enumerate(
        zip(
            *tree.query_radius(points, return_distance=True, r=search_radius),
            *tree.query(points, return_distance=True, k=1 + min_neighbors)
        )
    ):
        # Always search for k neighbors, this prevents strongly connected local areas
        # a little, attracting the eigenfield

        d, j = e, k
        k = j != i
        d, j = d[k], j[k]
        d **= 2
        d /= -2 * sigma**2
        graph[i, j] = d = np.exp(d)
        graph[j, i] = d[:, np.newaxis]

    assert 0 == (graph != graph.T).sum()

    return csgraph.laplacian(graph, normed=True)


def _laplacian(lapl: csgraph, n_eigenmodes: int = 5):
    """
    Compute the laplacian of a graph L

    Parameters
    ----------
    L : csgraph
        sparse cs graph from scipy
    n_eigenmodes : int
        number of eigenmodes to compute
    points : np.ndarray
        coordintes of graph nodes (only for plotting)

    Returns
    -------
    eigen_values : np.ndarray
        eingenvalues of the graph
    eigen_vecs : np.ndarray
        eigenvectors of each graph vector (iNode x nEigenmodes)
    """

    n_nonzero_eigenvalues = 0
    n_eigenvalues = int(n_eigenmodes * 1.5)

    eigen_vals = np.empty((0,))
    eigen_vecs = np.empty((0, 0))

    while n_nonzero_eigenvalues < n_eigenmodes:

        eigen_vals, eigen_vecs = map(np.real, eigsh(lapl, n_eigenvalues, which="SA"))

        i_start = np.argmax(eigen_vals > 1e-7)
        n_nonzero_eigenvalues = len(eigen_vals) - i_start

        if n_nonzero_eigenvalues >= n_eigenmodes:
            eigen_vecs = eigen_vecs[:, i_start : i_start + n_eigenmodes]
            eigen_vals = eigen_vals[i_start : i_start + n_eigenmodes]

        n_eigenvalues = int(n_eigenvalues * 1.5)

    return eigen_vals, eigen_vecs
