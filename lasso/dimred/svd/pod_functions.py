from typing import Tuple, Union

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from rich.progress import Progress, TaskID
from ...utils.richProgressBars import PlaceHolderBar


def svd_step_and_dim(s_mat: np.ndarray, k=10) -> np.ndarray:
    '''
    Performs a svds operation on the two dimensional s_mat

    Parameters
    ----------
    s_mat : ndarray
        2D array on which the svds operation shall be performed
    k : int, 10, optinal.
        The size of the POD

    Returns
    -------
    V : ndarray
        Array containing the right reduced order basis
    '''
    small_mat = csc_matrix(s_mat.astype(np.float64))

    _, _, V = svds(small_mat, k=k)

    V = V[::-1, :]

    return V


def calculate_V_and_Betas(
    stacked_sub_displ: np.ndarray,
    progress_bar: Union[None, Progress, PlaceHolderBar] = None,
    task_id: Union[None, TaskID] = None,
) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
    '''
    Calculates the right reduced order Basis V and up to 10 eigenvalues of the subsamples

    Parameters
    ----------
    stacked_sub_displ: np.ndarray
        np.ndarray containing all subsampled displacements
        shape must be (samples, timesteps, nodes, dims)

    Returns
    -------
    V_big: np.ndarray
        Reduced order basis to transform betas bag into subsamples
    betas: np.ndarray
        Projected simulation runs
    err_msg: str
        Error message if not enough samples where provided
    '''

    big_mat = stacked_sub_displ.reshape((stacked_sub_displ.shape[0], stacked_sub_displ.shape[1],
                                         stacked_sub_displ.shape[2]*stacked_sub_displ.shape[3]))

    diff_mat = np.stack([
        big_mat[:, 0, :] for _ in range(big_mat.shape[1])
    ]).reshape((big_mat.shape[0], big_mat.shape[1], big_mat.shape[2]))

    # We only want the difference in displacement
    big_mat = big_mat - diff_mat

    k = min(10, big_mat.shape[0]-1)
    if k < 1:
        return "Must provide more than 1 sample"

    if task_id is None and progress_bar:
        return "Progress requires a task ID"

    V_big = np.zeros(
        (k, big_mat.shape[1], big_mat.shape[2]))
    if progress_bar:
        progress_bar.advance(task_id)  # type: ignore
        for step in range(big_mat.shape[1]-1):
            V_big[:, step+1] = svd_step_and_dim(big_mat[:, step+1], k)
            progress_bar.advance(task_id)  # type: ignore
    else:
        for step in range(big_mat.shape[1]-1):
            V_big[:, step+1] = svd_step_and_dim(big_mat[:, step+1], k)

    betas_big = np.einsum('stn, ktn -> stk', big_mat, V_big)

    return V_big, betas_big
