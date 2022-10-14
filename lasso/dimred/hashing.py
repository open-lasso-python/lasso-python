import multiprocessing
import os
import time
from typing import List, Tuple, Union, Sequence

import h5py
import numpy as np
from scipy import integrate
from sklearn.neighbors import KDTree


from ..math.stochastic import jensen_shannon_entropy


def _match_modes(
    hashes1: np.ndarray,
    hashes2: np.ndarray,
    eigenvectors_sub1: np.ndarray,
    eigenvectors_sub2: np.ndarray,
):
    """Match the eigenvalue modes

    Parameters
    ----------
    hashes1 : np.ndarray
        hashes of first model
    hashes2 : np.ndarray
        hashes of second model
    eigenvectors_sub1 : np.ndarray
        eigenvector field of first model
    eigenvectors_sub2 : np.ndarray
        eigenvector field of second model

    Returns
    -------
    matches : list(tuple(int.int))
        indexes of the matched modes
    """

    matches = []
    mode1_hash_indexes = list(range(len(hashes1)))
    mode2_hash_indexes = list(range(len(hashes2)))

    for i_hash in mode1_hash_indexes:

        field1 = eigenvectors_sub1[:, i_hash]

        found_match = False
        for j_entry, j_hash in enumerate(mode2_hash_indexes):

            field2 = eigenvectors_sub2[:, j_hash]

            if is_mode_match(field1, field2):
                matches.append((i_hash, j_hash))
                del mode2_hash_indexes[j_entry]
                found_match = True
                break

        if not found_match:
            break

    return matches


def is_orientation_flip_required(eigenvectors1: np.ndarray, eigenvectors2: np.ndarray):
    """Checks whether the eigenfields require to be flipped

    Parameters
    ----------
    eigenvectors1 : np.ndarray
        eigenvector_field of mesh1.
    eigenvectors2 : np.ndarray
        eigenvector_field of mesh2.

    Returns
    -------
    flip_required : bool or list(bool)
        whether a flip of the eigenvector field is required.

    Note
    ----
        If the eigenvector field has multiple modes (e.g. shape n_nodes,n_modes)
        then for every mode a flip value is returned

        The eigenmodes require switching if the dot product of the knn-eigenfields yield
        a negative result.
    """
    assert eigenvectors1.shape == eigenvectors2.shape

    # one eigenmode only
    if eigenvectors1.ndim == 1:
        knn_error_basic = np.dot(eigenvectors1, eigenvectors2)
        return knn_error_basic < 0

    # multiple eigenmodes
    n_modes = min(eigenvectors1.shape[1], eigenvectors2.shape[1])
    errors = [
        np.dot(eigenvectors1[:, i_mode], eigenvectors2[:, i_mode]) for i_mode in range(n_modes)
    ]

    return np.array([err < 0 for err in errors])


def _compute_mode_similarities(
    hashes1: np.ndarray,
    hashes2: np.ndarray,
    eigenvectors_sub1: np.ndarray,
    eigenvectors_sub2: np.ndarray,
    matches: List[Tuple[int, int]],
) -> List[float]:
    """Compute the mode similarity between different meshes

    Parameters
    ----------
    hashes1 : np.ndarray
        hashes of first model
    hashes2 : np.ndarray
        hashes of second model
    eigenvectors_sub1 : np.ndarray
        eigenvector field of first model
    eigenvectors_sub2 : np.ndarray
        eigenvector field of second model
    matches : list(tuple(int, int))
        matches of modes (every match will be computed)

    Returns
    -------
    mode_similarities : list(float)
        similarities of the matched modes

    Notes
    -----
        This function cannot deal with unequal sampling of the input hashes.
    """

    mode_similarities = []
    for i_hash, j_hash in matches:

        assert hashes1.shape[2] == hashes2.shape[2]

        field1 = eigenvectors_sub1[:, i_hash]
        field2 = eigenvectors_sub2[:, j_hash]

        # flip orientation of eigenvector field and hash if required
        if is_orientation_flip_required(field1, field2):
            field2 *= -1
            # hdf5 can not handle negative slicing
            mode_ = np.array(hashes2[j_hash, 1], copy=True)
            mode_ = mode_[::-1]
        else:
            mode_ = hashes2[j_hash, 1, :]

        # Warning: x is usually originally hashes[i_mode, 0]
        x = np.linspace(0, 1, hashes1.shape[2])
        norm1 = curve_normalizer(x, hashes1[i_hash, 1])
        norm2 = curve_normalizer(x, mode_)
        if norm1 != 0 and norm2 != 0:
            mode_similarities.append(
                integrate.simps(hashes1[i_hash, 1] * mode_ / np.sqrt(norm1 * norm2), x=x)
            )
        else:
            mode_similarities.append(0)

    return mode_similarities


def _join_hash_comparison_thread_files(
    comparison_filepath: str, thread_filepaths: Sequence[str], n_runs: int
):
    # pylint: disable = too-many-locals

    if os.path.exists(comparison_filepath):
        if os.path.isfile(comparison_filepath):
            os.remove(comparison_filepath)
        else:
            raise OSError("Can not delete directory", comparison_filepath)

    with h5py.File(comparison_filepath, "w") as hdf5_file:
        smatrix = hdf5_file.create_dataset(
            "similarity_matrix",
            shape=(n_runs, n_runs, 25),
            maxshape=(n_runs, n_runs, None),
            dtype="float64",
            compression="gzip",
        )
        ds_matches = hdf5_file.create_dataset(
            "matches",
            shape=(n_runs, n_runs, 25, 2),
            maxshape=(n_runs, n_runs, None, 2),
            dtype="int64",
            compression="gzip",
        )
        ds_weights = hdf5_file.create_dataset(
            "weights",
            shape=(n_runs, n_runs, 25),
            maxshape=(n_runs, n_runs, None),
            dtype="float64",
            compression="gzip",
        )

        for thread_filepath in thread_filepaths:

            # open thread file
            with h5py.File(thread_filepath, "r") as thread_file:

                # insert matrix entries
                matrix_indexes = thread_file["matrix_indexes"]
                matrix_similarities = thread_file["matrix_similarities"]
                matrix_matches = thread_file["matrix_matches"]
                thread_weights = thread_file["weights"]

                for (i_row, i_col), values, matches in zip(
                    matrix_indexes, matrix_similarities, matrix_matches
                ):
                    smatrix[i_row, i_col] = values
                    ds_matches[i_row, i_col] = matches
                    ds_weights[i_row, i_col] = (thread_weights[i_row] + thread_weights[i_col]) / 2

            # delete thread file
            os.remove(thread_filepath)


def run_hash_comparison(
    comparison_filepath: str,
    hashes_filepaths: List[str],
    n_threads: int = 1,
    print_progress: bool = False,
):
    """Compare two hashes of a simulation run part

    Parameters
    ----------
    comparison_filepath: str
        filepath to the hdf5 in which the result of the comparison will be
        stored
    hashes_filepaths: List[str]
        filepath to the stored hashes
    n_threads: int
        number of threads used for the comparison
    print_progress: bool
        whether to print the progress
    """

    # pylint: disable = too-many-locals, too-many-statements

    assert n_threads > 0

    # fixed settings
    hdf5_dataset_compression = "gzip"

    # ! this is an inlined function !
    # the actual function starts way much down

    def _threading_run_comparison(run_indices, comparison_filepath, comm_q):
        # pylint: disable = too-many-statements

        n_comparisons_thread = len(run_indices)

        # setup storage file
        if os.path.exists(comparison_filepath):
            if os.path.isfile(comparison_filepath):
                os.remove(comparison_filepath)
            else:
                raise OSError("Can not delete directory", comparison_filepath)

        hdf5_file = h5py.File(comparison_filepath, "w")

        max_len = np.max([len(entry) for entry in hashes_filepaths])
        hashes_filepaths_ascii = [entry.encode("ascii", "ignore") for entry in hashes_filepaths]

        hdf5_file.require_dataset(
            "filepaths",
            data=hashes_filepaths_ascii,
            shape=(len(hashes_filepaths_ascii), 1),
            dtype=f"S{max_len}",
        )

        n_modes_estimated = 25

        # could be compressed to one per run only!
        ds_weights = hdf5_file.create_dataset(
            "weights",
            (n_runs, n_modes_estimated),
            maxshape=(n_runs, None),
            dtype="float64",
            compression=hdf5_dataset_compression,
        )
        ds_matrix_indexes = hdf5_file.create_dataset(
            "matrix_indexes",
            (n_comparisons_thread, 2),
            dtype="float64",
            compression=hdf5_dataset_compression,
        )
        ds_matrix_values = hdf5_file.create_dataset(
            "matrix_similarities",
            (n_comparisons_thread, n_modes_estimated),
            maxshape=(n_comparisons_thread, None),
            dtype="float64",
            compression=hdf5_dataset_compression,
        )

        # info only!
        ds_matrix_matches = hdf5_file.create_dataset(
            "matrix_matches",
            (n_comparisons_thread, n_modes_estimated, 2),
            maxshape=(n_comparisons_thread, None, 2),
            dtype="int64",
            compression=hdf5_dataset_compression,
        )

        def _save_data(computed_results, counter):

            start = counter + 1 - len(computed_results)
            for i_result, result in enumerate(computed_results):

                i_run, j_run = result["matrix_index"]
                similarities = result["similarities"]
                matches_tmp = result["matches"]

                ds_matrix_indexes[start + i_result, :] = i_run, j_run
                ds_matrix_values[start + i_result, : len(similarities)] = similarities
                ds_matrix_matches[start + i_result, : len(matches_tmp)] = matches_tmp
                weights1 = result["weights1"]
                n_weights1 = len(weights1)
                ds_weights[i_run, :n_weights1] = weights1
                weights2 = result["weights2"]
                n_weights2 = len(weights2)
                ds_weights[j_run, :n_weights2] = weights2

            computed_results.clear()

        # log
        computation_times = []
        io_times = []

        counter = None  # bugfix
        computed_results = []
        for counter, (i_run, j_run) in enumerate(run_indices):

            start = time.time()

            # get data (io)
            fp1 = h5py.File(hashes_filepaths[i_run], "r")
            fp2 = h5py.File(hashes_filepaths[j_run], "r")
            hashes1 = fp1["hashes"]
            hashes2 = fp2["hashes"]
            xyz1, xyz2 = fp1["subsample_xyz"], fp2["subsample_xyz"]
            eigenvalues1, eigenvalues2 = fp1["eigenvalues"], fp2["eigenvalues"]

            # hdf5 can only handle increasing indexes ... thus we need to copy the field
            eigenvectors_sub1 = np.array(fp1["eigenvectors"], copy=True)
            eigenvectors_sub2 = np.array(fp2["eigenvectors"], copy=True)

            # time
            io_times.append(time.time() - start)
            start = time.time()

            # match points roughly in xyz
            tree = KDTree(xyz1)
            knn_indexes = tree.query(xyz2, return_distance=False, k=1)
            eigenvectors_sub1 = np.squeeze(eigenvectors_sub1[knn_indexes])

            # match modes
            matches = _match_modes(hashes1, hashes2, eigenvectors_sub1, eigenvectors_sub2)

            # mode weights
            weights1 = get_mode_weights_inv(eigenvalues1)
            weights2 = get_mode_weights_inv(eigenvalues2)

            # compute mode similarity
            mode_similarities = _compute_mode_similarities(
                hashes1, hashes2, eigenvectors_sub1, eigenvectors_sub2, matches
            )

            # time
            computation_times.append(time.time() - start)

            # assemble computations
            computation_result = {
                "matrix_index": [i_run, j_run],
                "matches": matches,  # info only
                "similarities": mode_similarities,
                "weights1": weights1.tolist(),
                "weights2": weights2.tolist(),
            }
            computed_results.append(computation_result)

            # save to file occasionally
            if counter % 500 == 0:
                _save_data(computed_results, counter)

            # print status
            if comm_q and not comm_q.full():
                comm_q.put(
                    {
                        "i_entry": counter + 1,
                        "n_entries": len(run_indices),
                        "io_time": np.mean(io_times),
                        "computation_time": np.mean(computation_times),
                    },
                    False,
                )

        # dump at end (if anything was computed)
        if counter:
            _save_data(computed_results, counter)

    # <-- FUNCTION STARTS HERE

    # helper vars
    n_runs = len(hashes_filepaths)

    # THREADS
    if n_threads == 1:
        matrix_entries = []
        for i_run in range(n_runs):
            for j_run in range(i_run + 1, n_runs):
                matrix_entries.append((i_run, j_run))
        _threading_run_comparison(matrix_entries, comparison_filepath, None)
    else:
        # enlist runs
        thread_matrix_entries = [[] for i_thread in range(n_threads)]
        i_thread = 0
        for i_run in range(n_runs):
            for j_run in range(i_run + 1, n_runs):
                thread_matrix_entries[i_thread % n_threads].append((i_run, j_run))
                i_thread += 1

        # comm queues
        queues = [multiprocessing.Queue(1) for i_thread in range(n_threads)]

        # run threads
        thread_filepaths = [
            comparison_filepath + f"_thread{i_thread}" for i_thread in range(n_threads)
        ]
        threads = [
            multiprocessing.Process(
                target=_threading_run_comparison,
                args=(matrix_indexes, thread_filepaths[i_thread], queues[i_thread]),
            )
            for i_thread, matrix_indexes in enumerate(thread_matrix_entries)
        ]
        for thread in threads:
            thread.start()

        # logging
        if print_progress:
            thread_stats = [
                {
                    "i_entry": 0,
                    "n_entries": len(thread_matrix_entries[i_thread]),
                    "io_time": 0,
                    "computation_time": 0,
                }
                for i_thread in range(n_threads)
            ]

            while any(thread.is_alive() for thread in threads):

                # fetch data from channel
                for i_thread, comm_q in enumerate(queues):
                    if not comm_q.empty():
                        thread_stats[i_thread] = comm_q.get(False)

                # print msg
                # pylint: disable = consider-using-f-string
                thread_msg_list = [
                    (
                        f"Thread {i_thread}: "
                        f"{(100 * stats['i_entry'] / stats['n_entries']):.1f}% "
                        f"({stats['i_entry']}/{stats['n_entries']}) "
                        f"{stats['computation_time']:.2f}s | "
                    )
                    for i_thread, stats in enumerate(thread_stats)
                ]
                msg = "| " + "".join(thread_msg_list) + "\r"
                print(msg, end="")
                time.sleep(0.35)

            # print completion message
            thread_msg_list = [
                (
                    f"Thread {i_thread}: "
                    f"{(100 * stats['i_entry'] / stats['n_entries']):.1f}% "
                    f"({stats['i_entry']}/{stats['n_entries']}) "
                    f"{stats['computation_time']:.2f}s | "
                )
                for i_thread, stats in enumerate(thread_stats)
            ]
            msg = "| " + "".join(thread_msg_list) + "\r"
            print(msg, end="")

            print("")
            print("done.")

        # join thread worker files
        for thread in threads:
            thread.join()
        _join_hash_comparison_thread_files(comparison_filepath, thread_filepaths, n_runs)


def is_mode_match(
    eigenvectors1: np.ndarray,
    eigenvectors2: np.ndarray,
    knn_indexes: Union[np.ndarray, None] = None,
):
    """Detect a mode match from the eigenvector field

    Parameters
    ----------
    eigenvectors1 : np.ndarray
        subsample eigenvector field from model 1
    eigenvectors2 : np.ndarray
        subsample eigenvector field from model 2
    knn_indexes : np.ndarray
        kdd_indexes obtained for matching xyz1 and xyz2 of the eigenvectorfields
        so that only the coordinates of near points will be compared

    Returns
    -------
    is_matched : bool

    Notes
    -----
        A mode match is detected by watching the distribution
        of the eigenvector field errors. In case of a mode switch
        a correct orientation of the field is (obviously) not possible.
        In such a case will the probability distribution of the
        basic error and inverted error be quite similar, since both are wrong.

        A matching orientation (empirically) seems to have a normal
        distribution like character. A non-matching orientation
        is more like a uniform distribution (constantly wrong across
        the entire model).
    """

    # pylint: disable = too-many-locals

    # if the jensen-shannon-divergence is below this value
    # then a mode switch is assumed
    distance_limit = 0.1

    # number of bins for probability distribution
    n_bins = 25

    # (1) match sub-samples in xyz
    # tree = KDTree(xyz1)
    # indexes = tree.query(xyz2, return_distance=False, k=1)

    # (2) compute field errors of normal field and inverted field
    if knn_indexes:
        tmp1 = eigenvectors1[knn_indexes].flatten() - eigenvectors2
        tmp2 = eigenvectors1[knn_indexes].flatten() + eigenvectors2
    else:
        tmp1 = eigenvectors1 - eigenvectors2
        tmp2 = eigenvectors1 + eigenvectors2

    # (3) create a probability distribution for each error vector

    # bin the values
    xmin = min(tmp1.min(), tmp2.min())
    xmax = max(tmp1.max(), tmp2.max())
    bins = np.linspace(xmin, xmax, n_bins)
    indexes_p1 = np.digitize(tmp1, bins)
    indexes_p2 = np.digitize(tmp2, bins)
    p1 = np.bincount(indexes_p1) / len(tmp1)
    p2 = np.bincount(indexes_p2) / len(tmp2)

    # align bin vector size
    p1_tmp = np.zeros(max(len(p1), len(p2)))
    p2_tmp = np.zeros(max(len(p1), len(p2)))
    p1_tmp[: len(p1)] = p1
    p2_tmp[: len(p2)] = p2
    p1 = p1_tmp
    p2 = p2_tmp

    # compute similarity
    similarity_js = jensen_shannon_entropy(p1, p2)

    return similarity_js > distance_limit


def get_mode_weights_inv(vec: np.ndarray):
    """Inverse value weights (higher decay than softmax)"""
    val = 1.0 / (vec[:])
    return val / np.sum(val)


def curve_normalizer(x: np.ndarray, y: np.ndarray):
    """Compute the curve normalizer for a curve dot product

    Parameters
    ----------
    x : np.ndarray
        array of x values
    y : np.ndarray
        array of y values

    Returns
    -------
    norm : float
        normalizing factor
    """
    return integrate.simps(y**2, x=x)


def compute_hashes(
    eig_vecs: np.ndarray,
    result_field: np.ndarray,
    n_points: int = 100,
    bandwidth: float = 0.05,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute hashes for a result field

    Parameters
    ----------
    eig_vecs : np.ndarray
        eigenvector field of the component with (n_samples, n_modes)
    result_field : np.ndarray
        result field to hash
    n_points : resolution of the hash
        Number of equidistant points to use for smoothing.
        Should be determined from the mesh size (2.5 times average elem size).
    bandwidth : float
        Bandwidth in percent of the kernel.
        Recommended as 5 times global element size median.

    Returns
    -------
    hash_functions : list(tuple(np.ndarray, np.ndarray))
        list of the computed hash functions. Every item is the hash for
        an eigenmode. The hash consists of a pair of two functions: (x,y).
        For comparison, only y is usually used.
    """

    assert eig_vecs.shape[0] == len(result_field), f"{eig_vecs.shape[0]} != {len(result_field)}"

    # Note: needs to be vectorized to speed it up

    hash_functions = []
    for i_eigen in range(eig_vecs.shape[1]):

        xmin = eig_vecs[:, i_eigen].min()
        xmax = eig_vecs[:, i_eigen].max()

        x = np.linspace(xmin, xmax, n_points)
        y = np.zeros(n_points)

        local_bandwidth = bandwidth * (xmax - xmin)
        c = -0.5 / local_bandwidth**2

        for ii, point in enumerate(x):
            y[ii] = np.dot(result_field, np.exp(c * np.square(point - eig_vecs[:, i_eigen])))
        y /= np.sqrt(2 * np.pi) * bandwidth

        hash_functions.append((x, y))

    return hash_functions
