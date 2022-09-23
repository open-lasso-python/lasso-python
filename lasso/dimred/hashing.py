
import copy
import multiprocessing
import os
import re
import time
from typing import List, Tuple, Union, Sequence

import h5py
import numpy as np
from scipy import integrate
from sklearn.neighbors import KDTree

import tensorflow as tf

from ..math.sampling import homogenize_density, unique_subsamples
from ..math.stochastic import jensen_shannon_entropy
from ..mesh.dyna import get_results_from_dyna_hdf5
from .graph_laplacian import run_graph_laplacian


def _match_modes(hashes1: np.ndarray,
                 hashes2: np.ndarray,
                 eigenvectors_sub1: np.ndarray,
                 eigenvectors_sub2: np.ndarray):
    ''' Match the eigenvalue modes

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
    '''

    matches = []
    mode1_hash_indexes = [iHash for iHash in range(len(hashes1))]
    mode2_hash_indexes = [iHash for iHash in range(len(hashes2))]

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
    ''' Checks whether the eigenfields require to be flipped

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
    '''
    assert(eigenvectors1.shape == eigenvectors2.shape)

    # one eigenmode only
    if eigenvectors1.ndim == 1:

        knn_error_basic = np.dot(eigenvectors1, eigenvectors2)
        return knn_error_basic < 0

    # multiple eigenmodes
    else:
        n_modes = min(eigenvectors1.shape[1], eigenvectors2.shape[1])
        errors = [np.dot(eigenvectors1[:, i_mode], eigenvectors2[:, i_mode])
                  for i_mode in range(n_modes)]

        return np.array([err < 0 for err in errors])


def _compute_mode_similarities(hashes1: np.ndarray,
                               hashes2: np.ndarray,
                               eigenvectors_sub1: np.ndarray,
                               eigenvectors_sub2: np.ndarray,
                               matches: List[Tuple[int, int]]) -> List[float]:
    ''' Compute the mode similarity between different meshes

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
        similarities of the the matched modes
    '''

    # TODO integrate functions with unequal sampling

    mode_similarities = []
    for i_hash, j_hash in matches:

        assert(hashes1.shape[2] == hashes2.shape[2])

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

        # TODO Warning x is usually originally hashes[i_mode, 0]
        x = np.linspace(0, 1, hashes1.shape[2])
        norm1 = curve_normalizer(x, hashes1[i_hash, 1])
        norm2 = curve_normalizer(x, mode_)
        if norm1 != 0 and norm2 != 0:
            mode_similarities.append(
                integrate.simps(
                    hashes1[i_hash, 1] * mode_ / np.sqrt(norm1 * norm2), x=x)
            )
        else:
            mode_similarities.append(0)

    return mode_similarities


def _join_hash_comparison_thread_files(comparison_filepath: str,
                                       thread_filepaths: Sequence[str],
                                       n_runs: int):

    if os.path.exists(comparison_filepath):
        if os.path.isfile(comparison_filepath):
            os.remove(comparison_filepath)
        else:
            raise OSError("Can not delete directory", comparison_filepath)

    with h5py.File(comparison_filepath, 'w') as hdf5_file:
        smatrix = hdf5_file.create_dataset('similarity_matrix',
                                           shape=(n_runs, n_runs, 25),
                                           maxshape=(
                                               n_runs, n_runs, None),
                                           dtype="float64",
                                           compression='gzip')
        ds_matches = hdf5_file.create_dataset('matches',
                                              shape=(n_runs, n_runs, 25, 2),
                                              maxshape=(
                                                  n_runs, n_runs, None, 2),
                                              dtype="int64",
                                              compression='gzip')
        ds_weights = hdf5_file.create_dataset('weights',
                                              shape=(n_runs, n_runs, 25),
                                              maxshape=(
                                                  n_runs, n_runs, None),
                                              dtype="float64",
                                              compression='gzip')

        for i_thread in range(len(thread_filepaths)):

            # open thread file
            with h5py.File(thread_filepaths[i_thread], 'r') as thread_file:

                # insert matrix entries
                matrix_indexes = thread_file['matrix_indexes']
                matrix_similarities = thread_file['matrix_similarities']
                matrix_matches = thread_file['matrix_matches']
                thread_weights = thread_file['weights']

                for (i_row, i_col), values, matches in zip(matrix_indexes,
                                                           matrix_similarities,
                                                           matrix_matches):
                    smatrix[i_row, i_col] = values
                    ds_matches[i_row, i_col] = matches
                    ds_weights[i_row, i_col] = (
                        thread_weights[i_row] + thread_weights[i_col]) / 2

            # delete thread file
            os.remove(thread_filepaths[i_thread])


def run_hash_comparison(comparison_filepath: str,
                        hashes_filepaths: List[str],
                        n_threads: int = 1,
                        print_progress: bool = False):
    '''
    '''
    assert(n_threads > 0)

    # fixed settings
    hdf5_dataset_compression = "gzip"

    # ! this is an inlined function !
    # the actual function starts way much down

    def _threading_run_comparison(run_indices, comparison_filepath, comm_q):

        n_comparisons_thread = len(run_indices)

        # setup storage file
        if os.path.exists(comparison_filepath):
            if os.path.isfile(comparison_filepath):
                os.remove(comparison_filepath)
            else:
                raise OSError("Can not delete directory", comparison_filepath)

        hdf5_file = h5py.File(comparison_filepath, 'w')

        max_len = np.max([len(entry) for entry in hashes_filepaths])
        hashes_filepaths_ascii = [entry.encode("ascii", "ignore")
                                  for entry in hashes_filepaths]

        hdf5_file.require_dataset(
            'filepaths',
            data=hashes_filepaths_ascii,
            shape=(len(hashes_filepaths_ascii), 1),
            dtype='S{}'.format(max_len))

        n_modes_estimated = 25

        # could be compressed to one per run only!
        ds_weights = hdf5_file.create_dataset('weights', (n_runs, n_modes_estimated),
                                              maxshape=(n_runs, None),
                                              dtype='float64',
                                              compression=hdf5_dataset_compression)
        ds_matrix_indexes = hdf5_file.create_dataset('matrix_indexes',
                                                     (n_comparisons_thread, 2),
                                                     dtype='float64',
                                                     compression=hdf5_dataset_compression)
        ds_matrix_values = hdf5_file.create_dataset('matrix_similarities',
                                                    (n_comparisons_thread,
                                                     n_modes_estimated),
                                                    maxshape=(
                                                        n_comparisons_thread, None),
                                                    dtype='float64',
                                                    compression=hdf5_dataset_compression)

        # info only!
        ds_matrix_matches = hdf5_file.create_dataset('matrix_matches',
                                                     (n_comparisons_thread,
                                                      n_modes_estimated,
                                                      2),
                                                     maxshape=(
                                                         n_comparisons_thread, None, 2),
                                                     dtype='int64',
                                                     compression=hdf5_dataset_compression)

        def _save_data(computed_results, hdf5_file, counter):

            start = counter + 1 - len(computed_results)
            for i_result, result in enumerate(computed_results):

                i_run, j_run = result['matrix_index']
                similarities = result['similarities']
                matches_tmp = result["matches"]

                ds_matrix_indexes[start + i_result, :] = i_run, j_run
                ds_matrix_values[start + i_result,
                                 :len(similarities)] = similarities
                ds_matrix_matches[start + i_result,
                                  :len(matches_tmp)] = matches_tmp
                weights1 = result['weights1']
                n_weights1 = len(weights1)
                ds_weights[i_run, :n_weights1] = weights1
                weights1 = result['weights2']
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
            fp1 = h5py.File(hashes_filepaths[i_run], 'r')
            fp2 = h5py.File(hashes_filepaths[j_run], 'r')
            hashes1 = fp1['hashes']
            hashes2 = fp2['hashes']
            xyz1, xyz2 = fp1['subsample_xyz'], fp2['subsample_xyz']
            eigenvalues1, eigenvalues2 = fp1['eigenvalues'], fp2['eigenvalues']

            # hdf5 can only handle increasing indexes ... thus we need to copy the field
            eigenvectors_sub1 = np.array(fp1['eigenvectors'], copy=True)
            eigenvectors_sub2 = np.array(fp2['eigenvectors'], copy=True)
            # subsample_indexes1, subsample_indexes2 = fp1['subsample_indexes'], fp2['subsample_indexes']

            # subfields of eigenvectors
            # eigenvectors_sub1 = np.squeeze(
            #     eigenvectors1[subsample_indexes1, :])
            # eigenvectors_sub2 = np.squeeze(
            #     eigenvectors2[subsample_indexes2, :])

            # time
            io_times.append(time.time() - start)
            start = time.time()

            # match points roughly in xyz
            tree = KDTree(xyz1)
            knn_indexes = tree.query(xyz2, return_distance=False, k=1)
            eigenvectors_sub1 = np.squeeze(eigenvectors_sub1[knn_indexes])

            # match modes
            matches = _match_modes(
                hashes1,
                hashes2,
                eigenvectors_sub1,
                eigenvectors_sub2)

            # mode weights
            weights1 = get_mode_weights_inv(eigenvalues1)
            weights2 = get_mode_weights_inv(eigenvalues2)

            # compute mode similarity
            mode_similarities = _compute_mode_similarities(
                hashes1,
                hashes2,
                eigenvectors_sub1,
                eigenvectors_sub2,
                matches)

            # time
            computation_times.append(time.time() - start)

            # assemble computations
            computation_result = {
                'matrix_index': [i_run, j_run],
                'matches': matches,  # info only
                'similarities': mode_similarities,
                'weights1': weights1.tolist(),
                'weights2': weights2.tolist(),
            }
            computed_results.append(computation_result)

            # save to file occasionally
            if counter % 500 == 0:
                _save_data(computed_results, hdf5_file, counter)

            # print status
            if comm_q and not comm_q.full():
                comm_q.put({
                    'i_entry': counter + 1,
                    'n_entries': len(run_indices),
                    'io_time': np.mean(io_times),
                    'computation_time': np.mean(computation_times),
                }, False)

        # dump at end (if anything was computed)
        if counter:
            _save_data(computed_results, hdf5_file, counter)

    # <-- FUNCTION STARTS HERE

    # helper vars
    n_runs = len(hashes_filepaths)

    # THREADS
    if n_threads == 1:
        matrix_entries = []
        for i_run in range(n_runs):
            for j_run in range(i_run + 1, n_runs):
                matrix_entries.append((i_run, j_run))
        _threading_run_comparison(matrix_entries,
                                  comparison_filepath,
                                  None)
    else:
        # enlist runs
        thread_matrix_entries = [[] for i_thread in range(n_threads)]
        i_thread = 0
        for i_run in range(n_runs):
            for j_run in range(i_run + 1, n_runs):
                thread_matrix_entries[i_thread %
                                      n_threads].append((i_run, j_run))
                i_thread += 1

        # comm queues
        queues = [multiprocessing.Queue(1) for i_thread in range(n_threads)]

        # run threads
        thread_filepaths = [comparison_filepath + "_thread%d" %
                            i_thread for i_thread in range(n_threads)]
        threads = [multiprocessing.Process(target=_threading_run_comparison,
                                           args=(matrix_indexes,
                                                 thread_filepaths[i_thread],
                                                 queues[i_thread]))
                   for i_thread, matrix_indexes in enumerate(thread_matrix_entries)]
        [thread.start() for thread in threads]

        # logging
        if print_progress:
            thread_stats = [{
                'i_entry': 0,
                'n_entries': len(thread_matrix_entries[i_thread]),
                'io_time': 0,
                'computation_time': 0,
            } for i_thread in range(n_threads)]

            while any(thread.is_alive() for thread in threads):

                # fetch data from channel
                for i_thread, comm_q in enumerate(queues):
                    if not comm_q.empty():
                        thread_stats[i_thread] = comm_q.get(False)

                # print msg
                msg = '| ' + "".join("Thread {0}: {1}% ({2}/{3}) {4}s | ".format(
                    i_thread,
                    '%.1f' % (100 * stats['i_entry'] / stats['n_entries'],),
                    stats['i_entry'],
                    stats['n_entries'],
                    '%.2f' % stats['computation_time'],)
                    for i_thread, stats in enumerate(thread_stats)) + "\r"
                print(msg, end='')
                time.sleep(0.35)

            # print completion message
            msg = '| ' + "".join("Thread {0}: {1}% ({2}/{3}) {4}s | ".format(
                i_thread,
                '%d' % 100,
                stats['n_entries'],
                stats['n_entries'],
                '%.2f' % stats['computation_time'],)
                for i_thread, stats in enumerate(thread_stats)) + "\r"
            print(msg, end='')

            print("")
            print("done.")

        # join thread worker files
        [thread.join() for thread in threads]
        _join_hash_comparison_thread_files(comparison_filepath, thread_filepaths, n_runs)


def is_mode_match(eigenvectors1: np.ndarray,
                  eigenvectors2: np.ndarray,
                  knn_indexes: Union[np.ndarray, None] = None):
    ''' Detect a mode match from the eigenvector field

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
    '''

    # if the jensen-shannon-divergence is below this value
    # then a mode switch is assumed
    distance_limit = 0.1

    # number of bins for probability distribution
    nBins = 25

    # (1) match subsamples in xyz
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
    bins = np.linspace(xmin, xmax, nBins)
    indexes_p1 = np.digitize(tmp1, bins)
    indexes_p2 = np.digitize(tmp2, bins)
    p1 = np.bincount(indexes_p1) / len(tmp1)
    p2 = np.bincount(indexes_p2) / len(tmp2)

    # align bin vector size
    p1_tmp = np.zeros(max(len(p1), len(p2)))
    p2_tmp = np.zeros(max(len(p1), len(p2)))
    p1_tmp[:len(p1)] = p1
    p2_tmp[:len(p2)] = p2
    p1 = p1_tmp
    p2 = p2_tmp

    # compute similarity
    similarity_js = jensen_shannon_entropy(p1, p2)

    return similarity_js > distance_limit


def get_mode_weights_inv(vec: np.ndarray):
    ''' Inverse value weights (higher decay than softmax)
    '''
    val = 1. / (vec[:])
    return val / np.sum(val)


def curve_normalizer(x: np.ndarray, y: np.ndarray):
    ''' Compute the curve normalizer for a curve dot product

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
    '''
    return integrate.simps(y**2, x=x)


def hash_runs(hdf5_filepaths: Union[str, List[str]],
              result_field_request: Union[int, List[int]],
              workfolder: str,
              print_progress: bool = True,
              n_hashes: int = 25,
              i_timestep: int = -1,
              n_threads: int = 1):
    ''' Hash simulation result files

    Parameters
    ----------
    hdf5_filepaths : str or list(str)
        paths to the simulation files as hdf5
    partlist : int or list(int)
        ids of the parts to evaluate
    result_field_names : str or list(str)
        fields to be hashed
    workfolder : str
        folder where to store the working files
    print_progress : bool
        whether to print the progress to the console
    n_hashes : int
        detail degree of hashing
    '''

    # fixed settings
    hdf5_dataset_compression = "gzip"

    # checks
    if not isinstance(hdf5_filepaths, (list, tuple)):
        hdf5_filepaths = [hdf5_filepaths]
    # if not isinstance(partlist, (list, tuple)):
    #     partlist = [partlist]

    # extract run ids
    run_ids = [re.findall(r'\d+', filepath)[0] for filepath in hdf5_filepaths]

    # INLINE FUNCTION

    def _threading_hash_runs(local_hdf5_filepaths,
                             result_field_request,
                             workfolder,
                             n_hashes,
                             i_timestep,
                             local_run_ids,
                             comm_q):

        # vars
        n_files = len(local_hdf5_filepaths)
        result_field_request = copy.deepcopy(result_field_request)

        # statistics
        reduction_times = []

        for i_file, (filepath, run_id) in enumerate(zip(local_hdf5_filepaths, local_run_ids)):

            start = time.time()

            # extract data from mesh
            if 'beam' in result_field_request:
                element_type = 'beam'
            elif 'shell'in result_field_request:
                element_type = 'shell'
            else:
                raise ValueError(
                    "result_field_request contains neither a beam or shell category.")

            result_field_request[element_type]['fields'][element_type +
                                                         '_coordinates'] = True
            result_field_request[element_type]['size'] = True

            # heavy routine inc
            result = get_results_from_dyna_hdf5(
                filepath, result_field_request)

            # extract element xyz before other fields
            elem_xyz = result[element_type]['fields']['coordinates']
            del result[element_type]['fields']['coordinates']
            elem_size = result[element_type]['size']

            # now take fields
            n_neighbors = 3 if element_type == 'beam' else 18
            result_fields = [field
                             for _, field in result[element_type]['fields'].items()]

            # subsampling
            n_subsamples = min(2000, len(elem_xyz))
            subsample_indexes = unique_subsamples(
                0, len(elem_xyz), n_subsamples)

            # perform point cloud homogenization
            is_selected = homogenize_density(
                elem_xyz[subsample_indexes],
                dim=2.0,
                target_distance=np.median(elem_size),
                n_neighbors=n_neighbors)

            subsample_indexes = subsample_indexes[is_selected]
            elem_xyz = elem_xyz[subsample_indexes]
            result_fields = [field[subsample_indexes]
                             for field in result_fields]

            # LBO
            eigenvalues, eigenvectors = run_graph_laplacian(
                elem_xyz,
                n_eigenmodes=n_hashes)

            # computing hashes
            hashes = np.empty()
            for field in result_fields:
                hashes = compute_hashes(eigenvectors,
                                        field[:, i_timestep],
                                        n_points=100,
                                        bandwidth=0.05)
                # TODO save hashes differently

            # communicate log
            reduction_times.append(time.time() - start)
            if comm_q and not comm_q.full():
                comm_q.put((i_file, n_files, np.mean(reduction_times)), False)

            # save
            basename = os.path.basename(filepath)
            ending = '' if basename.split('.')[-1] in ('h5', 'hdf5') else '.h5'

            storage_filepath = os.path.join(workfolder, basename + ending)
            if os.path.exists(storage_filepath):
                if os.path.isfile(storage_filepath):
                    os.remove(storage_filepath)
                else:
                    raise OSError("Can not delete directory " + storage_filepath)

            with h5py.File(storage_filepath, 'w') as fp:

                # header
                ds = fp.create_dataset('info', data=np.zeros(1))
                ds.attrs['run_id'] = run_id

                # data
                fp.create_dataset('eigenvalues', data=eigenvalues,
                                  compression=hdf5_dataset_compression)
                fp.create_dataset('eigenvectors', data=eigenvectors,
                                  compression=hdf5_dataset_compression)
                fp.create_dataset(
                    'subsample_xyz', data=elem_xyz,
                    compression=hdf5_dataset_compression)
                fp.create_dataset('hashes',
                                  data=hashes,
                                  compression=hdf5_dataset_compression)

    # THREADS
    # if only 1 thread run function, this makes it possible to measure performance
    if n_threads == 1:
        _threading_hash_runs(hdf5_filepaths, result_field_request,
                             workfolder, n_hashes, i_timestep, run_ids, None)
    else:
        # comm queues
        queues = [multiprocessing.Queue(1) for i_thread in range(n_threads)]

        # run threads
        thread_filepaths = [[] for i_thread in range(n_threads)]
        thread_run_ids = [[] for i_thread in range(n_threads)]
        for i_file, filepath in enumerate(hdf5_filepaths):
            idx = i_file % n_threads
            thread_filepaths[idx].append(filepath)
            thread_run_ids[idx].append(i_file)

        threads = [multiprocessing.Process(target=_threading_hash_runs,
                                           args=(thread_filepaths_todo,
                                                 result_field_request,
                                                 workfolder,
                                                 n_hashes,
                                                 i_timestep,
                                                 thread_run_ids[i_thread],
                                                 queues[i_thread]))
                   for i_thread, thread_filepaths_todo in enumerate(thread_filepaths)]

        [thread.start() for thread in threads]

        # log status
        if print_progress:
            thread_stats = [{
                'runtime': "0",
                'i_file': 0,
                'n_files': len(thread_filepaths[i_thread]),
                'percent': "0",
            } for i_thread in range(n_threads)]

            while any(thread.is_alive() for thread in threads):

                # fetch thread data
                for i_thread, comm_q in enumerate(queues):
                    if not comm_q.empty():
                        i_file, n_files, runtime = comm_q.get(False)
                        thread_stats[i_thread]['runtime'] = '%.1f' % runtime
                        thread_stats[i_thread]['i_file'] = i_file + 1
                        thread_stats[i_thread]['n_files'] = n_files
                        thread_stats[i_thread]['percent'] = '%.1f' % (
                            i_file / n_files * 100,)

                # print msg
                msg = '| ' + "".join("Thread {0}: {1}/{2} ({3}%) {4}s | ".format(i_thread,
                                                                                 thread_stats[i_thread]['i_file'],
                                                                                 thread_stats[i_thread]['n_files'],
                                                                                 thread_stats[i_thread]['percent'],
                                                                                 thread_stats[i_thread]['runtime'],
                                                                                 )
                                     for i_thread in range(n_threads)) + "\r"
                print(msg, end='')
                time.sleep(0.35)

            # print completion msg
            msg = '| ' + "".join("Thread {0}: {1}/{2} ({3}%) {4}s | ".format(i_thread,
                                                                             thread_stats[i_thread]['n_files'],
                                                                             thread_stats[i_thread]['n_files'],
                                                                             100,
                                                                             thread_stats[i_thread]['runtime'],
                                                                             )
                                 for i_thread in range(n_threads)) + "\r"
            print(msg, end='')

            print("")
            print("done.")

        # join all threads
        [thread.join() for thread in threads]


def compute_hashes_tf(eig_vecs_tf: tf.TensorArray,
                      result_field_tf: tf.TensorArray,
                      n_points: int = 100,
                      bandwidth: float = 0.05):
    ''' Compute hashes for a result field with tensorflow

    Parameters
    ----------
    eig_vecs : tf.TensorArray
        eigenvector field of the component with (n_samples, n_modes)
    result_field : tf.TensorArray
        result field to hash
    n_points : int
        number of equidistant points to use for hashing
        (TODO) automate this selection from the mesh size
    bandwidth : float
        bandwidth in percent of the kernel
        (TODO) choose 5 times global element size median

    Returns
    -------
    hash_functions : tf.TensorArray
        operation to compute the hashes
    '''

    c = tf.constant(-0.5 / bandwidth**2, name='c')
    normalizer = tf.constant(
        1. / (np.sqrt(2 * np.pi) * bandwidth), name='normalizer', dtype=tf.float32)

    x = tf.lin_space(0., 1., n_points, name='x')
    # (P,)

    eig_vecs_tf = tf.expand_dims(eig_vecs_tf, axis=2)
    # (N, H, 1)

    delta_x = tf.math.squared_difference(x, eig_vecs_tf)
    # (N, H, P)

    exponent = tf.scalar_mul(c, delta_x)
    # (N, H, P)

    exponential = tf.math.exp(exponent)
    # (N, H, P)

    result_field_tf = tf.expand_dims(result_field_tf, axis=1)
    result_field_tf = tf.expand_dims(result_field_tf, axis=2)
    print(result_field_tf.shape)
    # (N, 1, 1)

    hashes = tf.multiply(result_field_tf, exponential)
    hashes = tf.reduce_sum(hashes, axis=0)
    hashes = tf.scalar_mul(normalizer, hashes)
    # hashes = tf.tensordot(result_field_tf, exponential, axes=0) # didn't work somehow
    # (H, P)

    return hashes


def compute_hashes(eig_vecs: np.ndarray,
                   result_field: np.ndarray,
                   # elem_size,
                   n_points: int = 100,
                   bandwidth: float = 0.05) -> List[Tuple[np.ndarray, np.ndarray]]:
    ''' Compute hashes for a result field

    Parameters
    ----------
    eig_vecs : np.ndarray
        eigenvector field of the component with (n_samples, n_modes)
    result_field : np.ndarray
        result field to hash
    n_points : resolution of the hash
        number of equidistant points to use for smoothing
        (TODO) automate this selection from the mesh size
    bandwidth : float
        bandwidth in percent of the kernel
        (TODO) choose 5 times global element size median

    Returns
    -------
    hash_functions : list(tuple(np.ndarray, np.ndarray))
        list of the computed hash functions. Every item is the hash for
        an eigenmode. The hash consists of a pair of two functions: (x,y).
        For comparison, only y is usually used.
    '''

    assert(eig_vecs.shape[0] == len(result_field)), "{} != {}".format(
        eig_vecs.shape[0], len(result_field))

    # TODO vectorize to speed it up
    hash_functions = []
    for iEigen in range(eig_vecs.shape[1]):

        xmin = eig_vecs[:, iEigen].min()
        xmax = eig_vecs[:, iEigen].max()

        x = np.linspace(xmin, xmax, n_points)
        y = np.zeros(n_points)

        local_bandwidth = bandwidth * (xmax - xmin)
        c = -0.5 / local_bandwidth**2

        for ii, point in enumerate(x):
            y[ii] = np.dot(result_field, np.exp(
                c * np.square(point - eig_vecs[:, iEigen])))
        y /= np.sqrt(2 * np.pi) * bandwidth

        hash_functions.append((x, y))

    return hash_functions