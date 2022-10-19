from typing import Sequence, Tuple, Union

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, KMeans, SpectralClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .keyword_types import ClusterType, DetectorType


def __apply_spectral_clustering(betas, runids, datasets, idsets, random_state=11, **kwargs):
    """
    Method to group the input Betas.
    Default keyword arguments: affinity='nearest_neighbors', random_state=11

    Parameters
    ----------
    betas: np.ndarray
        Betas that shall be grouped into clusters
    run_ids: np.ndarray
        Ids matching to each Beta
    datasets: list
        List where each grouped Betas will be added
    idsets: list
        List where the grouped ids corresponding to the grouped Betas will be saved
    **kwargs: keyword arguments
        Keyword arguments specific for the SpectralClustering algorythm

    See Also
    --------
    Detailed Documentation of the function parameters can be found on sklearn.
    Link: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
    """  # noqa pylint: disable = line-too-long
    clustering = SpectralClustering(random_state=random_state, **kwargs).fit(betas)

    indexes = clustering.labels_

    clusters = np.unique(indexes)

    for clump in clusters:
        clump_index = np.where(indexes == clump)[0]
        clump_betas = betas[clump_index]
        clump_runs = runids[clump_index]
        datasets.append(clump_betas)
        idsets.append(clump_runs.tolist())


def __apply_k_means(betas, runids, datasets, idsets, random_state=11, **kwargs):
    """
    Method to group the input Betas.
    Recommended keyword arguments: n_clusters=3, random_state=11

    Parameters
    ----------
    betas: np.ndarray
        Betas that shall be grouped into clusters
    run_ids: np.ndarray
        Ids matching to each Beta
    datasets: list
        List where each grouped Betas will be added
    idsets: list
        List where the grouped ids corresponding to the grouped Betas will be saved
    **kwargs: keyword arguments
        Keyword arguments specific fot the KMeans algorythm

    See Also
    --------
    Detailed Documentation of the function parameters can be found on sklearn.
    Link: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """  # noqa: E501 pylint: disable = line-too-long
    kmeans = KMeans(random_state=random_state, **kwargs).fit(betas)
    indexes = kmeans.labels_

    clusters = np.unique(indexes)

    for clump in clusters:
        clump_index = np.where(indexes == clump)[0]
        clump_betas = betas[clump_index]
        clump_runs = runids[clump_index]
        datasets.append(clump_betas)
        idsets.append(clump_runs.tolist())


def __apply_dbscan(betas, runids, datasets, idsets, **kwargs):
    """
    Method to group the input Betas.
    Defautl keyword arguments: eps=0.08

    Parameters
    ----------
    betas: np.ndarray
        Betas that shall be grouped into clusters
    run_ids: np.ndarray
        Ids matching to each Beta
    datasets: list
        List where each grouped Betas will be added
    idsets: list
        List where the grouped ids corresponding to the grouped Betas will be saved
    **kwags: keyword arguments
        Keyword arguments for the DBSCAN algorythm

    See Also
    --------
    Detailed Documentation of the function parameters can be found on sklearn.
    Link: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    """  # noqa: E501 pylint: disable = line-too-long
    deutsche_bahn = DBSCAN(**kwargs).fit(betas)
    indexes = deutsche_bahn.labels_

    clusters = np.unique(indexes)

    for clump in clusters:
        clump_index = np.where(indexes == clump)[0]
        clump_betas = betas[clump_index]
        clump_runs = runids[clump_index]
        datasets.append(clump_betas)
        idsets.append(clump_runs.tolist())


def __apply_optics(betas, runids, datasets, idsets, **kwargs):
    """
    Method to group the input Betas.
    Default keyword parameters: eps=0.05, min_cluster_size=10

    Parameters
    ----------
    betas: np.ndarray
        Betas that shall be grouped into clusters
    run_ids: np.ndarray
        Ids matching to each Beta
    datasets: list
        List where each grouped Betas will be added
    idsets: list
        List where the grouped ids corresponding to the grouped Betas will be saved
    **kwargs: keyword arguments
        Keyword arguments specific to the OPTICS function.

    See Also
    -------
    Detailed Documentation of the function parameters can be found on sklearn.
    Link: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS
    """  # noqa: E501 pylint: disable = line-too-long
    lense = OPTICS(**kwargs).fit(betas)
    indexes = lense.labels_

    clusters = np.unique(indexes)

    for clump in clusters:
        clump_index = np.where(indexes == clump)[0]
        clump_betas = betas[clump_index]
        clump_runs = runids[clump_index]
        datasets.append(clump_betas)
        idsets.append(clump_runs.tolist())


def __detect_outliers_isolation_forest(
    betas, ids, beta_clusters, id_clusters, random_state=11, **kwargs
):
    """
    Detects outliers based on the IsolationForest algorythm from sklearn.
    Detected outliers will be appended into the provided lists
    Default keyword parameters: random_state=12, behaviour="new", contamination=0.005

    Parameters
    ----------
    betas: np.ndarray
        Numpy array containing the betas
    ids: np.ndarray
        Numpy array containing the ids of each beta
    beta_clusters: list
        List where each cluster of betas will be appended
    id_clusters: list
        List where each cluster of ids will be appended
    **kwargs: keyword argument
        Keywords specific to the IsolationForest algorythm
    Returns
    -------
    inlier_betas: np.array
        Numpy array containing the betas that are not outliers
    inlier_ids: np.array
        Numpy array containing the ids of betas that are not outliers

    See Also
    --------
    Detailed Documentation of the function parameters can be found on sklearn.
    Link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """  # noqa: E501 pylint: disable = line-too-long
    outcasts = IsolationForest(random_state=random_state, **kwargs).fit(betas).predict(betas)

    outlier_key = np.where(outcasts == -1)[0]
    inlier_key = np.where(outcasts == 1)[0]
    beta_clusters.append(betas[outlier_key])
    id_clusters.append(ids[outlier_key].tolist())

    return betas[inlier_key], ids[inlier_key]


def __detect_outliers_local_outlier_factor(betas, ids, beta_clusters, id_clusters, **kwargs):
    """
    Detects outliers based on the LocalOutlierFactor algorythm from sklearn.
    Detected outliers will be appended into the provided lists
    Default keyword parameters: contamination=0.01

    Parameters
    ----------
    betas: np.ndarray
        Numpy array containing the betas
    ids: np.ndarray
        Numpy array containing the ids of each beta
    beta_clusters: list
        List where each cluster of betas will be appended
    id_clusters: list
        List where each cluster of ids will be appended
    **kwargs: keyword argument
        Keywords specific to the LocalOutlierFactor algorythm.
    Returns
    -------
    inlier_betas: np.ndarray
        Numpy array containing the betas that are not outliers
    inlier_ids: np.ndarray
        Numpy array containing the ids of betas that are not outliers

    See Also
    --------
    Detailed Documentation of the function parameters can be found on sklearn.
    Link:https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
    """  # noqa pylint: disable = line-too-long
    outcasts = LocalOutlierFactor(**kwargs).fit_predict(betas)

    outlier_key = np.where(outcasts == -1)[0]
    inlier_key = np.where(outcasts == 1)[0]
    beta_clusters.append(betas[outlier_key])
    id_clusters.append(ids[outlier_key].tolist())

    return betas[inlier_key], ids[inlier_key]


def __detect_outliers_one_class_svm(betas, ids, beta_clusters, id_clusters, **kwargs):
    """
    Detects outliers based on the OneClassSVM algorythm from sklearn.
    Detected outliers will be appended into the provided lists
    Defautl keyword arguments: gamma=0.1, nu=0.01

    Parameters
    ----------
    betas: np.ndarray
        Numpy array containing the betas
    ids: np.ndarray
        Numpy array containing the ids of each beta
    beta_clusters: list
        List where each cluster of betas will be appended
    id_clusters: list
        List where each cluster of ids will be appended
    **kwargs: keyword argument
        Keywords specific to the OneClassSVM algorythm.

    Returns
    -------
    inlier_betas: np.ndarray
        Numpy array containing the betas that are not outliers
    inlier_ids: np.ndarray
        Numpy array containing the ids of betas that are not outliers

    See Also
    --------
    Detailed Documentation of the function parameters can be found on sklearn.
    Link: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
    """  # noqa: E501 pylint: disable = line-too-long

    outcasts = OneClassSVM(**kwargs).fit_predict(betas)

    outlier_key = np.where(outcasts == -1)[0]
    inlier_key = np.where(outcasts == 1)[0]
    beta_clusters.append(betas[outlier_key])
    id_clusters.append(ids[outlier_key].tolist())

    return betas[inlier_key], ids[inlier_key]


def __experimental_outlier_detector(betas, ids, **kwargs):
    """
    Detects outliers by applying LocalOutlierFactor algorythm from sklearn over multiple slices of betas .
    Detected outliers will be appended into the provided lists
    Default keyword arguments:  contamination=0.01
    Parameters
    ----------
    betas: np.ndarray
        Numpy array containing the betas
    ids: np.ndarray
        Numpy array containing the ids of each beta
    **kwargs: keyword argument
        Keywords specific to the LocalOutlierFactor algorythm
    Returns
    -------
    outliers: np.array
        Numpy array containing the sample names identified as outliers
    outlier_index: np.array
        Array containing the indexes of outliers
    inlier_index: np.array
        Array of booleans to get inlier(not outliers) betas and IDs

    See Also
    --------
    Detailed Documentation of the function parameters can be found on sklearn.
    Link:https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
    """  # noqa pylint: disable = line-too-long

    # pylint: disable = too-many-locals

    loops = betas.shape[1] - 2
    alertlist = []
    for dadoop in range(loops):
        slicer = dadoop + 3
        beta_slice = betas[:, dadoop:slicer]

        sanitizer = LocalOutlierFactor(**kwargs).fit_predict(beta_slice)
        outlier_key = np.where(sanitizer == -1)[0]
        alertlist.append(outlier_key)

    suspects = np.concatenate(alertlist)
    individuals = np.unique(suspects)
    crimecounter = np.array([np.where(suspects == tracked)[0].shape[0] for tracked in individuals])

    the_cases = np.where(crimecounter > 2)[0]
    the_judged = ids[individuals[the_cases]]

    innocents = np.full(ids.shape, True)

    if the_judged.shape != (0,):
        judged_index = np.array([np.where(ids == convict)[0] for convict in the_judged])[:, 0]

        innocents[judged_index] = False
    else:
        return False

    return the_judged, judged_index, innocents


def __rescale_betas(betas):
    """
    utility function to rescale betas into the range of [0, 1].
    Expects only positive betas

    Parameters
    ----------
    betas: np.ndarray
        Numpy array containing the betas to be scaled. Expects betas of shape (samples, nr_betas)

    Returns
    -------
    betas_scaled: np.ndarray
        Betas scaled to range [0, 1]
    maxb: np.ndarray
        Array to rescale betas back to original values
    """
    assert len(betas.shape) == 2
    ref_betas = np.abs(betas)
    maxb = np.array([np.max(ref_betas[:, i]) for i in range(betas.shape[1])])
    # return np.array([(betas[:, i]/maxb[i]) for i in range(betas.shape[1])]).T
    return betas / (maxb.T), maxb.T


def list_detectors_and_cluster():
    """
    Prints out all keywords for outlier detection and clustering functions

    See Also
    --------
    list_detectors_and_cluster(keyword)"""

    print("Implemented Detectors:")
    for entry in __detector_dict:
        print("   " + entry)
    print("Implemented Clustering Functions")
    for entry in __cluster_dict:
        print("   " + entry)


def document_algorithm(keyword):
    """
    prints out the docstring of the function related to the input keyword

    Parameters
    ----------
    keyword: str
        String keyword referencing the outlier detection or clustering function

    See Also
    --------
    list_detectors_and_cluster()
    """
    print(__doc_dict[keyword])


__doc_dict = {
    DetectorType.IsolationForest: __detect_outliers_isolation_forest.__doc__,
    DetectorType.OneClassSVM: __detect_outliers_one_class_svm.__doc__,
    DetectorType.LocalOutlierFactor: __detect_outliers_local_outlier_factor.__doc__,
    # DetectorType.Experimental: __experimental_outlier_detector.__doc__,
    ClusterType.OPTICS: __apply_optics.__doc__,
    ClusterType.DBSCAN: __apply_dbscan.__doc__,
    ClusterType.KMeans: __apply_k_means.__doc__,
    ClusterType.SpectralClustering: __apply_spectral_clustering.__doc__,
}

__detector_dict = {
    DetectorType.IsolationForest: __detect_outliers_isolation_forest,
    DetectorType.OneClassSVM: __detect_outliers_one_class_svm,
    DetectorType.LocalOutlierFactor: __detect_outliers_local_outlier_factor,
    # DetectorType.Experimental: __experimental_outlier_detector
}
__cluster_dict = {
    ClusterType.OPTICS: __apply_optics,
    ClusterType.DBSCAN: __apply_dbscan,
    ClusterType.KMeans: __apply_k_means,
    ClusterType.SpectralClustering: __apply_spectral_clustering,
}


def create_cluster_arg_dict(args: Sequence[str]) -> Union[Tuple[str, dict], str]:
    """Determines which cluster to use and creates a python dictionary to use as cluster_params

    Parameters
    ----------
    args: Sequence[str]
        List of strings containing parameters and arguments

    Returns
    -------
    cluster_type: str
        determines which cluster algorithm to use
    cluster_arg_dict: dict
        dictionary containing arguments and values for specific cluster_type
    err_msg: str
        message containing error, mostly unrecognised keywords"""

    # first argument must be cluster type
    cluster_key = args[0].lower()
    cluster_arg_dict = {}
    cluster_type = None
    param_type = []

    # all following arguments are a parameter, followed by their respective value
    parameters = []
    values = []
    if len(args) % 3 == 0:
        # check if amount of parameters is valid
        err_msg = (
            "Invalid cluster arguments, first argument must be the chosen clustering algorithm,"
            " and each optional subsequent parameter must be followed by its type and value"
        )
        return err_msg
    if len(args) > 1:
        # check if we even have parameters
        parameters = args[1:-2:3]
        param_type = args[2:-1:3]
        values = args[3::3]

    for cluster_option in ClusterType.get_cluster_type_name():
        if cluster_key == cluster_option.lower():
            cluster_type = cluster_option

    if not cluster_type:
        err_msg = (
            f"No existing clustering method matching {args[0]}"
            f"possible clustering methods are: {str(ClusterType.get_cluster_type_name())[1:-1]}"
        )
        return err_msg

    for ind, param in enumerate(parameters):
        p_t = param_type[ind]
        v_type = None
        if p_t == "str":
            v_type = str
        elif p_t == "float":
            v_type = float
        elif p_t == "int":
            v_type = int
        else:
            err_msg = f"Clustering: Invalid type identifier {p_t}"
            return err_msg

        try:
            val = v_type(values[ind])
        except ValueError:
            err_msg = (
                f"Clustering: Invalid value {values[ind]} "
                f"for parameter {param} of type {v_type}"
            )
            return err_msg
        cluster_arg_dict[param] = val

    return cluster_type, cluster_arg_dict


def create_detector_arg_dict(args: Sequence[str]) -> Union[Tuple[str, dict], str]:
    """Determines which detector to use and creates a python dictionary to use as detector_params

    Parameters
    ----------
    args: Sequence[str]
        List of strings containing parameters and arguments

    Returns
    -------
    detector_type: str
        determines which cluster algorithm to use
    detector_arg_dict: dict
        dictionary containing arguments and values for specific cluster_type
    err_mgs: str
        message containing error, mostly unrecognised keywords"""

    # first argument must be detector type:
    detector_key = args[0].lower()
    detector_arg_dict = {}
    detector_type = None
    param_type = []

    #  all following arguments are a parameter, followed by their respective value
    parameters = []
    values = []
    if len(args) % 3 == 0:
        # check if amount of parameters is valid
        err_msg = (
            "Invalid outlier detector arguments, first argument must be "
            "the chosen detector algorithm, and each optional subsequent "
            "parameter must be followed by its type and value"
        )
        return err_msg
    if len(args) > 1:
        # check if we even have parameters
        parameters = args[1:-2:3]
        param_type = args[2:-1:3]
        values = args[3::3]

    for detector_option in DetectorType.get_detector_type_name():
        if detector_key == detector_option.lower():
            detector_type = detector_option

    if not detector_type:
        err_msg = (
            f"No existing outlier detection method matching {args[0]} "
            f"possible outlier detection methods are: "
            f"{str(DetectorType.get_detector_type_name())[1:-1]}"
        )
        return err_msg

    for ind, param in enumerate(parameters):
        p_t = param_type[ind]
        v_type = None
        if p_t == "str":
            v_type = str
        elif p_t == "float":
            v_type = float
        elif p_t == "int":
            v_type = int
        else:
            err_msg = f"Outlier Detection: Invalid type identifier {p_t}"
            return err_msg

        try:
            val = v_type(values[ind])
        except ValueError:
            err_msg = (
                f"Outlier Detection: Invalid value {values[ind]} "
                "for parameter {param} of type {v_type}"
            )
            return err_msg
        detector_arg_dict[param] = val

    return detector_type, detector_arg_dict


def group_betas(
    beta_index,
    betas,
    scale_betas=False,
    cluster=None,
    detector=None,
    cluster_params=None,
    detector_params=None,
) -> Union[Tuple[list, list], str]:
    """
    Base function to to group betas into groups, detect outliers. Provides that all different
    clustering and outlier detection algorythms are implemented in an easy to access environment.
    To select different clustering and outlier detection algoyrthms, please use appropriate
    KeywordTypes. A description of each function can be accessed with document_algorythm(keyword)
    A list of all functions can be accessed with list_detectors_and_clusters()

    Parameters
    ----------
    beta_index: np.ndarray
        Array containing the file names specific to the betas with the same index in the beta array
    betas: np.ndarray
        Numpy array containing the betas.
        Betas are expected to be of shape (samples, timestep, 3)
        The three entries per beta can either be dimesnions (x,y,z) or any three betas/eigenvalues
    cluster: str, optional, default : "KMeans".
        String specifying which clustering algorythm shall be applied.
        Use ClusterTypefor easier access
    detector: str, optional, default: None.
        String specifying which outlier detection algorythm shall be applied.
        Use DetectorType for easier access
    cluster_params: dict, optional
        Dictionary containing parameters for the clustering algorythms.
        See the sklearn documentation for the function to learn more.
    detector_params: dict, optional
        Dictionary containing parameters for the outlier detection algorythms.
        See the sklearn documentation for the function to learn more

    Returns
    -------
    beta_clusters: list
        List containing Numpy Arrays of betas in one cluster.
        If a detector was selected, or the clustering algorythm has its
        own outlier detection, the first entry in the list will be oultier betas
    id_clusters: list
        List containing lists of beta ids. Each id corresponds to the beta in
        the same place in the beta_clusters list
    err_msg: str
        Error message if wrong keywords for detector or cluster algorithms were used

    Notes
    --------
    document_algorithm:
        Prints docstring of each function into console
    list_detectors_and_clusters:
        Prints out all detection and clustering algorythms into console
    Sklearn Userguide chapter 2.3 Clustering:
        https://scikit-learn.org/stable/modules/clustering.html
        Detailed overview of different clustering algorythms
    Sklearn Examples outlier detection:
        https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html
        Example of different used outlier detection algorythms
    """

    # pylint: disable = too-many-arguments, too-many-locals, too-many-branches

    if cluster_params is None:
        cluster_params = {}

    if detector_params is None:
        detector_params = {}

    beta_clusters = []
    id_clusters = []

    if scale_betas:
        betas, _ = __rescale_betas(betas)

    if detector == "Experimental":

        experimental_results = __detector_dict[detector](betas, beta_index, **detector_params)
        if not isinstance(experimental_results, bool):
            outlier_betas, outlier_index, inlier_index = experimental_results
            beta_clusters.append(betas[outlier_index])
            id_clusters.append(outlier_betas.tolist())
            betas = betas[inlier_index]
            beta_index = beta_index[inlier_index]
        else:
            empy_list = []
            beta_clusters.append(empy_list)
            id_clusters.append(empy_list)

        detector = None

    if detector is not None:
        try:
            betas_det, index_det = __detector_dict[detector](
                betas, beta_index, beta_clusters, id_clusters, **detector_params
            )

        except TypeError as key_err:
            err_msg = (
                f"During Outlier Detection, a TypeError came up:\n{str(key_err)}\n"
                "Please check your outlier detection arguments"
            )
            return err_msg

        except ValueError as val_err:
            err_msg = (
                f"During Outlier Detection, a ValueError came up:\n{str(val_err)}\n"
                "Please check your outlier detection arguments"
            )
            return err_msg
    else:
        betas_det, index_det = betas, beta_index

    if cluster is not None:
        try:
            __cluster_dict[cluster](
                betas_det, index_det, beta_clusters, id_clusters, **cluster_params
            )
        except TypeError as key_err:
            err_msg = (
                f"During Clustering, a TypeError came up:\n{str(key_err)}\n"
                "Please check your outlier detection arguments"
            )
            return err_msg

        except ValueError as val_err:
            err_msg = (
                f"During Clustering, a ValueError came up:\n{str(val_err)}\n"
                "Please check your outlier detection arguments"
            )
            return err_msg
    else:
        beta_clusters, id_clusters = [*beta_clusters, betas_det], [*id_clusters, index_det]

    return beta_clusters, id_clusters
