import typing


class ClusterType:
    '''Specifies names of specific clustering algorythms '''

    OPTICS = "OPTICS"
    DBSCAN = "DBSCAN"
    KMeans = "KMeans"
    SpectralClustering = "SpectralClustering"

    @staticmethod
    def get_cluster_type_name() -> typing.List[str]:
        return [
            ClusterType.OPTICS,
            ClusterType.DBSCAN,
            ClusterType.KMeans,
            ClusterType.SpectralClustering,
        ]


class DetectorType:
    '''Specifies names of different outlier detector algorythms '''

    IsolationForest = "IsolationForest"
    OneClassSVM = "OneClassSVM"
    LocalOutlierFactor = "LocalOutlierFactor"
    # Experimental = "Experimental"

    @staticmethod
    def get_detector_type_name() -> typing.List[str]:
        return[
            DetectorType.IsolationForest,
            DetectorType.OneClassSVM,
            DetectorType.LocalOutlierFactor,
            # DetectorType.Experimental,
        ]


class FileNames:
    '''Makes sure, all functions use the same name when referencing the same file '''
    name_csv = 'mapping_names_to_id.csv'
    name_reference_sample = 'reference_subsample.npy'
    name_subsamples = 'all_subsamples.npy'
    name_betas = 'betas.npy'
    name_V_ROB = 'V_ROB.npy'

    @staticmethod
    def get_file_name() -> typing.List[str]:
        return[
            FileNames.name_csv,
            FileNames.name_reference_sample,
            FileNames.name_subsamples,
            FileNames.name_betas,
            FileNames.name_V_ROB,
        ]
