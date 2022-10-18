import typing


class ClusterType:
    """Specifies names of specific clustering algorithms"""

    OPTICS = "OPTICS"
    DBSCAN = "DBSCAN"
    KMeans = "KMeans"
    SpectralClustering = "SpectralClustering"

    @staticmethod
    def get_cluster_type_name() -> typing.List[str]:
        """Get the name of the clustering algorithms"""
        return [
            ClusterType.OPTICS,
            ClusterType.DBSCAN,
            ClusterType.KMeans,
            ClusterType.SpectralClustering,
        ]


class DetectorType:
    """Specifies names of different outlier detector algorythms"""

    IsolationForest = "IsolationForest"
    OneClassSVM = "OneClassSVM"
    LocalOutlierFactor = "LocalOutlierFactor"
    # Experimental = "Experimental"

    @staticmethod
    def get_detector_type_name() -> typing.List[str]:
        """Get the name of the detector algorithms"""
        return [
            DetectorType.IsolationForest,
            DetectorType.OneClassSVM,
            DetectorType.LocalOutlierFactor,
        ]
