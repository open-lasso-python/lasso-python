from unittest import TestCase
import numpy as np
from lasso.dimred.svd.clustering_betas import group_betas
from lasso.dimred.svd.keyword_types import DetectorType, ClusterType


class TestClustering(TestCase):
    def test_group_betas(self):
        """tests correct function of the group_betas function
        in clustering_betas.py"""

        fake_names = np.array(["betas_{i}".format(i=i) for i in range(25)])
        fake_cluster_0 = np.random.rand(12, 3) + 5
        fake_cluster_1 = np.random.rand(12, 3) - 5
        fake_betas = np.stack([*fake_cluster_0, *fake_cluster_1, np.array([0, 0, 0])])
        expected_clusters = 2
        expected_outliers = 1

        # test with recommended settings
        beta_clusters, name_clusters = group_betas(
            fake_names,
            fake_betas,
            cluster=ClusterType.KMeans,
            detector=DetectorType.LocalOutlierFactor,
            cluster_params=dict(
                n_clusters=expected_clusters,
            ),
        )

        # verify correct type of output
        self.assertIsInstance(beta_clusters, list)
        self.assertIsInstance(name_clusters, list)

        # verify that beta_clusters and name_clusters correspond to each other
        self.assertEqual(len(beta_clusters), len(name_clusters))
        # verify that beta_clusters contains as many clusters as searched for
        # inkluding one outlier
        self.assertEqual(len(beta_clusters), expected_clusters + expected_outliers)

        # verify that entries correspond to each other
        for c, cluster in enumerate(name_clusters):
            for e, entry in enumerate(cluster):
                index = np.where(fake_names == entry)[0]
                self.assertTrue((fake_betas[index] - beta_clusters[c][e]).max() == 0)

        # verify differen keyword kombinations

        for cluster_type in ClusterType.get_cluster_type_name():
            for detector_type in DetectorType.get_detector_type_name():
                beta_clusters, name_clusters = group_betas(
                    fake_names, fake_betas, cluster=cluster_type, detector=detector_type
                )

                # verify correct output
                self.assertIsInstance(beta_clusters, list)
                self.assertIsInstance(name_clusters, list)
                self.assertEqual(len(beta_clusters), len(name_clusters))
