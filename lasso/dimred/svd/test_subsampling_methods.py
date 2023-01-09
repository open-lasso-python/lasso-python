import os
import tempfile
from typing import Tuple
from unittest import TestCase

import numpy as np

from lasso.dimred.svd.subsampling_methods import create_reference_subsample, remap_random_subsample
from lasso.dimred.test_plot_creator import create_2_fake_plots


class TestSubsampling(TestCase):
    def test_create_reference_sample(self):
        """Tests the creation of reference sample"""

        with tempfile.TemporaryDirectory() as tmp_dir:

            create_2_fake_plots(tmp_dir, 500, 10)
            load_path = os.path.join(tmp_dir, "SVDTestPlot00/plot")
            n_nodes = 200

            result = create_reference_subsample(load_path, parts=[], nr_samples=n_nodes)

            # result should be tuple containing subsample, load time and total process time
            self.assertTrue(isinstance(result, Tuple))

            ref_sample, t_total, t_load = result

            # check for correct types
            self.assertTrue(isinstance(ref_sample, np.ndarray))
            self.assertTrue(isinstance(t_total, float))
            self.assertTrue(isinstance(t_load, float))

            # t_total should be greater than t_load
            self.assertTrue(t_total - t_load >= 0)

            # check for correct dimensions of ref_sample
            self.assertEqual(ref_sample.shape, (n_nodes, 3))

            # should return string error message if desired samplesize is greater
            # than avaiable nodes
            n_nodes = 5500
            result = create_reference_subsample(load_path, parts=[], nr_samples=n_nodes)

            self.assertTrue(isinstance(result, str))

            # should return string error message for nonexitant parts:
            n_nodes = 200
            result = create_reference_subsample(load_path, parts=[1], nr_samples=n_nodes)

            self.assertTrue(isinstance(result, str))

    def test_remap_random_subsample(self):
        """Verifies correct subsampling"""

        with tempfile.TemporaryDirectory() as tmp_dir:

            create_2_fake_plots(tmp_dir, 500, 10)
            ref_path = os.path.join(tmp_dir, "SVDTestPlot00/plot")
            sample_path = os.path.join(tmp_dir, "SVDTestPlot01/plot")
            n_nodes = 200

            ref_result = create_reference_subsample(ref_path, parts=[], nr_samples=n_nodes)

            ref_sample = ref_result[0]

            sub_result = remap_random_subsample(
                sample_path, parts=[], reference_subsample=ref_sample
            )

            # sub_result should be Tuple containing subsample, total process time,
            # and plot load time
            self.assertTrue(isinstance(sub_result, Tuple))

            subsample, t_total, t_load = sub_result

            # confirm correct types
            self.assertTrue(isinstance(subsample, np.ndarray))
            self.assertTrue(isinstance(t_total, float))
            self.assertTrue(isinstance(t_load, float))

            # t_total should be greater t_load
            self.assertTrue(t_total - t_load >= 0)

            # correct shape of subsample
            self.assertEqual(subsample.shape, (5, n_nodes, 3))

            # entries of subsmaple at timestep 0 should be the same as the reference sample
            # this is only true for for the dimredTestPlots, this might not be the case
            # with real plots we check if the difference is 0
            self.assertTrue((ref_sample - subsample[0]).max() == 0)

            # should return string error message for nonexistant parts:
            err_msg = remap_random_subsample(sample_path, parts=[1], reference_subsample=ref_sample)

            self.assertTrue(isinstance(err_msg, str))
