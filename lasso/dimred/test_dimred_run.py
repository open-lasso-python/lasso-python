import os
import tempfile
from unittest import TestCase

import h5py
import numpy as np

from lasso.dimred.dimred_run import DIMRED_STAGES, DimredRun, DimredRunError, HDF5FileNames
from lasso.dimred.test_plot_creator import create_50_fake_plots


class TestDimredRun(TestCase):
    def test_run(self):
        """Verifies correct function of DimredRun.py"""
        verification_hdf5_file = h5py.File("test/DimredRunTest/verificationFile.hdf5", "r")

        with tempfile.TemporaryDirectory() as tmpdir:

            # create simulation runs
            create_50_fake_plots(folder=tmpdir, n_nodes_x=500, n_nodes_y=10)

            # collect all simulation runs
            # sim_dir = "test/dimredTestPlots"
            sim_files = os.listdir(tmpdir)
            # sim_files.pop(sim_files.index("htmlTestPage.html"))
            sim_runs = []
            for sim in sim_files:
                sim_runs.append(os.path.join(tmpdir, sim, "plot"))

            test_run = DimredRun(
                reference_run=os.path.join(tmpdir, "SVDTestPlot00/plot"),
                simulation_runs=sim_runs,
                start_stage=DIMRED_STAGES[0],
                end_stage="CLUSTERING",
                console=None,
                project_dir="test/DimredRunTest",
                n_processes=5,
                cluster_args=["kmeans"],
            )

            with test_run:
                # verify creation of reference_subsample
                # to be able to reproduce results, each DimredRun must return same results
                test_run.process_reference_run()

                # check if reference subsamples match
                test_refsample = test_run.h5file[HDF5FileNames.SUBSAMPLE_SAVE_NAME.value]
                verification_refsample = verification_hdf5_file[
                    HDF5FileNames.SUBSAMPLE_SAVE_NAME.value
                ]
                self.assertEqual(test_refsample.shape, verification_refsample.shape)
                self.assertTrue((test_refsample[:] - verification_refsample[:]).max() == 0)

                # check if the expected reference run is chosen
                self.assertEqual(
                    os.path.abspath(os.path.join(tmpdir, "SVDTestPlot00/plot")),
                    test_run.reference_run,
                )

                # check if subsampled samples match
                test_run.subsample_to_reference_run()

                # get subsampled samples
                test_sub_group = test_run.h5file[HDF5FileNames.SUBSAMPLED_GROUP_NAME.value]
                test_subs = np.stack([test_sub_group[key][:] for key in test_sub_group.keys()])

                # check if shape is equal to (n_samples, timesteps, subsampled nodes, dims)
                # we have 50 sample, minus ref_run is 49
                # we have 5 timesteps
                # we subsample to 2000 nodes
                # we always have 3 spatial dimensions
                self.assertEqual(test_subs.shape, (49, 5, 2000, 3))

                # check if svd yields consistent results
                test_run.dimension_reduction_svd()

                # get test betas
                test_betas_group = test_run.h5file[HDF5FileNames.BETAS_GROUP_NAME.value]
                test_ids = np.stack([key for key in test_betas_group.keys()])
                test_betas = np.stack([test_betas_group[key][:] for key in test_betas_group.keys()])

                # we check if test_ids and test_betas are of correct shape
                # we have 44 samples, 5 timesteps and save the first 10 betas
                self.assertEqual(test_ids.shape, (49,))
                self.assertEqual(test_betas.shape, (49, 5, 10))

                test_v_rob = test_run.h5file[HDF5FileNames.V_ROB_SAVE_NAME.value][:]
                # shape of v_rob must be (eigen, timesteps, nodes)
                self.assertEqual(test_v_rob.shape, (10, 5, 2000 * 3))

                # verify that calculated betas are reproducable as expected
                # first, create displ mat containing difference in displ over time
                verify_displ_stacked = test_subs.reshape(49, 5, 2000 * 3)
                verify_diff_mat = np.stack(
                    [verify_displ_stacked[:, 0, :] for _ in range(5)]
                ).reshape(49, 5, 2000 * 3)
                verify_displ_stacked = verify_displ_stacked - verify_diff_mat

                # calculate betas and check if they are similar
                verify_betas = np.einsum("stn, ktn -> stk", verify_displ_stacked, test_v_rob)
                self.assertTrue(np.allclose(verify_betas, test_betas))

                # recalculate displ
                recalc_displ_stacked = np.einsum("stk, ktn -> stn", test_betas, test_v_rob)

                # Due to projection into eigenspace and back not using all avaiable eigenvectors,
                # a small error margin is inevitable
                self.assertTrue((verify_displ_stacked - recalc_displ_stacked).max() <= 1e-5)

                # checking clustering and html output makes little sense here,
                # but we know how the created plots are laid out: 25 bending up, 25 bending down
                # this should be presented in the betas
                # We will only look at the last timestep
                # We only check the first beta

                # first 24 betas point one direction (reference run is run 0 and points up)
                betas_up = test_betas[:24, -1]
                # other 25 betas point down
                betas_down = test_betas[24:, -1]

                # check that first beta has the same sign as others bending up
                is_pos_up = betas_up[0, 0] > 0
                for b in betas_up:
                    self.assertEqual(is_pos_up, b[0] > 0)

                # check that 25th betas has same sign as other bending down
                is_pos_down = betas_down[0, 0] > 0
                for b in betas_down:
                    self.assertEqual(is_pos_down, b[0] > 0)

                # verify that one group has negative and other group positive direction
                self.assertFalse(is_pos_down and is_pos_up)

                test_run.clustering_results()

            # check if glob pattern works correctly
            DimredRun(
                simulation_runs=os.path.join(tmpdir, "SVDTestPlot*/plot"),
                start_stage=DIMRED_STAGES[0],
                end_stage=DIMRED_STAGES[0],
                project_dir="test/DimredRunTest",
                console=None,
            )

    def test_for_errors(self):
        """Verifies correct error behaviour when facing incorrect parser arguments"""

        with tempfile.TemporaryDirectory() as tmpdir:
            # collect all simulation runs
            sim_files = os.listdir(tmpdir)
            sim_runs = []
            for sim in sim_files:
                sim_runs.append(os.path.join(tmpdir, sim, "plot"))

            # check invalid start_stage
            self.assertRaises(
                DimredRunError,
                DimredRun,
                reference_run="test/dimredTestPlots/SVDTestPlot0/plot",
                simulation_runs=sim_runs,
                start_stage="INVALID_START",
                end_stage=DIMRED_STAGES[-1],
                console=None,
                project_dir="test/DimredRunTest",
                n_processes=5,
            )

            # check invalid end_stage
            self.assertRaises(
                DimredRunError,
                DimredRun,
                reference_run="test/dimredTestPlots/SVDTestPlot0/plot",
                simulation_runs=sim_runs,
                start_stage=DIMRED_STAGES[0],
                end_stage="INVALID_END",
                console=None,
                project_dir="test/DimredRunTest",
                n_processes=5,
            )

            # check invalid start_stage after end_stage
            self.assertRaises(
                DimredRunError,
                DimredRun,
                reference_run="test/dimredTestPlots/SVDTestPlot0/plot",
                simulation_runs=sim_runs,
                start_stage=DIMRED_STAGES[-1],
                end_stage=DIMRED_STAGES[0],
                console=None,
                project_dir="test/DimredRunTest",
                n_processes=5,
            )

            # check invalid simulation runs
            self.assertRaises(
                DimredRunError,
                DimredRun,
                simulation_runs="test/dimredTestPlots200/plot",
                start_stage=DIMRED_STAGES[0],
                end_stage=DIMRED_STAGES[-1],
                console=None,
                project_dir="test/DimredRunTest",
                n_processes=5,
            )

            # check invalid cluster_args
            self.assertRaises(
                DimredRunError,
                DimredRun,
                simulation_runs=sim_runs,
                start_stage=DIMRED_STAGES[0],
                end_stage=DIMRED_STAGES[-1],
                console=None,
                project_dir="test/DimredRunTest",
                n_processes=5,
                cluster_args=["noMeans"],
            )

            # check invalid outlier-args
            self.assertRaises(
                DimredRunError,
                DimredRun,
                simulation_runs=sim_runs,
                start_stage=DIMRED_STAGES[0],
                end_stage=DIMRED_STAGES[-1],
                console=None,
                project_dir="test/DimredRunTest",
                n_processes=5,
                cluster_args=["kmeans"],
                outlier_args=["DoesNotExist"],
            )

            # check inexistent reference run
            self.assertRaises(
                DimredRunError,
                DimredRun,
                reference_run=os.path.join(tmpdir, "IDontExist"),
                simulation_runs=sim_runs,
                start_stage=DIMRED_STAGES[0],
                end_stage=DIMRED_STAGES[-1],
                console=None,
                project_dir="test/DimredRunTest",
                n_processes=5,
            )
            # check for empty simulation runs
            self.assertRaises(
                DimredRunError,
                DimredRun,
                simulation_runs="",
                start_stage=DIMRED_STAGES[0],
                end_stage=DIMRED_STAGES[-1],
                console=None,
                project_dir="test/DimredRunTest",
                n_processes=5,
            )

    def tearDown(self):
        # cleanup of created files
        test_files = os.listdir("test/DimredRunTest")
        test_files.pop(test_files.index("verificationFile.hdf5"))
        for entry in test_files:
            os.remove(os.path.join("test/DimredRunTest", entry))
