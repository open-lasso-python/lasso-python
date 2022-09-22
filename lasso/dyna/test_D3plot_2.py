
import os
import tempfile
from unittest import TestCase

import numpy as np

from lasso.dyna.D3plot import D3plot
from lasso.dyna.ArrayType import ArrayType


class D3plotTest2(TestCase):

    def test_write(self):

        self.maxDiff = None

        filepaths = [
            "test/simple_d3plot/d3plot",
            "test/d3plot_beamip/d3plot",
            "test/d3plot_node_temperature/d3plot",
            "test/d3plot_solid_int/d3plot",
        ]

        d3plot_kwargs_list = [
            {},
            {"buffered_reading": True},
        ]

        with tempfile.TemporaryDirectory() as dirpath:

            for d3plot_kwargs in d3plot_kwargs_list:
                for d3plot_filepath, d3plot_kwargs in zip(filepaths, d3plot_kwargs_list):

                    print(d3plot_filepath)

                    # read d3plot
                    d3plot1 = D3plot(d3plot_filepath, **d3plot_kwargs)

                    # rewrite d3plot
                    out_filepath = os.path.join(dirpath, "yay.d3plot")
                    d3plot1.write_d3plot(out_filepath)

                    # read it in again and compare
                    d3plot2 = D3plot(out_filepath, **d3plot_kwargs)
                    hdr_diff, array_diff = d3plot1.compare(d3plot2)

                    err_msg = f"{d3plot_filepath}: {d3plot_kwargs}"
                    self.assertDictEqual(hdr_diff, {}, err_msg)
                    self.assertDictEqual(array_diff, {}, err_msg)

    def test_write_new(self):

        self.maxDiff = None

        d3plot1 = D3plot()
        d3plot1.arrays[ArrayType.node_coordinates] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        d3plot1.arrays[ArrayType.element_shell_node_indexes] = np.array([[0, 2, 1, 1]])
        d3plot1.arrays[ArrayType.element_shell_part_indexes] = np.array([0])
        d3plot1.arrays[ArrayType.node_displacement] = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])

        with tempfile.TemporaryDirectory() as dirpath:
            filepath = os.path.join(dirpath, "yay.d3plot")

            # single file
            d3plot1.write_d3plot(filepath)
            d3plot2 = D3plot(filepath)
            hdr_diff, array_diff = d3plot1.compare(d3plot2)
            array_diff = {name: reason for name,
                          reason in array_diff.items() if "missing in original" not in reason}
            self.assertDictEqual(hdr_diff, {})
            self.assertDictEqual(array_diff, {})

            # multiple files
            d3plot1.write_d3plot(filepath, single_file=False)
            d3plot2 = D3plot(filepath)
            hdr_diff, array_diff = d3plot1.compare(d3plot2)
            array_diff = {name: reason for name,
                          reason in array_diff.items() if "missing in original" not in reason}
            self.assertDictEqual(hdr_diff, {})
            self.assertDictEqual(array_diff, {})
            self.assertTrue(os.path.isfile(filepath))
            self.assertTrue(os.path.isfile(filepath + "01"))

    def test_append_4_shell_hists_then_read_bug(self):

        self.maxDiff = None

        # we need some d3plot
        d3plot = D3plot()
        d3plot.arrays[ArrayType.node_coordinates] = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        d3plot.arrays[ArrayType.element_shell_node_indexes] = np.array([[0, 2, 1, 1]])
        d3plot.arrays[ArrayType.element_shell_part_indexes] = np.array([0])
        d3plot.arrays[ArrayType.node_displacement] = np.array(
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)

        # there was a bug occurring if more than 3 history vars
        # were added, these were written and read back in again
        n_history_vars = 4

        with tempfile.TemporaryDirectory() as dirpath:
            filepath1 = os.path.join(dirpath, "original")
            d3plot.write_d3plot(filepath1)

            # open it again to have a safe copy
            d3plot1 = D3plot(filepath1)
            n_timesteps, n_shells, n_layers = 1, d3plot1.header.n_shells, 3

            d3plot1.arrays[ArrayType.element_shell_history_vars] = np.random.random(
                (n_timesteps, n_shells, n_layers, n_history_vars))

            filepath2 = os.path.join(dirpath, "modified")
            d3plot1.write_d3plot(filepath2)

            d3plot_modif = D3plot(filepath2)
            self.assertTrue(ArrayType.element_shell_internal_energy not in d3plot_modif.arrays)

    def test_reading_selected_states(self):

        # read all states
        filepath = "test/d3plot_solid_int/d3plot"

        d3plot = D3plot(filepath)
        d3plot2 = D3plot(filepath, state_filter=np.arange(0, 22))

        hdr_diff, array_diff = d3plot.compare(d3plot2)

        self.assertDictEqual(hdr_diff, {})
        self.assertDictEqual(array_diff, {})

        # select first and last state
        d3plot = D3plot(filepath, state_filter={0, -1})

        node_id = 119
        disp_node_real = np.array(
            [[50., 70., 5.],
             [47.50418, 70., -10.000001]],
            dtype=np.float32)

        node_index = d3plot.arrays[ArrayType.node_ids].tolist().index(node_id)
        node_disp = d3plot.arrays[ArrayType.node_displacement][:, node_index]

        np.testing.assert_allclose(node_disp, disp_node_real, rtol=1E-4)
