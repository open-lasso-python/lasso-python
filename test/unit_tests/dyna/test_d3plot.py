import io
import os
import platform
import struct
import sys
import tempfile
from unittest import TestCase, skipUnless

import numpy as np

from lasso.dyna.array_type import ArrayType
from lasso.dyna.d3plot import D3plot, FilterType, _negative_to_positive_state_indexes
from lasso.femzip.femzip_api import FemzipAPI
from lasso.io.binary_buffer import BinaryBuffer


FEMZIP_SUPPORTED = sys.platform in {"linux", "win32"} and platform.machine().lower() in {
    "x86_64",
    "amd64",
}
FEMZIP_SKIP_REASON = "Bundled FEMZIP libraries require x86-64 Linux or Windows"


class D3plotTest(TestCase):
    def test_init(self):
        # settings
        self.maxDiff = None

        filepath = "test/test_data/simple_d3plot/d3plot"

        geometry_array_shapes = {
            "node_coordinates": (4915, 3),
            "element_solid_part_indexes": (0,),
            "element_solid_node_indexes": (0, 8),
            "element_tshell_node_indexes": (0, 8),
            "element_tshell_part_indexes": (0,),
            "element_beam_part_indexes": (0,),
            "element_beam_node_indexes": (0, 5),
            "element_shell_node_indexes": (4696, 4),
            "element_shell_part_indexes": (4696,),
            "node_ids": (4915,),
            "element_solid_ids": (0,),
            "element_beam_ids": (0,),
            "element_shell_ids": (4696,),
            "element_tshell_ids": (0,),
            "part_titles_ids": (1,),
            "part_titles": (1,),
        }

        state_array_shapes = {
            "timesteps": (1,),
            "global_kinetic_energy": (1,),
            "global_internal_energy": (1,),
            "global_total_energy": (1,),
            "global_velocity": (1, 3),
            "part_internal_energy": (1, 1),
            "part_kinetic_energy": (1, 1),
            "part_velocity": (1, 1, 3),
            "part_mass": (1, 1),
            "part_hourglass_energy": (1, 1),
            "node_displacement": (1, 4915, 3),
            "node_velocity": (1, 4915, 3),
            "node_acceleration": (1, 4915, 3),
            "element_shell_stress": (1, 4696, 3, 6),
            "element_shell_effective_plastic_strain": (1, 4696, 3),
            "element_shell_history_vars": (1, 4696, 3, 19),
            "element_shell_bending_moment": (1, 4696, 3),
            "element_shell_shear_force": (1, 4696, 2),
            "element_shell_normal_force": (1, 4696, 3),
            "element_shell_thickness": (1, 4696),
            "element_shell_unknown_variables": (1, 4696, 2),
            "element_shell_strain": (1, 4696, 2, 6),
            "element_shell_internal_energy": (1, 4696),
            "element_shell_is_alive": (1, 4696),
        }

        all_array_shapes = {**geometry_array_shapes, **state_array_shapes}

        # empty constructor
        D3plot()

        # file thing
        d3plot = D3plot(filepath)
        d3plot_shapes = {array_name: array.shape for array_name, array in d3plot.arrays.items()}
        self.assertDictEqual(d3plot_shapes, all_array_shapes)

        # limited buffer files
        d3plot = D3plot(filepath, buffered_reading=True)
        d3plot_shapes = {array_name: array.shape for array_name, array in d3plot.arrays.items()}
        self.assertDictEqual(d3plot_shapes, all_array_shapes)

        # test loading single state arrays
        for array_name, array_shape in state_array_shapes.items():
            d3plot = D3plot(filepath, state_array_filter=[array_name])
            d3plot_shapes = {array_name: array.shape for array_name, array in d3plot.arrays.items()}
            self.assertDictEqual(d3plot_shapes, {**geometry_array_shapes, array_name: array_shape})

        # test loading individual states
        d3plot = D3plot(filepath, state_filter={0})
        d3plot2 = D3plot(filepath)
        hdr_diff, array_diff = d3plot.compare(d3plot2)
        self.assertEqual(d3plot.n_timesteps, 1)
        self.assertDictEqual(hdr_diff, {})
        self.assertDictEqual(array_diff, {})

    def test_header(self):
        test_header_data = {
            "title": "                                    ",
            "runtime": 1472027823,
            "filetype": 1,
            "source_version": -971095103,
            "release_version": "R712",
            "version": 960.0,
            "ndim": 4,
            "numnp": 4915,
            "it": 0,
            "iu": 1,
            "iv": 1,
            "ia": 1,
            "nel2": 0,
            "nel4": 4696,
            "nel8": 0,
            "nelt": 0,
            "nel20": 0,
            "nel27": 0,
            "nel48": 0,
            "nv1d": 6,
            "nv2d": 102,
            "nv3d": 32,
            "nv3dt": 90,
            "nummat4": 1,
            "nummat8": 0,
            "nummat2": 0,
            "nummatt": 0,
            "icode": 6,
            "nglbv": 13,
            "numst": 0,
            "numds": 0,
            "neiph": 25,
            "neips": 19,
            "maxint": -10003,
            "nmsph": 0,
            "ngpsph": 0,
            "narbs": 9624,
            "ioshl1": 1000,
            "ioshl2": 1000,
            "ioshl3": 1000,
            "ioshl4": 1000,
            "ialemat": 0,
            "ncfdv1": 0,
            "ncfdv2": 0,
            "nadapt": 0,
            "nmmat": 1,
            "numfluid": 0,
            "inn": 1,
            "npefg": 0,
            "idtdt": 0,
            "extra": 0,
            "nt3d": 0,
            "neipb": 0,
        }

        d3plot = D3plot("test/test_data/simple_d3plot/d3plot")
        header = d3plot.header

        for name, value in test_header_data.items():
            self.assertEqual(header.raw_header[name], value, f"Invalid var {name}")

    def test_beam_integration_points(self):
        self.maxDiff = None

        filepath = "test/test_data/d3plot_beamip/d3plot"
        maxmin_test_values = {
            # "element_beam_shear_stress": (-0.007316963, 0.),
            "element_beam_shear_stress": (0.0, 0.0056635854),
            "element_beam_axial_stress": (-0.007316963, 0.0),
            "element_beam_plastic_strain": (0.0, 0.0056297667),
            "element_beam_axial_strain": (-0.0073745, 0.0),
        }

        d3plot = D3plot(filepath)

        for array_name, minmax in maxmin_test_values.items():
            array = d3plot.arrays[array_name]
            self.assertAlmostEqual(
                array.min(), minmax[0], msg=f"{array_name}: {array.min()} != {minmax[0]}"
            )
            self.assertAlmostEqual(
                array.max(), minmax[1], msg=f"{array_name}: {array.max()} != {minmax[1]}"
            )

    def test_correct_sort_of_more_than_100_state_files(self):
        filepath = "test/test_data/order_d3plot/d3plot"

        d3plot = D3plot(filepath)

        timesteps = d3plot.arrays[ArrayType.global_timesteps]
        self.assertListEqual(timesteps.astype(int).tolist(), [1, 2, 10, 11, 12, 22, 100])

    @skipUnless(FEMZIP_SUPPORTED, FEMZIP_SKIP_REASON)
    def test_femzip_basic(self):
        self.maxDiff = None

        filepath1 = "test/test_data/femzip/d3plot.fz"
        filepath2 = "test/test_data/femzip/d3plot"

        d3plot_kwargs_list = [{}, {"buffered_reading": True}, {"state_filter": [0]}]

        self.addCleanup(setattr, D3plot, "use_advanced_femzip_api", D3plot.use_advanced_femzip_api)
        D3plot.use_advanced_femzip_api = False

        for d3plot_kwargs in d3plot_kwargs_list:
            d3plot1 = D3plot(filepath1, **d3plot_kwargs)
            d3plot2 = D3plot(filepath2, **d3plot_kwargs)

            hdr_diff, array_diff = d3plot1.compare(d3plot2, array_eps=1e-2)

            self.assertDictEqual(hdr_diff, {})
            self.assertDictEqual(array_diff, {})

    @skipUnless(FEMZIP_SUPPORTED, FEMZIP_SKIP_REASON)
    def test_femzip_extended(self):
        self.maxDiff = None

        filepath1 = "test/test_data/femzip/d3plot.fz"
        filepath2 = "test/test_data/femzip/d3plot"

        d3plot_kwargs_list = [{}, {"buffered_reading": True}, {"state_filter": [0]}]

        api = FemzipAPI()
        if not api.has_femunziplib_license():
            self.skipTest("Extended FEMZIP API requires a FEMUNZIPLIB-DYNA license")

        self.addCleanup(setattr, D3plot, "use_advanced_femzip_api", D3plot.use_advanced_femzip_api)
        D3plot.use_advanced_femzip_api = True

        for d3plot_kwargs in d3plot_kwargs_list:
            d3plot1 = D3plot(filepath1, **d3plot_kwargs)
            d3plot2 = D3plot(filepath2, **d3plot_kwargs)

            hdr_diff, array_diff = d3plot1.compare(d3plot2, array_eps=1e-2)

            self.assertDictEqual(hdr_diff, {})
            self.assertDictEqual(array_diff, {})

    def test_part_filter(self):
        self.maxDiff = None

        filepath = "test/test_data/simple_d3plot/d3plot"
        part_ids = [1]

        d3plot = D3plot(filepath)
        shell_filter = d3plot.get_part_filter(FilterType.SHELL, part_ids)
        self.assertEqual(shell_filter.sum(), 4696)

        part_filter = d3plot.get_part_filter(FilterType.PART, part_ids)
        self.assertEqual(part_filter.sum(), 1)

        node_filter = d3plot.get_part_filter(FilterType.NODE, part_ids)
        self.assertEqual(len(node_filter), 4915)

    def test_read_solid_integration_points(self):
        filepath = "test/test_data/d3plot_solid_int/d3plot"

        # data from META
        stress_valid = np.array([
            2.132084e02,
            1.792203e02,
            1.397527e02,
            2.307352e02,
            2.132105e02,
            1.792210e02,
            1.397558e02,
            2.307304e02,
        ])
        pstrain_valid = np.array([
            2.227418e-02,
            2.576126e-03,
            1.909884e-02,
            3.695280e-02,
            2.227416e-02,
            2.576110e-03,
            1.909888e-02,
            3.695256e-02,
        ])
        last_timestep = -1
        first_element = 0
        stress_xx = 0

        d3plot = D3plot(filepath)

        stress = d3plot.arrays[ArrayType.element_solid_stress]
        np.array_equal(stress[last_timestep, first_element, :, stress_xx], stress_valid)

        pstrain = d3plot.arrays[ArrayType.element_solid_effective_plastic_strain]
        np.array_equal(pstrain[last_timestep, first_element, :], pstrain_valid)

    def test_negative_to_positive_state_indexes(self) -> None:
        indexes = set()
        new_indexes = _negative_to_positive_state_indexes(indexes, len(indexes))
        self.assertSetEqual(indexes, new_indexes)

        indexes = {0, 1, 2}
        with self.assertRaises(ValueError):
            new_indexes = _negative_to_positive_state_indexes(indexes, 2)

        indexes = {0, -1}
        new_indexes = _negative_to_positive_state_indexes(indexes, 8)
        self.assertSetEqual(new_indexes, {0, 7})

    def test_is_end_of_file_marker(self) -> None:
        # -999999. in float32
        bb = BinaryBuffer()
        bb.memoryview = memoryview(struct.pack("<f", -999999.0))

        result = D3plot._is_end_of_file_marker(bb, 0, np.float32)
        self.assertEqual(result, True)

        # -999999. in float64
        bb = BinaryBuffer()
        bb.memoryview = memoryview(struct.pack("<d", -999999.0))

        result = D3plot._is_end_of_file_marker(bb, 0, np.float64)
        self.assertEqual(result, True)

        # some other number
        bb = BinaryBuffer()
        bb.memoryview = memoryview(struct.pack("<d", -41.0))

        result = D3plot._is_end_of_file_marker(bb, 0, np.float64)
        self.assertEqual(result, False)

        # invalid dtype
        with self.assertRaises(ValueError):
            result = D3plot._is_end_of_file_marker(bb, 0, np.int32)

    def test_write(self):
        self.maxDiff = None

        filepaths = [
            "test/test_data/simple_d3plot/d3plot",
            "test/test_data/d3plot_beamip/d3plot",
            "test/test_data/d3plot_node_temperature/d3plot",
            "test/test_data/d3plot_solid_int/d3plot",
        ]

        d3plot_kwargs_list = [{}, {"buffered_reading": True}]

        with tempfile.TemporaryDirectory() as dirpath:
            for d3plot_kwargs in d3plot_kwargs_list:
                for d3plot_filepath, _ in zip(filepaths, d3plot_kwargs_list, strict=False):
                    print(d3plot_filepath)

                    # read d3plot
                    d3plot1 = D3plot(d3plot_filepath, **d3plot_kwargs)

                    # rewrite d3plot
                    out_filepath = os.path.join(dirpath, "yay.d3plot")
                    with self.assertLogs("lasso.dyna.d3plot", level="DEBUG"):
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
            array_diff = {
                name: reason
                for name, reason in array_diff.items()
                if "missing in original" not in reason
            }
            self.assertDictEqual(hdr_diff, {})
            self.assertDictEqual(array_diff, {})

            # multiple files
            d3plot1.write_d3plot(filepath, single_file=False)
            d3plot2 = D3plot(filepath)
            hdr_diff, array_diff = d3plot1.compare(d3plot2)
            array_diff = {
                name: reason
                for name, reason in array_diff.items()
                if "missing in original" not in reason
            }
            self.assertDictEqual(hdr_diff, {})
            self.assertDictEqual(array_diff, {})
            self.assertTrue(os.path.isfile(filepath))
            self.assertTrue(os.path.isfile(filepath + "01"))

    def test_append_4_shell_hists_then_read_bug(self):
        self.maxDiff = None

        # we need some d3plot
        d3plot = D3plot()
        d3plot.arrays[ArrayType.node_coordinates] = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
        )
        d3plot.arrays[ArrayType.element_shell_node_indexes] = np.array([[0, 2, 1, 1]])
        d3plot.arrays[ArrayType.element_shell_part_indexes] = np.array([0])
        d3plot.arrays[ArrayType.node_displacement] = np.array(
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32
        )

        # there was a bug occurring if more than 3 history vars
        # were added, these were written and read back in again
        n_history_vars = 4

        with tempfile.TemporaryDirectory() as dirpath:
            filepath1 = os.path.join(dirpath, "original")
            d3plot.write_d3plot(filepath1)

            # open it again to have a safe copy
            d3plot1 = D3plot(filepath1)
            n_timesteps, n_shells, n_layers = 1, d3plot1.header.n_shells, 3

            d3plot1.arrays[ArrayType.element_shell_history_vars] = np.random.random((
                n_timesteps,
                n_shells,
                n_layers,
                n_history_vars,
            ))

            filepath2 = os.path.join(dirpath, "modified")
            d3plot1.write_d3plot(filepath2)

            d3plot_modif = D3plot(filepath2)
            self.assertTrue(ArrayType.element_shell_internal_energy not in d3plot_modif.arrays)

    def test_reading_selected_states(self):
        # read all states
        filepath = "test/test_data/d3plot_solid_int/d3plot"

        d3plot = D3plot(filepath)
        d3plot2 = D3plot(filepath, state_filter=np.arange(0, 22))

        hdr_diff, array_diff = d3plot.compare(d3plot2)

        self.assertDictEqual(hdr_diff, {})
        self.assertDictEqual(array_diff, {})

        # select first and last state
        d3plot = D3plot(filepath, state_filter={0, -1})

        node_id = 119
        disp_node_real = np.array(
            [[50.0, 70.0, 5.0], [47.50418, 70.0, -10.000001]], dtype=np.float32
        )

        node_index = d3plot.arrays[ArrayType.node_ids].tolist().index(node_id)
        node_disp = d3plot.arrays[ArrayType.node_displacement][:, node_index]

        np.testing.assert_allclose(node_disp, disp_node_real, rtol=1e-4)

    def test_write_sph_geometry_only(self):
        d3plot = D3plot()
        d3plot.arrays[ArrayType.node_coordinates] = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        d3plot.arrays[ArrayType.sph_node_indexes] = np.array([0, 2, 3], dtype=np.int32)
        d3plot.arrays[ArrayType.sph_node_material_index] = np.array([0, 1, 1], dtype=np.int32)

        with tempfile.TemporaryDirectory() as dirpath:
            filepath = os.path.join(dirpath, "sph_geom.d3plot")
            d3plot.write_d3plot(filepath)

            loaded = D3plot(filepath)
            self.assertEqual(loaded.header.n_sph_nodes, 3)
            self.assertEqual(loaded.header.n_sph_materials, 2)
            np.testing.assert_array_equal(
                loaded.arrays[ArrayType.sph_node_indexes],
                d3plot.arrays[ArrayType.sph_node_indexes],
            )
            np.testing.assert_array_equal(
                loaded.arrays[ArrayType.sph_node_material_index],
                d3plot.arrays[ArrayType.sph_node_material_index],
            )
            np.testing.assert_allclose(
                loaded.arrays[ArrayType.node_coordinates],
                d3plot.arrays[ArrayType.node_coordinates],
            )

        # Also verify in-memory BinaryIO stream write
        stream = io.BytesIO()
        d3plot.write_d3plot(stream)
        self.assertGreater(stream.tell(), 0)

    def test_write_sph_full_and_partial_states(self):
        self.maxDiff = None
        rng = np.random.default_rng(42)
        n_states = 3
        n_nodes = 6
        n_sph = 4

        for wordsize, itype, ftype in ((4, np.int32, np.float32), (8, np.int64, np.float64)):
            for single_file in (True, False):
                d3plot = D3plot()
                d3plot.header.wordsize = wordsize
                d3plot.header.itype = itype
                d3plot.header.ftype = ftype

                # Mix shell element + SPH particles so neighboring section offsets are exercised
                d3plot.arrays[ArrayType.node_coordinates] = rng.normal(size=(n_nodes, 3)).astype(
                    ftype
                )
                d3plot.arrays[ArrayType.element_shell_node_indexes] = np.array(
                    [[0, 1, 2, 2]], dtype=itype
                )
                d3plot.arrays[ArrayType.element_shell_part_indexes] = np.array([0], dtype=itype)
                # Use non-monotonic node indexes and non-contiguous material indexes
                d3plot.arrays[ArrayType.sph_node_indexes] = np.array([5, 2, 4, 3], dtype=itype)
                d3plot.arrays[ArrayType.sph_node_material_index] = np.array(
                    [3, 0, 3, 1], dtype=itype
                )

                d3plot.arrays[ArrayType.global_timesteps] = np.array([0.0, 0.5, 1.0], dtype=ftype)
                d3plot.arrays[ArrayType.node_displacement] = rng.normal(
                    size=(n_states, n_nodes, 3)
                ).astype(ftype)
                d3plot.arrays[ArrayType.sph_deletion] = np.array(
                    [
                        [False, False, False, False],
                        [False, True, False, False],
                        [False, True, False, True],
                    ],
                    dtype=bool,
                )
                d3plot.arrays[ArrayType.sph_radius] = rng.uniform(
                    0.1, 1.0, size=(n_states, n_sph)
                ).astype(ftype)
                d3plot.arrays[ArrayType.sph_pressure] = rng.normal(size=(n_states, n_sph)).astype(
                    ftype
                )
                d3plot.arrays[ArrayType.sph_stress] = rng.normal(size=(n_states, n_sph, 6)).astype(
                    ftype
                )
                d3plot.arrays[ArrayType.sph_effective_plastic_strain] = rng.uniform(
                    0.0, 0.5, size=(n_states, n_sph)
                ).astype(ftype)
                d3plot.arrays[ArrayType.sph_density] = rng.uniform(
                    1.0, 8.0, size=(n_states, n_sph)
                ).astype(ftype)
                d3plot.arrays[ArrayType.sph_internal_energy] = rng.uniform(
                    0.0, 10.0, size=(n_states, n_sph)
                ).astype(ftype)
                d3plot.arrays[ArrayType.sph_n_neighbors] = rng.integers(
                    5, 40, size=(n_states, n_sph)
                ).astype(ftype)
                d3plot.arrays[ArrayType.sph_strain] = rng.normal(size=(n_states, n_sph, 6)).astype(
                    ftype
                )
                d3plot.arrays[ArrayType.sph_strainrate] = rng.normal(
                    size=(n_states, n_sph, 6)
                ).astype(ftype)
                d3plot.arrays[ArrayType.sph_mass] = rng.uniform(
                    0.01, 0.2, size=(n_states, n_sph)
                ).astype(ftype)

                with tempfile.TemporaryDirectory() as dirpath:
                    filepath = os.path.join(dirpath, "sph_full.d3plot")
                    d3plot.write_d3plot(filepath, single_file=single_file)

                    loaded = D3plot(filepath)
                    loaded_buffered = D3plot(filepath, buffered_reading=True)

                    for read_label, candidate in (
                        ("unbuffered", loaded),
                        ("buffered", loaded_buffered),
                    ):
                        hdr_diff, array_diff = d3plot.compare(candidate)
                        array_diff = {
                            name: reason
                            for name, reason in array_diff.items()
                            if "missing in original" not in reason
                        }
                        ctx = f"wordsize={wordsize}, single_file={single_file}, {read_label}"
                        self.assertDictEqual(hdr_diff, {}, ctx)
                        self.assertDictEqual(array_diff, {}, ctx)
                        self.assertEqual(candidate.header.n_sph_nodes, n_sph, ctx)
                        self.assertEqual(candidate.header.n_sph_materials, 3, ctx)
                        self.assertEqual(
                            candidate.arrays[ArrayType.sph_deletion].dtype, np.dtype(bool), ctx
                        )
                        np.testing.assert_array_equal(
                            candidate.arrays[ArrayType.sph_deletion],
                            d3plot.arrays[ArrayType.sph_deletion],
                        )

                    # Compare populated headers/arrays directly (non-empty raw_header)
                    buf_hdr_diff, buf_arr_diff = loaded.compare(loaded_buffered)
                    self.assertDictEqual(buf_hdr_diff, {})
                    self.assertDictEqual(buf_arr_diff, {})

                    # Second-hop rewrite verifies all 65 populated raw_header words round-trip
                    rewrite_path = os.path.join(dirpath, "sph_rewrite.d3plot")
                    loaded.write_d3plot(rewrite_path, single_file=single_file)
                    reloaded = D3plot(rewrite_path)
                    rew_hdr_diff, rew_arr_diff = loaded.compare(reloaded)
                    self.assertDictEqual(rew_hdr_diff, {})
                    self.assertDictEqual(rew_arr_diff, {})

                    # Filtered single-array reads including sph_strainrate and sph_deletion
                    for sph_array_name in (
                        ArrayType.sph_deletion,
                        ArrayType.sph_strainrate,
                        ArrayType.sph_mass,
                    ):
                        filtered = D3plot(filepath, state_array_filter=[sph_array_name])
                        np.testing.assert_allclose(
                            filtered.arrays[sph_array_name],
                            d3plot.arrays[sph_array_name],
                        )

        # Partial subset with true-strain flag and sph_history_vars (no sph_deletion or timesteps)
        partial = D3plot()
        partial._sph_info.has_true_strains = True
        partial.arrays[ArrayType.node_coordinates] = np.zeros((n_sph, 3), dtype=np.float32)
        partial.arrays[ArrayType.sph_node_indexes] = np.arange(n_sph, dtype=np.int32)
        partial.arrays[ArrayType.sph_node_material_index] = np.zeros(n_sph, dtype=np.int32)
        partial.arrays[ArrayType.sph_strain] = np.full((2, n_sph, 6), 0.125, dtype=np.float32)
        partial.arrays[ArrayType.sph_mass] = np.full((2, n_sph), 0.25, dtype=np.float32)
        expected_hist = np.arange(2 * n_sph * 2, dtype=np.float32).reshape(2, n_sph, 2) + 1.5
        partial.arrays[ArrayType.sph_history_vars] = expected_hist

        with tempfile.TemporaryDirectory() as dirpath:
            filepath = os.path.join(dirpath, "sph_partial.d3plot")
            partial.write_d3plot(filepath, single_file=False)
            loaded_partial = D3plot(filepath)
            self.assertTrue(loaded_partial._sph_info.has_true_strains)
            self.assertEqual(loaded_partial._sph_info.n_sph_history_vars, 2)
            np.testing.assert_allclose(
                loaded_partial.arrays[ArrayType.sph_strain],
                partial.arrays[ArrayType.sph_strain],
            )
            np.testing.assert_allclose(
                loaded_partial.arrays[ArrayType.sph_mass],
                partial.arrays[ArrayType.sph_mass],
            )
            self.assertFalse(np.any(loaded_partial.arrays[ArrayType.sph_deletion]))

            # Verify trailing sph_history_vars columns in raw state files (10 vars/particle)
            numsph_partial = 1 + 6 + 1 + 2
            for i_state, suffix in enumerate(("01", "02")):
                state_words = np.fromfile(filepath + suffix, dtype=np.float32)
                sph_block = state_words[1 : 1 + n_sph * numsph_partial].reshape(
                    n_sph, numsph_partial
                )
                np.testing.assert_allclose(sph_block[:, -2:], expected_hist[i_state])

        # Also verify sph_history_vars alone drives n_timesteps inference when timesteps omitted
        hist_only = D3plot()
        hist_only.arrays[ArrayType.node_coordinates] = np.zeros((2, 3), dtype=np.float32)
        hist_only.arrays[ArrayType.sph_node_indexes] = np.array([0, 1], dtype=np.int32)
        hist_only.arrays[ArrayType.sph_node_material_index] = np.array([0, 1], dtype=np.int32)
        hist_only.arrays[ArrayType.sph_history_vars] = np.array(
            [[[2.5, 3.5], [4.5, 5.5]], [[6.5, 7.5], [8.5, 9.5]]], dtype=np.float32
        )
        with tempfile.TemporaryDirectory() as dirpath:
            filepath = os.path.join(dirpath, "sph_hist_only.d3plot")
            hist_only.write_d3plot(filepath, single_file=False)
            loaded_hist_only = D3plot(filepath)
            self.assertEqual(loaded_hist_only.n_timesteps, 2)
            self.assertEqual(loaded_hist_only._sph_info.n_sph_history_vars, 2)

    def test_write_sph_validation_errors(self):
        # sph_strainrate without sph_strain cannot be represented by LS-DYNA isphfg9
        d3plot = D3plot()
        d3plot.arrays[ArrayType.node_coordinates] = np.zeros((2, 3), dtype=np.float32)
        d3plot.arrays[ArrayType.sph_node_indexes] = np.array([0, 1], dtype=np.int32)
        d3plot.arrays[ArrayType.sph_node_material_index] = np.array([0, 0], dtype=np.int32)
        d3plot.arrays[ArrayType.sph_strainrate] = np.zeros((1, 2, 6), dtype=np.float32)
        with self.assertRaises(ValueError):
            d3plot.write_d3plot(io.BytesIO())

        # Unpaired SPH geometry array must raise ValueError
        d3plot_unpaired = D3plot()
        d3plot_unpaired.arrays[ArrayType.node_coordinates] = np.zeros((2, 3), dtype=np.float32)
        d3plot_unpaired.arrays[ArrayType.sph_node_material_index] = np.array([0, 0], dtype=np.int32)
        with self.assertRaises(ValueError):
            d3plot_unpaired.write_d3plot(io.BytesIO())

        # Length-mismatched SPH geometry arrays must raise ValueError
        d3plot_mismatch = D3plot()
        d3plot_mismatch.arrays[ArrayType.node_coordinates] = np.zeros((3, 3), dtype=np.float32)
        d3plot_mismatch.arrays[ArrayType.sph_node_indexes] = np.array([0, 1, 2], dtype=np.int32)
        d3plot_mismatch.arrays[ArrayType.sph_node_material_index] = np.array([0, 1], dtype=np.int32)
        with self.assertRaises(ValueError):
            d3plot_mismatch.write_d3plot(io.BytesIO())
