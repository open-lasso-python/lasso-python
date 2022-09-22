
from lasso.io.BinaryBuffer import BinaryBuffer
from lasso.femzip.femzip_api import FemzipAPI
from unittest import TestCase

import struct
import numpy as np

from lasso.dyna.D3plot import D3plot, FilterType, _negative_to_positive_state_indexes
from lasso.dyna.ArrayType import ArrayType


class D3plotTest(TestCase):

    def test_init(self):

        # settings
        self.maxDiff = None

        filepath = "test/simple_d3plot/d3plot"

        geometry_array_shapes = {
            'node_coordinates': (4915, 3),
            'element_solid_part_indexes': (0,),
            'element_solid_node_indexes': (0, 8),
            'element_tshell_node_indexes': (0, 8),
            'element_tshell_part_indexes': (0,),
            'element_beam_part_indexes': (0,),
            'element_beam_node_indexes': (0, 5),
            'element_shell_node_indexes': (4696, 4),
            'element_shell_part_indexes': (4696,),
            'node_ids': (4915,),
            'element_solid_ids': (0,),
            'element_beam_ids': (0,),
            'element_shell_ids': (4696,),
            'element_tshell_ids': (0,),
            'part_titles_ids': (1,),
            'part_titles': (1,)
        }

        state_array_shapes = {
            'timesteps': (1,),
            'global_kinetic_energy': (1,),
            'global_internal_energy': (1,),
            'global_total_energy': (1,),
            'global_velocity': (1, 3),
            'part_internal_energy': (1, 1),
            'part_kinetic_energy': (1, 1),
            'part_velocity': (1, 1, 3),
            'part_mass': (1, 1),
            'part_hourglass_energy': (1, 1),
            'node_displacement': (1, 4915, 3),
            'node_velocity': (1, 4915, 3),
            'node_acceleration': (1, 4915, 3),
            'element_shell_stress': (1, 4696, 3, 6),
            'element_shell_effective_plastic_strain': (1, 4696, 3),
            'element_shell_history_vars': (1, 4696, 3, 19),
            'element_shell_bending_moment': (1, 4696, 3),
            'element_shell_shear_force': (1, 4696, 2),
            'element_shell_normal_force': (1, 4696, 3),
            'element_shell_thickness': (1, 4696),
            'element_shell_unknown_variables': (1, 4696, 2),
            'element_shell_strain': (1, 4696, 2, 6),
            'element_shell_internal_energy': (1, 4696),
            'element_shell_is_alive': (1, 4696)
        }

        all_array_shapes = {**geometry_array_shapes, **state_array_shapes}

        # empty constructor
        D3plot()

        # file thing
        d3plot = D3plot(filepath)
        d3plot_shapes = {array_name: array.shape for array_name,
                         array in d3plot.arrays.items()}
        self.assertDictEqual(d3plot_shapes, all_array_shapes)

        # limited buffer files
        d3plot = D3plot(filepath, buffered_reading=True)
        d3plot_shapes = {array_name: array.shape for array_name,
                         array in d3plot.arrays.items()}
        self.assertDictEqual(d3plot_shapes, all_array_shapes)

        # test loading single state arrays
        for array_name, array_shape in state_array_shapes.items():
            d3plot = D3plot(filepath, state_array_filter=[array_name])
            d3plot_shapes = {array_name: array.shape for array_name,
                             array in d3plot.arrays.items()}
            self.assertDictEqual(d3plot_shapes, {
                                 **geometry_array_shapes,
                                 array_name: array_shape})

        # test loading individual states
        d3plot = D3plot(filepath, state_filter={0})
        d3plot2 = D3plot(filepath)
        hdr_diff, array_diff = d3plot.compare(d3plot2)
        self.assertEqual(d3plot.n_timesteps, 1)
        self.assertDictEqual(hdr_diff, {})
        self.assertDictEqual(array_diff, {})

    def test_header(self):

        test_header_data = {
            'title': '                                    ',
            'runtime': 1472027823,
            'filetype': 1,
            'source_version': -971095103,
            'release_version': 'R712',
            'version': 960.0,
            'ndim': 4,
            'numnp': 4915,
            'it': 0,
            'iu': 1,
            'iv': 1,
            'ia': 1,
            'nel2': 0,
            'nel4': 4696,
            'nel8': 0,
            'nelt': 0,
            'nel20': 0,
            'nel27': 0,
            'nel48': 0,
            'nv1d': 6,
            'nv2d': 102,
            'nv3d': 32,
            'nv3dt': 90,
            'nummat4': 1,
            'nummat8': 0,
            'nummat2': 0,
            'nummatt': 0,
            'icode': 6,
            'nglbv': 13,
            'numst': 0,
            'numds': 0,
            'neiph': 25,
            'neips': 19,
            'maxint': -10003,
            'nmsph': 0,
            'ngpsph': 0,
            'narbs': 9624,
            'ioshl1': 1000,
            'ioshl2': 1000,
            'ioshl3': 1000,
            'ioshl4': 1000,
            'ialemat': 0,
            'ncfdv1': 0,
            'ncfdv2': 0,
            'nadapt': 0,
            'nmmat': 1,
            'numfluid': 0,
            'inn': 1,
            'npefg': 0,
            'idtdt': 0,
            'extra': 0,
            'nt3d': 0,
            'neipb': 0,
        }

        d3plot = D3plot("test/simple_d3plot/d3plot")
        header = d3plot.header

        for name, value in test_header_data.items():
            self.assertEqual(header.raw_header[name], value, "Invalid var %s" % name)

    def test_beam_integration_points(self):

        self.maxDiff = None

        filepath = "test/d3plot_beamip/d3plot"
        maxmin_test_values = {
            # "element_beam_shear_stress": (-0.007316963, 0.),
            "element_beam_shear_stress": (0., 0.0056635854),
            "element_beam_axial_stress": (-0.007316963, 0.),
            "element_beam_plastic_strain": (0., 0.0056297667),
            "element_beam_axial_strain": (-0.0073745, 0.),
        }

        d3plot = D3plot(filepath)

        for array_name, minmax in maxmin_test_values.items():
            array = d3plot.arrays[array_name]
            self.assertAlmostEqual(array.min(), minmax[0], msg="{0}: {1} != {2}".format(
                array_name, array.min(), minmax[0]))
            self.assertAlmostEqual(array.max(), minmax[1], msg="{0}: {1} != {2}".format(
                array_name, array.max(), minmax[1]))

    def test_correct_sort_of_more_than_100_state_files(self):

        filepath = "test/order_d3plot/d3plot"

        d3plot = D3plot(filepath)

        timesteps = d3plot.arrays[ArrayType.global_timesteps]
        self.assertListEqual(
            timesteps.astype(int).tolist(),
            [1, 2, 10, 11, 12, 22, 100])

    def test_femzip_basic(self):

        self.maxDiff = None

        filepath1 = "test/femzip/d3plot.fz"
        filepath2 = "test/femzip/d3plot"

        d3plot_kwargs_list = [
            {},
            {"buffered_reading": True},
            {"state_filter": [0]}
        ]

        D3plot.use_advanced_femzip_api = False

        for d3plot_kwargs in d3plot_kwargs_list:

            d3plot1 = D3plot(filepath1, **d3plot_kwargs)
            d3plot2 = D3plot(filepath2, **d3plot_kwargs)

            hdr_diff, array_diff = d3plot1.compare(d3plot2, array_eps=1E-2)

            self.assertDictEqual(hdr_diff, {})
            self.assertDictEqual(array_diff, {})

    def test_femzip_extended(self):

        self.maxDiff = None

        filepath1 = "test/femzip/d3plot.fz"
        filepath2 = "test/femzip/d3plot"

        d3plot_kwargs_list = [
            {},
            {"buffered_reading": True},
            {"state_filter": [0]}
        ]

        api = FemzipAPI()
        if not api.has_femunziplib_license():
            return

        D3plot.use_advanced_femzip_api = True

        for d3plot_kwargs in d3plot_kwargs_list:

            d3plot1 = D3plot(filepath1, **d3plot_kwargs)
            d3plot2 = D3plot(filepath2, **d3plot_kwargs)

            hdr_diff, array_diff = d3plot1.compare(d3plot2, array_eps=1E-2)

            self.assertDictEqual(hdr_diff, {})
            self.assertDictEqual(array_diff, {})

    def test_part_filter(self):

        self.maxDiff = None

        filepath = "test/simple_d3plot/d3plot"
        part_ids = [1]

        d3plot = D3plot(filepath)
        shell_filter = d3plot.get_part_filter(FilterType.SHELL, part_ids)
        self.assertEqual(shell_filter.sum(), 4696)

        part_filter = d3plot.get_part_filter(FilterType.PART, part_ids)
        self.assertEqual(part_filter.sum(), 1)

        node_filter = d3plot.get_part_filter(FilterType.NODE, part_ids)
        self.assertEqual(len(node_filter), 4915)

    def test_read_solid_integration_points(self):

        filepath = "test/d3plot_solid_int/d3plot"

        # data from META
        stress_valid = np.array([
            2.132084E+02, 1.792203E+02, 1.397527E+02,
            2.307352E+02, 2.132105E+02, 1.792210E+02,
            1.397558E+02, 2.307304E+02
        ])
        pstrain_valid = np.array([
            2.227418E-02, 2.576126E-03, 1.909884E-02,
            3.695280E-02, 2.227416E-02, 2.576110E-03,
            1.909888E-02, 3.695256E-02,
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
        bb.memoryview = memoryview(struct.pack("<f", -999999.))

        result = D3plot._is_end_of_file_marker(bb, 0, np.float32)
        self.assertEqual(result, True)

        # -999999. in float64
        bb = BinaryBuffer()
        bb.memoryview = memoryview(struct.pack("<d", -999999.))

        result = D3plot._is_end_of_file_marker(bb, 0, np.float64)
        self.assertEqual(result, True)

        # some other number
        bb = BinaryBuffer()
        bb.memoryview = memoryview(struct.pack("<d", -41.))

        result = D3plot._is_end_of_file_marker(bb, 0, np.float64)
        self.assertEqual(result, False)

        # invalid dtype
        with self.assertRaises(ValueError):
            result = D3plot._is_end_of_file_marker(bb, 0, np.int32)
