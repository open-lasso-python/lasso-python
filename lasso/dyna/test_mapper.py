
from typing import Dict, Set, Tuple, Union
from unittest import TestCase

import numpy as np

from ..femzip.fz_config import FemzipArrayType, FemzipVariableCategory
from .ArrayType import ArrayType
from .FemzipMapper import FemzipMapper

part_global_femzip_translations: Dict[Tuple[FemzipArrayType, FemzipVariableCategory], Set[str]] = {
    # GLOBAL
    (FemzipArrayType.global_data, FemzipVariableCategory.GLOBAL): {
        # ArrayType.global_timesteps,
        ArrayType.global_internal_energy,
        ArrayType.global_kinetic_energy,
        ArrayType.global_total_energy,
        ArrayType.global_velocity,
    },
    # PART
    (FemzipArrayType.part_results, FemzipVariableCategory.PART): {
        ArrayType.part_hourglass_energy,
        ArrayType.part_internal_energy,
        ArrayType.part_kinetic_energy,
        ArrayType.part_mass,
        ArrayType.part_velocity,
    },
}

element_nope_femzip_translations: Dict[Tuple[str, FemzipVariableCategory], str] = {
    # NODE
    (FemzipArrayType.node_displacement.value,
     FemzipVariableCategory.NODE): ArrayType.node_displacement,
    (FemzipArrayType.node_accelerations.value,
     FemzipVariableCategory.NODE): ArrayType.node_acceleration,
    (FemzipArrayType.node_velocities.value,
     FemzipVariableCategory.NODE): ArrayType.node_velocity,
    (FemzipArrayType.node_temperatures.value,
     FemzipVariableCategory.NODE): ArrayType.node_temperature,
    (FemzipArrayType.node_heat_flux.value,
     FemzipVariableCategory.NODE): ArrayType.node_heat_flux,
    (FemzipArrayType.node_mass_scaling.value,
     FemzipVariableCategory.NODE): ArrayType.node_mass_scaling,
    (FemzipArrayType.node_temperature_gradient.value,
     FemzipVariableCategory.NODE): ArrayType.node_temperature_gradient,
    # BEAM
    (FemzipArrayType.beam_axial_force.value,
     FemzipVariableCategory.BEAM): ArrayType.element_beam_axial_force,
    (FemzipArrayType.beam_s_bending_moment.value,
     FemzipVariableCategory.BEAM): ArrayType.element_beam_bending_moment,
    (FemzipArrayType.beam_t_bending_moment.value,
     FemzipVariableCategory.BEAM): ArrayType.element_beam_bending_moment,
    (FemzipArrayType.beam_s_shear_resultant.value,
     FemzipVariableCategory.BEAM): ArrayType.element_beam_shear_force,
    (FemzipArrayType.beam_t_shear_resultant.value,
     FemzipVariableCategory.BEAM): ArrayType.element_beam_shear_force,
    (FemzipArrayType.beam_torsional_moment.value,
     FemzipVariableCategory.BEAM): ArrayType.element_beam_torsion_moment,
    (FemzipArrayType.stress_x.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_stress,
    (FemzipArrayType.stress_y.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_stress,
    (FemzipArrayType.stress_z.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_stress,
    (FemzipArrayType.stress_xy.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_stress,
    (FemzipArrayType.stress_yz.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_stress,
    (FemzipArrayType.stress_xz.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_stress,
    (FemzipArrayType.eff_pstrain.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_effective_plastic_strain,
    (FemzipArrayType.history_vars.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_history_vars,
    (FemzipArrayType.bending_moment_mx.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_bending_moment,
    (FemzipArrayType.bending_moment_my.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_bending_moment,
    (FemzipArrayType.bending_moment_mxy.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_bending_moment,
    (FemzipArrayType.shear_force_x.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_shear_force,
    (FemzipArrayType.shear_force_y.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_shear_force,
    (FemzipArrayType.normal_force_x.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_normal_force,
    (FemzipArrayType.normal_force_y.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_normal_force,
    (FemzipArrayType.normal_force_xy.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_normal_force,
    (FemzipArrayType.thickness.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_thickness,
    (FemzipArrayType.unknown_1.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_unknown_variables,
    (FemzipArrayType.unknown_2.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_unknown_variables,
    (FemzipArrayType.strain_inner_x.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_inner_y.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_inner_z.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_inner_xy.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_inner_yz.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_inner_xz.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_outer_x.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_outer_y.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_outer_z.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_outer_xy.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_outer_yz.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.strain_outer_xz.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_strain,
    (FemzipArrayType.internal_energy.value,
     FemzipVariableCategory.SHELL): ArrayType.element_shell_internal_energy,
    # SOLID
    (FemzipArrayType.stress_x.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_stress,
    (FemzipArrayType.stress_y.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_stress,
    (FemzipArrayType.stress_z.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_stress,
    (FemzipArrayType.stress_xy.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_stress,
    (FemzipArrayType.stress_yz.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_stress,
    (FemzipArrayType.stress_xz.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_stress,
    (FemzipArrayType.eff_pstrain.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_effective_plastic_strain,
    (FemzipArrayType.strain_inner_x.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.strain_inner_y.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.strain_inner_z.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.strain_inner_xy.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.history_vars.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_history_variables,
    (FemzipArrayType.strain_inner_yz.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.strain_inner_xz.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.strain_outer_x.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.strain_outer_y.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.strain_outer_z.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.strain_outer_xy.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.strain_outer_yz.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain,
    (FemzipArrayType.strain_outer_xz.value,
     FemzipVariableCategory.SOLID): ArrayType.element_solid_strain
}


class MapperTest(TestCase):

    def validate(self,
                 fz: Dict[Tuple[str, FemzipVariableCategory], np.ndarray],
                 d3plot_shape: tuple,
                 data_index_positions: Union[tuple, slice] = slice(None)):
        """Validate that the arrays have the same shape and that the
        raw data has been allocated to the correct postions in the
        d3plot arrays.

        Parameters
        ----------
        fz:
            femzip data following the same schema as in the api
        d3plot_shape:
            shape of the d3plot array
        data_index_positions:
            postions of the raw data in the d3plot array
        """
        d3plot_name = []

        m = FemzipMapper()

        # filter out parts and globals if they exist
        for key in fz.keys():
            if key in element_nope_femzip_translations:
                d3plot_name.append(element_nope_femzip_translations[key])

        d3plot_name = set(d3plot_name).pop()

        m.map(fz)

        _ = m.d3plot_arrays[d3plot_name]

    def test_nodal_disp(self):
        m = FemzipMapper()

        # NODE DISPLACEMENT
        nd = np.random.randn(2, 10, 3)

        fz = {
            (1, FemzipArrayType.node_displacement.value, FemzipVariableCategory.NODE): nd
        }

        m.map(fz)

        result = m.d3plot_arrays
        self.assertTrue(np.allclose(nd, result["node_displacement"]))

    def test_tshells(self):
        d = np.random.randn(2, 2)

        fz = {
            (1, 'Sigma-z (IP 2)', FemzipVariableCategory.THICK_SHELL): d,
            (2, 'Epsilon-xy (inner)', FemzipVariableCategory.THICK_SHELL): d,
        }

        m = FemzipMapper()

        m.map(fz)

        r = m.d3plot_arrays

        self.assertEqual(r[ArrayType.element_tshell_stress].shape, (2, 2, 2, 3))
        self.assertTrue(np.allclose(r[ArrayType.element_tshell_stress][:, :, 1, 2], d))

        self.assertEqual(r[ArrayType.element_tshell_strain].shape, (2, 2, 1, 4))
        self.assertTrue(np.allclose(r[ArrayType.element_tshell_strain][:, :, 0, 3], d))

    def test_internal_shell_energy(self):
        interal_energy = np.array([[1, 1, .12, 2.121202, 2.1123, 7.213]]).reshape(2, 3)

        fz = {
            (1, 'internal_energy', FemzipVariableCategory.SHELL): interal_energy,
        }
        m = FemzipMapper()

        m.map(fz)

        r = m.d3plot_arrays[ArrayType.element_shell_internal_energy]
        self.assertTrue(np.allclose(r, interal_energy))

    def test_dependent_variable(self):
        d1 = np.random.randn(2, 1200)
        d2 = np.random.randn(2, 1200)

        fz: Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray] = {
            (1, 'element_dependent_variable_2', FemzipVariableCategory.SHELL): d2,
            (2, 'element_dependent_variable_1', FemzipVariableCategory.SHELL): d1,
        }
        m = FemzipMapper()

        m.map(fz)

        r = m.d3plot_arrays

        self.assertEqual(r[ArrayType.element_shell_unknown_variables].shape, (2, 1200, 2))

        self.assertTrue(
            np.allclose(d1, r[ArrayType.element_shell_unknown_variables][:, :, 0]))
        self.assertTrue(
            np.allclose(d2, r[ArrayType.element_shell_unknown_variables][:, :, 1]))

    def test_effective_p_strain(self):
        m = FemzipMapper()
        d1 = np.random.randn(2, 20000)
        d2 = np.random.randn(2, 20000)
        d3 = np.random.randn(2, 20000)

        fz: Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray] = {
            (1, 'Effective plastic strain (   1)', FemzipVariableCategory.SHELL): d1,
            (2, 'Effective plastic strain (   2)', FemzipVariableCategory.SHELL): d2,
            (3, 'Effective plastic strain (   3)', FemzipVariableCategory.SHELL): d3,
        }

        m.map(fz)

        r = m.d3plot_arrays

        self.assertEqual(r[ArrayType.element_shell_effective_plastic_strain].shape, (2, 20000, 3))

        self.assertTrue(
            np.allclose(d1, r[ArrayType.element_shell_effective_plastic_strain][:, :, 0]))
        self.assertTrue(
            np.allclose(d2, r[ArrayType.element_shell_effective_plastic_strain][:, :, 1]))
        self.assertTrue(
            np.allclose(d3, r[ArrayType.element_shell_effective_plastic_strain][:, :, 2]))

    def test_others(self):
        stress_1 = np.random.randn(2, 2)
        stress_2 = np.random.randn(2, 2)
        stress_3 = np.random.randn(2, 2)

        strain1 = np.random.randn(1, 2)
        strain2 = np.random.randn(1, 2)

        hisotry_vars = np.array([[1, 2],
                                 [0, 3],
                                 [12, 2]], dtype=np.float)

        hisotry_vars1 = np.random.randn(3, 2)
        hisotry_vars2 = np.random.randn(3, 2)

        fz: Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray] = {
            # stress
            (1, 'Sigma-x (IP 6)', FemzipVariableCategory.SOLID): stress_1,
            (2, 'Sigma-y (IP 3)', FemzipVariableCategory.SOLID): stress_2,
            (3, 'Sigma-x (IP 3)', FemzipVariableCategory.SOLID): stress_3,

            # history
            (4, 'extra_value_per_element  2 (IP 2)', FemzipVariableCategory.SOLID): hisotry_vars,
            (5, 'extra_value_per_element  21 (IP 15)', FemzipVariableCategory.SOLID): hisotry_vars1,
            (6, 'extra_value_per_element  4 (IP 3)', FemzipVariableCategory.SOLID): hisotry_vars2,

            # strain
            (7, 'Epsilon-xy (outer)', FemzipVariableCategory.SHELL): strain1,
            (8, 'Epsilon-z (outer)', FemzipVariableCategory.SHELL): strain2,

        }

        m = FemzipMapper()

        m.map(fz)

        r = m.d3plot_arrays

        self.assertEqual(r[ArrayType.element_solid_stress].shape, (2, 2, 6, 2))

        self.assertTrue(np.allclose(stress_1, r[ArrayType.element_solid_stress][:, :, 5, 0]))
        self.assertTrue(np.allclose(stress_2, r[ArrayType.element_solid_stress][:, :, 2, 1]))
        self.assertTrue(np.allclose(stress_3, r[ArrayType.element_solid_stress][:, :, 2, 0]))

        self.assertEqual(r[ArrayType.element_solid_history_variables].shape, (3, 2, 15, 21))

        self.assertTrue(
            np.allclose(hisotry_vars, r[ArrayType.element_solid_history_variables][:, :, 1, 1]))

        self.assertTrue(
            np.allclose(hisotry_vars1, r[ArrayType.element_solid_history_variables][:, :, 14, 20]))

        self.assertTrue(
            np.allclose(hisotry_vars2, r[ArrayType.element_solid_history_variables][:, :, 2, 3]))

        self.assertEqual(r[ArrayType.element_shell_strain].shape, (1, 2, 2, 4))

    def test_beam(self):
        axial_force = np.random.randn(5, 12)
        shear = np.random.randn(2, 4)

        bending = np.random.randn(3, 123)
        torsion = np.random.rand(2, 5)

        fz: Dict[Tuple[int, str, FemzipVariableCategory], np.ndarray] = {
            (1, 'axial_force', FemzipVariableCategory.BEAM): axial_force,
            (2, 's_shear_resultant', FemzipVariableCategory.BEAM): shear,
            (3, 't_bending_moment', FemzipVariableCategory.BEAM): bending,
            (4, 'torsional_resultant', FemzipVariableCategory.BEAM): torsion,
        }

        m = FemzipMapper()

        m.map(fz)

        r = m.d3plot_arrays

        # axial force
        self.assertTrue(np.allclose(r["element_beam_axial_force"], axial_force))

        self.assertEqual(r[ArrayType.element_beam_shear_force].shape, (2, 4, 1))
        self.assertTrue(np.allclose(
            r[ArrayType.element_beam_shear_force][:, :, 0],
            shear
        ))

        # bending moment
        self.assertEqual(r[ArrayType.element_beam_bending_moment].shape, (3, 123, 2))
        self.assertTrue(np.allclose(
            r[ArrayType.element_beam_bending_moment][:, :, 1],
            bending
        ))

        # torion
        self.assertEqual(r[ArrayType.element_beam_torsion_moment].shape, (2, 5))
        self.assertTrue(np.allclose(
            r[ArrayType.element_beam_torsion_moment],
            torsion
        ))

    #     # TODO
    #     # unknown1 and unknown2

    #     # shp

    #     # airbags

    #     # rigids
