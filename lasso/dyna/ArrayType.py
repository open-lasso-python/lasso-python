
import typing


class ArrayType:
    '''Specifies the names for specific arrays

    Enums from this class shall be used as a preferred practice
    instead of the string array names to ensure compatability.
    '''

    # global
    global_timesteps = "timesteps"  #: shape: (n_timesteps)
    global_kinetic_energy = "global_kinetic_energy"  #: shape: (n_timesteps)
    global_internal_energy = "global_internal_energy"  #: shape: (n_timesteps)
    global_total_energy = "global_total_energy"  #: shape: (n_timesteps)
    global_velocity = "global_velocity"  #: shape: (n_timesteps)
    # nodes
    node_ids = "node_ids"  #: shape: (n_nodes)
    node_coordinates = "node_coordinates"  #: shape: (n_nodes, x_y_z)
    node_displacement = "node_displacement"  #: shape: (n_states, n_nodes, x_y_z)
    node_velocity = "node_velocity"  #: shape: (n_states, n_nodes, x_y_z)
    node_acceleration = "node_acceleration"  #: shape: (n_states, n_nodes, x_y_z)
    node_is_alive = "node_is_alive"  #: shape: (n_states, n_nodes)
    node_temperature = "node_temperature"  #: shape: (n_states, n_nodes) or (n_states, n_nodes, 3)
    node_heat_flux = "node_heat_flux"  #: shape: (n_states, n_nodes, 3)
    node_mass_scaling = "node_mass_scaling"  #: shape: (n_states, n_nodes)
    node_temperature_gradient = "node_temperature_gradient"  #: shape(n_states, n_nodes)
    node_residual_forces = "node_residual_forces"  #: shape (n_states, n_nodes, fx_fy_fz)
    node_residual_moments = "node_residual_moments"  #: shape (n_states, n_nodes, mx_my_mz)
    # solids
    element_solid_node_indexes = "element_solid_node_indexes"  #: shape: (n_solids, 8)
    element_solid_part_indexes = "element_solid_part_indexes"  #: shape: (n_solids)
    element_solid_ids = "element_solid_ids"  #: shape: (n_solids)
    element_solid_thermal_data = \
        "element_solid_thermal_data"  #: shape: (n_states, n_solids, n_solids_thermal_vars)
    element_solid_stress = \
        "element_solid_stress"  #: shape: (n_states, n_solids, n_solid_layers, xx_yy_zz_xy_yz_xz)
    element_solid_effective_plastic_strain = \
        "element_solid_effective_plastic_strain"  #: shape: (n_states, n_solid_layers, n_solids)
    element_solid_history_variables = \
        "element_solid_history_variables"  #: shape: (n_states, n_solids, n_solid_layers, n_solids_history_vars)
    element_solid_strain = \
        "element_solid_strain"  #: shape: (n_states, n_solids, n_solid_layers, xx_yy_zz_xy_yz_xz)
    element_solid_plastic_strain_tensor = \
        "element_solid_plastic_strain_tensor"  #: shape: (n_states, n_solids, n_solid_layers, xx_yy_zz_xy_yz_xz)
    element_solid_thermal_strain_tensor = \
        "element_solid_thermal_strain_tensor"  #: shape: (n_states, n_solids, n_solid_layers, xx_yy_zz_xy_yz_xz)
    element_solid_is_alive = "element_solid_is_alive"  #: shape: (n_states, n_solids)
    element_solid_extra_nodes = "element_solid_extra_nodes"  #: shape: (n_solids, 2)
    element_solid_node10_extra_node_indexes = \
        "element_solid_node10_extra_node_indexes"  #: shape: (n_solids, 2)
    element_solid_node20_element_index = \
        "element_solid_node20_element_index"  #: shape: (n_node20_solids)
    element_solid_node20_extra_node_indexes = \
        "element_solid_node20_extra_node_indexes"  #: shape: (n_node20_solids, 12)
    element_solid_node27_element_index = \
        "element_solid_node27_element_index"  #: shape: (n_node27_solids)
    element_solid_node27_extra_node_indexes = \
        "element_solid_node27_extra_node_indexes"  #: shape: (n_node27_solids, 27)
    element_solid_node21_penta_element_index = \
        "element_solid_node21_penta_element_index"  #: shape: (n_node21p_solids)
    element_solid_node21_penta_extra_node_indexes = \
        "element_solid_node21_penta_extra_node_indexes"  #: shape: (n_node21p_solids, 21)
    element_solid_node15_tetras_element_index = \
        "element_solid_node15_tetras_element_index"  #: shape: (n_node15t_solids)
    element_solid_node15_tetras_extra_node_indexes = \
        "element_solid_node15_tetras_extra_node_indexes"  #: shape: (n_node15t_solids, 7)
    element_solid_node20_tetras_element_index = \
        "element_solid_node20_tetras_element_index"  #: shape: (n_node20t_solids)
    element_solid_node20_tetras_extra_node_indexes = \
        "element_solid_node20_tetras_extra_node_indexes"  #: shape: (n_node20t_solids, 20)
    element_solid_node40_pentas_element_index = \
        "element_solid_node40_pentas_element_index"  #: shape: (n_node40h_solids)
    element_solid_node40_pentas_extra_node_indexes = \
        "element_solid_node40_pentas_extra_node_indexes"  #: shape: (n_node40h_solids, 40)
    element_solid_node64_hexas_element_index = \
        "element_solid_node64_hexas_element_index"  #: shape: (n_node64h_solids)
    element_solid_node64_hexas_extra_node_indexes = \
        "element_solid_node64_hexas_extra_node_indexes"  #: shape: (n_node64h_solids, 64)

    # tshells
    element_tshell_part_indexes = "element_tshell_part_indexes"  #: shape: (n_tshells)
    element_tshell_node_indexes = "element_tshell_node_indexes"  #: shape: (n_tshells, 8)
    element_tshell_ids = "element_tshell_ids"  #: shape: (n_tshells)
    element_tshell_stress = \
        "element_tshell_stress"  #: shape: (n_states, n_tshells, n_tshells_layers, xx_yy_zz_xy_yz_xz)
    element_tshell_effective_plastic_strain = \
        "element_tshell_effective_plastic_strain"  #: shape: (n_states, n_tshells, n_tshells_layers)
    element_tshell_history_variables = \
        "element_tshell_history_variables"  #: shape: (n_states, n_tshells, n_tshells_layers, xx_yy_zz_xy_yz_xz)
    element_tshell_is_alive = "element_tshell_is_alive"  #: shape: (n_states, n_tshells)
    element_tshell_strain = \
        "element_tshell_strain"  #: shape: (n_states, n_tshells, upper_lower, xx_yy_zz_xy_yz_xz)
    # beams
    element_beam_part_indexes = "element_beam_part_indexes"  #: shape: (n_beams)
    element_beam_node_indexes = "element_beam_node_indexes"  #: shape: (n_beams, 5)
    element_beam_ids = "element_beam_ids"  #: shape: (n_beams)

    element_beam_axial_force = "element_beam_axial_force"  #: shape: (n_states, n_beams)
    element_beam_shear_force = "element_beam_shear_force"  #: shape: (n_states, n_beams, s_t)
    element_beam_bending_moment = "element_beam_bending_moment"  #: shape: (n_states, n_beams, s_t)
    element_beam_torsion_moment = "element_beam_torsion_moment"  #: shape: (n_states, n_beams)
    element_beam_shear_stress = \
        "element_beam_shear_stress"  #: shape: (n_states, n_beams, n_beams_layers, rs_rt)
    element_beam_axial_stress = \
        "element_beam_axial_stress"  #: shape: (n_states, n_beams, n_beams_layers)
    element_beam_plastic_strain = \
        "element_beam_plastic_strain"  #: shape: (n_states, n_beams, n_beams_layers)
    element_beam_axial_strain = \
        "element_beam_axial_strain"  #: shape: (n_states, n_beams, n_beams_layers)
    element_beam_history_vars = \
        "element_beam_history_vars"  #: shape: (n_states, n_beams, n_beams_layers+3, n_beams_history_vars)
    element_beam_is_alive = "element_beam_is_alive"  #: shape: (n_states, n_beams)
    # shells
    element_shell_part_indexes = "element_shell_part_indexes"  #: shape (n_shells, 4)
    element_shell_node_indexes = "element_shell_node_indexes"  #: shape (n_shells)
    element_shell_ids = "element_shell_ids"  #: shape (n_shells)
    element_shell_stress = \
        "element_shell_stress"  #: shape (n_states, n_shells_non_rigid, n_shell_layers, xx_yy_zz_xy_yz_xz)
    element_shell_effective_plastic_strain = \
        "element_shell_effective_plastic_strain"  #: shape (n_states, n_shells_non_rigid, n_shell_layers)
    element_shell_history_vars = \
        "element_shell_history_vars"  #: shape (n_states, n_shells_non_rigid, n_shell_layers, n_shell_history_vars)
    element_shell_bending_moment = \
        "element_shell_bending_moment"  #: shape (n_states, n_shells_non_rigid, mx_my_mxy)
    element_shell_shear_force = \
        "element_shell_shear_force"  #: shape (n_states, n_shells_non_rigid, qx_qy)
    element_shell_normal_force = \
        "element_shell_normal_force"  #: shape (n_states, n_shells_non_rigid, nx_ny_nxy)
    element_shell_thickness = "element_shell_thickness"  #: shape (n_states, n_shells_non_rigid)
    element_shell_unknown_variables = \
        "element_shell_unknown_variables"  #: shape (n_states, n_shells_non_rigid, 2)
    element_shell_internal_energy = \
        "element_shell_internal_energy"  #: shape (n_states, n_shells_non_rigid)
    element_shell_strain = \
        "element_shell_strain"  #: shape (n_states, n_shells_non_rigid, upper_lower, xx_yy_zz_xy_yz_xz)
    element_shell_is_alive = "element_shell_is_alive"  #: shape (n_states, n_shells_non_rigid)
    element_shell_node8_extra_node_indexes = \
        "element_shell_node8_extra_node_indexes"  #: shape (n_shells, 4)
    element_shell_node8_element_index = "element_shell_node8_element_index"  #: shape: (n_shells)
    element_shell_plastic_strain_tensor = \
        "element_shell_plastic_strain_tensor"  #: shape(n_states, n_shells_non_rigid, n_layers, 6)
    element_shell_thermal_strain_tensor = \
        "element_shell_thermal_strain_tensor"  #: shape(n_states, n_shells_non_rigid, 6)
    # parts
    part_material_type = "part_material_type"  #: shape: (n_parts)
    part_ids = "part_ids"  #: shape: (n_parts)
    part_ids_unordered = "part_ids_unordered"  #: shape: (n_parts)
    part_ids_cross_references = "part_ids_cross_references"  #: shape: (n_parts)
    part_titles = "part_titles"  #: shape: (n_parts)
    part_titles_ids = "part_titles_ids"  #: shape: (n_parts)

    part_internal_energy = "part_internal_energy"  #: shape: (n_states, n_parts)
    part_kinetic_energy = "part_kinetic_energy"  #: shape: (n_states, n_parts)
    part_velocity = "part_velocity"  #: shape: (n_states, n_parts, x_y_z)
    part_mass = "part_mass"  #: shape: (n_states, n_parts)
    part_hourglass_energy = "part_hourglass_energy"  #: shape: (n_states, n_parts)
    # sph
    sph_node_indexes = "sph_node_indexes"  #: shape: (n_sph_nodes)
    sph_node_material_index = "sph_node_material_index"  #: shape: (n_sph_nodes)
    sph_is_alive = "sph_is_alive"  #: shape: (n_states, n_sph_particles)
    sph_radius = "sph_radius"  #: shape: (n_states, n_sph_particles)
    sph_pressure = "sph_pressure"  #: shape: (n_states, n_sph_particles)
    sph_stress = "sph_stress"  #: shape: (n_states, n_sph_particles, xx_yy_zz_xy_yz_xz)
    sph_effective_plastic_strain = "sph_effective_plastic_strain"  #:
    sph_density = "sph_density"  #: shape: (n_states, n_sph_particles)
    sph_internal_energy = "sph_internal_energy"  #: shape: (n_states, n_sph_particles)
    sph_n_neighbors = "sph_n_neighbors"  #: (n_states, n_sph_particles)
    sph_strain = "sph_strain"  #: shape: (n_states, n_sph_particles, xx_yy_zz_xy_yz_xz)
    sph_strainrate = "sph_strainrate"  #: shape: (n_states, n_sph_particles, xx_yy_zz_xy_yz_xz)
    sph_mass = "sph_mass"  #: shape: (n_states, n_sph_particles)
    sph_deletion = "sph_deletion"  #: shape: (n_states, n_sph_particles)
    sph_history_vars = \
        "sph_history_vars"  #: shape: (n_states, n_sph_particles, n_sph_history_vars)
    # airbag
    airbag_variable_names = "airbag_variable_names"  #: shape: (n_variables)
    airbag_variable_types = "airbag_variable_types"  #: shape: (n_variables)

    airbags_first_particle_id = "airbags_first_particle_id"  #: shape: (n_airbags)
    airbags_n_particles = "airbags_n_particles"  #: shape: (n_airbags)
    airbags_ids = "airbags_ids"  #: shape: (n_airbags)
    airbags_n_gas_mixtures = "airbags_n_gas_mixtures"  #: shape: (n_airbags)
    airbags_n_chambers = "airbags_n_chambers"  #: shape: (n_airbags)

    airbag_n_active_particles = "airbag_n_active_particles"  #: shape: (n_states, n_airbags)
    airbag_bag_volume = "airbag_bag_volume"  #: shape: (n_states, n_airbags)

    airbag_particle_gas_id = "airbag_particle_gas_id"  #: shape: (n_states, n_airbag_particles)
    airbag_particle_chamber_id = \
        "airbag_particle_chamber_id"  #: shape: (n_states, n_airbag_particles)
    airbag_particle_leakage = "airbag_particle_leakage"  #: shape: (n_states, n_airbag_particles)
    airbag_particle_mass = "airbag_particle_mass"  #: shape: (n_states, n_airbag_particles)
    airbag_particle_radius = "airbag_particle_radius"  #: shape: (n_states, n_airbag_particles)
    airbag_particle_spin_energy = \
        "airbag_particle_spin_energy"  #: shape: (n_states, n_airbag_particles)
    airbag_particle_translation_energy = \
        "airbag_particle_translation_energy"  #: shape: (n_states, n_airbag_particles)
    airbag_particle_nearest_segment_distance = \
        "airbag_particle_nearest_segment_distance"  #: shape: (n_states, n_airbag_particles)
    airbag_particle_position = \
        "airbag_particle_position"  #: shape: (n_states, n_airbag_particles, x_y_z)
    airbag_particle_velocity = \
        "airbag_particle_velocity"  #: shape: (n_states, n_airbag_particles, x_y_z)
    # rigid roads
    rigid_road_node_ids = "rigid_road_node_ids"  #: shape: (rigid_road_n_nodes)
    rigid_road_node_coordinates = \
        "rigid_road_node_coordinates"  #: #: shape: (rigid_road_n_nodes, x_y_z)
    rigid_road_ids = "rigid_road_ids"  #: shape: (n_roads)
    rigid_road_n_segments = "rigid_road_n_segments"  #: shape: (n_roads)
    rigid_road_segment_node_ids = \
        "rigid_road_segment_node_ids"  #: list!: (n_roads, 4*n_road_segments)
    rigid_road_segment_road_id = "rigid_road_segment_road_id"  #: list!: (n_raods, n_road_segments)

    rigid_road_displacement = "rigid_road_displacement"  #: shape: (n_states, n_roads, x_y_z)
    rigid_road_velocity = "rigid_road_velocity"  #: shape: (n_states, n_roads, x_y_z)
    # rigid body
    rigid_body_part_indexes = "rigid_body_part_index"  #: shape: (n_rigid_bodies)
    rigid_body_n_nodes = "rigid_body_n_nodes"  #: shape: (n_rigid_bodies)
    #: list!: (n_rigid_bodies, n_rigid_body_nodes (differs))
    rigid_body_node_indexes_list = "rigid_body_node_indexes_list"
    rigid_body_n_active_nodes = "rigid_body_n_active_nodes"  #: shape: (n_rigid_bodies)
    rigid_body_active_node_indexes_list = "rigid_body_active_node_indexes_list"  #:

    rigid_body_coordinates = "rigid_body_coordinates"  #: shape: (n_states, n_rigid_bodies, x_y_z)
    rigid_body_rotation_matrix = \
        "rigid_body_rotation_matrix"  #: shape: (n_states, n_rigid_bodies, 9)
    rigid_body_velocity = "rigid_body_velocity"  #: shape: (n_states, n_rigid_bodies, x_y_z)
    rigid_body_rot_velocity = \
        "rigid_body_rotational_velocity"  #: shape: (n_states, n_rigid_bodies, x_y_z)
    rigid_body_acceleration = \
        "rigid_body_acceleration"  #: shape: (n_states, n_rigid_bodies, x_y_z)
    rigid_body_rot_acceleration = \
        "rigid_body_rotational_acceleration"  #: shape: (n_states, n_rigid_bodies, x_y_z)
    # contact info
    contact_title_ids = "contact_title_ids"  #:
    contact_titles = "contact_titles"  #:
    # ALE
    ale_material_ids = "ale_material_ids"  #:
    # rigid wall
    rigid_wall_force = "rigid_wall_force"  #: shape: (n_states, n_rigid_walls)
    rigid_wall_position = "rigid_wall_position"  #: shape: (n_states, n_rigid_walls, x_y_z)

    @staticmethod
    def get_state_array_names() -> typing.List[str]:
        return [
            # global
            ArrayType.global_timesteps,
            ArrayType.global_kinetic_energy,
            ArrayType.global_internal_energy,
            ArrayType.global_total_energy,
            ArrayType.global_velocity,
            # parts
            ArrayType.part_internal_energy,
            ArrayType.part_kinetic_energy,
            ArrayType.part_velocity,
            ArrayType.part_mass,
            ArrayType.part_hourglass_energy,
            # rigid wall
            ArrayType.rigid_wall_force,
            ArrayType.rigid_wall_position,
            # nodes
            ArrayType.node_temperature,
            ArrayType.node_heat_flux,
            ArrayType.node_mass_scaling,
            ArrayType.node_displacement,
            ArrayType.node_velocity,
            ArrayType.node_acceleration,
            ArrayType.node_temperature_gradient,
            ArrayType.node_residual_forces,
            ArrayType.node_residual_moments,
            # solids
            ArrayType.element_solid_thermal_data,
            ArrayType.element_solid_stress,
            ArrayType.element_solid_effective_plastic_strain,
            ArrayType.element_solid_history_variables,
            ArrayType.element_solid_strain,
            ArrayType.element_solid_is_alive,
            ArrayType.element_solid_plastic_strain_tensor,
            ArrayType.element_solid_thermal_strain_tensor,
            # thick shells
            ArrayType.element_tshell_stress,
            ArrayType.element_tshell_effective_plastic_strain,
            ArrayType.element_tshell_history_variables,
            ArrayType.element_tshell_strain,
            ArrayType.element_tshell_is_alive,
            # beams
            ArrayType.element_beam_axial_force,
            ArrayType.element_beam_shear_force,
            ArrayType.element_beam_bending_moment,
            ArrayType.element_beam_torsion_moment,
            ArrayType.element_beam_shear_stress,
            ArrayType.element_beam_axial_stress,
            ArrayType.element_beam_plastic_strain,
            ArrayType.element_beam_axial_strain,
            ArrayType.element_beam_history_vars,
            ArrayType.element_beam_is_alive,
            # shells
            ArrayType.element_shell_stress,
            ArrayType.element_shell_effective_plastic_strain,
            ArrayType.element_shell_history_vars,
            ArrayType.element_shell_bending_moment,
            ArrayType.element_shell_shear_force,
            ArrayType.element_shell_normal_force,
            ArrayType.element_shell_thickness,
            ArrayType.element_shell_unknown_variables,
            ArrayType.element_shell_internal_energy,
            ArrayType.element_shell_strain,
            ArrayType.element_shell_is_alive,
            ArrayType.element_shell_plastic_strain_tensor,
            ArrayType.element_shell_thermal_strain_tensor,
            # sph
            ArrayType.sph_deletion,
            ArrayType.sph_radius,
            ArrayType.sph_pressure,
            ArrayType.sph_stress,
            ArrayType.sph_effective_plastic_strain,
            ArrayType.sph_density,
            ArrayType.sph_internal_energy,
            ArrayType.sph_n_neighbors,
            ArrayType.sph_strain,
            ArrayType.sph_mass,
            # airbag
            ArrayType.airbag_n_active_particles,
            ArrayType.airbag_bag_volume,
            ArrayType.airbag_particle_gas_id,
            ArrayType.airbag_particle_chamber_id,
            ArrayType.airbag_particle_leakage,
            ArrayType.airbag_particle_mass,
            ArrayType.airbag_particle_radius,
            ArrayType.airbag_particle_spin_energy,
            ArrayType.airbag_particle_translation_energy,
            ArrayType.airbag_particle_nearest_segment_distance,
            ArrayType.airbag_particle_position,
            ArrayType.airbag_particle_velocity,
            # rigid road
            ArrayType.rigid_road_displacement,
            ArrayType.rigid_road_velocity,
            # rigid body
            ArrayType.rigid_body_coordinates,
            ArrayType.rigid_body_rotation_matrix,
            ArrayType.rigid_body_velocity,
            ArrayType.rigid_body_rot_velocity,
            ArrayType.rigid_body_acceleration,
            ArrayType.rigid_body_rot_acceleration,
        ]
