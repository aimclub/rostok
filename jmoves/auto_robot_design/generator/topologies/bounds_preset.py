from auto_robot_design.generator.topologies.graph_manager_2l import GraphManager2L


def set_preset_bounds(graph_manager, bounds):
    """
    Set preset bounds for the joint points in the graph manager.

    Args:
        graph_manager (GraphManager): The graph manager object.
        bounds (dict): A dictionary containing the bounds for each generator info.
            The keys are the names of the generators, and the values are tuples
            representing the lower and upper bounds for each generator.

    Returns:
        None
    """
    nam2jp = {jp.name: jp for jp in graph_manager.generator_dict.keys()}
    
    for name, (init_coord, range) in bounds.items():
        jp = nam2jp[name]
        graph_manager.generator_dict[jp].mutation_range = range
        if init_coord != "Skip":
            graph_manager.generator_dict[jp].initial_coordinate = init_coord

bounds_3n2p_02 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.07)]),
    "Main_knee": ("Skip", [None, None, (-0.1, 0.1)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_1": (None, [(-0.05, 0.1), None, (-0.3, -0.1)])
}
bounds_3n2p_12 = {
    "Main_knee": ("Skip", [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_1": (None, [(-0.05, 0.15), None, (-0.1, 0.1)])
}

bounds_6n4p_s_012 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ("Skip", [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_0": (None, [(-0.1, 0.05), None, (-0.25, -0.01)]),
    "branch_1": (None, [(-0.1, -0.02), None, (-0.1, 0.1)]),
    "branch_2": (None, [(-0.1, -0.02), None, (0.05, 0.15)])
}
bounds_6n4p_a_012 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ("Skip", [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_0": (None, [(-0.1, 0.05), None, (-0.15, -0.01)]),
    "branch_1": (None, [(-0.1, -0.02), None, (-0.15, 0.05)]),
    "branch_2": (None, [(-0.15, -0.02), None, (0.05, 0.15)])
}
bounds_6n4p_a_120 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ("Skip", [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_2": (None, [(-0.1, 0.05), None, (-0.15, -0.01)]),
    "branch_0": (None, [(-0.1, -0.02), None, (-0.15, 0.0)]),
    "branch_1": (None, [(-0.15, -0.02), None, (0.05, 0.15)])
}
bounds_6n4p_a_102 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ("Skip", [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_1": (None, [(-0.1, 0.05), None, (-0.25, -0.01)]),
    "branch_0": (None, [(-0.1, -0.02), None, (-0.1, 0.1)]),
    "branch_2": (None, [(-0.1, -0.02), None, (0.05, 0.15)])
}
bounds_6n4p_a_210 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ("Skip", [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_2": (None, [(-0.1, 0.05), None, (-0.15, 0.05)]),
    "branch_1": (None, [(-0.1, -0.02), None, (-0.1, 0.1)]),
    "branch_0": (None, [(-0.15, -0.02), None, (0.08, 0.2)])
}
bounds_6n4p_a_201 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ("Skip", [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_1": (None, [(-0.1, 0.05), None, (-0.15, 0.05)]),
    "branch_2": (None, [(-0.1, -0.02), None, (-0.1, 0.1)]),
    "branch_0": (None, [(-0.15, -0.02), None, (0.08, 0.2)])
}

bounds_preset_6n4p_a_021 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ("Skip", [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_0": (None, [(-0.1, 0.05), None, (-0.2, -0.05)]),
    "branch_2": (None, [(-0.1, -0.02), None, (-0.1, 0.1)]),
    "branch_1": (None, [(-0.15, -0.02), None, (0.08, 0.2)])
}

def get_preset_by_index_with_bounds(idx: int):
    if idx == -1:
        gm = GraphManager2L()
        gm.name =f"topology_{idx}"
        gm.reset()
        gm.build_main(0.4, fully_actuated=True)
        gm.set_mutation_ranges()
        return gm

    if idx == 0:
        gm = GraphManager2L()
        gm.name =f"topology_{idx}"
        gm.reset()
        gm.build_main(0.4)
        gm.build_3n2p_branch([0, 2])
        set_preset_bounds(gm, bounds_3n2p_02)
        gm.set_mutation_ranges()
        return gm

    if idx == 1:
        gm = GraphManager2L()
        gm.name =f"topology_{idx}"
        gm.reset()
        gm.build_main(0.4)
        gm.build_3n2p_branch([1, 2])
        set_preset_bounds(gm, bounds_3n2p_12)
        gm.set_mutation_ranges()
        return gm

    if idx == 2:
        gm = GraphManager2L()
        gm.name =f"topology_{idx}"
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_symmetric([0, 1, 2])
        set_preset_bounds(gm, bounds_6n4p_s_012)
        gm.set_mutation_ranges()
        return gm

    if idx == 3:
        gm = GraphManager2L()
        gm.name =f"topology_{idx}"
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([0, 1, 2])
        set_preset_bounds(gm, bounds_6n4p_a_012)
        gm.set_mutation_ranges()
        return gm

    if idx == 4:
        gm = GraphManager2L()
        gm.name =f"topology_{idx}"
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([0, 2, 1])
        set_preset_bounds(gm, bounds_preset_6n4p_a_021)
        gm.set_mutation_ranges()
        return gm

    if idx == 5:
        gm = GraphManager2L()
        gm.name =f"topology_{idx}"
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([1, 0, 2])
        set_preset_bounds(gm, bounds_6n4p_a_102)
        gm.set_mutation_ranges()
        return gm

    if idx == 6:
        gm = GraphManager2L()
        gm.name =f"topology_{idx}"
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([1, 2, 0])
        set_preset_bounds(gm, bounds_6n4p_a_120)
        gm.set_mutation_ranges()
        return gm

    if idx == 7:
        gm = GraphManager2L()
        gm.name =f"topology_{idx}"
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([2, 0, 1])
        set_preset_bounds(gm, bounds_6n4p_a_201)
        gm.set_mutation_ranges()
        return gm

    if idx == 8:
        gm = GraphManager2L()
        gm.name =f"topology_{idx}"
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([2, 1, 0])
        set_preset_bounds(gm, bounds_6n4p_a_210)
        gm.set_mutation_ranges()
        return gm
