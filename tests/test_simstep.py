import random
#from rostok.control_chrono.controller import YaxisShaker

from test_ruleset import (get_terminal_graph_no_joints,
                          get_terminal_graph_three_finger,
                          get_terminal_graph_two_finger)

from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.block_parameters import FrameTransform, Material
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono


def test_control_bind_and_create_sim():
    """
        Test for simulation class, control binder and control generator
    """

    mechs = [
        get_terminal_graph_three_finger, get_terminal_graph_no_joints, get_terminal_graph_two_finger
    ]

    for get_graph in mechs:
        graph = get_graph()
        n_joints = len(get_joint_vector_from_graph(graph))
        const_torque_koef = [random.random() for _ in range(n_joints)]
        controll_parameters = {"initial_value": const_torque_koef}
        times_step = 1e-3
        mat = Material()
        mat.Friction = 0.65
        mat.DampingF = 0.65
        obj_bp = EnvironmentBodyBlueprint(material=mat,
                                          pos=FrameTransform([0, 1, 0], [0, -0.048, 0.706, 0.706]))

        sim = RobotSimulationChrono([])
        sim.add_design(graph, controll_parameters)
        #shake = YaxisShaker(2, 2)
        sim.add_object(creator().create_environment_body(obj_bp), True)

        sim_output = sim.simulate(10000, times_step, 10)


