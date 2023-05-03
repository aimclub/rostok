import numpy as np

import pychrono as chrono
from example_vocabulary import (get_terminal_graph_three_finger, get_terminal_graph_two_finger, get_terminal_graph_no_joints)
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph

from rostok.block_builder_api.block_parameters import Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono
from rostok.block_builder_chrono.block_builder_chrono_api import ChronoBlockCreatorInterface as creator
from simple_designs import get_one_link_shifted_one_finger, get_two_link_one_finger
mechs = [get_terminal_graph_three_finger, get_terminal_graph_two_finger]

mechs = [get_two_link_one_finger]
def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]


for get_graph in mechs:

    graph = get_graph()
    print(get_joint_vector_from_graph(graph))
    controll_parameters = []
    # for _ in range(len(get_joint_vector_from_graph(graph))):
    #     controll_parameters.append([2.0,np.random.normal(0,1,1)[0]])

    for _ in range(len(get_joint_vector_from_graph(graph))):
        controll_parameters.append(np.random.normal(0,1,1)[0])

    controll_parameters = {"initial_value": [0, 0], "sin_parameters" : [[1, 0.1], [2,0.2]]}
    print(controll_parameters)
    #control_trajectories = [chrono.ChFunction_Sine(0,0.1,1), chrono.ChFunction_Sine(0,0.1,1)]
    sim = RobotSimulationChrono([])
    # Create object to grasp
    mat = Material()
    mat.Friction = 0.65
    mat.DampingF = 0.65
    obj = EnvironmentBodyBlueprint(shape=Box(3, 0.2, 3),
                                   material=mat,
                                   pos=FrameTransform([0, -0.4, 0], [1, 0, 0, 0]))
    sim.add_object(creator.init_block_from_blueprint(obj))
    sim.add_design(graph, controll_parameters, None, FrameTransform([0, 2.5, 0], rotation_x(180)))
    #print(sim.robot.sensor.joint_body_map)
    sim.simulate(100, 0.01, 10, True)
    #print(sim.robot.sensor.trajectories)
    print(sim.robot.data_storage.get_data("joint_trajectories"))
