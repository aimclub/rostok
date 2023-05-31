import numpy as np
import matplotlib.pyplot as plt
import pychrono as chrono
from example_vocabulary import (get_terminal_graph_no_joints,
                                get_terminal_graph_three_finger,
                                get_terminal_graph_two_finger)

from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.block_parameters import FrameTransform, Material
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.block_builder_chrono_alt.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.control_chrono.controller import ConstController, PIDController
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from rostok.library.rule_sets.simple_designs import (get_one_link_one_finger,
                                                     get_two_link_one_finger)
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono

mechs = [get_terminal_graph_three_finger, get_terminal_graph_two_finger]
from rostok.graph_grammar.graph_utils import plot_graph, plot_graph_ids
from rostok.library.rule_sets.ruleset_locomotion import (get_bip,
                                                         get_bip_single,
                                                         get_box,
                                                         get_box_joints,
                                                         get_box_one_joint)
from rostok.virtual_experiment.sensors import DataStorage
mechs = [get_one_link_one_finger]
mechs = [get_two_link_one_finger]
mechs = [get_bip]
def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]


for get_graph in mechs:

    graph = get_graph()
    # plot_graph(graph)
    # plot_graph_ids(graph)
    print(get_joint_vector_from_graph(graph))
    controll_parameters = []
    # for _ in range(len(get_joint_vector_from_graph(graph))):
    #     controll_parameters.append([2.0,np.random.normal(0,1,1)[0]])

    for _ in range(len(get_joint_vector_from_graph(graph))):
        controll_parameters.append(np.random.normal(0,1,1)[0])

    controll_parameters = {"PID_parameters" : [[1000, 100, 1, chrono.ChFunction_Const(0)],[0, 0, 0, chrono.ChFunction_Const(0)],
                [1000, 100, 1,chrono.ChFunction_Sine(0.1, 0.5, 0.5)],[0, 0, 0, chrono.ChFunction_Const(0)],[0, 0, 0, chrono.ChFunction_Const(0)],
                [1000, 100, 1, chrono.ChFunction_Const(0)],[0, 0, 0, chrono.ChFunction_Const(0)],
                [1000,100,1,chrono.ChFunction_Sine(0.1, 0.5, 0.5)],[0, 0, 0, chrono.ChFunction_Const(0)],[0, 0, 0, chrono.ChFunction_Const(0)]]}
    # controll_parameters = {"PID_parameters" : [[1, 1, 1, chrono.ChFunction_Const(0)],[1, 1, 1, chrono.ChFunction_Const(0)],
    #             [1, 1, 1, chrono.ChFunction_Const(0)],[1, 1, 1, chrono.ChFunction_Const(0)],[1, 1, 1, chrono.ChFunction_Const(0)],
    #             [1, 1, 1, chrono.ChFunction_Const(0)],[1, 1, 1, chrono.ChFunction_Const(0)],
    #             [1, 1, 1, chrono.ChFunction_Const(0)],[1, 1, 1, chrono.ChFunction_Const(0)],[1, 1, 1, chrono.ChFunction_Const(0)]]}
    #controll_parameters = {"initial_value": [0, 0, 0, 0, 0, 0 ,0 ,0], "sin_parameters" : [[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1],[0.000, 0.1, 0.1]]}

    #controll_parameters = {"initial_value": [0], "sin_parameters" : [[0.001, 0.1, 0.1]]}
    #print(controll_parameters)
    sim = RobotSimulationChrono([])
    # Create object to grasp
    mat = Material()
    mat.Friction = 0.65
    mat.DampingF = 0.65
    obj = EnvironmentBodyBlueprint(shape=Box(3, 0.2, 3),
                                   material=mat,
                                   pos=FrameTransform([0, 0, 0], [1, 0, 0, 0]))
    sim.add_object(creator.init_block_from_blueprint(obj))
    sim.objects[0].body.SetBodyFixed(True)
    sim.add_design(graph, controll_parameters, PIDController, FrameTransform([0, 2.14, 0], rotation_x(180)), False)
    i=0
    # for idx, body in sim.robot.get_graph().body_map_ordered.items():
    #     if i == 2:
    #         body.body.SetBodyFixed(True)
    #     i+=1
    #print(sim.robot.sensor.joint_body_map)
    sim_result = sim.simulate(10000, 0.001, 1, None, True)
    #print(sim.robot.sensor.trajectories)
    #print(sim.robot.data_storage.get_data("joint_trajectories"))
    for idx, value in sim.robot.data_storage.get_data('forces').items():
        print(idx)
    time_list = list(np.linspace(0, 1, 1001))
    robot_joint_data: DataStorage = sim_result[2]
    path = robot_joint_data.make_time_dependent_path()
    robot_joint_data.save()
    print(sim.robot.controller.torque_array[0])
    print(sim.robot.controller.torque_array[2])
    # for idx, data in robot_joint_data.items():
    #     fig.add_subplot(2, 5, i)
    #     plt.plot(time_list, data)
    #     i+=1
    # plt.suptitle('joints')
    # plt.show()
    # print()



