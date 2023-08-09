import numpy as np

import pychrono as chrono
from example_vocabulary import (get_terminal_graph_three_finger, get_terminal_graph_two_finger,
                                get_terminal_graph_no_joints)
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
import matplotlib.pyplot as plt
from rostok.block_builder_api.block_parameters import Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono, SystemPreviewChrono
from rostok.block_builder_chrono.block_builder_chrono_api import ChronoBlockCreatorInterface as creator

from rostok.library.rule_sets.simple_designs_cm import get_two_link_one_finger, get_2_3_link_2_finger, get_one_link_one_finger, get_two_link_one_finger_transform_0, get_5_link_one_finger

from rostok.graph_grammar.graph_utils import plot_graph, plot_graph_ids
from rostok.control_chrono.controller import ConstController, SinControllerChrono, YaxisShaker
from rostok.control_chrono.tendon_controller import PulleyKey, PulleyParamsFinger_2p, RelativeSetting_2p, TendonController_2p, create_pulley_params_finger_2p, create_pulley_params_relative_finger_2p
from rostok.virtual_experiment.sensors import SensorCalls, SensorObjectClassification

mechs = [get_two_link_one_finger]


def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]



def setup_mech_two_link_one_finger():
    graph = get_two_link_one_finger()
    pp0 = PulleyParamsFinger_2p(0, (-0.01, -0.02, 0), (-0.01, 0.02, 0))
    finger_parametrs_list = [pp0]
    pulley_d = create_pulley_params_finger_2p(graph, finger_parametrs_list)
    controll_parameters = {
        "initial_value": [0, 0],
        "pulley_params_dict": pulley_d,
        "force_finger_dict": {
            0: 1
        }
    }
    return graph, controll_parameters

def setup_mech_one_link_one_finger():
    graph = get_one_link_one_finger()
    pp0 = PulleyParamsFinger_2p(0, (-0.01, -0.02, 0), (-0.01, 0.02, 0))
    finger_parametrs_list = [pp0]
    pulley_d = create_pulley_params_finger_2p(graph, finger_parametrs_list)
    controll_parameters = {
        "initial_value": [0, 0],
        "pulley_params_dict": pulley_d,
        "force_finger_dict": {
            0: 1
        }
    }
    return graph, controll_parameters


def setup_mech_2_3_link_2_finger(is_rel = False):
    graph = get_2_3_link_2_finger()
    pp1 = PulleyParamsFinger_2p(1, (-0.02*2, -0.03, 0), (-0.02*2, 0.03, 0))
    pp0 = PulleyParamsFinger_2p(0, (-0.01*2, -0.02, 0), (-0.01*2, 0.02, 0))
    finger_parametrs_list = [pp0, pp1]
    pulley_d = create_pulley_params_finger_2p(graph, finger_parametrs_list)
    if is_rel:
        param = RelativeSetting_2p(0.6, 0.6, -1)
        pulley_d = create_pulley_params_relative_finger_2p(graph, param)
    #del pulley_d[PulleyKey(0, 39, 1)] 
    #del pulley_d[PulleyKey(1, 25, 1)] 
    controll_parameters = {
        "initial_value": [0, 0],
        "pulley_params_dict": pulley_d,
        "force_finger_dict": {
            0: 5,
            1: 5  # Long finger
        }
    }
    return graph, controll_parameters

def setup_mech_two_link_one_finger_transform_0():
    graph = get_two_link_one_finger_transform_0()
    pp0 = PulleyParamsFinger_2p(0, (-0.01, -0.02, 0), (-0.01, 0.02, 0))
    finger_parametrs_list = [pp0]
    pulley_d = create_pulley_params_finger_2p(graph, finger_parametrs_list)
    controll_parameters = {
        "initial_value": [0, 0],
        "pulley_params_dict": pulley_d,
        "force_finger_dict": {
            0: 2
        }
    }
    return graph, controll_parameters



def setup_mech_5_link_one_finger():
    graph = get_5_link_one_finger()
    pp0 = PulleyParamsFinger_2p(0, (-0.01, -0.02, 0), (-0.01, 0.02, 0))
    finger_parametrs_list = [pp0]
    pulley_d = create_pulley_params_finger_2p(graph, finger_parametrs_list)
    controll_parameters = {
        "initial_value": [0, 0],
        "pulley_params_dict": pulley_d,
        "force_finger_dict": {
            0: 10
        }
    }
 
    return graph, controll_parameters



def setup_mech_5_link_one_finger_rel():
    graph = get_5_link_one_finger()
    param = RelativeSetting_2p(0.9, 0.9, -0.9)
    pulley_d = create_pulley_params_relative_finger_2p(graph, param)
 
    controll_parameters = {
        "initial_value": [0, 0],
        "pulley_params_dict": pulley_d,
        "force_finger_dict": {
            0: 10
        }
    }

    return graph, controll_parameters
STEPS = 500*10
graph, controll_parameters = setup_mech_2_3_link_2_finger(True)
 
 
sim = RobotSimulationChrono([])
env_data_dict = {
    "pos": (SensorCalls.BODY_TRAJECTORY, SensorObjectClassification.BODY),
    "j_pos": (SensorCalls.JOINT_TRAJECTORY, SensorObjectClassification.JOINT),
    "contact": (SensorCalls.AMOUNT_FORCE, SensorObjectClassification.BODY)
}

#sim.add_env_data_type_dict(env_data_dict)
sim.add_robot_data_type_dict(env_data_dict)
 
# Create object to grasp
mat = Material()
mat.Friction = 0.65
mat.DampingF = 0.65
obj = EnvironmentBodyBlueprint(shape=Box(0.1, 0.07, 0.1),
                                material=mat,
                                pos=FrameTransform([0, -0.5, 0], [1, 0, 0, 0]))
shake = YaxisShaker(0, -1)
added_obj =  creator.init_block_from_blueprint(obj)
sim.add_object(added_obj,
                force_torque_controller=shake,
                )

sim.add_design(graph,
                controll_parameters,
                control_cls=TendonController_2p,
                Frame=FrameTransform([0, 0, 0], rotation_x(180)),
                is_fixed=True)

res = sim.simulate(STEPS, 0.0001, 1, None, True)

j_pos1 = np.array(sim.result.robot_final_ds.main_storage["j_pos"][38])
plt.figure()
plt.plot(j_pos1 * 180 / np.pi)
plt.show()
