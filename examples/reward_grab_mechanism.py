import random
import pickle

import pychrono as chrono
from example_vocabulary import (get_terminal_graph_three_finger)

from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.criterion.criterion_calculation import (ForceCriterion, LateForceAmountCriterion,
                                                    LateForceCriterion, ObjectCOGCriterion,
                                                    SimulationReward, TimeCriterion)
from rostok.criterion.simulation_flags import (FlagContactTimeOut, FlagFlyingApart, FlagSlipout)
from rostok.simulation_chrono.simulation_scenario import ConstTorqueGrasp
from rostok.trajectory_optimizer.control_optimizer import (CounterGraphOptimization,
                                                           CounterWithOptimization,
                                                           CalculatorWithOptimizationDirect)

from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from example_vocabulary import (get_terminal_graph_no_joints, get_terminal_graph_three_finger,
                                get_terminal_graph_two_finger)


from rostok.block_builder_api.block_parameters import DefaultFrame, Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint

from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
# Constants
MAX_TIME = 3
TIME_STEP = 1e-3

# Graph initialization
graph = get_terminal_graph_three_finger()
grasp_object_blueprint = get_object_parametrized_sphere(0.4, 0.7)

# configurate the simulation manager
simulation_manager = ConstTorqueGrasp(TIME_STEP, MAX_TIME)
simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
    grasp_object_blueprint)
simulation_manager.add_flag(FlagContactTimeOut(1))
simulation_manager.add_flag(FlagFlyingApart(10))
simulation_manager.add_flag(FlagSlipout(0.8))
#create criterion manager
simulation_rewarder = SimulationReward()
#create criterions and add them to manager
simulation_rewarder.add_criterion(TimeCriterion(MAX_TIME), 1)
simulation_rewarder.add_criterion(ForceCriterion(), 1)
simulation_rewarder.add_criterion(ObjectCOGCriterion(), 1)
simulation_rewarder.add_criterion(LateForceCriterion(0.5, 3), 1)
simulation_rewarder.add_criterion(LateForceAmountCriterion(0.5), 1)

control_optimizer = CalculatorWithOptimizationDirect(simulation_manager, simulation_rewarder, (6, 15), 5)

# Create trajectory
number_trq = len(get_joint_vector_from_graph(graph))
# const_torque_koef = [random.random() for _ in range(number_trq)]
const_torque_koef = [0, 0, 0, 0, -1, 6]


reward = control_optimizer.calculate_reward(graph)

print(reward)
