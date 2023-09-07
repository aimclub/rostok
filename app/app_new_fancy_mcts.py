import numpy as np
import hyperparameters as hp
from rostok.graph_generators.environments.design_environment import DesignEnvironment, SubDesignEnvironment, SubStringDesignEnvironment
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.mcts_manager import MCTSManager
from rostok.graph_generators.search_algorithms.random_search import RandomSearch

from rostok.library.rule_sets.ruleset_old_style_nsc import create_rules
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import (get_object_parametrized_sphere, get_object_cylinder, get_object_box,
                                              get_object_ellipsoid)
import sys

from mcts_run_setup import config_combination_force_tendon_multiobject

rule_vocabulary = create_rules()
grasp_object_blueprint = []
# grasp_object_blueprint.append(get_object_parametrized_sphere(0.11))
grasp_object_blueprint.append(get_object_ellipsoid(0.10, 0.08, 0.14, 10))
grasp_object_blueprint.append(get_object_cylinder(0.07, 0.09, 0))
grasp_object_blueprint.append(get_object_box(0.12, 0.12, 0.1, 0))
# create reward counter using run setup function
control_optimizer = config_combination_force_tendon_multiobject(grasp_object_blueprint, [ 1, 1, 1])

init_graph = GraphGrammar()
env = SubStringDesignEnvironment(rule_vocabulary, control_optimizer, 13, init_graph, 4)
env = SubStringDesignEnvironment(rule_vocabulary, control_optimizer, 13, init_graph, 4)

mcts = MCTS(env)
name_directory = input("enter directory name")
mcts_manager = MCTSManager(mcts, name_directory,verbosity=4)
mcts_manager.save_information_about_search(hp, grasp_object_blueprint)

for i in range(10):
    mcts_manager.run_search(10, 1, iteration_checkpoint=1, num_test=3)
    mcts_manager.save_results()
# state = env.initial_state
# trajectory = [state]
# while not env.is_terminal_state(state)[0]:
#     for __ in range(10):
#         mcts.search(state)
    
#     pi = mcts.get_policy(state)
#     a = max(env.actions, key=lambda x: pi[x])
#     state, reward, is_terminal_state, __ = env.next_state(state, a)
#     print(f"State: {state}, Reward: {reward}, is_terminal_state: {is_terminal_state}")
#     trajectory.append(state)
# env.save_environment("test")
# mcts.save("test")
# print(f"Trajectory: {trajectory}")