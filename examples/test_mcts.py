import os
import pickle
import sys
import numpy as np

from rostok.graph_generators.environments.design_environment import SubStringDesignEnvironment
from rostok.graph_generators.mcts_manager import MCTSManager, load_last_state
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.search_algorithms.random_search import RandomSearch

from rostok.library.rule_sets.ruleset_old_style import create_rules
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere

sys.path.append('app')
from mcts_run_setup import config_with_standard, config_tendon
import hyperparameters as hp

rule_vocabulary = create_rules()
grasp_object_blueprint = get_object_parametrized_sphere(0.5)
# create reward counter using run setup function
control_optimizer = config_tendon(grasp_object_blueprint)

init_graph = GraphGrammar()
env = SubStringDesignEnvironment(rule_vocabulary, control_optimizer, 5, init_graph, 2)

mcts = MCTS(env)

path = "D:\\Work_be2r_lab\\rostok\\results\\MCTS\\test_checkpoint\\checkpoint"
mcts_manager = MCTSManager(mcts, "test_checkpoint", verbosity=2, use_date=False)
# mcts_manager.save_information_about_search(hp, grasp_object_blueprint)
mcts.load(path)
state = load_last_state(path)

for i in range(1):
    mcts_manager.run_search(5, 1, 1, 2, state = state)
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