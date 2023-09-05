import numpy as np

from rostok.graph_generators.environments.design_environment import DesignEnvironment, SubDesignEnvironment
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.search_algorithms.random_search import RandomSearch

from rostok.library.rule_sets.ruleset_old_style_smc import create_rules
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_object_cylinder, get_object_box
import sys

from mcts_run_setup import config_combination_force_tendon_multiobject

rule_vocabulary = create_rules()
grasp_object_blueprint = []
grasp_object_blueprint.append(get_object_parametrized_sphere(0.11))
grasp_object_blueprint.append(get_object_cylinder(0.07, 0.09, 0))
grasp_object_blueprint.append(get_object_box(0.12, 0.12, 0.1, 0))
# create reward counter using run setup function
control_optimizer = config_combination_force_tendon_multiobject(grasp_object_blueprint, [ 1, 1, 1])

init_graph = GraphGrammar()
env = SubDesignEnvironment(rule_vocabulary, control_optimizer, 5, init_graph, 2)

mcts = MCTS(env)

state = env.initial_state
trajectory = [state]
while not env.is_terminal_state(state)[0]:
    for __ in range(10):
        mcts.search(state)
    
    pi = mcts.get_policy(state)
    a = max(env.actions, key=lambda x: pi[x])
    state, reward, is_terminal_state, __ = env.next_state(state, a)
    print(f"State: {state}, Reward: {reward}, is_terminal_state: {is_terminal_state}")
    trajectory.append(state)
env.save_environment("test")
mcts.save("test")
print(f"Trajectory: {trajectory}")