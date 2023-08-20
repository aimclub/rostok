import numpy as np

from rostok.graph_generators.environments.design_environment import DesignEnvironment, SubDesignEnvironment
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.search_algorithms.random_search import RandomSearch

from rostok.library.rule_sets.ruleset_old_style import create_rules
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
import sys
sys.path.append('d:\\lab_be2r\\rostok_09_01_2023\\rostok\\app')
from mcts_run_setup import config_with_standard

rule_vocabulary = create_rules()
grasp_object_blueprint = get_object_parametrized_sphere(0.5)
# create reward counter using run setup function
control_optimizer = config_with_standard(grasp_object_blueprint)

init_graph = GraphGrammar()
env = SubDesignEnvironment(rule_vocabulary, control_optimizer, 5, init_graph, 2)

mcts = MCTS(env)

state = env.initial_state
trajectory = [state]
while not env.is_terminal_state(state)[0]:
    for __ in range(10):
        mcts.search(state)
    
    pi = mcts.get_policy(state)
    a = np.random.choice(env.actions, p=pi)
    state, reward, is_terminal_state, __ = env.next_state(state, a)
    print(f"State: {state}, Reward: {reward}, is_terminal_state: {is_terminal_state}")
    trajectory.append(state)
env.save_environment("test")
print(f"Trajectory: {trajectory}")