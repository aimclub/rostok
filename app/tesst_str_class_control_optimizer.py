from rostok.graph_generators.environments.design_environment import DesignEnvironment, SubDesignEnvironment
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.search_algorithms.random_search import RandomSearch

from rostok.library.rule_sets.ruleset_old_style import create_rules
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere
from rostok.graph_generators.mcts_manager import MCTSManager
import sys

from mcts_run_setup import config_with_standard, config_cable

rule_vocabulary = create_rules()
grasp_object_blueprint = get_object_parametrized_sphere(0.5)
# create reward counter using run setup function
ctrl_optimizer = config_cable(grasp_object_blueprint)

init_graph = GraphGrammar()
env = SubDesignEnvironment(rule_vocabulary, ctrl_optimizer, 5, init_graph, 2)

mcts = MCTS(env)


mcts_manager = MCTSManager(mcts, "test_240823", verbosity=3, use_date=False)

str(ctrl_optimizer)