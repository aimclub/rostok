import numpy as np

from rostok.graph_generators.environments.design_environment import (DesignEnvironment, SubDesignEnvironment, SubStringDesignEnvironment)
from rostok.graph_generators.mcts_manager import MCTSManager
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.search_algorithms.random_search import RandomSearch

from rostok.library.rule_sets.ruleset_simple_fingers import create_rules
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_ellipsoid, get_object_cylinder, get_object_box
import sys

from mcts_run_setup import config_combination_force_tendon_multiobject_parallel

if __name__ == "__main__":
    rule_vocabulary = create_rules()
    grasp_object_blueprint = []
    grasp_object_blueprint.append(get_object_ellipsoid(0.10, 0.08, 0.14, 10))
    grasp_object_blueprint.append(get_object_cylinder(0.07, 0.09, 0))
    grasp_object_blueprint.append(get_object_box(0.12, 0.12, 0.1, 0))
    # create reward counter using run setup function
    control_optimizer = config_combination_force_tendon_multiobject_parallel(grasp_object_blueprint, [ 1, 1, 1])

    init_graph = GraphGrammar()
    env = SubStringDesignEnvironment(rule_vocabulary, control_optimizer, 13, init_graph, 0)

    mcts = MCTS(env)
    # name_directory = input("enter directory name: ")
    mcts_manager = MCTSManager(mcts, "test_parallel",verbosity=0, use_date=False)
    # mcts_manager.save_information_about_search(hp, grasp_object_blueprint)

    for i in range(1):
        mcts_manager.run_search(10, 1)
    # mcts_manager.save_results()