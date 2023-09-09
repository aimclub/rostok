import numpy as np
import hyperparameters as hp
from rostok.graph_generators.environments.design_environment import (DesignEnvironment,
                                                                     SubDesignEnvironment,
                                                                     SubStringDesignEnvironment)
from rostok.graph_generators.mcts_manager import MCTSManager
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.search_algorithms.random_search import RandomSearch

from rostok.library.rule_sets.ruleset_simple_fingers import create_rules
# from rostok.library.rule_sets.ruleset_old_style_smc import create_rules
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import (get_object_ellipsoid, get_object_cylinder,
                                              get_object_box, get_object_parametrized_dipyramid_3,
                                              get_object_parametrized_trapezohedron)
import sys

from mcts_run_setup import config_combination_force_tendon_multiobject_parallel

if __name__ == "__main__":
    rule_vocabulary = create_rules()
    grasp_object_blueprint = []
    grasp_object_blueprint.append(get_object_parametrized_trapezohedron(0.15))
    grasp_object_blueprint.append(get_object_box(0.14, 0.19, 0.28, 0, mass = 0.268))
    grasp_object_blueprint.append(get_object_ellipsoid(0.14, 0.14, 0.22, 0, mass=0.188))
    # create reward counter using run setup function
    control_optimizer = config_combination_force_tendon_multiobject_parallel(
        grasp_object_blueprint, [1.1, 1.2, 0.5])

    init_graph = GraphGrammar()
    env = SubStringDesignEnvironment(rule_vocabulary, control_optimizer, 13, init_graph, 2)

    mcts = MCTS(env)
    name_directory = input("enter directory name: ")
    mcts_manager = MCTSManager(mcts, name_directory, verbosity=2, use_date=True)
    mcts_manager.save_information_about_search(hp, grasp_object_blueprint)

    for i in range(30):
        mcts_manager.run_search(10, 1, iteration_checkpoint=1, num_test=3)
        mcts_manager.save_results()