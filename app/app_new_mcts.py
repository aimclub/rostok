
from rostok.graph_generators.environments.design_environment import (SubStringDesignEnvironment)
from rostok.graph_generators.mcts_manager import MCTSManager
from rostok.graph_generators.search_algorithms.mcts import MCTS

from rostok.library.rule_sets.rulset_simple_fingers import create_rules
from rostok.graph_grammar.node import GraphGrammar

from tendon_graph_evaluators import evaluator_tendon_standart, evaluator_tendon_standart_parallel
from tendon_graph_evaluators import mcts_hyper_default, evaluator_tendon_fast_debug
import tendon_driven_cfg

if __name__ == "__main__":
    rule_vocabulary = create_rules()

    init_graph = GraphGrammar()
    env = SubStringDesignEnvironment(rule_vocabulary, evaluator_tendon_fast_debug,
                                     mcts_hyper_default.max_number_rules, init_graph, 0)

    mcts = MCTS(env, mcts_hyper_default.C)
    name_directory = input("enter directory name: ")
    mcts_manager = MCTSManager(mcts, name_directory, verbosity=2, use_date=True)
    mcts_manager.save_information_about_search(
        mcts_hyper_default, tendon_driven_cfg.default_grasp_objective.object_list)

    for i in range(mcts_hyper_default.full_loop):
        mcts_manager.run_search(mcts_hyper_default.base_iteration, 1, 1, 2)
        mcts_manager.save_results()