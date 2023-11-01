import hyperparameters as hp
from rostok.graph_generators.environments.design_environment import (SubStringDesignEnvironment)
from rostok.graph_generators.mcts_manager import MCTSManager
from rostok.graph_generators.search_algorithms.mcts import MCTS

from rostok.library.rule_sets.rulset_simple_fingers import create_rules
from rostok.graph_grammar.node import GraphGrammar
import hyperparameters as hp
from tendon_graph_evaluators import evaluator_tendon_standart, evaluator_tendon_standart_parallel
import tendon_driven_cfg

if __name__ == "__main__":
    rule_vocabulary = create_rules()

    init_graph = GraphGrammar()
    env = SubStringDesignEnvironment(rule_vocabulary, evaluator_tendon_standart_parallel,
                                     hp.MAX_NUMBER_RULES, init_graph, 0)

    mcts = MCTS(env, hp.MCTS_C)
    name_directory = input("enter directory name: ")
    mcts_manager = MCTSManager(mcts, name_directory, verbosity=2, use_date=True)
    #mcts_manager.save_information_about_search(
    #    hp, tendon_driven_cfg.default_grasp_objective.object_list)

    for i in range(hp.FULL_LOOP_MCTS):
        mcts_manager.run_search(hp.BASE_ITERATION_LIMIT_TENDON, 1, 1, 2)
        mcts_manager.save_results()