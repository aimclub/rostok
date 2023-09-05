from rostok.graph_generators.environments.design_environment import DesignEnvironment, SubDesignEnvironment, SubStringDesignEnvironment
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.search_algorithms.random_search import RandomSearch

from rostok.library.rule_sets.ruleset_old_style_smc import create_rules
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_object_parametrized_sphere_smc
from rostok.graph_generators.mcts_manager import MCTSManager
import sys
sys.path.append('../app')
from mcts_run_setup import config_with_tendon

rule_vocabulary = create_rules()
grasp_object_blueprint = get_object_parametrized_sphere_smc(0.08)
# create reward counter using run setup function
ctrl_optimizer = config_with_tendon(grasp_object_blueprint)

init_graph = GraphGrammar()

str_env = SubStringDesignEnvironment(rule_vocabulary, ctrl_optimizer, 13, init_graph, 3)
mcts = MCTS(str_env)
mcts_manager = MCTSManager(mcts, "test_040923", verbosity=3, use_date=False)
# mcts.load("D:\\Work_be2r_lab\\rostok\\examples\\results\\MCTS\\test_240823_\\checkpoint")

mcts_manager.run_search(100, 1, iteration_checkpoint=1, num_test=3)