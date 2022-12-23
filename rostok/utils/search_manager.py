"""The manager class that controls the flow of search algorithm"""
# imports from standard libs
import matplotlib.pyplot as plt
import mcts
import networkx as nx

#imports from rostok lib
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
import rostok.graph_generators.graph_environment as env
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
from rostok.utils.result_saver import MCTSReporter


class SearchManager:
    __instance = None

    @staticmethod
    def get_instance():
        if SearchManager.__instance is None:
            SearchManager()
        return SearchManager.__instance

    def __init__(self):
        if SearchManager.__instance is not None:
            raise Exception("Attempt to create an instance of manager class - use get_instance method instead!")
        
        self._object_to_grasp_callback = None
        self._rule_vocabulary = None
        self._optimizer_config = None
        self._control_opimizer = None
        SearchManager.__instance = self

    @property
    def object_to_grasp_callback(self):
        return self._object_to_grasp_callback

    @object_to_grasp_callback.setter
    def object_to_grasp_callback(self, fun):
        self._object_to_grasp_callback = fun

    @property
    def rule_vocabulary(self):
        return self._rule_vocabulary

    @rule_vocabulary.setter
    def rule_vocabulary(self, rules):
        self._rule_vocabulary = rules

    @property
    def optimizer_config(self):
        return self._optimizer_config

    @optimizer_config.setter    
    def optimizer_config(self, conf):
        self._optimizer_config = conf
    
    @property
    def control_opimizer(self):
        return self._control_opimizer

    def create_optimizer(self):
        self._control_opimizer = ControlOptimizer(self.optimizer_config)

    def check_setup(self):
        if self._object_to_grasp_callback is None:
            print('object_to_grasp_callback not definded')
            return False

        return True 

    def run_search(self, *args):
        parameter_list = args
        if not self.check_setup():
            print("The search configuration is not complete\n revisit setup and try again!")

        reporter = MCTSReporter.get_instance()
        run_mcts(parameter_list[0], parameter_list[1], self.rule_vocabulary, self.control_opimizer)
        reporter

        
def run_mcts(iteration_limit, max_number_of_rules, rule_vocabulary, control_optimizer):
    searcher = mcts.mcts(iterationLimit=iteration_limit)
    G = GraphGrammar()
    graph_env = env.GraphVocabularyEnvironment(G, rule_vocabulary, max_number_of_rules)
    graph_env.set_control_optimizer(control_optimizer)
    finish = False
    iter = 0
    while not finish:
        action = searcher.search(initialState=graph_env)
        finish, final_graph, opt_trajectory, path = graph_env.step(action, False)
        iter += 1
        print(
            f"number iteration: {iter}, counter actions: {graph_env.counter_action}, reward: {graph_env.reward}"
        )



if __name__ == "__main__":
    manager = SearchManager.get_instance()
    manager.rule_vocabulary = RuleVocabulary()
