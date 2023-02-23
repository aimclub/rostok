import numpy as np
import networkx as nx

from copy import deepcopy
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
from rostok.graph_grammar.node import GraphGrammar
from rostok.neural_network.converter import ConverterToPytorchGeometric

import sys

sys.path.append("./alpha-zero-general")

from Game import Game


class GraphGrammarGame(Game):

    def __init__(self, rule_vocabulary: RuleVocabulary, control_optimization,
                 max_no_terminal_rules: int):

        self.rule_vocabulary = rule_vocabulary
        self._converter: ConverterToPytorchGeometric = ConverterToPytorchGeometric(
            self.rule_vocabulary.node_vocab)

        sorted_name_rule = sorted(self.rule_vocabulary.rule_dict.keys())
        self.id2rule: dict[int, str] = {t[0]: t[1] for t in list(enumerate(sorted_name_rule))}
        self.rule2id: dict[str, int] = {t[1]: t[0] for t in self.id2rule.items()}
        self.max_no_terminal_rules: int = max_no_terminal_rules
        self.counter_actions: int = 0
        self.control_optimization = control_optimization
        self.terminal_graphs: dict[list[list[str]], tuple(float, list[list[float]])] = {}

    def getBoardSize(self):
        return super().getBoardSize()

    def getInitBoard(self):
        initial_graph = GraphGrammar()
        return initial_graph

    def getNextState(self, graph: GraphGrammar, player: int, action: int):

        name_rule = self.id2rule[action]
        rule = self.rule_vocabulary.rule_dict[name_rule]
        next_state_graph = deepcopy(graph)
        next_state_graph.apply_rule(rule)
        if name_rule in self.rule_vocabulary.rules_nonterminal_node_set:
            self.counter_actions += 1

        return (next_state_graph, -player)

    def getActionSize(self):
        return len(self.id2rule)

    def getValidMoves(self, graph: GraphGrammar, player: int):

        if self.counter_actions < self.max_no_terminal_rules:
            possible_rules_name = self.rule_vocabulary.get_list_of_applicable_rules(graph)
        else:
            possible_rules_name = self.rule_vocabulary.get_list_of_applicable_terminal_rules(graph)

        mask_valid_movies = [0 for __ in range(len(self.id2rule))]

        for rule in possible_rules_name:
            id_rule = self.rule2id[rule]
            mask_valid_movies[id_rule] = 1

        return np.array(mask_valid_movies)

    def getGameEnded(self, graph: GraphGrammar, player):

        if len(self.rule_vocabulary.get_list_of_applicable_terminal_rules(graph)) != 0:
            return 0

        if (len(self.rule_vocabulary.get_list_of_applicable_nonterminal_rules(graph)) and
                self.counter_actions < self.max_no_terminal_rules):
            return 0

        flatten_graph = self.stringRepresentation(graph)
        self.counter_actions = 0
        if flatten_graph in set(self.terminal_graphs.keys()):
            reward, movments_trajectory = self.terminal_graphs[flatten_graph]
        else:
            result_optimizer = self.control_optimization.start_optimisation(graph)
            reward = -result_optimizer[0]
            if reward == 0:
                reward = 0.01
            movments_trajectory = result_optimizer[1]
            self.terminal_graphs[flatten_graph] = (reward, movments_trajectory)

        return reward

    def stringRepresentation(self, graph: GraphGrammar):
        flatten_graph, __ = self._converter.flatting_sorted_graph(graph)

        return "".join(flatten_graph)
    
    def getCanonicalForm(self, graph: GraphGrammar, player):
        return graph