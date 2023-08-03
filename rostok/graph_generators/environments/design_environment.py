from copy import deepcopy
import pickle
from abc import ABC, abstractmethod
from typing import Any, Union
import os
from datetime import datetime
import numpy as np
import networkx as nx

from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import GraphRewardCalculator

STATESTYPE = Union[int, str]


class Environment(ABC):

    def __init__(self, initial_state: STATESTYPE, actions: np.ndarray):
        self.initial_state = initial_state
        self.actions = actions
        self.terminal_states: dict[STATESTYPE, tuple[float, Any]] = {}
        self.transition_function: dict[tuple[STATESTYPE, int], STATESTYPE] = {}

    def is_terminal_state(self, state) -> tuple[bool, bool]:

        in_terminal_table = state in self.terminal_states
        is_state_terminal = in_terminal_table or self._check_terminal_state(state)

        return is_state_terminal, in_terminal_table

    def get_reward(self, state: STATESTYPE) -> float:

        is_terminal, is_known = self.is_terminal_state(state)

        if is_terminal:
            if is_known:
                reward, __ = self.terminal_states[state]
            else:
                reward, data = self._calculate_reward(state)
                self.terminal_states[state] = (reward, data)

        if not is_terminal:
            reward = 0.0

        return reward

    @abstractmethod
    def next_state(self, state: STATESTYPE, action: int) -> STATESTYPE:
        return 0

    @abstractmethod
    def data2state(self, data) -> STATESTYPE:
        return 0

    @abstractmethod
    def get_available_actions(self, state: STATESTYPE) -> np.ndarray:
        return np.array([])

    @abstractmethod
    def _calculate_reward(self, state: STATESTYPE) -> tuple[float, Any]:
        return 0.0, None

    @abstractmethod
    def _check_terminal_state(self, state: STATESTYPE) -> bool:
        return False


class DesignEnvironment(Environment):

    def __init__(self,
                 rule_vocabulary: RuleVocabulary,
                 control_optimizer: GraphRewardCalculator,
                 initial_graph: GraphGrammar = GraphGrammar()):

        self.rule_vocabulary = rule_vocabulary
        sorted_name_rule = sorted(self.rule_vocabulary.rule_dict.keys())

        self.action2rule: dict[int, str] = {t[0]: t[1] for t in list(enumerate(sorted_name_rule))}

        actions = np.array(list(self.action2rule.keys()))
        initial_state = self.data2state(initial_graph)
        super().__init__(initial_state, actions)

        self.node2id = {
            name_node: id
            for id, name_node in enumerate(self.rule_vocabulary.node_vocab.node_dict.keys())
        }
        self.control_optimizer = control_optimizer

        self.state2graph: dict[STATESTYPE, GraphGrammar] = {
            self.initial_state: deepcopy(initial_graph)
        }

    def next_state(self, state: STATESTYPE, action: int) -> STATESTYPE:
        if (state, action) in self.transition_function:
            next_state = self.transition_function[(state, action)]
        else:
            name_rule = self.action2rule[action]
            rule = self.rule_vocabulary.rule_dict[name_rule]
            graph = self.state2graph[state]
            new_graph = deepcopy(graph)
            new_graph.apply_rule(rule)
            next_state = self.data2state(new_graph)
            self.update_environment(graph, action, new_graph)
        return next_state

    def possible_next_state(self, state: STATESTYPE, mask_actions=None) -> list[STATESTYPE]:
        if mask_actions is None:
            mask_actions = self.get_available_actions(state)

        avb_actions = self.actions[mask_actions == 1]
        possible_next_s = []
        for a in avb_actions:
            possible_next_s.append(self.next_state(state, a))

        return possible_next_s

    def get_available_actions(self, state: STATESTYPE) -> np.ndarray:
        graph = self.state2graph[state]
        available_rules = self.rule_vocabulary.get_list_of_applicable_rules(graph)
        mask_available_actions = np.zeros_like(self.actions)

        rule_list = np.array(list(self.action2rule.values()))
        mask = [rule in available_rules for rule in rule_list]
        mask_available_actions[mask] = 1

        return mask_available_actions

    def get_nonterminal_actions(self) -> np.ndarray:
        nonterminal_rules = self.rule_vocabulary.nonterminal_rule_dict.keys()
        mask_nonterminal_actions = np.zeros_like(self.actions)
        rule_list = np.array(list(self.action2rule.values()))
        mask = [rule in nonterminal_rules for rule in rule_list]
        mask_nonterminal_actions[mask] = 1

        return mask_nonterminal_actions

    def get_terminal_actions(self) -> np.ndarray:
        terminal_rules = self.rule_vocabulary.terminal_rule_dict.keys()
        mask_terminal_actions = np.zeros_like(self.actions)
        rule_list = np.array(list(self.action2rule.values()))
        mask = [rule in terminal_rules for rule in rule_list]
        mask_terminal_actions[mask] = 1

        return mask_terminal_actions

    def _calculate_reward(self, state: STATESTYPE) -> tuple[float, Any]:
        result_optimizer = self.control_optimizer.calculate_reward(self.state2graph[state])
        reward = result_optimizer[0]
        movments_trajectory = result_optimizer[1]
        return reward, movments_trajectory

    def _check_terminal_state(self, state: STATESTYPE) -> bool:
        terminal_nodes = [
            node[1]["Node"].is_terminal for node in self.state2graph[state].nodes.items()
        ]
        return sum(terminal_nodes) == len(terminal_nodes)

    def data2state(self, data: GraphGrammar) -> STATESTYPE:

        sorted_id_nodes = list(
            nx.lexicographical_topological_sort(data, key=lambda x: data.get_node_by_id(x).label))
        id_nodes = list(map(lambda x: self.node2id[data.get_node_by_id(x).label], sorted_id_nodes))

        return int(''.join([str(n) for n in id_nodes]))

    def update_environment(self, graph: GraphGrammar, action: int, next_graph: GraphGrammar):
        state = self.data2state(graph)
        next_state = self.data2state(next_graph)
        if next_state not in self.state2graph:
            self.state2graph[next_state] = deepcopy(next_graph)
        if (state, action) not in self.transition_function:
            self.transition_function[(state, action)] = next_state
        reward = self.get_reward(next_state)
        return reward

    def save_environment(
            self,
            prefix,
            path="./rostok/graph_generators/graph_heuristic_search/dataset_design_space"):
        current_date = datetime.now()
        folder = f"{prefix}__{current_date.hour}h{current_date.minute}m_{current_date.second}s_date_{current_date.day}d{current_date.month}m{current_date.year}y"
        os_path = os.path.join(path, folder)
        os.mkdir(os_path)

        file_names = [
            "actions.p", "terminal_states.p", "transition_function.p", "action2rule.p",
            "state2graph.p", "rule_vocabulary.p"
        ]
        variables = [
            self.actions, self.terminal_states, self.transition_function, self.action2rule,
            self.state2graph, self.rule_vocabulary, self.control_optimizer
        ]
        for file, var in zip(file_names, variables):
            with open(os.path.join(os_path, file), "wb") as f:
                pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_environment(self, path_to_folder):
        file_names = ["actions.p", "action2rule.p", "rule_vocabulary.p"]
        variables = [self.actions, self.action2rule, self.rule_vocabulary, self.control_optimizer]

        for file, var in zip(file_names, variables):
            with open(os.path.join(path_to_folder, file), "rb") as f:
                var = pickle.load(f)

        with open(os.path.join(path_to_folder, "terminal_states.p"), "rb") as f:
            s_t = pickle.load(f)
        with open(os.path.join(path_to_folder, "transition_function.p"), "rb") as f:
            p_sa = pickle.load(f)
        with open(os.path.join(path_to_folder, "state2graph.p"), "rb") as f:
            s2g = pickle.load(f)

        self.terminal_states.update(s_t)
        self.transition_function.update(p_sa)
        self.state2graph.update(s2g)
