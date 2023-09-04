from collections import defaultdict
from copy import deepcopy
import pickle
from abc import ABC, abstractmethod
from typing import Any, Union, TypeAlias
import os
from datetime import datetime

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import GraphRewardCalculator

STATESTYPE = Union[int, str]

StepType: TypeAlias = tuple[STATESTYPE, float, bool, bool]
TransitionFunctionType: TypeAlias = dict[tuple[STATESTYPE, int], tuple[STATESTYPE, float, bool]]

class Environment(ABC):
    def __init__(self, initial_state: STATESTYPE, actions: np.ndarray, verbosity=0):
        """Abstract class environment for defining design space of the problem.

        Args:
            initial_state (STATESTYPE): initial state of the environment
            actions (np.ndarray): array of all possible actions
            verbosity (int, optional): Define verbosity of info and state_info. Maximum iss 3. Defaults to 0.
        """
        self.initial_state = initial_state
        self.actions = actions
        self.terminal_states: dict[STATESTYPE, tuple[float, Any]] = {}
        self.transition_function: TransitionFunctionType = {}
        self.verbosity = verbosity

    def is_terminal_state(self, state: STATESTYPE) -> tuple[bool, bool]:
        """Check if state is terminal. If state is terminal, return True and True if state is in terminal_states table, else False.

        Args:
            state (STATESTYPE): state to check

        Returns:
            tuple[bool, bool]: condition of terminal state and condition of state in terminal_states table
        """
        in_terminal_table = state in self.terminal_states
        is_state_terminal = in_terminal_table or self._check_terminal_state(state)

        return is_state_terminal, in_terminal_table

    def get_reward(self, state: STATESTYPE) -> tuple[float, bool]:
        """Get reward of state. If state is terminal, return reward and True if state is in terminal_states table, else False.
        For nonterminal states return 0.0 and False.

        Args:
            state (STATESTYPE): state to get reward

        Returns:
            tuple[float, bool]: reward of state and condition of state in terminal_states table
        """
        is_terminal, is_known = self.is_terminal_state(state)

        if is_terminal:
            if is_known:
                reward, __ = self.terminal_states[state]
            else:
                reward, data = self._calculate_reward(state)
                self.terminal_states[state] = (reward, data)

        if not is_terminal:
            reward = 0.0

        return reward, is_known

    def info(self, verbosity=None) -> str:
        """Get info about environment.

        Args:
            verbosity (int, optional): Define verbosity of information. If it is None, use the value of self.verbosity. Defaults to None.


        Returns:
            str: Information about environment
        """
        if verbosity is None:
            verbosity = self.verbosity
        info_out = ""
        if verbosity > 0:
            info_out = f"Number of terminal states: {len(self.terminal_states)};"
            if verbosity > 1:
                info_out += f" Initial state: {self.initial_state};"
                info_out += f"Best state: {self.get_best_states()[0]}; Reward: {self.terminal_states[self.get_best_states()[0]][0]:.4f};"
            info_out += "\n"

        return info_out

    def get_info_state(self, state: STATESTYPE, verbosity=None) -> str:
        """Get info about state.

        Args:
            state (STATESTYPE): state to get info
            verbosity (int, optional): Define verbosity of information. If it is None, use the value of self.verbosity. Defaults to None.

        Returns:
            str: Information about state
        """
        if verbosity is None:
            verbosity = self.verbosity
        info_out = ""
        if verbosity > 0:
            info_out = f"State: {state};"
            if verbosity > 1:
                info_out += f" Is terminal: {self.is_terminal_state(state)[0]};"
                info_out += f" Reward: {self.get_reward(state)[0]:.4f};"
            info_out += "\n"

        return info_out

    def get_best_states(self, num=1) -> list[STATESTYPE]:
        """Get best states by reward.

        Args:
            num (int, optional): Number of best states. Defaults to 1.  

        Returns:
            list[STATESTYPE]: List of best states
        """
        best_states = sorted(self.terminal_states,
                            key=lambda x: self.terminal_states[x][0],
                            reverse=True)[:num]
        return best_states

    @abstractmethod
    def next_state(self, state: STATESTYPE, action: int) -> StepType:
        """Get next state by action.

        Args:
            state (STATESTYPE): state to get next state
            action (int): action to get next state

        Returns:
            StepType: tuple of next state, reward, is_terminal_state, bool if state is in terminal_states table
        """
        return 0, 0.0, False, False

    @abstractmethod
    def data2state(self, data) -> STATESTYPE:
        """Convert data to state.

        Args:
            data (_type_): data to convert

        Returns:
            STATESTYPE: state
        """
        return 0

    @abstractmethod
    def get_available_actions(self, state: STATESTYPE) -> np.ndarray:
        """Get mask of available actions for state.

        Args:
            state (STATESTYPE): state to get mask of available actions

        Returns:
            np.ndarray: mask of available actions
        """
        return np.array([])

    @abstractmethod
    def _calculate_reward(self, state: STATESTYPE) -> tuple[float, Any]:
        """Calculate reward of state. Need to override in child class.

        Args:
            state (STATESTYPE): state to calculate reward

        Returns:
            tuple[float, Any]: reward and data of state
        """
        return 0.0, None

    @abstractmethod
    def _check_terminal_state(self, state: STATESTYPE) -> bool:
        """Check if state is terminal. Need to override in child class.

        Args:
            state (STATESTYPE): state to check

        Returns:
            bool: condition of terminal state
        """
        return False



class DesignEnvironment(Environment):

    def __init__(self,
                 rule_vocabulary: RuleVocabulary,
                 control_optimizer: GraphRewardCalculator,
                 initial_graph: GraphGrammar = GraphGrammar(),
                 verbosity=0):
        """Environment for design space of mechanism. Create dictionary of rules and convert it to actions by sorted name of rules.
        Create dictionary of nodes and convert it to states by sorted name of nodes.
        Save graph of state in state2graph dictionary.

        Args:
            rule_vocabulary (RuleVocabulary): Vocabulary of rules
            control_optimizer (GraphRewardCalculator): Class for optimizate control of mechanism
            initial_graph (GraphGrammar, optional): Initial state of environment. Defaults to GraphGrammar().
            verbosity (int, optional): Information verbosity. Defaults to 0.
        """
        self.rule_vocabulary = rule_vocabulary
        sorted_name_rule = sorted(self.rule_vocabulary.rule_dict.keys())

        self.action2rule: dict[int, str] = {t[0]: t[1] for t in list(enumerate(sorted_name_rule))}

        self.node2id = {
            name_node: id
            for id, name_node in enumerate(self.rule_vocabulary.node_vocab.node_dict.keys())
        }

        actions = np.array(list(self.action2rule.keys()))
        initial_state = self.data2state(initial_graph)
        super().__init__(initial_state, actions, verbosity)

        self.control_optimizer = control_optimizer

        self.state2graph: dict[STATESTYPE, GraphGrammar] = {
            self.initial_state: deepcopy(initial_graph)
        }

    def next_state(self, state: STATESTYPE, action: int) -> StepType:
        """Get next state by action. If next state is not in state2graph dictionary, apply rule to graph of state and save it in state2graph dictionary.
        If next state is not in transition_function dictionary, calculate reward and save it in transition_function dictionary.

        Args:
            state (STATESTYPE): State to get next state
            action (int): Action to get next state

        Returns:
            StepType: tuple of next state, reward, is_terminal_state, bool if state is in terminal_state dictionary
        """
        if (state, action) in self.transition_function:
            next_state, reward, is_terminal_state = self.transition_function[(state, action)]
            is_known = True
        else:
            name_rule = self.action2rule[action]
            rule = self.rule_vocabulary.rule_dict[name_rule]
            graph = self.state2graph[state]
            new_graph = deepcopy(graph)
            new_graph.apply_rule(rule)
            next_state = self.data2state(new_graph)
            reward, is_known = self.update_environment(graph, action, new_graph)
            is_terminal_state, __ = self.is_terminal_state(next_state)
        return (next_state, reward, is_terminal_state, is_known)

    def possible_next_state(self, state: STATESTYPE, mask_actions=None) -> list[STATESTYPE]:
        """Return list of possible next states by mask of available actions.

        Args:
            state (STATESTYPE): state to get possible next states
            mask_actions (np.ndarray, optional): Mask of desired actions. Defaults to None. None means all available actions.

        Returns:
            list[STATESTYPE]: List of possible next states
        """
        if mask_actions is None:
            mask_actions = self.get_available_actions(state)

        avb_actions = self.actions[mask_actions == 1]
        possible_next_s = []
        for a in avb_actions:
            possible_next_s.append(self.next_state(state, a)[0])

        return possible_next_s

    def get_available_actions(self, state: STATESTYPE) -> np.ndarray:
        """Get mask of available actions for state.

        Args:
            state (STATESTYPE): state to get mask of available actions

        Returns:
            np.ndarray: mask of available actions
        """
        graph = self.state2graph[state]
        available_rules = self.rule_vocabulary.get_list_of_applicable_rules(graph)
        mask_available_actions = np.zeros_like(self.actions)

        rule_list = np.array(list(self.action2rule.values()))
        mask = [rule in available_rules for rule in rule_list]
        mask_available_actions[mask] = 1

        return mask_available_actions

    def get_nonterminal_actions(self) -> np.ndarray:
        """Get mask of nonterminal actions. Nonterminal actions are actions that apply nonterminal rules.

        Returns:
            np.ndarray: mask of nonterminal actions
        """
        nonterminal_rules = self.rule_vocabulary.nonterminal_rule_dict.keys()
        mask_nonterminal_actions = np.zeros_like(self.actions)
        rule_list = np.array(list(self.action2rule.values()))
        mask = [rule in nonterminal_rules for rule in rule_list]
        mask_nonterminal_actions[mask] = 1

        return mask_nonterminal_actions

    def get_terminal_actions(self) -> np.ndarray:
        """Get mask of terminal actions. Terminal actions are actions that apply terminal rules.

        Returns:
            np.ndarray: mask of terminal actions
        """
        terminal_rules = self.rule_vocabulary.terminal_rule_dict.keys()
        mask_terminal_actions = np.zeros_like(self.actions)
        rule_list = np.array(list(self.action2rule.values()))
        mask = [rule in terminal_rules for rule in rule_list]
        mask_terminal_actions[mask] = 1

        return mask_terminal_actions

    def _calculate_reward(self, state: STATESTYPE) -> tuple[float, Any]:
        """Calculate reward of state. Use control_optimizer to calculate reward.
        Don't use the method directly. Use get_reward instead.

        Args:
            state (STATESTYPE): state to calculate reward

        Returns:
            tuple[float, Any]: reward and data of state
        """
        result_optimizer = self.control_optimizer.calculate_reward(self.state2graph[state])
        reward = result_optimizer[0]
        movments_trajectory = result_optimizer[1]
        # movments_trajectory: list[float] = []
        # reward = state + 0.0
        return reward, movments_trajectory

    def _check_terminal_state(self, state: STATESTYPE) -> bool:
        """Check if state is terminal. If state is terminal, return True, else False.
        Don't use the method directly. Use is_terminal_state instead.

        Args:
            state (STATESTYPE): state to check

        Returns:
            bool: condition of terminal state
        """
        terminal_nodes = [
            node[1]["Node"].is_terminal for node in self.state2graph[state].nodes.items()
        ]
        return sum(terminal_nodes) == len(terminal_nodes)

    def data2state(self, data: GraphGrammar) -> STATESTYPE:
        """Convert data to state. Convert graph to int of sorted id of nodes.

        Args:
            data (GraphGrammar): graph to convert

        Returns:
            STATESTYPE: state of graph
        """
        # if len(self.state2graph) != len(set(self.state2graph.values())):

        #     print("WARNING: There are repeated states")

        sorted_id_nodes = list(
            nx.lexicographical_topological_sort(data, key=lambda x: data.get_node_by_id(x).label))
        id_nodes = list(map(lambda x: self.node2id[data.get_node_by_id(x).label], sorted_id_nodes))

        return int(''.join([str(n) for n in id_nodes]))

    def update_environment(self, graph: GraphGrammar, action: int, next_graph: GraphGrammar):
        """Update environment. If next state is not in state2graph dictionary, save it in state2graph dictionary.
        If next state is not in transition_function dictionary, calculate reward and save it in transition_function dictionary.
        Save reward and data of state in terminal_states dictionary.

        Args:
            graph (GraphGrammar): Previous state in the form of a graph
            action (int): Action that was applied to the previous state
            next_graph (GraphGrammar): Next state in the form of a graph

        Returns:
            _type_: reward and bool if state is in terminal_states table
        """
        state = self.data2state(graph)
        next_state = self.data2state(next_graph)
        if next_state not in self.state2graph:
            self.state2graph[next_state] = deepcopy(next_graph)
        reward, is_known = self.get_reward(next_state)
        if (state, action) not in self.transition_function:
            self.transition_function[(state, action)] = (next_state, reward,
                                                         self.is_terminal_state(next_state)[0])
        return reward, is_known

    def info(self, verbosity=None) -> str:
        """Get info about environment. If verbosity is None, use the value of self.verbosity.
        

        Args:
            verbosity (int, optional): Define verbosity of info and state_info. Maximum is 3. Defaults to None.

        Returns:
            str: String with information about environment
        """
        if verbosity is None:
            verbosity = self.verbosity
        info_out = ""

        if verbosity > 0:
            info_out = super().info(verbosity=verbosity)
            if verbosity > 1:
                info_out += f"Number of states: {len(self.state2graph)};"
                info_out += "\n"

        return info_out

    def get_info_state(self, state: STATESTYPE, verbosity=None) -> str:
        """Get info about state. If verbosity is None, use the value of self.verbosity.

        Args:
            state (STATESTYPE): state to get info
            verbosity (int, optional): Define verbosity of info and state_info. Maximum is 3. Defaults to None.

        Returns:
            str: String with information about state
        """
        if verbosity is None:
            verbosity = self.verbosity
        info_out = ""
        if verbosity > 0:
            info_out = super().get_info_state(state, verbosity=verbosity)
            if verbosity > 1:
                info_out += f"Number of possible next states: {len(self.possible_next_state(state))};\n"
            if verbosity > 2:
                info_out += f"Graph representation: {self.state2graph[state].get_uniq_representation()};\n"

                mask_action = self.get_available_actions(state)
                possible_actions = self.actions[mask_action == 1]
                name_actions = [self.action2rule[a] for a in possible_actions]

                info_out += f"Actions: {name_actions};"
                info_out += "\n"

        return info_out

    def save_environment(self, prefix, path="./environments/", rewrite=False, use_date=True):
        """Save environment to folder. If folder does not exist, create it. If folder exist, create new folder with postfix.

        Args:
            prefix (str): Prefix of folder name
            path (str, optional): Folder to save environment. Defaults to "./environments/".
            rewrite (bool, optional): Rewrite folder if it exist. Defaults to False.
            use_date (bool, optional): Use date in folder name. Defaults to True.
        """
        os.path.split(path)
        if not os.path.exists(path):
            print(f"Path {path} does not exist. Creating...")
            os.mkdir(path)

        if use_date:
            current_date = datetime.now()
            folder = f"{prefix}__{current_date.hour}h{current_date.minute}m_{current_date.second}s_date_{current_date.day}d{current_date.month}m{current_date.year}y"
        else:
            folder = prefix
        os_path = os.path.join(path, folder)
        if not os.path.exists(os_path):
            print(f"Saving environment to {os_path}")
            os.mkdir(os_path)
        elif rewrite:
            print(f"Rewriting environment to {os_path}")
        else:
            postfix_folder = 1
            os_path = os_path + f"_{postfix_folder}"
            while os.path.exists(os_path):
                os_path = os_path.replace(f"_{postfix_folder-1}", f"_{postfix_folder}")
                postfix_folder += 1
            print(f"Saving environment to {os_path}")

        file_names = [
            "actions.p", "terminal_states.p", "transition_function.p", "action2rule.p",
            "state2graph.p", "rule_vocabulary.p"
        ]
        variables = [
            self.actions, self.terminal_states, self.transition_function, self.action2rule,
            self.state2graph, self.rule_vocabulary
        ]
        for file, var in zip(file_names, variables):
            with open(os.path.join(os_path, file), "wb") as f:
                pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_environment(self, path_to_folder):
        """Load environment from folder.
        The method don't clear the current state of the environment. It add saved data to current state.

        Args:
            path_to_folder (str): Path to folder with saved environment
        """
        file_names = ["actions.p", "action2rule.p", "rule_vocabulary.p"]
        variables = [self.actions, self.action2rule, self.rule_vocabulary]

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
    
    def plot_metrics(self, save, path, name):
        """Plot metrics of environment.

        Args:
            save (bool): Save plot
            path (str): Path to save plot
            name (str): Name of plot
        """

        plt.figure(figsize=(10, 5))
        plt.plot(metrics)
        plt.title(name)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        if save:
            plt.savefig(os.path.join(path, name))
        plt.show()


class SubDesignEnvironment(DesignEnvironment):

    def __init__(self,
                 rule_vocabulary: RuleVocabulary,
                 control_optimizer: GraphRewardCalculator,
                 max_number_nonterminal_rules,
                 initial_graph: GraphGrammar = GraphGrammar(),
                 verbosity=0):
        """Environment for design subspace of mechanism. Create dictionary of rules and convert it to actions by sorted name of rules.
        Max number of nonterminal rules is restriction the design space.

        Args:
            rule_vocabulary (RuleVocabulary): Vocabulary of rules
            control_optimizer (GraphRewardCalculator): Class for optimizate control of mechanism
            max_number_nonterminal_rules (_type_): Max number of nonterminal rules
            initial_graph (GraphGrammar, optional): Initial state of environment. Defaults to GraphGrammar().
            verbosity (int, optional): Information verbosity. Defaults to 0.
        """
        super().__init__(rule_vocabulary, control_optimizer, initial_graph, verbosity)
        self.max_number_nonterminal_rules = max_number_nonterminal_rules
        self.counter_nonterminal_rules: dict[STATESTYPE, int] = defaultdict(int)
        self.counter_nonterminal_rules[self.initial_state] = 0

    def get_available_actions(self, state: STATESTYPE) -> np.ndarray:
        """Get mask of available actions for state. If number of nonterminal rules of state is greater than max_number_nonterminal_rules, return mask of terminal actions.
        
        Args:
            state (STATESTYPE): state to get mask of available actions
            
        Returns:
            np.ndarray: mask of available actions
        """
        graph = self.state2graph[state]
        available_rules = self.rule_vocabulary.get_list_of_applicable_rules(graph)
        mask_available_actions = np.zeros_like(self.actions)

        rule_list = np.array(list(self.action2rule.values()))
        mask = [rule in available_rules for rule in rule_list]
        mask_available_actions[mask] = 1

        mask_terminal = self.get_terminal_actions()

        if self.max_number_nonterminal_rules <= self.counter_nonterminal_rules[state]:
            mask_available_actions *= mask_terminal

        return mask_available_actions

    def update_environment(self, graph: GraphGrammar, action: int, next_graph: GraphGrammar):
        """Update environment. If next state is not in state2graph dictionary, save it in state2graph dictionary.

        Args:
            graph (GraphGrammar): Previous state in the form of a graph
            action (int): Action that was applied to the previous state
            next_graph (GraphGrammar): Next state in the form of a graph

        Returns:
            tuple[float, bool]: reward and bool if state is in terminal_states table
        """
        state = self.data2state(graph)
        next_state = self.data2state(next_graph)
        if next_state not in self.state2graph:
            self.state2graph[next_state] = deepcopy(next_graph)
        reward, is_known = self.get_reward(next_state)
        if (state, action) not in self.transition_function:
            self.transition_function[(state, action)] = (next_state, reward,
                                                         self.is_terminal_state(next_state)[0])

        if self.get_nonterminal_actions()[action] == 1:
            self.counter_nonterminal_rules[next_state] = self.counter_nonterminal_rules[state] + 1
        elif self.get_terminal_actions()[action] == 1:
            self.counter_nonterminal_rules[next_state] = self.counter_nonterminal_rules[state]

        return reward, is_known

    def save_environment(self, prefix, path="./environments/", rewrite=False, use_date=True):
        """Save environment to folder. If folder does not exist, create it. If folder exist, create new folder with postfix.
        
        Args:
            prefix (str): Prefix of folder name
            path (str, optional): Folder to save environment. Defaults to "./environments/".
            rewrite (bool, optional): Rewrite folder if it exist. Defaults to False.
            use_date (bool, optional): Use date in folder name. Defaults to True.
        """
        os.path.split(path)
        if not os.path.exists(path):
            print(f"Path {path} does not exist. Creating...")
            os.mkdir(path)
        if use_date:
            current_date = datetime.now()
            folder = f"{prefix}__{current_date.hour}h{current_date.minute}m_{current_date.second}s_date_{current_date.day}d{current_date.month}m{current_date.year}y"
        else:
            folder = prefix
        os_path = os.path.join(path, folder)
        if not os.path.exists(os_path):
            print(f"Saving environment to {os_path}")
            os.mkdir(os_path)
        elif rewrite:
            print(f"Rewriting environment to {os_path}")
        else:
            postfix_folder = 1
            os_path = os_path + f"_{postfix_folder}"
            while os.path.exists(os_path):
                os_path = os_path.replace(f"_{postfix_folder-1}", f"_{postfix_folder}")
                postfix_folder += 1
            print(f"Saving environment to {os_path}")
        file_names = [
            "actions.p", "terminal_states.p", "transition_function.p", "action2rule.p",
            "state2graph.p", "rule_vocabulary.p", "counter_nonterminal_rules.p"
        ]
        variables = [
            self.actions, self.terminal_states, self.transition_function, self.action2rule,
            self.state2graph, self.rule_vocabulary, self.counter_nonterminal_rules
        ]
        for file, var in zip(file_names, variables):
            with open(os.path.join(os_path, file), "wb") as f:
                pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)

    def info(self, verbosity=None) -> str:
        """Get info about environment. If verbosity is None, use the value of self.verbosity.

        Args:
            verbosity (int, optional): Define verbosity of info and state_info. Maximum is 3. Defaults to None.

        Returns:
            str: String with information about environment
        """
        if verbosity is None:
            verbosity = self.verbosity
        info_out = ""

        if verbosity > 0:
            info_out = super().info(verbosity=verbosity)
        if verbosity > 1:
            info_out += f"Max number of nonterminal rules: {self.max_number_nonterminal_rules};"
            info_out += f" Max number of nonterminal rules of states: {max(self.counter_nonterminal_rules.values())};"
            info_out += "\n"
        return info_out

    def get_info_state(self, state: STATESTYPE, verbosity=None) -> str:
        """Get info about state. If verbosity is None, use the value of self.verbosity.

        Args:
            state (STATESTYPE): state to get info
            verbosity (int, optional): Define verbosity of info and state_info. Maximum is 3. Defaults to None.

        Returns:
            str: String with information about state
        """
        if verbosity is None:
            verbosity = self.verbosity
        info_out = ""
        if verbosity > 0:
            info_out = super().get_info_state(state, verbosity=verbosity)
        if verbosity > 1:
            info_out += f"Number nonterminal rules of states: {self.counter_nonterminal_rules.get(state, None)};"
            info_out += "\n"
        return info_out

    def load_environment(self, path_to_folder):
        """Load environment from folder. The method don't clear the current state of the environment. It add saved data to current state.


        Args:
            path_to_folder (str): Path to folder with saved environment
        """
        file_names = ["actions.p", "action2rule.p", "rule_vocabulary.p"]
        variables = [self.actions, self.action2rule, self.rule_vocabulary]

        for file, var in zip(file_names, variables):
            with open(os.path.join(path_to_folder, file), "rb") as f:
                var = pickle.load(f)

        with open(os.path.join(path_to_folder, "terminal_states.p"), "rb") as f:
            s_t = pickle.load(f)
        with open(os.path.join(path_to_folder, "transition_function.p"), "rb") as f:
            p_sa = pickle.load(f)
        with open(os.path.join(path_to_folder, "state2graph.p"), "rb") as f:
            s2g = pickle.load(f)
        with open(os.path.join(path_to_folder, "counter_nonterminal_rules.p"), "rb") as f:
            counter_non_rules = pickle.load(f)

        self.terminal_states.update(s_t)
        self.transition_function.update(p_sa)
        self.state2graph.update(s2g)
        self.counter_nonterminal_rules.update(counter_non_rules)
        
def prepare_state_for_optimal_simulation(state: STATESTYPE, env: DesignEnvironment) -> tuple:
    """Prepare state for simulation. Convert state to data and graph.

    Args:
        state (STATESTYPE): state to prepare
        env (DesignEnvironment): environment

    Returns:
        tuple: data and graph of state
    """
    graph = env.state2graph[state]
    control = env.terminal_states[state][1]
    data = env.control_optimizer.optim_parameters2data_control(control, graph)
    return (data, graph)