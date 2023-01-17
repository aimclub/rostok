"""Module includes classes and functions that control saving and reading data"""
import os
import pickle
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary


# State here is a list of rules that can be used to create robots
class RobotState():
    """The class that represents a mechanism within a given rule set.

    This class represents a mechanism through the ordered list of rules that is used to
    build a graph. It requires a RuleVocabulary object that controls the building of the
    graph from state. The State can represent the unfinished design.

    Attributes:
        rule_list (list[str]): the list of rules that determines the state

        rules (RuleVocabulary): the rule set that to use for the state
    """
    def __init__(self,  rules: RuleVocabulary, rule_list: list[str] = []):
        """Create an empty state or a state from the given rule list.

        The in list is copied and can be modified outside the state object.

        Args:
            rules (RuleVocabulary): the rule set for the state.
            rule_list (list[str]): the rule list to create a new state.

        Raises:
            Exception: one of rules in the rule_list does not belong to given the rule vocabulary.
        """

        self.__rules = rules
        list_of_rules = rules.rule_dict.keys()
        for rule in rule_list:
            if rule not in list_of_rules:
                raise Exception("Attempt to add a rule absent in the rule_vocabulary to a state")
        self.rule_list: list[str] = rule_list.copy()

    def add_rule(self, rule: str):
        """Add a rule to a list of the current state.

        Args:
            rule (str): the new rule to add to the state

        Raises:
            Exception: the rule does not belong to the rule vocabulary assigned to the state
        """

        if rule not in self.__rules.rule_dict.keys():
            raise Exception("Attempt to add to a state a rule that is absent in the" + 
                "rule_vocabulary")

        self.rule_list.append(rule)

    def create_and_add(self, rule:str):
        """Create a new state and add rule to it, return the new state.

        Args:
            rule (str): the new rule to add to the state

        Raises:
            Exception: the rule does not belong to the rule vocabulary assigned to the state
        """

        if rule not in self.__rules.rule_dict.keys():
            raise Exception("Attempt to add to a state a rule that is absent in the"+
                "rule_vocabulary")

        new = RobotState(self.__rules, self.rule_list)
        new.add_rule(rule)
        return new

    def make_graph(self):
        """Return a GraphGrammar object built with the rule list and rule vocabulary"""

        graph = GraphGrammar()
        for rule in self.rule_list:
            graph.apply_rule(self.__rules.get_rule(rule))

        self.__rules.make_graph_terminal(graph)
        return graph

    @property
    def rules(self):
        return self.__rules

    def __hash__(self):
        answer = ''
        answer = answer.join(self.rule_list)
        return hash(answer)

@dataclass
class OptimizedState():
    """Class that represents the state with a calculated reward

    Attributes:
        state (RobotState): the state to calculate reward
        reward (float): the calculated reward of the state
        control (list[float]): the parameters of the optimized control"""

    state: RobotState
    reward: float
    control: Any


class OptimizedGraph():
    """Class that represents the graph with a calculated reward"""

    def __init__(self, graph, reward, control):
        self.__graph = deepcopy(graph)
        self.__reward = reward
        self.__control = deepcopy(control)

    @property
    def graph(self):
        return self.__graph

    @property
    def reward(self):
        return self.__reward

    @property
    def control(self):
        return self.__control


class MCTSReporter():
    """A Singleton class to gather and dump information about MCTS search process

    Attributes:
        instance: the instance of the class to call
        seen_graphs: list of graphs with already calculated rewards
        rule_vocabulary: set of rules that is used in MCTS search
        current_rewards: list of states with calculated rewards at current step
        rewards: dictionary with step number as a key and list of rewards as values
        main_state: the state currently generated with MCTS search
        main_simulated_state: the state and reward/control built in the MCTS search
        best_simulated_state: the state with the best reward obtained during the search
        path: path for saving data
        """
    __instance = None

    @staticmethod
    def get_instance():
        if MCTSReporter.__instance is None:
            MCTSReporter()
        return MCTSReporter.__instance

    def __init__(self) -> None:
        if MCTSReporter.__instance is not None:
            raise Exception("Attempt to create an instance of manager class -"+
                "use get_instance method instead!")

        self.seen_graphs: list[OptimizedGraph] = []
        self.__rule_vocabulary = None
        self.current_rewards: list[OptimizedState] = []
        self.rewards: dict[int, list[OptimizedState]] = {}
        self.main_state = None
        self.main_simulated_state = None
        self.best_simulated_state = None
        self.path = Path("./results")
        MCTSReporter.__instance = self

    @property
    def rule_vocabulary(self):
        return self.__rule_vocabulary

    @rule_vocabulary.setter
    def rule_vocabulary(self, rules):
        self.__rule_vocabulary = rules

    def initialize(self):
        """ Create and set initial values for main and best states

        Should be called after the rule vocabulary is set"""

        if self.rule_vocabulary is None:
            raise Exception("Attempt to initialize MCTSReporter without rule vocabulary")

        self.main_state = RobotState(rules=self.rule_vocabulary)
        self.main_simulated_state = OptimizedState(self.main_state, 0, None)
        self.best_simulated_state = OptimizedState(self.main_state, 0, None)
        time = datetime.now()
        time = str(time.date()) + "_" + str(time.hour) + "-" + str(time.minute) + "-" + str(
            time.second)
        self.path = Path(self.path, "MCTS_report_" + datetime.now().strftime("%yy_%mm_%dd_%HH_%MM"))
        self.path.mkdir(parents=True, exist_ok=True)

    def set_best_state(self, state, reward, control):
        """Set the values for best state

        Args:
            state (RobotState): the state of the best design
            reward (float): the best reward obtained during MCTS search
            control: parameters of the control for best design"""

        if control is None:
            control = []
        elif isinstance(control, (float,int)):
            control = [control]

        control = list(control)
        self.best_simulated_state = OptimizedState(state, reward, control)

    def set_main_optimized_state(self, state, reward, control):
        """Set the values for main state

        Args:
            state (RobotState): the state of the main design
            reward (float): the main reward obtained during MCTS search
            control: parameters of the control for main design"""

        if control is None:
            control = []
        elif isinstance(control, (float, int)):
            control = [control]

        control = list(control)
        self.main_simulated_state = OptimizedState(state, reward, control)

    def add_graph(self, graph, reward, control):
        """Add a graph, reward and control to seen_graph

        Args:
            graph (GraphGrammar): the state of the main design
            reward (float): the main reward obtained during MCTS search
            control: parameters of the control for main design"""

        if control is None:
            control = []
        elif isinstance(control, (float, int)):
            control = [control]

        control = list(control)
        new_optimized_graph = OptimizedGraph(graph, reward, control)
        self.seen_graphs.append(new_optimized_graph)

    def check_graph(self, new_graph):
        """Check if the graph is already in seen_graphs

        Args:
            new_graph: the graph to check"""

        if len(self.seen_graphs) > 0:
            for optimized_graph in self.seen_graphs:
                if optimized_graph.graph == new_graph:
                    reward = optimized_graph.reward
                    control = optimized_graph.control
                    print('seen reward:', reward)
                    return True, reward, control

        return False, 0, []

    def add_state(self, state: RobotState, reward: float, control):
        """Add a state, reward and control to current_rewards

        state: a new state to add
        reward: a new calculated reward
        control: control parameters for the new state
        """
        if control is None:
            control = []
        elif isinstance(control, (float, int)):
            control = [control]

        control = list(control)
        new_optimized_state = OptimizedState(state, reward, control)
        self.current_rewards.append(new_optimized_state)
        if reward > self.best_simulated_state.reward:
            self.set_best_state(state, reward, control)

    def make_step(self, rule, step_number):
        """Save the information about the MCTS step

        The function must be called after each MCTS step. It also update the temp files

        Args:
            rule: the rule chosen in the search process
            step_number (int): the MCTS step number that would be a key in rewards dict"""

        self.rewards[step_number] = self.current_rewards
        if self.main_state is not None:
            self.main_state.add_rule(rule)
        else:
            self.main_state = RobotState(self.rule_vocabulary, [rule])

        # save current state of the reporter
        path_temp_state = Path(self.path, "temp_state.pickle")
        with open(path_temp_state, 'wb+') as file:
            pickle.dump(self, file)

        self.current_rewards = []
        print(f'step {step_number} finished')

    def plot_means(self):
        """Plot the mean rewards for steps of MCTS search"""

        mean_rewards = []
        for item in self.rewards.items():
            rewards = [result.reward for result in item[1]]
            mean_rewards.append(mean(rewards))

        plt.figure()
        plt.plot(mean_rewards)
        plt.show()

    def save_result(self):
        path_to_file = Path(self.path, "mcts_result.txt")
        with open(path_to_file, 'w', encoding='utf-8') as file:
            original_stdout = sys.stdout
            sys.stdout = file
            print('main_result:')
            print('rules:', *self.main_simulated_state.state.rule_list)
            print('control:', *self.main_simulated_state.control)
            print('reward:', self.main_simulated_state.reward)
            print()
            print('best_result:')
            print('rules:', *self.best_simulated_state.state.rule_list)
            print('control:', *self.best_simulated_state.control)
            print('reward:', self.best_simulated_state.reward)
            sys.stdout = original_stdout

    def delete_temp(self):
        path_temp_state = Path(self.path, "temp_state.pickle")
        os.remove(path_temp_state)

    def get_best_info(self):
        """Get graph, reward and control for the best state"""
        graph = self.best_simulated_state.state.make_graph()
        return graph, self.best_simulated_state.reward, self.best_simulated_state.control

    def save_reporter(self):
        """Saves the Reporter object to file

        Args:
            path: path to the directory where the Reporter should be saved"""

        path = Path(self.path, 'reporter_state.pickle')
        with open(path, 'wb+') as file:
            pickle.dump(self, file)

    def save_rule_vocabulary(self):
        path = Path(self.path, "rules.pickle")
        with open(path, 'wb+') as file:
            pickle.dump(self.rule_vocabulary, file)
    
    def save_seen_graphs(self):
        path = Path(self.path, "seen_graphs.pickle")
        with open(path, "wb+") as file:
            pickle.dump(self.seen_graphs)

    def load_rule_set(self, path):
        """Load a rule vocabulary to Reporter object

        Args:
            path: path to directory with rule vocabulary saved"""

        rule_path = Path(path, "rules.pickle")
        rule_file = open(rule_path, 'rb')
        self.rule_vocabulary = pickle.load(rule_file)
        rule_file.close()

    def load_seen_graphs(self, path, temp = False):
        """Load list of graphs with calculated rewards to Reporter object

        Args:
            path: path to directory with rule vocabulary saved"""

        if temp:
            graph_path = Path(path, "temp_graphs.pickle")
        else:
            graph_path = Path(path, "seen_graphs.pickle")
        with open(graph_path,'rb') as graphs:
            seen_graphs = pickle.load(graphs)

        return seen_graphs

def load_reporter(path, temp = False):
    """Creates a Reporter object using previously saved state

    Args:
        path: path to the directory with saved reporter
        temp: flag to use the temp or total report"""

    if temp:
        path_to_report = Path(path, "temp_state.pickle")
    else:
        path_to_report = Path(path, "reporter_state.pickle")

    with open(path_to_report,'rb') as report:
        return pickle.load(report)

# if __name__ == "__main__":
#     reporter = MCTSReporter()
#     state1 = RobotState(["Rule1", "Rule2"])
#     state2 = RobotState(["Rule1", "Rule2", "Rule3"])
#     reporter.add_reward(state1, 10, [14, 25])
#     reporter.add_reward(state2, 15, [14, 25])
#     reporter.make_step("Rule1", 1)
#     reporter.dump_results()
#     reporter.plot_means()
