import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from rostok.graph_generators.graph_environment import \
    GraphVocabularyEnvironment
from rostok.graph_grammar.graph_utils import (plot_graph_reward,
                                              save_graph_plot_reward)
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
from rostok.trajectory_optimizer.control_optimizer import GraphRewardCalculator
from rostok.utils.pickle_save import Saveable
from rostok.utils.states import (MCTSOptimizedState, OptimizedGraph,
                                 OptimizedState, RobotState)


def convert_control_to_list(control):
    """Turn control parameters into list.

    Args:
        control: control parameters in the form returned by ControlOptimizer
    """
    if control is None:
        control = []
    elif isinstance(control, (float, int)):
        control = [control]

    return list(control)


class OptimizedGraphReport(Saveable):
    """Class to contain and update the list of OptimizedGraph objects

    Attributes:
        path (Path): a path to directory for saving the object
        file_name (str): name of the file to save the object
        graph_list (list[OptimizedGraph]): list of graphs"""

    def __init__(self, path=Path("./results")) -> None:
        """Create an object with empty graph_list.

        Args:
            path (Path): path for saving the object"""
        super().__init__(path, 'optimized_graph_report')
        self.graph_list: list[OptimizedGraph] = []

    def add_graph(self, graph, reward, control):
        """Add a graph, reward and control to graph_list

        The function does not check if the graph is already in list, use check_graph
        to check if a graph is already in graph_list

        Args:
            graph (GraphGrammar): the state of the main design
            reward (float): the main reward obtained during MCTS search
            control: parameters of the control for main design
        """
        control = convert_control_to_list(control)
        new_optimized_graph = OptimizedGraph(graph, reward, control)
        self.graph_list.append(new_optimized_graph)

    def check_graph(self, new_graph):
        """Check if the graph is already in graph_list

        Args:
            new_graph: the graph to check
        """
        if len(self.graph_list) > 0:
            for optimized_graph in self.graph_list:
                if optimized_graph.graph == new_graph:
                    reward = optimized_graph.reward
                    control = optimized_graph.control
                    return True, reward, control

        return False, 0, []


class OptimizedMCTSStateReport(Saveable):
    """Class to contain and update the list of MCTSOptimizedState objects

    Attributes:
        path (Path): a path to directory for saving the object
        file_name (str): name of the file to save the object
        graph_list (list[MCTSOptimizedState]): list of states
    """

    def __init__(self, path=Path("./results")) -> None:
        """Create an object with empty graph_list.

        Args:
            path (Path): path for saving the object"""
        super().__init__(path, 'optimized_MCTS_state_report')
        self.state_list: list[MCTSOptimizedState] = []

    def add_state(self, state: RobotState, reward: float, control, step_counter):
        """Add a state, reward and control to current_rewards

        state: a new state to add
        reward: a new calculated reward
        control: control parameters for the new state
        """
        control = convert_control_to_list(control)
        new_optimized_state = MCTSOptimizedState(state, reward, control, step_counter)
        self.state_list.append(new_optimized_state)


class MCTSSaveable(Saveable):
    """Class include all the information that should be saved as a result of MCTS search.

    Attributes:
        seen_graphs (OptimizedGraphReport): graphs obtained in the search
        seen_states (OptimizedMCTSStateReport): states obtained in the search
        main_state (RobotState): the main state of the MCTS search

    """

    def __init__(self, rule_vocabulary, path) -> None:
        super().__init__(path, 'MCTS_data')
        self.seen_graphs: OptimizedGraphReport = OptimizedGraphReport(path)
        self.seen_states: OptimizedMCTSStateReport = OptimizedMCTSStateReport(path)
        self.main_state: RobotState = RobotState(rules=rule_vocabulary)
        self.main_simulated_state = OptimizedState(self.main_state, 0, [])
        self.best_simulated_state = OptimizedState(self.main_state, 0, [])
        self.non_terminal_rules_limit: int = 0
        self.search_parameter = 0

    def get_best_info(self):
        """Get graph, reward and control for the best state."""
        graph = self.best_simulated_state.state.make_graph()
        return graph, self.best_simulated_state.reward, self.best_simulated_state.control

    def get_main_info(self):
        """Get graph, reward and control for the main state."""
        graph = self.main_simulated_state.state.make_graph()
        return graph, self.main_simulated_state.reward, self.main_simulated_state.control

    def draw_best_graph(self):
        """Draw best graph with plot title based on the reward."""
        graph, reward, _ = self.get_best_info()
        plot_graph_reward(graph, reward)

    def plot_means(self):
        """Plot the mean rewards for steps of MCTS search."""

        rewards = []
        for state in self.seen_states.state_list:
            i = state.step
            if len(rewards) == i:
                rewards.append([state.reward])
            else:
                rewards[i].append(state.reward)

            mean_rewards = [mean(on_step_rewards) for on_step_rewards in rewards]

        plt.figure()
        plt.plot(mean_rewards)
        plt.show()

    def save_means(self):
        """Plot the mean rewards for steps of MCTS search"""

        rewards = []
        for state in self.seen_states.state_list:
            i = state.step
            if len(rewards) == i:
                rewards.append([state.reward])
            else:
                rewards[i].append(state.reward)

            mean_rewards = [mean(on_step_rewards) for on_step_rewards in rewards]

        plt.figure()
        plt.plot(mean_rewards)
        plt.title(
            f'Iterations: {self.search_parameter}. Non-terminal rules: {self.non_terminal_rules_limit}'
        )
        plt.xlabel('Steps')
        plt.ylabel('Rewards')
        plt.savefig(Path(self.path, "step_means.png"))

    def save_visuals(self):
        """Saves graphs and info for main and best states."""
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
            print()
            print('max number of non-terminal rules:', self.non_terminal_rules_limit,
                  'search parameter:', self.search_parameter)
            print()
            print("Number of unique mechanisms tested in current MCTS run: ",
                  len(self.seen_graphs.graph_list))
            print("Number of states ", len(self.seen_states.state_list))
            sys.stdout = original_stdout

        path_to_best_graph = Path(self.path, "best_graph.png")
        best_graph, reward, _ = self.get_best_info()
        save_graph_plot_reward(best_graph, reward, path_to_best_graph)
        path_to_main_graph = Path(self.path, "main_graph.png")
        main_graph, reward, _ = self.get_main_info()
        save_graph_plot_reward(main_graph, reward, path_to_main_graph)

    def save_lists(self):
        """Saves lists of graphs and states."""
        self.seen_graphs.set_path(self.path)
        self.seen_graphs.save()
        self.seen_states.set_path(self.path)
        self.seen_states.save()

    def save_all(self):
        """Save all information in the object but not object itself."""
        self.save_visuals()
        self.save_lists()


class MCTSHelper():
    """Class that accumulates information about the MCTS search process.
    
    The instance of a class should be created for each MCTS run.

    Attributes:
        actions (RuleVocabulary): the rule_vocabulary used in the MCTS run
        optimizer (ControlOptimizer): the optimizer used in the MCTS run
        step_counter (int): counter of MCTS steps
        report (MCTSSaveable): a saveable object that contains all information about graphs, 
            states and rewards obtained in the MCTS run
    """

    def __init__(self,
                 rule_vocabulary: RuleVocabulary,
                 optimizer: GraphRewardCalculator,
                 path=Path("./results")) -> None:
        """Initialize empty instance of the MCTSHelper.

        Args:
            rule_vocabulary (RuleVocabulary): should be the same as one used in the MCTS search
            optimizer (ControlOptimizer): should be the same as one used in the MCTS search
            path (Path): path to save the results of the MCTS run
        """
        self.actions: RuleVocabulary = rule_vocabulary
        self.optimizer: GraphRewardCalculator = optimizer
        self.step_counter: int = 0
        self.report: MCTSSaveable = MCTSSaveable(rule_vocabulary, path)

    def convert_control_to_list(self, control):
        """Convert control to the form acceptable for pickle."""
        if control is None:
            control = []
        elif isinstance(control, (float, int)):
            control = [control]
        elif isinstance(control, (np.ndarray | np.matrix)):
            control = control.tolist()

        return list(control)

    def set_best_state(self, state, reward, control):
        """Set the values for best state

        Args:
            state (RobotState): the state of the best design
            reward (float): the best reward obtained during MCTS search
            control: parameters of the control for best design"""

        control = self.convert_control_to_list(control)
        self.report.best_simulated_state = OptimizedState(state, reward, control)

    def set_main_optimized_state(self, state, reward, control):
        """Set the values for main state

        Args:
            state (RobotState): the state of the main design
            reward (float): the main reward obtained during MCTS search
            control: parameters of the control for main design"""

        control = self.convert_control_to_list(control)
        self.report.main_simulated_state = OptimizedState(state, reward, control)

    def add_state(self, state: RobotState, reward: float, control):
        """Add a state, reward and control to current_rewards

        state: a new state to add
        reward: a new calculated reward
        control: control parameters for the new state
        """
        control = self.convert_control_to_list(control)
        self.report.seen_states.add_state(state, reward, control, self.step_counter)
        if reward > self.report.best_simulated_state.reward:
            self.set_best_state(state, reward, control)


class MCTSGraphEnvironment(GraphVocabularyEnvironment):
    """Class that represents the state with methods and attributes required by MCTS algorithm

    Attributes:
        init_graph (GraphGrammar): the initial graph for MCTS search
        helper (MCTSHelper): helper object for the state
        actions (RuleVocabulary): rules for the search
        optimizer (ControlOptimizer): optimizer for simulation of the mechanism
        max_actions_not_terminal (int): max number of non-terminal rules for the MCTS run"""

    def __init__(self,
                 initial_graph: GraphGrammar,
                 helper: MCTSHelper,
                 graph_vocabulary: RuleVocabulary,
                 optimizer: GraphRewardCalculator,
                 max_numbers_rules_non_terminal: int = 20):
        """Create state from the graph

        Args:
            initial_graph (GraphGrammar): initial graph for the state
        helper (MCTSHelper): helper object for the state
        graph_vocabulary (RuleVocabulary): rules for the search
        optimizer (ControlOptimizer): optimizer for simulation of the mechanism
        max_numbers_rules_non_terminal (int): max number of non-terminal rules for the MCTS run
        """
        super().__init__(initial_graph, graph_vocabulary, optimizer, max_numbers_rules_non_terminal)
        self.helper: MCTSHelper = helper

    def getReward(self):
        """Make optimization and calculate reward for the graph of the state.
        
        It also adds the graph to the seen_graph of the helper.report object
        """
        report = self.helper.report.seen_graphs.check_graph(self.graph)
        if report[0]:
            self.reward = report[1]
            self.movements_trajectory = report[2]
            self.helper.add_state(self.state, self.reward, self.movements_trajectory)
            print('seen reward:', self.reward)
            return self.reward

        result_optimizer = self.helper.optimizer.calculate_reward(self.graph)
        self.reward = result_optimizer[0]
        self.movments_trajectory = result_optimizer[1]
        self.helper.report.seen_graphs.add_graph(self.graph, self.reward, self.movments_trajectory)
        self.helper.add_state(self.state, self.reward, self.movments_trajectory)
        print(self.reward)
        return self.reward

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["actions", "optimizer", "helper"]:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result


def prepare_mcts_state_and_helper(graph: GraphGrammar,
                                  rule_vocabulary: RuleVocabulary,
                                  optimizer: GraphRewardCalculator,
                                  num_of_rules: int,
                                  path: Path = Path("./results")):
    """Set the MCTSHelper and initial MCTSGraphEnvironment

    Args:
        graph (GraphGrammar): initial graph for the MCTS run
        rule_vocabulary (RuleVocabulary): rule set for the MCTS run
        optimizer (ControlOptimizer): simulation tool for the graph
        num_of_rules (int): number of non-terminal rules for MCTS
        path (Path): path for saving the MCTS run information
    """
    mcts_helper = MCTSHelper(rule_vocabulary, optimizer, path)
    mcts_state = MCTSGraphEnvironment(graph, mcts_helper, rule_vocabulary, optimizer, num_of_rules)
    return mcts_state


def make_mcts_step(searcher, state: MCTSGraphEnvironment, counter):
    """Start MCTS search for the state and return the new state corresponding to the action

    Args:
        searcher: search algorithm
        state (MCTSGraphEnvironment): starting state for the search
        counter: counter of the steps
    """
    state.helper.step_counter = counter
    action = searcher.search(initialState=state)
    rule_action = action.get_rule
    rule_dict = state.actions.rule_dict
    rule_name = list(rule_dict.keys())[list(rule_dict.values()).index(rule_action)]
    state.helper.report.main_state.add_rule(rule_name)
    new_state: GraphVocabularyEnvironment = state.takeAction(action)
    done = new_state.isTerminal()
    if done:
        main_reward = new_state.getReward()
        main_control = new_state.movments_trajectory
        state.helper.set_main_optimized_state(new_state.state, main_reward, main_control)

    return done, new_state
