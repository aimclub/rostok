import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from rostok.graph_generators.graph_environment import \
    GraphVocabularyEnvironment
from rostok.graph_grammar.graph_utils import (plot_graph_reward,
                                              save_graph_plot_reward)
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
from rostok.utils.pickle_save import Saveable
from rostok.utils.states import (MCTSOptimizedState, OptimizedGraph,
                                 OptimizedState, RobotState)


def convert_control_to_list(control):
    if control is None:
        control = []
    elif isinstance(control, (float, int)):
        control = [control]

    return list(control)


class OptimizedGraphReport(Saveable):

    def __init__(self, path=Path("./results")) -> None:
        super().__init__(path, 'optimized_graph_report')
        self.graph_list: list[OptimizedGraph] = []

    def add_graph(self, graph, reward, control):
        """Add a graph, reward and control to seen_graph

        Args:
            graph (GraphGrammar): the state of the main design
            reward (float): the main reward obtained during MCTS search
            control: parameters of the control for main design"""

        control = convert_control_to_list(control)
        new_optimized_graph = OptimizedGraph(graph, reward, control)
        self.graph_list.append(new_optimized_graph)

    def check_graph(self, new_graph):
        """Check if the graph is already in seen_graphs

        Args:
            new_graph: the graph to check"""

        if len(self.graph_list) > 0:
            for optimized_graph in self.graph_list:
                if optimized_graph.graph == new_graph:
                    reward = optimized_graph.reward
                    control = optimized_graph.control
                    print('seen reward:', reward)
                    return True, reward, control

        return False, 0, []


class OptimizedMCTSStateReport(Saveable):

    def __init__(self, path=Path("./results")) -> None:
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

    def __init__(self, rule_vocabulary, path) -> None:
        super().__init__(path, 'MCTS_data')
        self.seen_graphs: OptimizedGraphReport = OptimizedGraphReport(path)
        self.seen_states: OptimizedMCTSStateReport = OptimizedMCTSStateReport(path)
        self.main_state = RobotState(rules=rule_vocabulary)
        self.main_simulated_state = OptimizedState(self.main_state, 0, None)
        self.best_simulated_state = OptimizedState(self.main_state, 0, None)
        self.non_terminal_rules_limit: int = 0
        self.search_parameter = 0

    def get_best_info(self):
        """Get graph, reward and control for the best state"""
        graph = self.best_simulated_state.state.make_graph()
        return graph, self.best_simulated_state.reward, self.best_simulated_state.control

    def get_main_info(self):
        """Get graph, reward and control for the main state"""
        graph = self.main_simulated_state.state.make_graph()
        return graph, self.main_simulated_state.reward, self.main_simulated_state.control

    def draw_best_graph(self):
        graph, reward, _ = self.get_best_info()
        plot_graph_reward(graph, reward)

    def plot_means(self):
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
        plt.title(f'Iterations: {self.search_parameter}. Non-terminal rules: {self.non_terminal_rules_limit}')
        plt.xlabel('Steps')
        plt.ylabel('Rewards')
        plt.savefig(Path(self.path, "step_means.png"))

    def save_visuals(self):
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
            print("Number of unique mechanisms tested in current MCTS run: ", len(self.seen_graphs.graph_list))
            print("Number of states ", len(self.seen_states.state_list))
            sys.stdout = original_stdout

        path_to_best_graph = Path(self.path, "best_graph.png")
        best_graph, reward, _ = self.get_best_info()
        save_graph_plot_reward(best_graph, reward, path_to_best_graph)
        path_to_main_graph = Path(self.path, "main_graph.png")
        main_graph, reward, _ = self.get_main_info()
        save_graph_plot_reward(main_graph, reward, path_to_main_graph)

    def save_lists(self):
        self.seen_graphs.set_path(self.path)
        self.seen_graphs.save()
        self.seen_states.set_path(self.path)
        self.seen_states.save()

    def save_all(self):
        self.save_visuals()
        self.save_lists()


class MCTSHelper():

    def __init__(self, rule_vocabulary, optimizer, path=Path("./results")) -> None:
        self.actions: RuleVocabulary = rule_vocabulary
        self.optimizer: ControlOptimizer = optimizer
        self.step_counter: int = 0
        self.report: MCTSSaveable = MCTSSaveable(rule_vocabulary, path)

    def convert_control_to_list(self, control):
        if control is None:
            control = []
        elif isinstance(control, (float, int)):
            control = [control]

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


class MCTSGraphEnviromnent(GraphVocabularyEnvironment):

    def __init__(self,
                 initilize_graph: GraphGrammar,
                 helper: MCTSHelper,
                 graph_vocabulary,
                 optimizer,
                 max_numbers_rules_non_terminal=20):

        super().__init__(initilize_graph, graph_vocabulary, optimizer,
                         max_numbers_rules_non_terminal)
        self.helper: MCTSHelper = helper

    def getReward(self):
        report = self.helper.report.seen_graphs.check_graph(self.graph)
        if report[0]:
            self.reward = report[1]
            self.movments_trajectory = report[2]
            self.helper.add_state(self.state, self.reward, self.movments_trajectory)
            print('seen reward:', self.reward)
            return self.reward

        result_optimizer = self.helper.optimizer.start_optimisation(self.graph)
        self.reward = -result_optimizer[0]
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
                                  optimizer,
                                  num_of_rules: int,
                                  path: Path = Path("./results")):

    mcts_helper = MCTSHelper(rule_vocabulary, optimizer, path)
    mcts_state = MCTSGraphEnviromnent(graph, mcts_helper, rule_vocabulary, optimizer, num_of_rules)
    return mcts_state


def make_mcts_step(searcher, state: MCTSGraphEnviromnent, counter):
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
