"""Module includes classes and functions that control saving and reading data"""
#import json
from dataclasses import dataclass
import pickle
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import networkx as nx

from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary


# State here is a list of rules that can be used to create robots
class RobotState():
    """The class that represents a mechanism within a given ruleset.

    This class represents a mechanism through the ordered list of rules that is used to
    build a graph. It requires a RuleVocabulary object that controls the building of the
    graph from state. The State can represent the unfinished design.
    """
    def __init__(self,  rules: RuleVocabulary, rule_list: list[str] = []):
        self.rule_list: list[str] = rule_list.copy()
        self.rules = rules

    def add_rule(self, rule: str):
        self.rule_list.append(rule)

    def create_and_add(self, rule):
        new = RobotState(self.rule_list)
        new.add_rule(rule)
        return new

    def __hash__(self):
        answer = ''
        answer = answer.join(self.rule_list)
        return hash(answer)

    def make_graph(self):
        graph = GraphGrammar()
        for rule in self.rule_list:
            graph.apply_rule(self.rules.get_rule(rule))
        self.rules.make_graph_terminal(graph)
        return graph

@dataclass
class OptimizedState():
    def __init__(self, state: RobotState, reward = 0, control = None):
        self.rules = state.rules
        self.state = state
        self.reward = reward
        self.control = control

@dataclass
class OptimizedGraph():
    def __init__(self, graph, reward, control):
        self.graph = deepcopy(graph)
        self.reward = reward
        self.control = deepcopy(control)

class MCTSReporter():
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
        self._rule_vocabulary = None
        self.current_rewards: list[OptimizedState] = []
        self.rewards:dict[int, list[OptimizedState]] = {}
        self.main_state = None
        self.main_simulated_state = None
        self.best_simulated_state = None
        self.path = Path("./results")
        MCTSReporter.__instance = self

    @property
    def rule_vocabulary(self):
        return self._rule_vocabulary

    def initialize(self):
        self.main_state = RobotState(rules=self.rule_vocabulary)
        self.main_simulated_state = OptimizedState(self.main_state, 0, None)
        self.best_simulated_state = OptimizedState(self.main_state, 0, None)

    @rule_vocabulary.setter
    def rule_vocabulary(self, rules):
        self._rule_vocabulary = rules

    def set_best_state(self, graph, reward, control):
        if control is None:
            control = []
        elif isinstance(control, (float,int)):
            control = [control]

        control = list(control)
        self.best_simulated_state = OptimizedState(graph,reward,control)

    def set_main_optimized_state(self, graph, reward, control):
        if control is None:
            control = []
        elif isinstance(control, (float, int)):
            control = [control]

        control = list(control)
        self.main_simulated_state = OptimizedState(graph,reward,control)

    def add_graph(self, graph, reward, control):
        if control is None:
            control = []
        elif isinstance(control, (float, int)):
            control = [control]

        control = list(control)
        new_optimized_graph = OptimizedGraph(graph, reward, control)
        self.seen_graphs.append(new_optimized_graph)

    def check_graph(self, new_graph):
        if len(self.seen_graphs) > 0:
            #seen_graphs_t = list(zip(*self.seen_graphs))
            i = 0
            for optimized_graph in self.seen_graphs:
                if optimized_graph.graph == new_graph:
                    reward = optimized_graph.reward
                    control = optimized_graph.control
                    #self.add_reward(self.state, self.reward, self.movements_trajectory)
                    print('seen reward:', reward)
                    return True, reward, control
                i += 1

        return False, 0, []

    def add_reward(self, state: RobotState, reward: float, control):
        if control is None:
            control = []
        elif isinstance(control, (float, int)):
            control = [control]

        control = list(control)
        new_optimized_state = OptimizedState(state, reward, control)
        self.current_rewards.append(new_optimized_state)

    def make_step(self, rule, step_number):
        #add to the dict of steps
        self.rewards[step_number] = self.current_rewards
        if self.main_state is not None:
            self.main_state.add_rule(rule)
        else:
            self.main_state = RobotState(self.rule_vocabulary, [rule])

        # create directory for the temp state
        path = Path(self.path, "temp_state")
        path.mkdir(parents=True, exist_ok=True)
        # save the rule set
        rules_path = Path(path,"rules.pickle")
        file = open(rules_path,'wb+')
        pickle.dump(self.rule_vocabulary, file)
        file.close()
        # save current state of the reporter
        path_temp_state = Path(path, "temp_state.pickle")
        with open(path_temp_state, 'wb+') as file:
            pickle.dump(self, file)

        path_temp_graphs = Path(path, "temp_graphs.pickle")
        with open(path_temp_graphs, "wb+") as file:
            pickle.dump(self.seen_graphs, file)

        path_temp_main = Path(path,"temp_main.txt")
        with open(path_temp_main, 'w', encoding='utf-8') as file:
            original_stdout = sys.stdout
            sys.stdout = file
            print()
            print('main_result:')
            print('rules:\n', *self.main_state.rule_list)
            sys.stdout = original_stdout

        self.current_rewards = []
        print(f'step {step_number} finished')

    def plot_means(self):
        print("plot_means started")
        mean_rewards = []
        for item in self.rewards.items():
            rewards = [result.reward for result in item[1]]
            mean_rewards.append(mean(rewards))

        plt.figure()
        plt.plot(mean_rewards)
        plt.show()

    def dump_results(self):
        print("dump_results started")
        time = datetime.now()
        time = str(time.date()) + "_" + str(time.hour) + "-" + str(time.minute) + "-" + str(
            time.second)
        path = Path(self.path,"MCTS_report_" + datetime.now().strftime("%yy_%mm_%dd_%HH_%MM"))
        path.mkdir(parents=True, exist_ok=True)
        rules_path = Path(path,"rules.pickle")
        file = open(rules_path,'wb+')
        pickle.dump(self.rule_vocabulary, file)
        file.close()
        path_to_file = Path(path, "mcts_result.txt")
        # with open(path_to_file, 'w', encoding='utf-8') as file:
        #     original_stdout = sys.stdout
        #     sys.stdout = file
        #     print('MCTS report generated at: ', str(time))
        #     print()
        #     print('main_result:')
        #     print('rules:', *self.main_simulated_state.state.rule_list)
        #     print('control:', *self.main_simulated_state.control)
        #     print('reward:', self.main_simulated_state.reward)
        #     print()
        #     print('best_result:')
        #     print('rules:', *self.best_simulated_state.state.rule_list)
        #     print('control:', *self.best_simulated_state.control)
        #     print('reward:', self.best_simulated_state.reward)
        #     sys.stdout = original_stdout

        path_seen_graphs = Path(path, "seen_graphs.pickle")
        with open(path_seen_graphs, "wb+") as file:
            pickle.dump(self.seen_graphs,file)

        temp_path = self.path / "temp_state"
        if temp_path.exists():
            shutil.rmtree(temp_path)

        self.save_object(path)
        print("dump_results finished")
        return path

    def get_best_info(self):
        graph = self.best_simulated_state.state.make_graph()
        return graph, self.best_simulated_state.reward, self.best_simulated_state.control

    def save_object(self, path):
        path = Path(path, 'reporter_state.pickle')
        file = open(path, 'wb+')
        pickle.dump(self, file)
        file.close()

    # def create_json_report(self, path):
    #     path = Path(path, 'report.json')
    #     file = open(path, 'w')
    #     json.dump(self.rewards, file)
    #     file.close()

    def load_rule_set(self, path):
        rule_path = Path(path, "rules.pickle")
        rule_file = open(rule_path, 'rb')
        self.rule_vocabulary = pickle.load(rule_file)
        rule_file.close()

    def load_seen_graphs(self, path, temp = False):
        if temp:
            graph_path = Path(path, "temp_graphs.pickle")
        else:
            graph_path = Path(path, "seen_graphs.pickle")
        with open(graph_path,'rb') as graphs:
            seen_graphs = pickle.load(graphs)

        return seen_graphs

    # def read_report(path, rules: RuleVocabulary):
    #     path_to_log = Path(path, "mcts_log.txt")
    #     with open(path_to_log,'r') as report:
    #         final_state = None
    #         lines = report.readlines()
    #         for i in range(len(lines)):
    #             line = lines[i]
    #             if line == 'best_result:\n':
    #                 current_rules = (lines[i + 1]).split()
    #                 del current_rules[0]
    #                 final_state = RobotState(current_rules)
    #                 control = (lines[i + 2]).split()
    #                 del control[0]
    #                 reward = (lines[i + 3]).split()
    #                 del reward[0]
    #         final_graph = final_state.make_graph(rules)
            
    #     path_to_pic=Path(path,"best_result_graph.jpg")
    #     plt.figure()
    #     nx.draw_networkx(final_graph,
    #                     pos=nx.kamada_kawai_layout(final_graph, dim=2),
    #                     node_size=800,
    #                     labels={n: final_graph.nodes[n]["Node"].label for n in final_graph})
    #     plt.savefig(path_to_pic)
    #     return final_graph, control, reward

def load_reporter(path, temp = False):
    # path is a path to catalog where u have
    if temp:
        path_to_report = Path(path, "temp_state.pickle")
    else:
        path_to_report = Path(path, "reporter_state.pickle")

    with open(path_to_report,'rb') as report:
        return pickle.load(report)

if __name__ == "__main__":

    reporter = MCTSReporter()
    state1 = RobotState(["Rule1", "Rule2"])
    state2 = RobotState(["Rule1", "Rule2", "Rule3"])
    reporter.add_reward(state1, 10, [14, 25])
    reporter.add_reward(state2, 15, [14, 25])
    reporter.make_step("Rule1", 1)
    reporter.dump_results()
    reporter.plot_means()
