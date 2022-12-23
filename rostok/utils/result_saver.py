"""Module includes classes and functions that control saving and reading data"""
import pickle
import json
import sys
import shutil
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
    build a grath. It requires a RuleVocabulary object that controls the building of the
    graph from state.
    """
    def __init__(self, rule_list: list[str] = []):
        self.rule_list = rule_list.copy()

    def add_rule(self, rule: str):
        self.rule_list.append(rule)

    def create_and_add(self, rule):
        new = RobotState(self.rule_list)
        new.add_rule(rule)
        return new

    def __hash__(self):
        answ = ''
        for rule in self.rule_list:
            answ += rule
        return hash(answ)

    def make_graph(self, rules: RuleVocabulary):
        graph = GraphGrammar()
        for rule in self.rule_list:
            graph.apply_rule(rules.get_rule(rule))
        rules.make_graph_terminal(graph)
        return graph


class MCTSReporter():
    __instance = None

    @staticmethod
    def get_instance():
        if MCTSReporter.__instance is None:
            MCTSReporter()
        return MCTSReporter.__instance

    def __init__(self):
        if MCTSReporter.__instance is not None:
            raise Exception("Attempt to create an instance of manager class -"+
                "use get_instance method instead!")

        self.seen_graphs: list[list[GraphGrammar, float, list[float]]] = []
        self.current_rewards = []
        self.rewards = dict()
        self.main_state = RobotState()
        self.main_reward = 0.
        self.main_control = []
        self.best_state = RobotState()
        self.best_control = []
        self.best_reward = 0.
        self.path = Path("./results")
        self.rule_vocabulary = None
        MCTSReporter.__instance = self

    def set_rule_vocabulary(self, rules):
        self.rule_vocabulary = rules
    
    def add_graph(self, graph, reward, control):
        self.seen_graphs.append([deepcopy(graph), reward, deepcopy(control)])
    
    def check_graph(self, new_garph):
        if len(self.seen_graphs)>0:
            seen_graphs_t = list(zip(*self.seen_graphs))
            i = 0
            for graph in seen_graphs_t[0]:
                if graph == new_garph:
                    reward = seen_graphs_t[1][i]
                    control = seen_graphs_t[2][i]
                    #self.add_reward(self.state, self.reward, self.movments_trajectory)
                    print('seen reward:', reward)
                    return True, reward, control
                i += 1
        return False, 0, []

    def add_reward(self, state: RobotState, reward: float, control):
        self.current_rewards.append([state.rule_list, reward, control])

    def make_step(self, rule, step_number):
        self.rewards[step_number] = self.current_rewards
        self.main_state.add_rule(rule)
        path = Path(self.path, "temp_logs")
        path.mkdir(parents=True, exist_ok=True)
        rules_path = Path(path,"rules.pickle")
        file = open(rules_path,'wb+')
        pickle.dump(self.rule_vocabulary,file)
        file.close()
        path_log = Path(path,"temp_log.txt")
        with open(path_log, 'a') as file:
            original_stdout = sys.stdout
            sys.stdout = file
            print(step_number)
            for design in self.current_rewards:
                print('rules:', *design[0])
                control = design[2]
                if control is None:
                    print('control:', "no joints")
                else:
                    print('control:', control)
                print('reward:', design[1])

            print()
            print('main_result:')
            print('rules:', *self.main_state.rule_list)
            sys.stdout = original_stdout

        self.current_rewards = []
        print(f'step {step_number} finished')

    def plot_means(self):
        print("plot_means started")
        mean_rewards = []
        for key in self.rewards:
            rewards = [result[1] for result in self.rewards[key]]
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
        pickle.dump(self.rule_vocabulary,file)
        file.close()
        path_to_file = Path(path, "mcts_result.txt")
        with open(path_to_file, 'w') as file:
            original_stdout = sys.stdout
            sys.stdout = file
            print('MCTS report generated at: ', str(time))
            # for key in self.rewards:
            #     print(str(key))
            #     for design in self.rewards[key]:
            #         print('rules:', *design[0])
            #         control = design[2]
            #         if control is None:
            #             print('control:', "no joints")
            #         else:
            #             print('control:', control)
            #         print('reward:', design[1])
            print()
            print('main_result:')
            print('rules:', *self.main_state.rule_list)
            if self.main_control is None:
                print('control:', "no joints")
            else:
                print('control:', self.main_control)
            print('reward:', self.main_reward)
            print()
            print('best_result:')
            print('rules:', *self.best_state.rule_list)
            if self.best_control is None:
                print('control:', "no joints")
            else:
                print('control:', *self.best_control)
            print('reward:', self.best_reward)
            sys.stdout = original_stdout
        temp_path = self.path / "temp_logs"
        #shutil.rmtree(temp_path)
        self.create_json_report(path)
        print("dump_results finished")
        return path

    def get_best_info(self):
        graph = self.best_state.make_graph(self.rule_vocabulary)
        return graph, self.best_control, self.best_reward

    def create_json_report(self, path):
        path = Path(path, 'report.json')
        file = open(path, 'w')
        json.dump(self.rewards,file)
        file.close()

    def load_report(self, path):
        # path is a path to catalog where u have
        path_to_report = Path(path, "mcts_result.txt")
        with open(path_to_report,'r') as report:
            lines = report.readlines()
            print('read', lines[0])
            rule_list = lines[3].split()
            del rule_list[0]
            self.main_state = RobotState(rule_list)
            reward = lines[4].split()
            del reward[0]
            self.main_reward = float(reward[0])
            control = lines[5].split()
            del control[0]
            self.main_control = [float(x) for x in control]
            rule_list = lines[8].split()
            del rule_list[0]
            self.best_state = RobotState(rule_list)
            reward = lines[9].split()
            del reward[0]
            self.main_reward = float(reward[0])
            control = lines[10].split()
            del control[0]
            self.main_control = [float(x) for x in control]
        
        rule_path = Path(path, "rules.pickle")
        rule_file = open(rule_path, 'rb')
        self.rule_vocabulary = pickle.load(rule_file)
        rule_file.close()
        rewards_file = Path(path, 'report.json')
        with open(rewards_file,'r') as rewards:
            self.rewards = json.load(rewards)

        self.seen_graphs = []
        for step in self.rewards.items():
            for robot in step[1]:
                state = RobotState(robot[0])
                graph = state.make_graph(self.rule_vocabulary)
                reward = float(robot[1])
                if hasattr(robot[2],'__iter__'): 
                    control = [float(x) for x in robot[2]]
                else:
                     control = float(robot[2])
                self.add_graph(graph, reward, control)

        # path_to_report = Path(path, "mcts_result.txt")
        # with open(path_to_report,'r') as report:
        #     lines = report.readlines()
        #     print('read', lines[0])
        #     self.main_state = RobotState(lines[3])
        #     self.main_reward = float(lines[4])
        #     self.main_control = [float(x) for x in lines[5]]
        #     self.best_state = RobotState(lines[8])
        #     self.main_reward = float(lines[9])
        #     self.main_control = [float(x) for x in lines[10]]

        # line_count = 0
        # for line in report:
        #     if line_count == 0:
        #         print('read', line)
        #         continue

        # with open(path_to_log,'r') as report:
        #     lines = report.readlines()

    def read_report(path, rules: RuleVocabulary):
        path_to_log = Path(path, "mcts_log.txt")
        with open(path_to_log,'r') as report:
            final_state = None
            lines = report.readlines()
            for i in range(len(lines)):
                line = lines[i]
                if line == 'best_result:\n':
                    current_rules = (lines[i + 1]).split()
                    del current_rules[0]
                    final_state = RobotState(current_rules)
                    control = (lines[i + 2]).split()
                    del control[0]
                    reward = (lines[i + 3]).split()
                    del reward[0]
            final_graph = final_state.make_graph(rules)
            
        path_to_pic=Path(path,"best_result_graph.jpg")
        plt.figure()
        nx.draw_networkx(final_graph,
                        pos=nx.kamada_kawai_layout(final_graph, dim=2),
                        node_size=800,
                        labels={n: final_graph.nodes[n]["Node"].label for n in final_graph})
        plt.savefig(path_to_pic)
        return final_graph, control, reward


if __name__ == "__main__":

    reporter = MCTSReporter()
    state1 = RobotState(["Rule1", "Rule2"])
    state2 = RobotState(["Rule1", "Rule2", "Rule3"])
    reporter.add_reward(state1, 10, [14, 25])
    reporter.add_reward(state2, 15, [14, 25])
    reporter.make_step("Rule1", 1)
    reporter.dump_results()
    reporter.plot_means()
