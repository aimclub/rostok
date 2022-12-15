from statistics import mean
from pathlib import Path 
from datetime import datetime
import sys
import networkx as nx
import matplotlib.pyplot as plt
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary

# State here is a list of rules that can be used to create robots
class RobotState():
    def __init__(self, rule_list:list[str]=[]):
        self.rule_list = rule_list.copy()
    
    def add_rule(self, rule:str):
        self.rule_list.append(rule)

    def create_and_add(self, rule):
        new = RobotState(self.rule_list)
        new.add_rule(rule)
        return new
    
    def __hash__(self):
        #print(self.rule_list)
        answ = ''
        for rule in self.rule_list:
            answ+=rule
        return hash(answ)

    def make_graph(self, rules: RuleVocabulary):
        graph = GraphGrammar()
        for rule in self.rule_list:
            graph.apply_rule(rules.get_rule(rule))
        rules.make_graph_terminal(graph)
        return graph


class MCTSReporter():
    def __init__(self):
        #self.path = path
        self.current_rewards = []
        self.rewards=dict()
        self.main_state = RobotState()
        self.main_reward = 0.
        self.main_control = []
        self.best_state = RobotState()
        self.best_control = []
        self.best_reward = 0.
        
    def add_reward(self, state:RobotState, reward: float, control):
        self.current_rewards.append([state.rule_list, reward, control])

    
    def make_step(self, rule, step_number):
        #print(step_number)
        #print(self.current_rewards)
        self.rewards[step_number] = self.current_rewards
        self.current_rewards=[]
        self.main_state.add_rule(rule)


    def plot_means(self):
        print("plot_means started")
        mean_rewards=[]
        for key in self.rewards:
            rewards = [result[1] for result in self.rewards[key]]
            mean_rewards.append(mean(rewards))

        plt.figure()
        plt.plot(mean_rewards)
        plt.show()

    def dump_results(self):
        print("dump_results started")
        time = datetime.now()
        time = str(time.date())+"_"+str(time.hour)+"-"+str(time.minute)+"-"+str(time.second)
        path = Path("./results/MCTS_report_" + datetime.now().strftime("%yy_%mm_%dd_%HH_%MM"))
        path. mkdir(parents=True, exist_ok=True)
        path_to_file = Path(path, "mcts_log.txt")
        with open(path_to_file, 'w') as file:
            original_stdout = sys.stdout
            sys.stdout = file
            print('MCTS report generated at: ', str(time))
            for key in self.rewards:
                print(str(key))
                for design in self.rewards[key]:
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
            if self.main_control is None:
                print('control:', "no joints")   
            else:
                print('control:', self.main_control)
            print('reward:',self.main_reward)
            print()
            print('best_result:')
            print('rules:', *self.best_state.rule_list)
            if self.best_control is None:
                print('control:', "no joints")
            else:
                print('control:', *self.best_control) 
            print('reward:',self.best_reward)
            sys.stdout = original_stdout
        return path


def read_report(path, rules: RuleVocabulary):
    path_to_log = Path(path, "mcts_log.txt")
    with open(path_to_log,'r') as report:
        final_state = None
        lines = report.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line == 'best_result:\n':
                current_rules = (lines[i+1]).split()
                del current_rules[0]
                final_state = RobotState(current_rules)
                control = (lines[i+2]).split()
                del control[0]
                reward = (lines[i+3]).split()
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

if __name__ =="__main__":

    reporter = MCTSReporter()
    state1 = RobotState(["Rule1", "Rule2"])
    state2 = RobotState(["Rule1", "Rule2", "Rule3"])
    reporter.add_reward(state1, 10, [14, 25])
    reporter.add_reward(state2, 15, [14, 25])
    reporter.make_step("Rule1", 1)
    reporter.dump_results()
    reporter.plot_means()
    

                
            







    


