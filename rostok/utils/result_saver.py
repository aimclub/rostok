from statistics import mean
from datetime import datetime
import matplotlib.pyplot as plt
import sys
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


class MCTSReporter():
    def __init__(self, path):
        self.path = path
        self.current_rewards = []
        self.rewards=dict()
        self.main_state = RobotState()
        self.main_reward = -10
        self.main_control = []
        
    def add_reward(self, state:RobotState, reward: float, control):
        self.current_rewards.append([state.rule_list, reward, control])

    
    def make_step(self, rule, step_number):
        print(step_number)
        print(self.current_rewards)
        self.rewards[step_number] = self.current_rewards
        self.current_rewards=[]
        self.main_state.add_rule(rule)

    def plot_means(self):
        print("plot_means started")
        mean_rewards=[]
        for key in self.rewards:
            rewards = [result[1] for result in self.rewards[key]]
            print(rewards)
            mean_rewards.append(mean(rewards))

        plt.figure()
        plt.plot(mean_rewards)
        plt.show()

    def dump_results(self):
        print("dump_results started")
        time = datetime.now()
        time = str(time.date())+"_"+str(time.hour)+"-"+str(time.minute)+"-"+str(time.second)
        with open(self.path + "mcts_log_"+time+".txt", 'w') as file:
            original_stdout = sys.stdout
            sys.stdout = file
            
            print('MCTS report generated at: ', str(time))
            for key in self.rewards:
                print(str(key))
                for design in self.rewards[key]:
                    print('rules:', *design[0])
                    if design[2] != None:
                        print('control:', *design[2])
                    else: 
                        print('control:', "no joints")
                    print('reward:', design[1])
            print()
            print('main_result:')
            print('rules:', *self.main_state.rule_list)
            if self.main_control != None:
                print('control:', *self.main_control)
            else:
                print('control:', "no joints")
            print('reward:',self.main_reward)
            sys.stdout = original_stdout


def read_report(path):
    with open(path,'r') as report:
        first_line = report.readline()
        print(first_line)
        line = first_line
        while line:
            if line == 'main_result':
                rules = report.readline().split().pop(0)
                final_state = RobotState(rules)
                control = report.readline().split().pop(0)
                reward = report.readline().split().pop(0)

            line = report.readline()
    return final_state, control, reward

if __name__ =="__main__":
    reporter = MCTSReporter("results/")
    state1 = RobotState(["Rule1", "Rule2"])
    state2 = RobotState(["Rule1", "Rule2", "Rule3"])
    reporter.add_reward(state1, 10, [14, 25])
    reporter.add_reward(state2, 15, [14, 25])
    reporter.make_step("Rule1", 1)
    reporter.dump_results()
    reporter.plot_means()
    

                
            







    


