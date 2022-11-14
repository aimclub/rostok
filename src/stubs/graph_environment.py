from copy import deepcopy
from engine.node import *
from stubs.graph_reward import Reward
from engine.rule_vocabulary import RuleVocabulary

# Function finding non terminal rules 
def rule_is_terminal(rule: Rule):
    terminal_rule = [node[1]["Node"].is_terminal for node in rule.graph_insert.nodes.items()]
    return sum(terminal_rule)
# Class action like rule grammar

class RuleAction:
    def __init__(self,rule):
        self.__rule = rule
        
    @property
    def get_rule(self):
        return self.__rule

    def get_replaced_node(self):
        return self.__rule.replaced_node
    
    def is_terminal(self):
        return self.__rule.is_terminal
    
    # For this algorithm need override hash method
    
    def __hash__(self):
        return hash(self.__rule)

# Class "environment" graph

class GraphEnvironment():
    def __init__(self, initilize_graph, rule_vocab: RuleVocabulary, node_vocab, max_numbers_rules_non_terminal = 20):
        self.init_graph = initilize_graph
        self.graph = initilize_graph
        self.__actions = [RuleAction(rule_vocab.rule_dict[r]) for r in rule_vocab.rule_dict.keys()]
        self.max_actions_not_terminal = max_numbers_rules_non_terminal
        self.map_nodes_reward = {}
        self.current_player = 1
        self.reward = 0
        self.counter_action = 0
        self.node_vocab = node_vocab
        
    # Need override for mcts libary
    
    def getCurrentPlayer(self):
        return self.current_player
    
    # getter possible actions for current state
    
    def getPossibleActions(self):
        
        def filter_exist_node(action):
            out = False
            flags_max_actions = self.counter_action >= self.max_actions_not_terminal
            if action.get_replaced_node().label == node:
                if action.is_terminal():
                    out = True
                if not flags_max_actions:
                    out = True
                
            return out
        label_nodes = {node[1]["Node"].label for node in self.graph.nodes.items()}
        possible_actions = []
        for node in label_nodes:
            possible_actions += set(filter(filter_exist_node, self.__actions))
        
        return possible_actions
    
    # take action and return new state environment
    
    def takeAction(self, action):
        rule_action = action.get_rule
        new_state = deepcopy(self)
        new_state.graph.apply_rule(rule_action)
        if not action.is_terminal():
            new_state.counter_action += 1
        return new_state
    
    # Condition on terminal graph
    
    def isTerminal(self):
        terminal_nodes = [node[1]["Node"].is_terminal for node in self.graph.nodes.items()]
        return sum(terminal_nodes) == len(terminal_nodes)
    
    # getter reward
    
    def getReward(self): # Add calculate reward
        # Reward in number (3) of nodes graph mechanism
        if self.map_nodes_reward:
            reward = self.function_reward(self.graph, self.map_nodes_reward, self.node_vocab)
            self.reward = reward
        else:
            nodes = [node[1]["Node"] for node in self.graph.nodes.items()]
            self.reward = 10 if len(nodes) == 4 else 0
        return self.reward
    
    # Move current environment to new state
    
    def step(self, action: RuleAction, render = False):
        new_state = self.takeAction(action)
        self.graph = new_state.graph
        self.reward = new_state.reward
        self.counter_action = new_state.counter_action
        done = new_state.isTerminal()
        
        if render:
            plt.figure()
            nx.draw_networkx(self.graph, pos=nx.kamada_kawai_layout(self.graph, dim=2), node_size=800,
                 labels={n: self.graph.nodes[n]["Node"].label for n in self.graph})
            plt.show()
        return done, self.graph
    
    def set_node_rewards(self, map_of_reward: map, func_reward: Reward.complex):
        self.map_nodes_reward = map_of_reward
        self.function_reward = func_reward
    
    # Reset environment (experimental version)
    def reset(self, new_rules = None):
        self.graph = self.init_graph
        self.reward = 0
        self.counter_action = 0
        if new_rules:
            self.__actions = [RuleAction(r) for r in new_rules]
        
    
