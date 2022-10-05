from copy import deepcopy
from engine.node import *

class RuleAction():
    def __init__(self,rule):
        self.rule = rule
    
    def get_replaced_node(self):
        return self.rule.replaced_node
    
    def __hash__(self):
        return hash(self.rule)

class GraphEnvironment():
    def __init__(self, initilize_graph, rules):
        self.init_graph = initilize_graph
        self.graph = initilize_graph
        self.__actions = [RuleAction(r) for r in rules]
        self.current_player = 1
        self.reward = 0 
        
    
    def getCurrentPlayer(self):
        return self.current_player
    
    def getPossibleActions(self):
        
        def filter_exist_node(action):
            if action.get_replaced_node().label == node:
                return True
            else: 
                return False
            
        label_nodes = {node[1]["Node"].label for node in self.graph.nodes.items()}
        possible_actions = []
        for node in label_nodes:
            possible_actions += set(filter(filter_exist_node, self.__actions))
        
        return possible_actions
    
    def takeAction(self, action):
        rule_action = action.rule

        new_state = deepcopy(self)
        new_state.graph.apply_rule(rule_action)
        return new_state
    
    def isTerminal(self):
        terminal_nodes = [node[1]["Node"].is_terminal for node in self.graph.nodes.items()]
        return sum(terminal_nodes) == len(terminal_nodes)
    
    def getReward(self): # Add calculate reward
        # Reward in number (3) of nodes graph mechanism 
        nodes = [node[1]["Node"] for node in self.graph.nodes.items()]
        self.reward = 10 if len(nodes) == 3 else 0
        return self.reward
    
    def step(self, action: RuleAction, render = False):
        new_state = self.takeAction(action)
        self.graph = new_state.graph
        self.reward = new_state.reward
        done = new_state.isTerminal()
        
        if render:
            plt.figure()
            nx.draw_networkx(self.graph, pos=nx.kamada_kawai_layout(self.graph, dim=2), node_size=800,
                 labels={n: self.graph.nodes[n]["Node"].label for n in self.graph})
            plt.show()
        return done, self.graph
    
    def reset(self, new_rules = None):
        self.graph = self.init_graph
        self.reward = 0
        if new_rules:
            self.__actions = [RuleAction(r) for r in new_rules]
        
    
