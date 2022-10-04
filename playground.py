from copy import deepcopy
import random as rnd

class GraphPlayground():
    def __init__(self, initilize_graph, rules):
        self.init_state = initilize_graph
        self.actions = [RuleAction(r) for r in rules]
        self.count = 0
        # for r in rules:
            # self.actions.add(RuleAction(r))
        
    
    def getCurrentPlayer(self):
        return 1
    
    def getPossibleActions(self): # add input state 
        #set_replaces_node = {(a.get_replaced_node()) for a in self.actions}
        #for replace_node in set_replaces_node:
        
        def filter_exist_node(action):
            if action.get_replaced_node().label == node:
                return True
            else: 
                return False
            
        label_nodes = {node[1]["Node"].label for node in self.init_state.nodes.items()}
        possible_actions = []
        for node in label_nodes:
            possible_actions += list(filter(filter_exist_node, self.actions))
        
        return possible_actions
    
    def takeAction(self, action):
        rule_action = action.rule

        new_state = deepcopy(self)
        new_state.init_state.apply_rule(rule_action)
        return new_state
    
    def isTerminal(self):
        self.count += 1
        terminal_nodes = [node[1]["Node"].is_terminal for node in self.init_state.nodes.items()]
        #return sum(terminal_nodes) == len(terminal_nodes)
        return self.count > 5
    
    def getReward(self): # Add calculate reward
        reward = 1#rnd.randint(0, 10)
        return reward#rnd.randint(0, 10)
        
    
class RuleAction():
    def __init__(self,rule):
        self.rule = rule
    
    def get_replaced_node(self):
        return self.rule.replaced_node
    
    def __hash__(self):
        return hash(self.rule)
    
