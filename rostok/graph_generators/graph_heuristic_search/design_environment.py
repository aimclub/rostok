from typing import Union
from copy import deepcopy

import numpy as np

from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import GraphRewardCalculator

class DesignEnvironment:
    def __init__(self, rule_vocabulary: RuleVocabulary, control_optimizer: GraphRewardCalculator,
                max_no_terminal_rules: int, initial_state: GraphGrammar = GraphGrammar()):
        
        self.rule_vocabulary = rule_vocabulary
        sorted_name_rule = sorted(self.rule_vocabulary.rule_dict.keys(), key=lambda x: hash(self.rule_vocabulary.rule_dict[x]))
        
        self.id2rule: dict[int, str] = {t[0]: t[1] for t in list(enumerate(sorted_name_rule))}
        self.rule2id: dict[str, int] = {t[1]: t[0] for t in self.id2rule.items()}
        
        self.max_no_terminal_rules: int = max_no_terminal_rules
        self.terminal_graphs: dict[list[str], tuple(float, list[list[float]])] = {}
        self.initial_state = initial_state
        self.control_optimizer = control_optimizer
        
    def next_state(self, state: GraphGrammar, action: Union[int,str]) -> GraphGrammar:
        
        if isinstance(action, int):
            name_rule = self.id2rule[action]
        elif isinstance(action,str):
            name_rule = action
        else:
            ValueError(f"Wrong type action: {type(action)}. Action have to be int or str type")
        
        rule = self.rule_vocabulary.rule_dict[name_rule]
        next_state_graph = deepcopy(state)
        state.apply_rule(rule)
        
        return next_state_graph
    
    def get_action_size(self):
        return len(self.id2rule)
    
    def get_available_actions(self, state: GraphGrammar) -> np.ndarray:
        
        if state.counter_nonterminal_rules < self.max_no_terminal_rules:
            available_actions = self.rule_vocabulary.get_list_of_applicable_rules(state)
        else:
            available_actions = self.rule_vocabulary.get_list_of_applicable_terminal_rules(state)
            
        mask_available_rules = np.zeros(len(self.id2rule))
        
        for rule in available_actions:
            id_rule = self.id2rule[rule]
            mask_available_rules[id_rule] = 1
            
        return mask_available_rules
    
    def get_game_ended(self, graph: GraphGrammar) -> float:
        
        terminal_nodes = [node[1]["Node"].is_terminal for node in graph.nodes.items()]
        
        if sum(terminal_nodes) == len(terminal_nodes):
            flatten_graph = self.graph_to_state_represitation(graph)
            if flatten_graph in set(self.terminal_graphs.keys()):
                reward, movments_trajectory = self.terminal_graphs[flatten_graph]
            else:
                result_optimizer = self.control_optimizer.calculate_reward(graph)
                reward = result_optimizer[0]
                movments_trajectory = result_optimizer[1]
                self.terminal_graphs[flatten_graph] = (reward, movments_trajectory)

            return reward
        else:
            return 0
        
    def graph_to_state_represitation(self, graph: GraphGrammar):
        return hash(graph)