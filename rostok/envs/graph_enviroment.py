from copy import deepcopy
import gymnasium as gym
import networkx as nx
import gymnasium.spaces as spaces
from engine.node import GraphGrammar

from engine.rule_vocabulary import RuleVocabulary, NodeVocabulary

def networkx_2_gym_digraph(nx_graph: nx.DiGraph, table_nodes_id: dict[int, str]):
    pass    


class GraphGrammarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, node_vocabulary: NodeVocabulary, rule_vocabulary: RuleVocabulary, render_mode=None):
        
        self.node_vocabulary = deepcopy(node_vocabulary)
        self.rule_vocabulary = deepcopy(rule_vocabulary)
        self.graph_grammar = GraphGrammar()
        
        dict_rules = self.rule_vocabulary.rule_dict
        dict_nodes = self.node_vocabulary.node_dict
        
        amount_nodes = len(dict_nodes.keys())
        node_space = spaces.Discrete(amount_nodes)
        self.table_nodes = dict(map(lambda x, y: (x, y), range(amount_nodes), dict_nodes.keys()))
        
        edges = spaces.Discrete(1, start=1)
        #FIXME: Change observation space
        
        self.observation_space = spaces.Graph(node_space,edges)
        
        amount_rules = len(dict_rules.keys())
        self.action_space = spaces.Discrete(amount_rules)
        self.table_rules = dict(map(lambda x, y: (x, y), range(amount_rules), dict_rules.keys()))
        
        assert render_mode is None or render_mode is self.metadata["render_modes"]
        
        
    
        
        
    def _get_initial_obs(self):
        observation = spaces.GraphInstance()
        observation.nodes = [0,]
        observation.edges = None
        observation.edge_links = None
        return observation
    
    def _get_obs(self):
        observation = spaces.GraphInstance()
        observation.nodes = [0,]
        observation.edges = None
        observation.edge_links = None
        return observation
    
    def _get_info(self):
        pass
    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        
        
        