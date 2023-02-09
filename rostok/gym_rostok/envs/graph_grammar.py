from copy import deepcopy
from abc import ABC

import gymnasium as gym
import gymnasium.spaces as spaces

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary


def networkx_2_gym_digraph(nx_graph: GraphGrammar,
                           table_nodes_id: dict[str, int]) -> spaces.GraphInstance:
    """Convert GraphGrammar object to GraphInstance from gym/gymnasium

    Args:
        nx_graph (GraphGrammar): graph is converted to gym graph 
        table_nodes_id (dict[str, int]): table of nodes id {"name_node": "id for gym graph", "}

    Returns:
        gymnsaium.spaces.GraphInstance: Gymnasium Graph for environment
    """
    list_id_nodes = np.asarray(
        list(table_nodes_id[node[1]["Node"].label] for node in nx_graph.nodes.items()))

    table_id_index_nodes = dict((id[1], id[0]) for id in enumerate(nx_graph.nodes.keys()))
    nx_edges = list(nx_graph.edges)

    list_id_edge_links = np.asarray(
        list(map(lambda x: [table_id_index_nodes[n] for n in x], nx_edges)))

    list_edges = np.asarray([1 for _ in range(len(list_id_edge_links))])

    gym_graph = spaces.GraphInstance(list_id_nodes, list_edges, list_id_edge_links)

    return gym_graph


class GraphGrammarEnv(gym.Env, ABC):
    r"""Abstract class gym graph grammar environment project rostok.
    
    The auxiliraty methods for specific graph grammar environments that users of this class need to know are:

    - :meth:`get_mask_possible_actions` - Return mask of possible actions for the current observation state,
    - :meth:`set_max_number_nonterminal_rules` - You can set max number nonterminal rules for finite optimization.

    Environments have additional attributes for users to understand the implementation

    - :attr:`rule_vocabulary` - Main class for manipulate with graph.
    - :attr:`graph_grammar` - GraphGrammar object in current state.
    - :attr:`table_nodes` - A dictionary name nodes and it id in gym graph representation.
    - :attr:`table_rules` - A dictionary name rules and it id in gym discrete actions representation.
    - :attr:`controller` - Object of the controller for robot, made up of the graph.  
    - :attr:`max_number_nonterminal_rules` - Max number of nonterminal rules. 
        It is setted :meth:`set_max_number_nonterminal_rules`. 
        Default by infity.
    """
    metadata = {"render_modes": ["grammar", "simulation", "grammar&simulation"], "render_fps": 4}

    def __init__(self,
                 rule_vocabulary: RuleVocabulary = RuleVocabulary(),
                 controller=None,
                 render_mode=None):
        """Abstract class gym graph grammar environment project rostok.

        Args:
            - rule_vocabulary (RuleVocabulary): Instance of RuleVocabulary for manipulation graph and apply rules.
            - controller (_type_): Controller instance for controlling robot, made up of the graph
            - render_mode (string, optional): Define type of rendering modifying graph on each step (`grammar`),
            or result robot simulation on each episodes (`simulation`). May set both (`grammar&simulation`).
            Defaults to None.
        """

        self.rule_vocabulary = deepcopy(rule_vocabulary)
        self._graph_grammar = GraphGrammar()

        dict_rules = self.rule_vocabulary.rule_dict
        dict_nodes = self.rule_vocabulary.node_vocab.node_dict

        #FIXME: Change observation space
        amount_nodes = len(dict_nodes.keys())
        node_space = spaces.Discrete(amount_nodes)
        self.table_nodes = dict(map(lambda x, y: (x, y), range(amount_nodes), dict_nodes.keys()))
        self.__auxiliraty_table_nodes = dict(
            map(lambda x, y: (x, y), dict_nodes.keys(), range(amount_nodes)))

        edges = spaces.Discrete(1, start=1)

        self.observation_space = spaces.Graph(node_space, edges)

        amount_rules = len(dict_rules.keys())
        self.action_space = spaces.Discrete(amount_rules)
        self.table_rules = dict(map(lambda x, y: (x, y), range(amount_rules), dict_rules.keys()))

        self.controller = controller

        self.max_number_nonterminal_rules = float("inf")
        self._number_nonterminal_rules = 0
        self._number_terminal_rules = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if render_mode in ("grammar", "grammar&simulation"):
            self._grammar_render = plt.subplots()
            self._grammar_render[0].show()
            self._grammar_render[1].axis("off")

    def _get_initial_obs(self):
        observation = spaces.GraphInstance([0], None, None)
        return observation

    def _get_obs(self):
        observation = networkx_2_gym_digraph(self._graph_grammar, self.__auxiliraty_table_nodes)
        return observation

    def _sim(self):
        #TODO: Add simulation and getting reward
        reward = 0
        return reward

    def _get_info(self):
        return {
            "num_nodes": len(list(self._graph_grammar.nodes)),
            "num_applicable_rules": {
                "nonterminal":
                    len(
                        self.rule_vocabulary.get_list_of_applicable_nonterminal_rules(
                            self._graph_grammar)),
                "terminal":
                    len(
                        self.rule_vocabulary.get_list_of_applicable_terminal_rules(
                            self._graph_grammar))
            },
            "num_aplied_rules": {
                "nonterminal": self._number_nonterminal_rules,
                "terminal": self._number_terminal_rules
            }
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._graph_grammar = GraphGrammar()
        self._number_nonterminal_rules = 0
        self._number_terminal_rules = 0
        observation = self._get_initial_obs()
        info = self._get_info()
        return observation, info

    def render(self, terminated):
        if self.render_mode in ("grammar", "grammar&simulation"):
            self._grammar_render[1].clear()
            nx.draw_networkx(
                self._graph_grammar,
                pos=nx.kamada_kawai_layout(self._graph_grammar, dim=2),
                node_size=800,
                labels={n: self._graph_grammar.nodes[n]["Node"].label for n in self._graph_grammar})
            plt.pause(0.1)
        if self.render_mode in ("simulation", "grammar&simulation") and terminated is True:
            #TODO: Add method to visualizer
            pass

    def close(self):
        if self.render_mode in ("grammar", "grammar&simulation"):
            plt.close(self._grammar_render[0])
            self._grammar_render = plt.subplots()

    def step(self, action):
        mask = list(map(lambda x: bool(x), self.get_mask_possible_actions()))
        actions = np.asarray(list(self.table_rules.keys()))
        if action in actions[mask]:
            str_rule = self.table_rules[action]
            if str_rule in self.rule_vocabulary.get_list_of_applicable_nonterminal_rules(
                    self._graph_grammar):
                self._number_nonterminal_rules += 1
            else:
                self._number_terminal_rules += 1
            self._graph_grammar.apply_rule(self.rule_vocabulary.get_rule(str_rule))

        info = self._get_info()
        terminated = (info["num_applicable_rules"]["nonterminal"] +
                      info["num_applicable_rules"]["terminal"] == 0)

        if terminated:
            reward = self._sim()
        else:
            reward = 0.

        self.render(terminated)

        observation = self._get_obs()

        return observation, reward, terminated, False, info

    def get_mask_possible_actions(self):
        """Return mask of possible actions for the current observation state

        Returns:
            np.array[np.int8]: Mask array with possible actions
        """
        is_num_rules_more_max = self._number_nonterminal_rules > self.max_number_nonterminal_rules
        if not is_num_rules_more_max:
            applicable_rules = self.rule_vocabulary.get_list_of_applicable_rules(
                self._graph_grammar)
        else:
            applicable_rules = self.rule_vocabulary.get_list_of_applicable_terminal_rules(
                self._graph_grammar)
        mask_applicable_rules = map(lambda x: x in applicable_rules, self.table_rules.values())
        mask_applicable_rules = np.asarray(list(mask_applicable_rules), dtype=np.int8)
        return mask_applicable_rules

    def set_max_number_nonterminal_rules(self, max_number: int):
        """Setter max number nonterminal rules for finite optimization.


        Args:
            max_number (int): Max number of nonterminal rules
        """
        self.max_number_nonterminal_rules = max_number
