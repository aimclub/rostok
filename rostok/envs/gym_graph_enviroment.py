from copy import deepcopy
from time import sleep
import gymnasium as gym
import gymnasium.spaces as spaces
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from rostok.graph_grammar.node import GraphGrammar

from rostok.graph_grammar.rule_vocabulary import RuleVocabulary


def networkx_2_gym_digraph(nx_graph: GraphGrammar, table_nodes_id: dict[str, int]):
    list_id_nodes = np.asarray(
        list(table_nodes_id[node[1]["Node"].label] for node in nx_graph.nodes.items()))

    table_id_index_nodes = dict((id[1], id[0]) for id in enumerate(nx_graph.nodes.keys()))
    nx_edges = list(nx_graph.edges)

    list_id_edge_links = np.asarray(
        list(map(lambda x: [table_id_index_nodes[n] for n in x], nx_edges)))

    list_edges = np.asarray([1 for _ in range(len(list_id_edge_links))])

    gym_graph = spaces.GraphInstance(list_id_nodes, list_edges, list_id_edge_links)

    return gym_graph


class GraphGrammarEnv(gym.Env):
    metadata = {"render_modes": ["grammar", "simulation", "grammar&simulation"], "render_fps": 4}

    def __init__(self, rule_vocabulary: RuleVocabulary, controller, render_mode=None):

        self.rule_vocabulary = deepcopy(rule_vocabulary)
        self.graph_grammar = GraphGrammar()

        dict_rules = self.rule_vocabulary.rule_dict
        dict_nodes = self.rule_vocabulary.node_vocab.node_dict

        amount_nodes = len(dict_nodes.keys())
        node_space = spaces.Discrete(amount_nodes)
        self.table_nodes = dict(map(lambda x, y: (x, y), range(amount_nodes), dict_nodes.keys()))
        self.__auxiliraty_table_nodes = dict(
            map(lambda x, y: (x, y), dict_nodes.keys(), range(amount_nodes)))

        edges = spaces.Discrete(1, start=1)
        #FIXME: Change observation space

        self.observation_space = spaces.Graph(node_space, edges)

        amount_rules = len(dict_rules.keys())
        self.action_space = spaces.Discrete(amount_rules)
        self.table_rules = dict(map(lambda x, y: (x, y), range(amount_rules), dict_rules.keys()))

        self.controller = controller

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if render_mode in ("grammar", "grammar&simulation"):
            self._grammar_render = plt.subplots()

    def _get_initial_obs(self):
        observation = spaces.GraphInstance([0], None, None)
        return observation

    def _get_obs(self):
        observation = networkx_2_gym_digraph(self.graph_grammar, self.__auxiliraty_table_nodes)
        return observation

    def _sim(self):
        #FIXME: Change to interface controller
        result_optimizer = self.controller.start_optimisation(self.graph_grammar)
        reward = -result_optimizer[0]
        movments_trajectory = result_optimizer[1]
        return reward, movments_trajectory

    def _get_info(self):
        return {
            "num_nodes": len(list(self.graph_grammar.nodes)),
            "num_applicable_rules": {
                "nonterminal":
                    len(
                        self.rule_vocabulary.get_list_of_applicable_nonterminal_rules(
                            self.graph_grammar)),
                "terminal":
                    len(
                        self.rule_vocabulary.get_list_of_applicable_terminal_rules(
                            self.graph_grammar))
            },
            #"num_aplied_rules": {"nonterminal": self.rule_vocabulary}
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.graph_grammar = GraphGrammar()

        observation = self._get_initial_obs()
        info = self._get_info()
        return observation, info

    def render(self, terminated, movment_trajectory=None):
        if self.render_mode in ("grammar", "grammar&simulation"):
            self._grammar_render[0].show()
            self._grammar_render[1].clear()
            nx.draw_networkx(
                self.graph_grammar,
                pos=nx.kamada_kawai_layout(self.graph_grammar, dim=2),
                node_size=800,
                labels={n: self.graph_grammar.nodes[n]["Node"].label for n in self.graph_grammar})
            plt.pause(0.5)
        if self.render_mode in ("simulation", "grammar&simulation") and terminated is True:
            func_reward = self.controller.create_reward_function(self.graph_grammar)
            if movment_trajectory is None:
                raise Exception("To set movment trajectory ror simulation render")
            func_reward(movment_trajectory, True)

    def step(self, action):

        applicable_rules = self.rule_vocabulary.get_list_of_applicable_rules(self.graph_grammar)
        mask_applicable_rules = map(lambda x: x in applicable_rules, self.table_rules.values())
        mask_applicable_rules = np.asarray(list(mask_applicable_rules))
        actions = np.asarray(list(self.table_rules.keys()))
        if action in actions[mask_applicable_rules]:
            str_rule = self.table_rules[action]
            self.graph_grammar.apply_rule(self.rule_vocabulary.get_rule(str_rule))

        trajectory = None

        info = self._get_info()
        terminated = (info["num_applicable_rules"]["nonterminal"] +
                      info["num_applicable_rules"]["terminal"] == 0)

        if terminated:
            reward, trajectory = self._sim()
        else:
            reward = 0.

        self.render(terminated, trajectory)

        observation = self._get_obs()

        return observation, reward, terminated, False, info
