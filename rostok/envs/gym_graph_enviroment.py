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
    list_id_nodes = np.asarray(list(table_nodes_id[node[1]["Node"].label] for node in nx_graph.nodes.items()))

    table_id_index_nodes = dict((id[1], id[0]) for id in enumerate(nx_graph.nodes.keys()))
    nx_edges = list(nx_graph.edges)

    list_id_edge_links = np.asarray(
        list(map(lambda x: [table_id_index_nodes[n] for n in x], nx_edges)))

    list_edges = np.asarray([1 for _ in range(len(list_id_edge_links))])

    gym_graph = spaces.GraphInstance(list_id_nodes, list_edges, list_id_edge_links)

    return gym_graph


class GraphGrammarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

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

        assert render_mode is None or render_mode is self.metadata["render_modes"]

    def _get_initial_obs(self):
        observation = spaces.GraphInstance([0], None, None)
        return observation

    def _get_obs(self):
        observation = networkx_2_gym_digraph(self.graph_grammar, self.__auxiliraty_table_nodes)
        return observation

    def _sim(self):
        # FIXME: Change to interface controller
        result_optimizer = self.controller.start_optimisation(self.graph_grammar)
        reward = -result_optimizer[0]
        movments_trajectory = result_optimizer[1]

        # TODO: Render mode
        # if self.render_modes == "human":
        #     func_reward = self.optimizer.create_reward_function(self.graph)
        #     func_reward(self.movments_trajectory, True)
        return reward, movments_trajectory

    def _get_info(self):
        return {
            "num_nodes": len(list(self.graph_grammar.nodes)),
            "num_applicable_rules": {
                "nonterminal":
                    self.rule_vocabulary.get_list_of_applicable_nonterminal_rules(self.graph_grammar
                                                                                 ),
                "terminal":
                    self.rule_vocabulary.get_list_of_applicable_terminal_rules(self.graph_grammar)
            },
            #"num_aplied_rules": {"nonterminal": self.rule_vocabulary}
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.graph_grammar = GraphGrammar()

        observation = self._get_initial_obs()
        info = self._get_info()
        return observation, info

    # TODO: Render mode
    def render(self):
        if self.render_mode == "human":
            plt.figure()
            nx.draw_networkx(
                self.graph_grammar,
                pos=nx.kamada_kawai_layout(self.graph, dim=2),
                node_size=800,
                labels={n: self.graph_grammar.nodes[n]["Node"].label for n in self.graph_grammar})
            plt.show()
            sleep(5)
            plt.close()

    def step(self, action):

        str_rule = self.table_rules[action]
        self.graph_grammar.apply_rule(self.rule_vocabulary.get_rule(str_rule))

        terminated = (len(
            self.rule_vocabulary.get_list_of_applicable_nonterminal_rules(self.graph_grammar)) == 0)

        self.render()
        if terminated:
            reward, _ = self._sim()
        else:
            reward = 0.

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
