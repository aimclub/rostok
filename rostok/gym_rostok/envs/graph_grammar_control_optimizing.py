import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
from rostok.gym_rostok.envs.graph_grammar import GraphGrammarEnv
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer


class GGrammarControlOpimizingEnv(GraphGrammarEnv):

    def __init__(self,
                 rule_vocabulary: RuleVocabulary,
                 controller: ControlOptimizer,
                 render_mode=None):
        """Class gym graph grammar environment. Instead using controller, aplies control optimization.

        Args:
            - rule_vocabulary (RuleVocabulary): Instance of RuleVocabulary for manipulation graph and apply rules.
            - controller (ControlOptimizer): Instance of ControlOptimizer for optimization robot control 
            - render_mode (string, optional): Define type of rendering modifying graph on each step (`grammar`),
            or result robot simulation on each episodes (`simulation`). May set both (`grammar&simulation`).
            Defaults to None.
        """

        super().__init__(rule_vocabulary, controller, render_mode)

    def _sim(self):
        result_optimizer = self.controller.start_optimisation(self._graph_grammar)
        reward = -result_optimizer[0]
        movments_trajectory = result_optimizer[1]
        return reward, movments_trajectory

    def render(self, terminated, movment_trajectory=None):
        if self.render_mode in ("grammar", "grammar&simulation"):
            self._grammar_render[1].clear()
            nx.draw_networkx(
                self._graph_grammar,
                pos=nx.kamada_kawai_layout(self._graph_grammar, dim=2),
                node_size=800,
                labels={n: self._graph_grammar.nodes[n]["Node"].label for n in self._graph_grammar})
            plt.pause(0.1)
        if self.render_mode in ("simulation", "grammar&simulation") and terminated is True:
            # self.close()
            func_reward = self.controller.create_reward_function(self._graph_grammar)
            if movment_trajectory is None:
                raise Exception("To set movment trajectory ror simulation render")
            func_reward(movment_trajectory, True)

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
