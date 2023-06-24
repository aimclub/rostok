from copy import deepcopy
from datetime import datetime
import time
import os
from typing import Union
import numpy as np
import numpy.random as random

from rostok.graph_generators.graph_heuristic_search.torch_adapter import TorchAdapter
from design_environment import DesignEnvironment
from rostok.graph_grammar.node import GraphGrammar


class GraphHeuristicSearch:

    def __init__(self, nnet_adapter: TorchAdapter, nnet, args):
        self.nnet = nnet
        self.args = args
        self.nnet_adapter = nnet_adapter

        self.hat_v = {}

        self.dataset = []
        self.candidate_design_P = {}

        self.best_design = None
        self.best_reward = 0

        self.max_nonterminal_actions = args["max_nonterminal_actions"]

        self.history_best_reward = []
        self.history_reward = []
        self.history_loss = []
        self.prediction_error = []

    def eps_greedy_choose_action(self, state: int, mask_actions, eps,
                                 design_environment: DesignEnvironment):

        if random.random_sample() <= eps:
            action = random.choice(mask_actions)
        else:
            next_states = design_environment.possible_next_state(state, mask_actions)
            next_graphs = [design_environment.state2graph[s] for s in next_states]
            data_next_state = self.nnet_adapter.list_gg2data_loader(
                next_graphs, self.args["batch_size"]).to(self.args["device"])
            estiamation_v = self.nnet.predict(data_next_state)
            estiamation_v = estiamation_v.cpu().detach().numpy()
            action = design_environment.actions[np.argmax(estiamation_v)]
        return action

    def eps_greedy_choose_designs(self, states: list[int], eps,
                                  design_environment: DesignEnvironment):

        if random.random_sample() <= eps:
            design = random.choice(states)
        else:
            next_graphs = [design_environment.state2graph[s] for s in states]
            data_states = self.nnet_adapter.list_gg2data_loader(
                next_graphs, self.args["batch_size"]).to(self.args["device"])
            estiamation_V = self.nnet.predict(data_states)
            design = states[np.argmax(estiamation_V.cpu().detach().numpy())]

        return design

    def update_hat_v(self, designs: Union[list[int], int], value: float):

        if isinstance(designs, list):
            set_of_known_design = set(self.hat_v.keys())
            new_designs = set(designs) - set_of_known_design
            old_designs = set(designs) - new_designs
            for design in new_designs:
                self.hat_v[design] = value
            for design in old_designs:
                old_value = self.hat_v[design]
                self.hat_v[design] = np.max([old_value, value])
        elif isinstance(designs, int):
            if designs in self.hat_v.keys():
                old_value = self.hat_v[designs]
                self.hat_v[designs] = np.max([old_value, value])
            else:
                self.hat_v[designs] = value

    def design_phase(self, num_designs, eps, design_environment: DesignEnvironment):
        designs = {}
        mask_terminal = design_environment.get_terminal_actions()
        mask_nonterminal = design_environment.get_nonterminal_actions()

        for __ in range(num_designs):
            design = design_environment.initial_state
            nonterminal_actions = 0
            parents = []
            while not design_environment.is_terminal_state(design):
                parents.append(design)

                mask = design_environment.get_available_actions(design)
                if nonterminal_actions >= self.max_nonterminal_actions:
                    mask *= mask_terminal

                a = self.eps_greedy_choose_action(design, mask, eps, design_environment)
                if mask_nonterminal[a] == 1:
                    nonterminal_actions += 1
                design = design_environment.next_state(design, a)

            designs[design] = parents

        return designs

    def evaluation_phase(self, design_candidates, eps, design_environment: DesignEnvironment):

        design_for_estimation = self.eps_greedy_choose_designs(design_candidates.keys(), eps,
                                                               design_environment)
        reward = design_environment.get_reward(design_for_estimation)

        if reward >= self.best_reward:
            self.best_reward = reward
            self.best_design = design_for_estimation

        self.history_best_reward.append(self.best_reward)
        self.history_reward.append(reward)

        return design_candidates[design_for_estimation] + design_for_estimation, reward

    def learning_phase(self, size_batch, opt_iter, design_environment: DesignEnvironment):
        dataset = np.array(list(self.hat_v.items()))
        dataset = np.random.choice(dataset, size=size_batch)
        states = dataset[:, 0]
        y = dataset[:, 1]
        graphs = list([design_environment.state2graph[s] for s in states])
        data_graph = self.nnet_adapter.list_gg2data_loader(graphs, size_batch, y)

        self.history_loss.append(self.nnet.history_loss[-1])
        for __ in opt_iter:
            self.nnet.update(data_graph)

    def search(self, num_iteration, design_environment: DesignEnvironment):

        for epochs in num_iteration:
            t_start = time.time()

            eps = self.args["end_eps"] + (self.args["start_eps"] - self.args["end_eps"]) * np.exp(
                -1 * epochs / num_iteration / self.args["eps_decay"])

            designs = self.design_phase(self.args["num_designs"], eps, design_environment)
            est_designs, value = self.evaluation_phase(designs, self.args["eps_design"],
                                                       design_environment)
            self.update_hat_v(est_designs, value)
            self.learning_phase(self.args["minibatch"], self.args["opt_iter"], design_environment)
            t_finish = time.time() - t_start

            print(f"Epochs: {epochs}, time epoch: {t_finish}")
            print(f"Current reward: {value}, Best reward: {self.best_reward}")
            print(
                f"Size hat-V: {len(self.hat_v)}, Num seen designs {len(design_environment.state2graph)}"
            )
            print(f"Loss: {self.nnet.history_loss[-1]}, eps: {eps}")
            print("===============")

    def save_history(self,
                     path='./rostok/graph_generators/graph_heuristic_search/history_random_search'):
        current_date = datetime.now()
        file = f"history_random_search_{current_date.hour}h{current_date.minute}m_date_{current_date.day}d{current_date.month}m{current_date.year}y"
        full_path = os.path.join(path, file)

        self.history_best_reward = []
        self.history_reward = []
        self.history_loss = []
        self.prediction_error = []

        np_reward = np.array(self.reward_history)
        np_best_reward = np.array(self.history_best_reward)
        np_time = np.array(self.time_history)
        with open(full_path, "wb") as f:
            np.save(f, np_best_reward)
            np.save(f, np_time)
            np.save(f, np_reward)