from collections import defaultdict

import numpy as np

from rostok.graph_generators.environments.design_environment import STATESTYPE, DesignEnvironment

EPS = 1e-8


class MCTS:

    def __init__(self, environment: DesignEnvironment, max_nonterminal_action: int, c=1.4):

        self.c = c
        self.environment = environment
        self.max_nonterminal_action = max_nonterminal_action

        self.Qsa = defaultdict(float)  # total reward of each state-action pair
        self.Nsa = defaultdict(int)  # total visit count for each state-action pair
        self.Ns = defaultdict(int)  # total visit count for each state
        self.Vs = defaultdict(float)  # total reward of each state

    def get_actions_probalities(self, state: STATESTYPE):
        pi = np.ones_like(self.environment.actions)
        for a in self.environment.get_available_actions(state):
            if self.Nsa[(state, a)] == 0:
                pi[a] = 1.0 / len(self.environment.get_available_actions(state))
            else:
                pi[a] = self.Nsa[(state, a)]
        return pi

    def search(self, state: STATESTYPE, amount_nonterminal_actions):

        if state not in self.Vs:
            is_terminal_s, __ = self.environment.is_terminal_state(state)
            self.Vs[state] = self.environment.terminal_states[state][0] if is_terminal_s else 0.0

        if self.Vs[state] != 0.0:
            return self.Vs[state]

        if state not in self.Ns:
            hat_V = self.default_policy(state, amount_nonterminal_actions)

            return hat_V

        best_action = self.tree_policy(state)
        
        new_state = self.environment.next_state(state, best_action)[0]

        v = self.search(new_state, amount_nonterminal_actions-1)

        self.update_Q_function(state, best_action, v)
        return v

    def update_Q_function(self, state, action, reward):

        if (state, action) in self.Qsa:
            self.Qsa[(
                state,
                action)] += (reward - self.Qsa[(state, action)]) / (self.Nsa[(state, action)] + 1)
            self.Nsa[(state, action)] += 1
        else:
            self.Qsa[(state, action)] = reward
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1

    def default_policy(self, state, amount_nonterminal_actions):
        rewards = []
        mask_terminal = self.environment.get_terminal_actions()
        mask_nonterminal = self.environment.get_nonterminal_actions()
        
        mask = self.environment.get_available_actions(state)
        available_actions = self.environment.actions[mask == 1]
        
        for a in available_actions:
            nonterminal_actions = amount_nonterminal_actions
            if mask_nonterminal[a] == 1:
                nonterminal_actions += 1

            s, reward, is_terminal_state, is_known = self.environment.next_state(state, a)
            while not is_terminal_state:
                mask = self.environment.get_available_actions(s)
                if nonterminal_actions >= self.max_nonterminal_action:
                    mask *= mask_terminal
                available_actions = self.environment.actions[mask == 1]
                rnd_action = np.random.choice(available_actions)

                if mask_nonterminal[a] == 1:
                    nonterminal_actions += 1

                s, reward, is_terminal_state, is_known = self.environment.next_state(s, rnd_action)

            rewards.append(reward)
            self.update_Q_function(state, a, rewards[-1])

        return np.mean(rewards)
    
    def tree_policy(self, state):
        curr_best = -float('inf')
        best_action = -1
        
        mask = self.environment.get_available_actions(state)
        available_actions = self.environment.actions[mask == 1]
        
        for a in available_actions:
            if (state, a) in self.Qsa:
                u = self.Qsa[(state,
                              a)] + self.c * np.sqrt(np.log(self.Ns[state]) / self.Nsa[(state, a)])
            else:
                u = self.c * np.sqrt(np.log(self.Ns[state]+EPS))

            if u > curr_best:
                curr_best = u
                best_action = a
        return best_action