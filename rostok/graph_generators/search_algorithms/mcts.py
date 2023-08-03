

from collections import defaultdict
from rostok.graph_generators.environments.design_environment import STATESTYPE

class MCTS:
    def __init__(self, environments, root, c=1.4):
        self.root = root
        self.c = c
        self.environments = environments
        
        self.Qsa = defaultdict(float)  # total reward of each state-action pair
        self.Nsa = defaultdict(int) # total visit count for each state-action pair
        self.Ns = defaultdict(int) # total visit count for each state
        self.Ps = defaultdict(float)  # initial policy
        self.Vs = defaultdict(float) # total reward of each state
        self.Es = defaultdict(bool) # is terminal state
        
    def get_actions_probalities(self, state: STATESTYPE):
        for a in self.environments.get_available_actions(state):
            if self.Nsa[(state, a)] == 0:
                self.Ps[(state, a)] = 1.0 / len(self.environments.get_available_actions(state))
            else:
                self.Ps[(state, a)] = self.Qsa[(state, a)] / self.Nsa[(state, a)]
        return self.Ps