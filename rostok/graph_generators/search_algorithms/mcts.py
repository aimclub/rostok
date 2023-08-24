from collections import defaultdict
import os
from datetime import datetime
import pickle

import numpy as np

from rostok.graph_generators.mcts_helper import MCTSHelper
from rostok.graph_generators.environments.design_environment import STATESTYPE, DesignEnvironment

EPS = 1e-8


class MCTS:

    def __init__(self,
                 environment: DesignEnvironment,
                 c=1.4):
        """Class for Monte Carlo Tree Search algorithm. 

        Args:
            environment (DesignEnvironment): Environment for MCTS.
            c (float, optional): Exploration coefficient. Defaults to 1.4.
        """
        self.c = c
        self.environment = environment

        self.Qsa = defaultdict(float)  # total reward of each state-action pair
        self.Nsa = defaultdict(int)  # total visit count for each state-action pair
        self.Ns = defaultdict(int)  # total visit count for each state
        self.Vs = defaultdict(float)  # total reward of each state

    def get_policy(self, state: STATESTYPE):
        """Get policy for state. Policy is a probability distribution over actions.
        Probability of action is proportional to number of visits of this action.
        
            Args:
                state (STATESTYPE): State for which we want to get policy.
                
            Returns:
                pi (np.ndarray): Policy for state.
        """
        pi = np.zeros_like(self.environment.actions, dtype=np.float32)
        mask_actions = self.environment.get_available_actions(state)
        for a in self.environment.actions[mask_actions == 1]:
            pi[a] = 0.0 if self.Nsa[(state, a)] == 0 else self.Nsa[(state, a)]
        
        if np.sum(pi) == 0.0:
            pi = np.ones_like(self.environment.actions, dtype=np.float32)
            pi *= mask_actions
        pi /= np.sum(pi)
        return pi

    def search(self, state: STATESTYPE, num_actions = 0):
        """Search for best action for state. The method use recursive tree search.
        If state is not in tree, then we use default policy for this state.
        If state is in tree, then we use tree policy for this state.
        If state is terminal, then we return reward of this state.
        
            Args:
                state (STATESTYPE): State for which we want explore tree of actions.
                
            Returns:
                float: Value reward of state.
        """
        if state not in self.Vs:
            is_terminal_s, __ = self.environment.is_terminal_state(state)
            self.Vs[state] = self.environment.terminal_states[state][0] if is_terminal_s else 0.0

        if self.Vs[state] != 0.0:
            return self.Vs[state]

        if state not in self.Ns:
            hat_V = self.default_policy(state, num_actions)

            return hat_V

        best_action = self.tree_policy(state)

        new_state = self.environment.next_state(state, best_action)[0]

        v = self.search(new_state)

        self.update_Q_function(state, best_action, v)
        return v

    def update_Q_function(self, state, action, reward):
        """Update Q function for pair (state, action).

        Args:
            state: State for which we want update Q function.
            action: Action for which we want update Q function.
            reward: Reward for pair (state, action) based on Monte Carlo estimation.
        """
        if (state, action) in self.Qsa:
            self.Qsa[(
                state,
                action)] += (reward - self.Qsa[(state, action)]) / (self.Nsa[(state, action)])
            self.Nsa[(state, action)] += 1
        else:
            self.Qsa[(state, action)] = reward
            self.Nsa[(state, action)] = 1

        if state in self.Ns:
            self.Ns[state] += 1
        else:
            self.Ns[state] = 1

    def default_policy(self, state, num_actions = 0):
        """Default policy for unkown states. We use random actions until we reach terminal state.
        If num_actions = 0, then we explore all actions. Otherwise we explore random num_actions actions.

        Args:
            state: Root state for which we want to get default policy to terminal state.
            num_actions (int, optional): Number of actions which be explored. Defaults to 0.

        Returns:
            float: Return mean reward on actions.
        """
        rewards = []

        mask = self.environment.get_available_actions(state)
        available_actions = self.environment.actions[mask == 1]
        if num_actions != 0:
            available_actions = np.random.choice(available_actions, num_actions)

        for a in available_actions:

            s, reward, is_terminal_state, __ = self.environment.next_state(state, a)
            while not is_terminal_state:
                mask = self.environment.get_available_actions(s)
                available_actions = self.environment.actions[mask == 1]
                rnd_action = np.random.choice(available_actions)

                s, reward, is_terminal_state, __ = self.environment.next_state(s, rnd_action)

            rewards.append(reward)
            self.update_Q_function(state, a, reward)

        if len(rewards) == 0:
            return self.environment.get_reward(state)[0]

        return np.mean(rewards)

    def tree_policy(self, state):
        """Tree policy for known states. We use UCT formula for choosing best action.
        
            Args:
                state: State for which we want to get best action.
                
        Returns:
            best_action: Best action for state.
        """
        mask = self.environment.get_available_actions(state)
        available_actions = self.environment.actions[mask == 1]
        uct_score = self.uct_score(state)
        best_action = available_actions[np.argmax(uct_score)]

        return best_action
    
    def uct_score(self, state):
        """UCT formula for choosing best action.

        Args:
            state: State for which we want to get UCT score.

        Returns:
            float: uct score for each action.
        """
        mask = self.environment.get_available_actions(state)
        available_actions = self.environment.actions[mask == 1]
        
        state_action = [(state, a) for a in available_actions]
        Q = np.array([self.Qsa.get(sa, 0) for sa in state_action])
        N = np.array([self.Nsa.get(sa, 0) for sa in state_action])
        
        uct_scores = Q + self.c * np.sqrt(np.abs(np.log(self.Ns[state]) / (N + EPS)))
        
        return uct_scores
    
    def save(self, prefix, path = "./LearnedMCTS/", rewrite=False, use_date = True):
        """Save MCTS data in path. If path does not exist, then create it.
        
            Args:
                prefix (str): Prefix for folder name.
                path (str, optional): Path to folder where we want to save MCTS data. Defaults to "./LearnedMCTS/".
                rewrite (bool, optional): If True, then rewrite data in path. Defaults to False.
                use_date (bool, optional): If True, then add date to folder name. Defaults to True.
                
            Returns:
                str: Path to folder where we save MCTS data.
        """
        os.path.split(path)
        if not os.path.exists(path):
            print(f"Path {path} does not exist. Creating...")
            os.mkdir(path)

        if use_date:
            current_date = datetime.now()
            folder = f"{prefix}__{current_date.hour}h{current_date.minute}m_{current_date.second}s_date_{current_date.day}d{current_date.month}m{current_date.year}y"
        else:
            folder = prefix
        os_path = os.path.join(path, folder)
        if not os.path.exists(os_path):
            print(f"Create dictionary {os_path} and save MCTS")
            os.mkdir(os_path)
        elif rewrite:
            print(f"Rewite mcts data in {os_path}")
        else:
            postfix_folder = 1
            os_path = os_path + f"_{postfix_folder}"
            while os.path.exists(os_path):
                os_path = os_path.replace(f"_{postfix_folder-1}", f"_{postfix_folder}")
                postfix_folder += 1
            print(f"Create dictionary {os_path} and save MCTS")
        
        self.environment.save_environment(prefix="MCTS_env", path=os_path, use_date=False)
        
        file_names = [
            "Qsa.p", "Nsa.p", "Ns.p", "Vs.p"
        ]
        variables = [
            self.Qsa, self.Nsa, self.Ns, self.Vs
        ]
        for file, var in zip(file_names, variables):
            with open(os.path.join(os_path, file), "wb") as f:
                pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return os_path
                
    def load(self, path):
        """Load MCTS data from path.

        Args:
            path (str): Path to folder where we want to load MCTS data.
        """
        self.environment.load_environment(os.path.join(path, "MCTS_env"))
        
        file_names = [
            "Qsa.p", "Nsa.p", "Ns.p", "Vs.p"
        ]
        variables = [
            self.Qsa, self.Nsa, self.Ns, self.Vs
        ]

        for file, var in zip(file_names, variables):
            with open(os.path.join(path, file), "rb") as f:
                var.update(pickle.load(f))
    
    def get_data_state(self, state: STATESTYPE):
        """Get data for state. Data is a dictionary with keys:
        "Qa" - Q function for each action,
        "pi" - policy for state,
        "V" - value of state,
        "N" - number of visits of state,
        "Na" - number of visits of each action.
        
            Args:
                state (STATESTYPE): State for which we want to get data.
                
            Returns:
                dict: Dictionary with data for state.
        """
        mask = self.environment.get_available_actions(state)
        possible_actions = self.environment.actions[mask == 1]
        Q = {self.environment.action2rule[a]: self.Qsa.get((state, a), 0) for a in possible_actions}
        pi = self.get_policy(state)
        V = sum([self.Qsa.get((state, a), 0) * pi[a] for a in possible_actions])
        N = self.Ns[state]
        Na = {self.environment.action2rule[a]: self.Nsa.get((state, a), 0) for a in possible_actions}
        return {"Qa": Q, "pi": pi, "V": V, "N": N, "Na": Na}
    

