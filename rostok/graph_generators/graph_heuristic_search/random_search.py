import datetime
import os
import time
import numpy as np

from rostok.graph_generators.graph_heuristic_search.design_environment import DesignEnvironment

class RandomSearch:
    def __init__(self, max_nonterminal_actions):
        self.best_reward = 0
        self.best_state = 0
        
        self.max_nonterminal_actions  = max_nonterminal_actions
        self.reward_history = []
        self.best_reward_history = []
        self.time_history = []
        
    def search(self, design_environment: DesignEnvironment, max_iteration):
        mask_terminal = design_environment.get_terminal_actions()
        mask_nonterminal = design_environment.get_nonterminal_actions()
        
        for iter in range(max_iteration):
            t_start = time.time()
            state = design_environment.initial_state
            nonterminal_actions = 0
            while not design_environment.is_terminal_state(state):
                mask = design_environment.get_available_actions(state)
                if nonterminal_actions >= self.max_nonterminal_actions:
                    mask *= mask_terminal
                avb_actions = design_environment.actions[mask ==1]
                a = np.random.choice(avb_actions)
                if mask_nonterminal[a] == 1:
                    nonterminal_actions += 1
                state = design_environment.next_state(state, a)

            reward = design_environment.terminal_states[state][0]
            if reward >= self.best_reward:
                self.best_reward = reward
                self.best_state = state

            t_finish = time.time() - t_start
            self.reward_history.append(reward)
            self.best_reward_history.append(self.best_reward)
            self.time_history.append(t_finish)

            print(f"Iter: {iter}, Iteration time {t_finish}, Current reward: {reward}, Best reward: {self.best_reward}")
            print(f"Num terminal states: {len(design_environment.terminal_states)}, Num seen designs {len(design_environment.state2graph)}")
            print(f"Amount nonterminal actions: {nonterminal_actions}")
            print("===========")


    def save_history(self, path='./rostok/graph_generators/graph_heuristic_search/history_random_search' ):
        current_date = datetime.now()
        file = f"history_random_search_{current_date.hour}h{current_date.minute}m_date_{current_date.day}d{current_date.month}m{current_date.year}y"
        full_path = os.path.join(path,file)
        np_reward = np.array(self.reward_history)
        np_best_reward = np.array(self.best_reward_history)
        np_time = np.array(self.time_history)
        with open(full_path, "wb") as f:
            np.save(f, np_best_reward)
            np.save(f, np_time)
            np.save(f, np_reward)