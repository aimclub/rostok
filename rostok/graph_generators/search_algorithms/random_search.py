from datetime import datetime
import os
import time
import numpy as np

from rostok.graph_generators.environments.design_environment import DesignEnvironment


class RandomSearch:

    def __init__(self, max_nonterminal_actions, save_history_iteration = 0, verbosity=0):
        self.best_reward = 0
        self.best_state = 0

        self.max_nonterminal_actions = max_nonterminal_actions
        self.reward_history = []
        self.best_reward_history = []
        self.time_history = []
        self.verbosity = verbosity
        self.save_history_iter = save_history_iteration

    def search(self, design_environment: DesignEnvironment, max_iteration):

        for iter in range(max_iteration):
            t_start = time.time()
            state = design_environment.initial_state
            is_terminal_state, is_known_state = design_environment.is_terminal_state(state)
            while not is_terminal_state:
                mask = design_environment.get_available_actions(state)
                avb_actions = design_environment.actions[mask == 1]
                # print(f"Available actions: {avb_actions}")
                a = np.random.choice(avb_actions)
                state, reward, is_terminal_state, is_known_state = design_environment.next_state(state, a)
                # print(f"State: {state}, Reward: {reward}, is_terminal_state: {is_terminal_state}, is_known_state: {is_known_state}")
                

            if reward >= self.best_reward:
                self.best_reward = reward
                self.best_state = state

            t_finish = time.time() - t_start
            self.reward_history.append(reward)
            self.best_reward_history.append(self.best_reward)
            self.time_history.append(t_finish)
            
            if self.save_history_iter != 0 and (iter+1) % self.save_history_iter == 0:
                self.save_history()
            if self.verbosity > 0:
                print(
                    f"Iter: {iter}, Iteration time {t_finish}, Current reward: {reward}, Best reward: {self.best_reward}"
                )
                print(design_environment.info())
                if self.verbosity > 1:
                    print(f"Best state: {self.best_state}")
                    print(design_environment.get_info_state(state))
                print("===========")

    def save_history(self,
                     prefix="",
                     path='./results/random_search'):
        current_date = datetime.now()
        file = f"{prefix}_{current_date.hour}h{current_date.minute}m_{current_date.second}s_date_{current_date.day}d{current_date.month}m{current_date.year}y"
        full_path = os.path.join(path, file)
        np_reward = np.array(self.reward_history)
        np_best_reward = np.array(self.best_reward_history)
        np_time = np.array(self.time_history)
        with open(full_path, "wb") as f:
            np.save(f, np_best_reward)
            np.save(f, np_time)
            np.save(f, np_reward)