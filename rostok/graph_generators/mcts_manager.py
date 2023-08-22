import sys
import os
import time

from rostok.graph_generators.search_algorithms.mcts import MCTS


class MCTSManager:

    def __init__(self, mcts_algorithm: MCTS, folder_name: str, verbosity=1, use_date: bool = True):

        self.mcts_algorithm = mcts_algorithm

        date = time.strftime("%Y-%m-%d_%H-%M-%S") if use_date else ""

        self.folder_name = folder_name + "_" + date
        self.path = self._prepare_path(self.folder_name)

        self.trajectories: list = []
        self.verbosity = verbosity

    def _prepare_path(self, folder_name: str):
        folders = ["results", "MCTS", folder_name]
        path = "./"
        for folder in folders:
            path = os.path.join(path, folder)
            if not os.path.exists(path):
                os.mkdir(path)

        print(f"MCTS data will be in {path}")
        return path

    def run_search(self,
                   max_iteration_mcts: int,
                   max_simulation: int,
                   iteration_checkpoint: int = 0):
        env = self.mcts_algorithm.environment
        state = env.initial_state

        trajectory = []
        is_terminal_state = env.is_terminal_state(state)[0]
        iterator = 0
        while not is_terminal_state:
            time_start = time.time()
            for iter in range(max_iteration_mcts):
                self.mcts_algorithm.search(state, num_actions=max_simulation)

                if self.verbosity > 0:
                    print(f"Search iteration: {iter}")
                    print(env.info())
                    if self.verbosity > 1:
                        print(env.get_info_state(state))
                    if self.verbosity > 2:
                        state_data = self.mcts_algorithm.get_data_state(state)
                        print(state_data)
                    print("===========")
            finish_time = time.time() - time_start

            if iteration_checkpoint != 0 and iterator % (iteration_checkpoint + 1) == 0:
                self.save_checkpoint(iter, state, finish_time)

            pi = self.mcts_algorithm.get_policy(state)
            a = max(env.actions, key=lambda x: pi[x])
            trajectory.append((state, a))
            state, __, is_terminal_state, __ = env.next_state(state, a)

            state_data = self.mcts_algorithm.get_data_state(state)
            print(f"Search time: {finish_time}")
            print(f"State: {state}, V-function: {state_data['V']}, N: {state_data['N']}\n")
            iterator += 1
        trajectory.append((state, -1))
        self.trajectories.append(trajectory)

    def save_checkpoint(self, iteration: int, state, time_search):
        env = self.mcts_algorithm.environment
        path_to_log = os.path.join(self.path, "log.txt")
        with open(path_to_log, "a", encoding="utf-8") as file:
            original_stdout = sys.stdout
            sys.stdout = file
            print()
            print(f"Search iteration: {iteration}, search_time: {time_search}, state: {state}")
            print(env.info(verbosity=2))
            print()
            print(env.get_info_state(state, verbosity=5))
            print()
            data_state = self.mcts_algorithm.get_data_state(state)
            print(data_state)
            print("===========")
            sys.stdout = original_stdout

        self.mcts_algorithm.save("checkpoint", self.path, rewrite=True, use_date=False)

    def save_information_about_search(self, hyperparameters):

        ctrl_optim = self.mcts_algorithm.environment.control_optimizer
        dict_hp = {
            item: getattr(hyperparameters, item)
            for item in dir(hyperparameters)
            if not item.startswith("__") and not item.endswith("__")
        }
        path_to_info = os.path.join(self.path, "info.txt")
        with open(path_to_info, "a", encoding="utf-8") as file:
            original_stdout = sys.stdout
            sys.stdout = file
            print()

            for key, value in dict_hp.items():
                print(key, " = ", value)
            print()
            print(str(ctrl_optim))
            sys.stdout = original_stdout
            
    def test_mcts(self, num_test):
        env = self.mcts_algorithm.environment
        for num_test in range(num_test):
            state = env.initial_state
            is_terminal_state = env.is_terminal_state(state)[0]
            while not is_terminal_state:
                pi = self.mcts_algorithm.get_policy(state)
                a = max(env.actions, key=lambda x: pi[x])
                state, reward, is_terminal_state,  = env.next_state(state, a)
