import sys
import os
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint

from rostok.graph_generators.search_algorithms.mcts import MCTS


class MCTSManager:

    def __init__(self, mcts_algorithm: MCTS, folder_name: str, verbosity=1, use_date: bool = True):
        """Class for managing MCTS algorithm.

        Args:
            mcts_algorithm (MCTS): The MCTS algorithm.
            folder_name (str): The name of the folder where the results will be saved.
            verbosity (int, optional): The level of verbosity. Defaults to 1.
            use_date (bool, optional): If True, the date will be added to the folder name. Defaults to True.
        """
        self.mcts_algorithm = mcts_algorithm

        date = "_" + time.strftime("%Y-%m-%d_%H-%M-%S") if use_date else ""

        self.folder_name = folder_name + date
        self.path = self._prepare_path(self.folder_name)

        self.trajectories: list = []
        self.verbosity = verbosity
        self.tests_mcts = []

    def _prepare_path(self, folder_name: str):
        """Create a folder for saving results.

        Args:
            folder_name (str): The name of the folder where the results will be saved.

        Returns:
            str: The path to the folder.
        """
        folders = ["results", "MCTS", folder_name]
        path = "./"
        for folder in folders:
            path = os.path.join(path, folder)
            if not os.path.exists(path):
                os.mkdir(path)
        path = os.path.abspath(path)
        print(f"MCTS data will be in {path}")
        
        return path

    def run_search(self,
                   max_iteration_mcts: int,
                   max_simulation: int,
                   iteration_checkpoint: int = 0,
                   num_test: int = 0):
        """Run the MCTS algorithm for a given number of iterations. Max simulation is the number of simulations in one iteration.
        The search stores the trajectory of the states and actions.


        Args:
            max_iteration_mcts (int): max number of iterations of the MCTS algorithm.
            max_simulation (int): max number of simulations in one iteration.
            iteration_checkpoint (int, optional): The number of iterations after which the checkpoint will be saved. Defaults to 0.
            num_test (int, optional): The number of tests after which the mean and std of the reward will be calculated. Defaults to 0.
        """
        env = self.mcts_algorithm.environment
        state = env.initial_state

        trajectory = []
        is_terminal_state = env.is_terminal_state(state)[0]
        iterator = 0

        if not self.tests_mcts:
            test_iter_offset = 0
        else:
            test_iter_offset = self.tests_mcts[-1][0] + 1

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

            pi = self.mcts_algorithm.get_policy_by_Q(state)
            a = max(env.actions, key=lambda x: pi[x])
            trajectory.append((state, a))
            state, __, is_terminal_state, __ = env.next_state(state, a)

            state_data = self.mcts_algorithm.get_data_state(state)
            print(f"Search time: {finish_time}")
            print(f"State: {state}, V-function: {state_data['V']}, N: {state_data['N']}\n")
            iterator += 1
            
            if num_test != 0:
                rwd_mean, rwd_std = self.test_mcts(num_test)
                print(f"Test MCTS - mean reward: {rwd_mean:.4f}, std reward: {rwd_std:.4f}")
                self.tests_mcts.append((iterator + test_iter_offset, rwd_mean, rwd_std))
                
        trajectory.append((state, -1))
        self.trajectories.append(trajectory)

    def save_checkpoint(self, iteration: int, state, time_search):
        """Save the checkpoint of the MCTS algorithm. The checkpoint contains the state of the MCTS algorithm and the state of the environment.
        Write the log of the search to the file.

        Args:
            iteration (int): The number of iterations of the MCTS algorithm.
            state: The state of the environment.
            time_search: The time of the search.
        """
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

    def save_information_about_search(self, hyperparameters, grasp_object: EnvironmentBodyBlueprint | list[EnvironmentBodyBlueprint]):
        """Save the information about the search to the file.
        
            Args:
                hyperparameters: The hyperparameters of the MCTS algorithm.
                grasp_object (EnvironmentBodyBlueprint): The object to grasp.
        """
        ctrl_optim = self.mcts_algorithm.environment.control_optimizer
        dict_hp = {
            item: getattr(hyperparameters, item)
            for item in dir(hyperparameters)
            if not item.startswith("__") and not item.endswith("__")
        }
        path_to_info = os.path.join(self.path, "info.txt")
        with open(path_to_info, "w+", encoding="utf-8") as file:
            original_stdout = sys.stdout
            sys.stdout = file
            print()

            for key, value in dict_hp.items():
                print(key, " = ", value)
            print()
            print(str(ctrl_optim))
            print()
            print(str(grasp_object))
            sys.stdout = original_stdout
            
    def test_mcts(self, num_test):
        """Test the MCTS algorithm. The test is to run the MCTS algorithm for a given number of iterations and calculate the mean and std of the reward.

        Args:
            num_test (int): The number of tests.

        Returns:
            tuple[float, float]: The mean and std of the reward.
        """
        rewards = []
        env = self.mcts_algorithm.environment
        for num_test in range(num_test):
            state = env.initial_state
            is_terminal_state = env.is_terminal_state(state)[0]
            while not is_terminal_state:
                pi = self.mcts_algorithm.get_policy_by_Q(state)
                a = max(env.actions, key=lambda x: pi[x])
                state, reward, is_terminal_state, __ = env.next_state(state, a)
            rewards.append(reward)
        return np.mean(rewards), np.std(rewards)
    
    def plot_test_mcts(self, save=False, name="test_mcts.svg"):
        """Plot the mean and std of the reward for the test of the MCTS algorithm.

        Args:
            save (bool, optional): If True, the plot will be saved. Defaults to False.
            name (str, optional): The name of file. Defaults to "test_mcts.svg".
        """
        if not self.tests_mcts:
            print("No tests")
            return
        x, y_mean, y_std = zip(*self.tests_mcts)
        plt.figure(figsize=(10, 5))
        plt.errorbar(x, y_mean, yerr=y_std, fmt='o', color='black',
                     ecolor='lightgray', elinewidth=3, capsize=0)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("MCTS test")
        if save:
            plt.savefig(os.path.join(self.path, name))
        else:
            plt.show()
            time.sleep(20)
        plt.close()
    
    def plot_v_trajectory(self, trajectory, save=False, name="v_trajectory.svg"):
        """Plot the V-function and Q-function for the trajectory.

        Args:
            trajectory (list, np.ndarray): The trajectory of the states and actions.
            save (bool, optional): If True, the plot will be saved. Defaults to False.
            name (str, optional): The name of file. Defaults to "v_trajectory.svg".
        """
        if not self.trajectories:
            print("No trajectories")
            return
        plt.figure(figsize=(10, 5))
        v_traj = []
        q_traj = []
        for state, a in trajectory:
            state_data = self.mcts_algorithm.get_data_state(state)
            v_traj.append(state_data["V"])
            if a != -1:
                rule = self.mcts_algorithm.environment.action2rule[a]
                q_traj.append(state_data["Qa"][rule])
        plt.plot(q_traj, label=f"Q-function")
        plt.plot(v_traj, label=f"V-function")
        plt.xlabel("Iteration")
        plt.ylabel("V-function")
        plt.title("Trajectory")
        plt.legend()
        if save:
            plt.savefig(os.path.join(self.path, name))
        else:
            plt.show()
            time.sleep(20)
        plt.close()
        
    def save_results(self, save_plot=True):
        """Save the trajectories of the states and actions to the file.
        """
        date = time.strftime("%Y-%m-%d_%H-%M-%S")
        path_result = os.path.join(self.path, "result_" + date)
        os.mkdir(path_result)
        print(f"Results are saved in {path_result}")
        
        path_to_trajectories = os.path.join(path_result, "trajectories.p")
        with open(path_to_trajectories, "wb") as file:
            pickle.dump(self.trajectories, file)

        
        if save_plot:
            path_plot = os.path.join(path_result, "plot")
            os.mkdir(path_plot)
            
            self.plot_test_mcts(save=True, name=os.path.join(path_plot, "test_mcts.svg"))
            for num, traj in enumerate(self.trajectories):
                self.plot_v_trajectory(traj, save=True, name=os.path.join(path_plot, f"v_trajectory_{num}.svg"))
        
        self.mcts_algorithm.save("final", path_result, rewrite=True, use_date=False)