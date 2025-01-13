import time
import concurrent
import numpy as np
import pandas as pd

from auto_robot_design.optimization.rewards.reward_base import NotReacablePoints, RewardManager
from auto_robot_design.user_interface.check_in_ellips import (
    Ellipse,
)
from auto_robot_design.motion_planning.dataset_generator import (
    Dataset,
    set_up_reward_manager,
)
from auto_robot_design.description.builder import (
    jps_graph2pinocchio_robot_3d_constraints,
)


def calc_criteria(id_design, joint_poses, graph_manager, builder, reward_manager):
    """
    Calculate the criteria for a given design based on joint poses and reward management.
    Args:
        id_design (int): Identifier for the design.
        joint_poses (list): List of joint poses.
        graph_manager (GraphManager): Instance of GraphManager to handle graph operations.
        builder (Builder): Instance of Builder to construct robots.
        reward_manager (RewardManager): Instance of RewardManager to calculate rewards.
    Returns:
        tuple: A tuple containing the design identifier and partial rewards.
    """
    graph = graph_manager.get_graph(joint_poses)
    fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(graph, builder)
    reward_manager.precalculated_trajectories = None
    try:
        _, partial_rewards, _ = reward_manager.calculate_total(
            fixed_robot, free_robot, builder.actuator["default"]
        )
    except NotReacablePoints as e:
        partial_rewards = [0]

    return id_design, partial_rewards


def parallel_calculation_rew_manager(indexes, dataset, reward_manager):
    """
    Perform parallel calculations on a subset of a dataset using a reward manager.
    This function utilizes a process pool executor to parallelize the computation
    of criteria for a subset of the dataset. The results are then aggregated into
    a new DataFrame with updated reward values.
    Args:
        indexes (list): List of indexes to select the subset of the dataset.
        dataset (object): The dataset object containing the data and associated parameters.
        reward_manager (object): The reward manager object used for calculating rewards.
    Returns:
        pd.DataFrame: A new DataFrame containing the subset of the dataset with updated reward values.
    """
    rwd_mgrs = [reward_manager] * len(indexes)
    sub_df = dataset.df.loc[indexes]
    designs = sub_df.values[:, : dataset.params_size].round(4)
    grph_mngrs = [dataset.graph_manager] * len(indexes)
    bldrs = [dataset.builder] * len(indexes)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                calc_criteria, list(indexes), designs, grph_mngrs, bldrs, rwd_mgrs
            )
        )
    new_df = pd.DataFrame(columns=dataset.df.columns)
    for k, res in results:
        new_df.loc[k] = sub_df.loc[k]
        new_df.at[k, "reward"] = np.sum(res)
    new_df = new_df.dropna()
    return new_df


class ManyDatasetAPI:

    def __init__(self, path_to_dirs):
        """
        Initializes the DatasetGenerator with a list of directories.
        Args:
            path_to_dirs (list of str): A list of directory paths where datasets are located.
        Attributes:
            datasets (list of Dataset): A list of Dataset objects created from the provided directory paths.
        """
        self.paths = path_to_dirs
        self.datasets = [] + [Dataset(path) for path in path_to_dirs]

    def get_indexes_cover_ellipse(self, ellipse: Ellipse):
        """
        Get the indexes of all designs that cover the given ellipse.
        Args:
            ellipse (Ellipse): The ellipse object for which to find covering design indexes.
        Returns:
            list: A list of indexes from all datasets that cover the given ellipse.
        """

        list_indexes_2d = [
            dataset.get_all_design_indexes_cover_ellipse(ellipse)
            for dataset in self.datasets
        ]

        return self._index_2d_to_1d(list_indexes_2d)

    def _indexes_1d_to_2d(self, list_indexes_1d):
        list_indexes_2d = [[] for __ in range(len(self.datasets))]
        for index in list_indexes_1d:
            list_indexes_2d[index[0]].append(index[1])
        list_indexes_2d = tuple(
            [np.array(index_list) for index_list in list_indexes_2d]
        )
        return list_indexes_2d

    def _index_2d_to_1d(self, list_indexes_2d):
        list_indexes_1d = []
        for id_design, indexes in enumerate(list_indexes_2d):
            list_indexes_1d += [(id_design, index) for index in indexes]
        np.random.shuffle(list_indexes_1d)
        return list_indexes_1d

    def sorted_indexes_by_reward(self, indexes, num_samples, reward_manager):
        """
        Sorts and returns indexes based on rewards for each dataset.
        Args:
            indexes (list of np.ndarray): A list of numpy arrays where each array contains indexes for corresponding datasets.
            num_samples (int): The number of samples to randomly choose from each dataset.
            reward_manager (RewardManager): An instance of RewardManager to calculate rewards.
        Returns:
            list of pd.Index: A list of pandas Index objects, each containing sorted indexes based on rewards for the corresponding dataset.
        """

        if len(indexes) == 0:
            return []

        indexes = self._indexes_1d_to_2d(indexes)

        samples = []
        for k, dataset in enumerate(self.datasets):

            if len(indexes[k]) > 0:
                sample_indexes = np.random.choice(indexes[k].flatten(), num_samples)
                df = parallel_calculation_rew_manager(
                    sample_indexes, dataset, reward_manager
                )

                df.sort_values(["reward"], ascending=False, inplace=True)
                df = df[df["reward"] > 0]
                samples += [
                    (k, index, reward)
                    for index, reward in zip(df.index, df["reward"].values)
                ]
        sorted_samples = sorted(samples, key=lambda x: x[-1], reverse=True)
        return sorted_samples

    def indexes2graph(self, indexes):

        if len(indexes) == 0:
            return []

        list_graphs = []
        for index in indexes:
            dataset = self.datasets[index[0]]
            jps = dataset.df.loc[index[1]].values[: dataset.params_size]

            graph = dataset.graph_manager.get_graph(jps)

            if len(index) > 2:
                list_graphs.append((graph, *index[2:]))
        return list_graphs
    
    def get_indexes_in_bound(self, indexes, bounds):

        if len(indexes) == 0:
            return []

        indexes = self._indexes_1d_to_2d(indexes)

        indexes_in_bounds = []
        for k, dataset in enumerate(self.datasets):

            if len(indexes[k]) > 0:
                index_in_bound = dataset.get_filtered_df_with_jps_limits(bounds[k], indexes[k])
                indexes_in_bounds.append(index_in_bound)

        return self._index_2d_to_1d(indexes_in_bounds)


def get_sorted_graph_from_datasets(
    many_dataset_api: ManyDatasetAPI, ellipse: Ellipse, rewards: RewardManager
):
    valid_design_indexes = many_dataset_api.get_indexes_cover_ellipse(ellipse)
    sorted_design_indexes_with_rewards = many_dataset_api.sorted_indexes_by_reward(
        valid_design_indexes
    )
    sorted_graphs = many_dataset_api.indexes2graph(sorted_design_indexes_with_rewards)
    return sorted_graphs


def test_dataset_functionality(path_to_dir):

    dataset = Dataset(path_to_dir)

    df_upd = dataset.df.assign(
        total_ws=lambda x: np.sum(x.values[:, dataset.params_size :], axis=1)
        / dataset.ws_grid_size
    )

    df_upd = df_upd[df_upd["total_ws"] > 100 / dataset.ws_grid_size]
    df_upd = df_upd.sort_values(["total_ws"], ascending=False)
    from auto_robot_design.pinokla.default_traj import add_auxilary_points_to_trajectory

    des_point = np.array([-0.1, -0.35])
    traj = np.array(
        add_auxilary_points_to_trajectory(([des_point[0]], [des_point[1]]))
    ).T
    test_ws = dataset.get_workspace_by_indexes([0])[0]
    traj_6d = test_ws.robot.motion_space.get_6d_traj(traj)

    reward_manager = set_up_reward_manager(traj_6d)
    time_start = time.perf_counter()
    parallel_calculation_rew_manager(df_upd.head(200).index, dataset, reward_manager)
    time_end = time.perf_counter()

    print(f"Time spent {time_end - time_start}")


def test_many_dataset_api(list_paths):

    many_dataset = ManyDatasetAPI(list_paths)

    cover_design_indexes = many_dataset.get_indexes_cover_ellipse(
        Ellipse(np.array([0.05, -0.21]), 0, np.array([0.1, 0.04]))
    )
    from auto_robot_design.pinokla.default_traj import add_auxilary_points_to_trajectory

    des_point = np.array([-0.1, -0.35])
    traj = np.array(
        add_auxilary_points_to_trajectory(([des_point[0]], [des_point[1]]))
    ).T
    test_ws = many_dataset.datasets[0].get_workspace_by_indexes([0])[0]
    traj_6d = test_ws.robot.motion_space.get_6d_traj(traj)

    reward_manager = set_up_reward_manager(traj_6d)

    sorted_indexes = many_dataset.sorted_indexes_by_reward(
        cover_design_indexes, 10, reward_manager
    )

    # for desing in sorted_indexes:
    #     print(desing)
    #     for ind, rew in desing.items():
    #         print(ind, rew)


if __name__ == "__main__":

    paths = [
        "/var/home/yefim-work/Documents/auto-robotics-design/top_5",
        "/var/home/yefim-work/Documents/auto-robotics-design/top_8",
    ]

    test_many_dataset_api(paths)
