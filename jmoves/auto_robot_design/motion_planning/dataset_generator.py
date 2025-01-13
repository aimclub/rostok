import os
import csv
import glob
import pathlib
import time
from copy import deepcopy
import concurrent.futures
from typing import Optional
import dill
from joblib import cpu_count
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pinocchio as pin


from auto_robot_design.description.builder import (
    MIT_CHEETAH_PARAMS_DICT,
    ParametrizedBuilder,
    URDFLinkCreater3DConstraints,
    jps_graph2pinocchio_robot_3d_constraints,
)


from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.motion_planning.bfs_ws import BreadthFirstSearchPlanner
from auto_robot_design.motion_planning.utils import Workspace, build_graphs
from auto_robot_design.user_interface.check_in_ellips import (
    Ellipse,
    check_points_in_ellips,
)
from auto_robot_design.utils.append_saver import chunk_list
from auto_robot_design.utils.bruteforce import get_n_dim_linspace
from presets.MIT_preset import get_mit_builder


WORKSPACE_ARGS_NAMES = ["bounds", "resolution", "dexterous_tolerance", "grid_shape"]


class WorkspaceOutBodunds(Exception):
    pass

class DatasetGenerator:
    def __init__(self, graph_manager, path, workspace_args):
        """
        Initializes the DatasetGenerator.
        Args:
            graph_manager (GraphManager): The manager responsible for handling the graph operations.
            path (str): The directory path where the dataset and related files will be saved.
            workspace_args (tuple): Arguments required to initialize the workspace.
        Attributes:
            ws_args (tuple): Stored workspace arguments.
            graph_manager (GraphManager): Stored graph manager.
            path (pathlib.Path): Path object for the directory where files will be saved.
            builder (ParametrizedBuilder): Builder for creating URDF links with 3D constraints.
            params_size (int): Size of the parameters generated from the mutation range.
            ws_grid_size (int): Size of the workspace grid.
            field_names (list): List of field names for the dataset CSV file.
        Operations:
            - Creates the directory if it does not exist.
            - Draws and saves a graph image.
            - Serializes the graph manager to a pickle file.
            - Saves workspace arguments to a .npz file and writes them to an info.txt file.
            - Initializes the dataset CSV file with appropriate headers.
        """

        self.ws_args = workspace_args
        self.graph_manager = graph_manager
        self.path = pathlib.Path(path)
        workspace = Workspace(None, *self.ws_args[:-1])

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
        draw_joint_point(
            self.graph_manager.get_graph(
                self.graph_manager.generate_central_from_mutation_range()
            )
        )
        plt.savefig(self.path / "graph.png")
        with open(self.path / "graph.pkl", "wb") as file:
            dill.dump(self.graph_manager, file)
        wrt_lines = []
        arguments = self.ws_args + (workspace.mask_shape,)
        for nm, vls in zip(WORKSPACE_ARGS_NAMES, arguments):
            wrt_lines.append(nm + ": " + str(np.round(vls, 3)) + "\n")
        np.savez(
            self.path / "workspace_arguments.npz",
            bounds=arguments[0],
            resolution=arguments[1],
            dexterous_tolerance=arguments[2],
            grid_shape=arguments[3],
        )

        with open(self.path / "info.txt", "w") as file:
            file.writelines(wrt_lines)
        self.builder = get_mit_builder()

        self.params_size = len(self.graph_manager.generate_random_from_mutation_range())
        self.ws_grid_size = np.prod(workspace.mask_shape)

        dataset_fields_names = ["jp_" + str(i) for i in range(self.params_size)]
        dataset_fields_names += ["ws_" + str(i) for i in range(self.ws_grid_size)]
        self.field_names = dataset_fields_names
        with open(self.path / "dataset.csv", "a", newline="") as f_object:
            # Pass the file object and a list of column names to DictWriter()

            dict_writer_object = csv.DictWriter(f_object, fieldnames=self.field_names)
            # If the file is empty or you are adding the first row, write the header

            if f_object.tell() == 0:
                dict_writer_object.writeheader()

    def _find_workspace(self, joint_positions: np.ndarray):
        graph = self.graph_manager.get_graph(joint_positions)
        robot, _ = jps_graph2pinocchio_robot_3d_constraints(graph, self.builder)
        workspace = Workspace(robot, *self.ws_args[:-1])
        ws_search = BreadthFirstSearchPlanner(workspace, 0, self.ws_args[-1])

        q = pin.neutral(robot.model)
        pin.framesForwardKinematics(robot.model, robot.data, q)
        id_ee = robot.model.getFrameId(robot.ee_name)
        start_pos = robot.data.oMf[id_ee].translation[[0, 2]]

        workspace = ws_search.find_workspace(start_pos, q)

        return joint_positions, workspace.reachabilty_mask.flatten()

    def save_batch_to_dataset(self, batch, postfix=""):
        """
        Save a batch of data to the dataset file.
        This method processes a batch of data, combining joint positions and workspace grid data,
        and saves it to a CSV file. The data is rounded to three decimal places before saving.
        Args:
            batch (list): A list of tuples, where each tuple contains joint positions and workspace grid data.
            postfix (str, optional): A string to append to the dataset filename. Defaults to "".
        Returns:
            None
        """

        joints_pos_batch = np.zeros((len(batch), self.params_size))
        ws_grid_batch = np.zeros((len(batch), self.ws_grid_size))
        for k, el in enumerate(batch):
            joints_pos_batch[k, :] = el[0]
            ws_grid_batch[k, :] = el[1]
        sorted_batch = np.hstack((joints_pos_batch, ws_grid_batch)).round(3)
        file_dataset = self.path / ("dataset" + postfix + ".csv")
        with open(file_dataset, "a", newline="") as f_object:
            # Pass the file object and a list of column names to DictWriter()

            dict_writer_object = csv.DictWriter(f_object, fieldnames=self.field_names)
            # If the file is empty or you are adding the first row, write the header

            if f_object.tell() == 0:
                dict_writer_object.writeheader()

            writer = csv.writer(f_object)
            writer.writerows(sorted_batch)

    def _parallel_calculate_batch(self, joint_poses_batch: np.ndarray):
        bathch_result = []
        cpus = cpu_count() - 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
            futures = [
                executor.submit(self._find_workspace, i) for i in joint_poses_batch
            ]
            for future in concurrent.futures.as_completed(futures):
                bathch_result.append(future.result())
        return bathch_result

    def _calculate_batches(self, batches: np.ndarray, postfix=""):
        for batch in batches:
            bathch_result = []
            for i in batch:
                bathch_result.append(self._find_workspace(i))
            self.save_batch_to_dataset(bathch_result, postfix)

    def start(self, num_points, size_batch):
        """
        Generates a dataset by creating points within specified mutation ranges and processes them in batches.
        Args:
            num_points (int): The number of points to generate.
            size_batch (int): The size of each batch.
        Raises:
            Exception: If an error occurs during batch processing.
        Writes:
            A file named "info.txt" containing the number of points generated.
            A file named "dataset.csv" containing the concatenated results of all processed batches.
        """

        self.graph_manager.generate_central_from_mutation_range()
        low_bnds = [value[0] for value in self.graph_manager.mutation_ranges.values()]
        up_bnds = [value[1] for value in self.graph_manager.mutation_ranges.values()]
        vecs = get_n_dim_linspace(up_bnds, low_bnds, num_points)
        batches = list(chunk_list(vecs, size_batch))

        with open(self.path / "info.txt", "a") as file:
            file.writelines("Number of points: " + str(num_points) + "\n")
            file.writelines(
                "Lower bounds mutation JPs: " + str(np.round(low_bnds, 3)) + "\n"
            )
            file.writelines(
                "Upper bounds mutation JPs: " + str(np.round(up_bnds, 3)) + "\n"
            )

        cpus = cpu_count() - 1 if cpu_count() - 1 < len(batches) else len(batches)
        batches_chunks = list(chunk_list(batches, (len(batches) // cpus) + 1))
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
                futures = [
                    executor.submit(
                        self._calculate_batches, batches, "_" + str(m // cpus)
                    )
                    for m, batches in enumerate(batches_chunks)
                ]
        except Exception as e:
            print(e)
        finally:
            all_files = glob.glob(os.path.join(self.path, "*.csv"))
            df = pd.concat(
                (pd.read_csv(f, low_memory=False) for f in all_files),
                ignore_index=True,
            )

        for file in all_files:
            os.remove(file)

        pd.DataFrame(df).to_csv(self.path / "dataset.csv", index=False)

        # for num, batch in tqdm(enumerate(batches)):
        #     try:
        #         batch_results = self._parallel_calculate_batch(batch)
        #         self.save_batch_to_dataset(batch_results)
        #     except Exception as e:
        #         print(e)


class Dataset:
    def __init__(self, path_to_dir):
        """
        Initializes the DatasetGenerator with the specified directory path.
        Args:
            path_to_dir (str): The path to the directory containing the dataset and other necessary files.
        Attributes:
            path (pathlib.Path): The path to the directory as a pathlib.Path object.
            df (pd.DataFrame): The dataset loaded from 'dataset.csv'.
            dict_ws_args (dict): The workspace arguments loaded from 'workspace_arguments.npz'.
            ws_args (list): The list of workspace arguments.
            workspace (Workspace): The Workspace object initialized with the workspace arguments.
            graph_manager (GraphManager): The graph manager loaded from 'graph.pkl'.
            params_size (int): The size of the parameters generated by the graph manager.
            ws_grid_size (int): The size of the workspace grid.
            builder (ParametrizedBuilder): The builder object initialized with URDFLinkCreater3DConstraints.
        """
        self.path = pathlib.Path(path_to_dir)

        self.df = pd.read_csv(self.path / "dataset.csv", nrows=2e4)
        self.dict_ws_args = np.load(self.path / "workspace_arguments.npz")
        self.ws_args = [self.dict_ws_args[name] for name in WORKSPACE_ARGS_NAMES[:-1]]
        self.workspace = Workspace(None, *self.ws_args[:-1])

        with open(self.path / "graph.pkl", "rb") as f:
            self.graph_manager = dill.load(f)
        self.params_size = len(self.graph_manager.generate_random_from_mutation_range())
        self.ws_grid_size = np.prod(self.workspace.mask_shape)

        self.builder = ParametrizedBuilder(URDFLinkCreater3DConstraints)

    def get_workspace_by_indexes(self, indexes):
        """
        Generates a list of workspace objects based on the provided indexes.
        Args:
            indexes (list): A list of indexes to retrieve workspace data.
        Returns:
            list: A list of workspace objects with updated robot and reachable index information.
        The function performs the following steps:
        1. Initializes an empty list to store reachable indexes.
        2. Iterates over the provided indexes to extract workspace masks and calculates reachable indexes.
        3. Retrieves graphs corresponding to the provided indexes.
        4. Builds robot configurations from the graphs.
        5. Creates a deep copy of the workspace for each index.
        6. Updates each workspace copy with the corresponding robot configuration and reachable indexes.
        """
        arr_reach_indexes = []
        for k in indexes:
            ws_mask = (
                self.df.loc[k]
                .values[self.params_size : self.params_size + self.ws_grid_size]
                .reshape(self.dict_ws_args["grid_shape"])
            )
            arr_reach_indexes.append(
                {
                    self.workspace.calc_grid_index_with_index(ind): ind
                    for ind in np.argwhere(ws_mask == 1).tolist()
                }
            )
        graphs = self.get_graphs_by_indexes(indexes)
        robot_list = list(
            build_graphs(graphs, self.builder, jps_graph2pinocchio_robot_3d_constraints)
        )
        arr_ws_outs = [deepcopy(self.workspace) for _ in range(len(indexes))]

        for k, ws_out in enumerate(arr_ws_outs):
            ws_out.robot = robot_list[k][0]
            ws_out.reachable_index = arr_reach_indexes[k]
        return arr_ws_outs

    def get_all_design_indexes_cover_ellipse(self, ellipse: Ellipse, indexes: Optional[list] = None):
        """
        Get all design indexes that cover the given ellipse.
        This method calculates the indexes of designs that cover the specified ellipse
        within the workspace. It first verifies that all points on the ellipse are within
        the workspace bounds. Then, it creates a mask for the workspace points that fall
        within the ellipse and uses this mask to find the relevant design indexes.
        Args:
            ellipse (Ellipse): The ellipse object for which to find covering design indexes.
        Returns:
            numpy.ndarray: An array of indexes corresponding to designs that cover the given ellipse.
        Raises:
            Exception: If any point on the ellipse is out of the workspace bounds.
        """
        points_on_ellps = ellipse.get_points(0.1).T

        if indexes is None:
            df = self.df
        else:
            df = self.df.loc[indexes]
        for pt in points_on_ellps:
            if not self.workspace.point_in_bound(pt):
                raise WorkspaceOutBodunds("Input ellipse out of workspace bounds")
        ws_points = self.workspace.points
        mask_ws_n_ellps = check_points_in_ellips(ws_points, ellipse, 0.1)
        ellips_mask = np.zeros(self.workspace.mask_shape, dtype=bool)
        for point in ws_points[mask_ws_n_ellps, :]:
            index = self.workspace.calc_index(point)
            ellips_mask[tuple(index)] = True
        ws_bool_flatten = np.asarray(
            df.values[:, self.params_size : self.params_size + self.ws_grid_size],
            dtype=bool,
        )
        ell_mask_2_d = ellips_mask.flatten()[np.newaxis :]
        indexes = np.argwhere(
            np.sum(ell_mask_2_d * ws_bool_flatten, axis=1) == np.sum(ell_mask_2_d)
        )
        return df.index[indexes.flatten()].values

    def get_design_parameters_by_indexes(self, indexes):
        """
        Retrieve design parameters based on provided indexes.
        Args:
            indexes (list or array-like): The indexes of the rows to retrieve from the dataframe.
        Returns:
            numpy.ndarray: A 2D array containing the design parameters for the specified indexes.
        """
        return self.df.loc[indexes].values[:, : self.params_size]

    def get_graphs_by_indexes(self, indexes):
        """
        Retrieve graphs based on the provided indexes.
        Args:
            indexes (list): A list of indexes to retrieve the corresponding design parameters.
        Returns:
            list: A list of graphs corresponding to the design parameters obtained from the provided indexes.
        """
        desigm_parameters = self.get_design_parameters_by_indexes(indexes)
        return [
            deepcopy(self.graph_manager.get_graph(des_param)) for des_param in desigm_parameters
        ]
    
    def get_filtered_df_with_jps_limits(self, limits:np.ndarray, indexes: Optional[list] = None):
        
        if indexes is None:
            df = self.df
        else:
            df = self.df.loc[indexes]

        def filter_func(df):
            jps = df.values[:, :self.params_size]
            arr_higher_low = np.all(jps >= limits[:,0], axis=1)
            arr_lower_upper = np.all(jps <= limits[:,1], axis=1)
            arr_in_limits = np.logical_and(arr_higher_low, arr_lower_upper)
            return arr_in_limits
        
        filt_df = df[filter_func(df)]

        return filt_df


def set_up_reward_manager(traj_6d, reward):
    from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import (
        HeavyLiftingReward,
        MinAccelerationCapability,
    )

    from auto_robot_design.optimization.rewards.reward_base import RewardManager

    from auto_robot_design.pinokla.calc_criterion import (
        ActuatedMass,
        EffectiveInertiaCompute,
        ManipJacobian,
        MovmentSurface,
        NeutralPoseMass,
    )

    from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
    from auto_robot_design.utils.configs import get_standard_builder, get_mesh_builder, get_standard_crag, get_standard_rewards
    # dict_trajectory_criteria = {
    #     "MASS": NeutralPoseMass(),
    #     "POS_ERR": TranslationErrorMSE()  # MSE of deviation from the trajectory
    # }
    # # criteria calculated for each point on the trajectory
    # dict_point_criteria = {
    #     # Impact mitigation factor along the axis
    #     "IMF": ImfCompute(ImfProjections.Z),
    #     "MANIP": ManipCompute(MovmentSurface.XZ),
    #     "Effective_Inertia": EffectiveInertiaCompute(),
    #     "Actuated_Mass": ActuatedMass(),
    #     "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ)
    # }
    # # special object that calculates the criteria for a robot and a trajectory

    # crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
    crag = get_standard_crag()
    # set the rewards and weights for the optimization task

    # acceleration_capability = MinAccelerationCapability(
    #     manipulability_key="Manip_Jacobian",
    #     trajectory_key="traj_6d",
    #     error_key="error",
    #     actuated_mass_key="Actuated_Mass",
    # )

    # heavy_lifting = HeavyLiftingReward(
    #     manipulability_key='Manip_Jacobian', mass_key='MASS', reachability_key="is_reach")
    

    reward_manager = RewardManager(crag=crag)
    reward_manager.add_trajectory(traj_6d, 0)

    reward_manager.add_reward(reward, 0, 1)
    # reward_manager.add_reward(heavy_lifting, 0, 1)

    return reward_manager


def test_dataset_generator(name_path):
    from auto_robot_design.generator.topologies.bounds_preset import (
        get_preset_by_index_with_bounds,
    )

    gm = get_preset_by_index_with_bounds(0)
    ws_agrs = (
        np.array([[-0.05, 0.05], [-0.4, -0.3]]),
        np.array([0.01, 0.01]),
        np.array([0, np.inf]),
    )
    dataset_generator = DatasetGenerator(gm, name_path, ws_agrs)

    # jp_batch = []
    # for __ in range(10):
    #     jp_batch.append(gm.generate_random_from_mutation_range())
    # res = dataset_generator._calculate_batch(jp_batch)
    # dataset_generator.save_batch_to_dataset(res)

    dataset_generator.start(3, 50)


if __name__ == "__main__":
    pass
