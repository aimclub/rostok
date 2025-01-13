    
import os, pathlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
from auto_robot_design.user_interface.check_in_ellips import (
    Ellipse,
    check_points_in_ellips
)


def create_bounded_box(center: np.ndarray, bound_range: np.ndarray):
    bounds = np.array(
        [
            [-bound_range[0] / 2 - 0.001, bound_range[0] / 2],
            [-bound_range[1] / 2, bound_range[1] / 2],
        ]
    )
    bounds[0, :] += center[0]
    bounds[1, :] += center[1]

    return bounds


class Workspace:
    def __init__(self, robot, bounds, resolution: np.ndarray):
        ''' Class for working workspace of robot like grid with `resolution` and `bounds`. 
        Grid's indices go from bottom-right to upper-left corner of bounds

        '''
        self.robot = robot
        self.resolution = resolution
        self.bounds = bounds

        num_indexes = (np.max(bounds, 1) - np.min(bounds, 1)) / self.resolution
        self.mask_shape = np.zeros_like(num_indexes)
        self.bounds = np.zeros_like(bounds)
        # Bounds correction for removing ucertainties with indices. Indices was calculated with minimal `bounds` and `resolution`
        for id, idx_value in enumerate(num_indexes):
            residue_div = np.round(idx_value % 1, 6)

            check_bound_size = np.isclose(residue_div, 1.0)
            check_min_bound = np.isclose(
                bounds[id, 0] % self.resolution[id], 0)
            check_max_bound = np.isclose(
                bounds[id, 1] % self.resolution[id], 0)
            if check_bound_size and check_min_bound and check_max_bound:
                self.bounds[id, :] = bounds[id, :]
                self.mask_shape[id] = num_indexes[id]
            else:
                self.bounds[id, 1] = np.round(
                    bounds[id, 1] + bounds[id, 1] % self.resolution[id], 4)
                self.bounds[id, 0] = np.round(
                    bounds[id, 0] - bounds[id, 0] % self.resolution[id], 4)
                self.mask_shape[id] = np.ceil(
                    (self.bounds[id, 1] - self.bounds[id, 0]) /
                    self.resolution[id]
                )
        self.mask_shape = np.asarray(self.mask_shape.round(4), dtype=int) + 1
        self.bounds = self.bounds.round(4)
        self.set_nodes = {}
        self.reachable_index = {}
        # self.grid_nodes = np.zeros(tuple(self.mask_shape), dtype=object)

    def calc_grid_position(self, indexes):

        pos = indexes * self.resolution + self.bounds[:, 0]

        return pos

    def calc_index(self, pos):
        return np.round((pos - self.bounds[:, 0]) / self.resolution).astype(int)

    def calc_grid_index(self, pos):
        idx = self.calc_index(pos)
        grid_index = 0
        for k, ind in enumerate(idx):
            grid_index += ind * np.prod(self.mask_shape[:k])

        return grid_index
    
    def calc_grid_index_with_index(self, index):
        grid_index = 0
        for k, ind in enumerate(index):
            grid_index += ind * np.prod(self.mask_shape[:k])
        return grid_index

    def point_in_bound(self, point: np.ndarray):
        return np.all(point >= self.bounds[:, 0] - self.resolution*0.9) and np.all(point <= self.bounds[:, 1] + self.resolution*0.9)
    # def update_by_reach_mask(reachable_mask): 
    
    
    def check_points_in_ws(self, points: np.ndarray):

        check_array = np.zeros(points.shape[0], dtype=int)
        grid_indexes = np.zeros(points.shape[0], dtype=int)
        for idx, point in enumerate(points):
            check_array[idx] = 1 if self.point_in_bound(point) else 0
            grid_indexes[idx] = self.calc_grid_index(point)
        check_in_bound_points = np.all(check_array == 1)
        check_reachable_points = set(grid_indexes.tolist()) <= set(self.reachable_index)
        check_points_in_ws = False
        if check_in_bound_points and check_reachable_points:
            check_points_in_ws = True
        return check_points_in_ws

    @property
    def reachabilty_mask(self):
        mask = np.zeros(tuple(self.mask_shape), dtype=bool)

        for index in self.reachable_index.values():
            mask[tuple(index)] = True

        return mask

    @property
    def points(self):
        points = []
        point = self.bounds[:, 0]
        for m in range(self.mask_shape[0]):
            for k in range(self.mask_shape[1]):
                point = self.bounds[:, 0] + np.array(
                    self.resolution) * np.array([m, k])
                points.append(point)
        # while point[1] <= self.bounds[1, 1]:

        #     while point[0] <= self.bounds[0, 1]:
        #         points.append(point)
        #         m += 1
        #         point = self.bounds[:, 0] + np.array(
        #             self.resolution) * np.array([m, k])
        #     k += 1
        #     m = 0
        #     point = self.bounds[:, 0] + np.array(
        #         self.resolution) * np.array([m, k])

        points = np.array(points)
        return points
    
    @property
    def reachable_points(self):
        points = np.zeros((len(self.reachable_index), self.mask_shape.size), dtype=float)
        for k, reach_index in enumerate(self.reachable_index.values()):
            points[k,:] = self.calc_grid_position(reach_index)
        return points
    

# def save_workspace(workspace: Workspace, path):
#     init_points = workspace.bounds[:,0]
#     resolution = workspace.resolution
#     reachable_mask = workspace.reachabilty_mask
    
#     path = pathlib.Path(path)
#     if not path.exists():
#         path.mkdir(parents=True, exist_ok=True)
    
#     name_file = 'workspace_data.npz'
#     np.savez_compressed(path / name_file, init_points=init_points, resolution=resolution, reachable_mask=reachable_mask)
#     with open(path / "robot.pkl", "wb") as f:
#         pickle.dump(workspace.robot, f)


# def load_workspace(path):
#     path = pathlib.Path(path)
#     file_data = np.load(path / 'workspace_data.npz')
    
#     with open(path / "robot.pkl", "rb") as f:
#         robot = pickle.load(f)
    
#     init_points = file_data["init_points"]
#     resolution =  file_data["resolution"]
#     reachable_mask =  file_data["reachabilty_mask"]
    
#     bounds = np.zeros((init_points.size, 2), dtype=float)
#     bounds[:,0] = init_points
#     bounds[:,1] = init_points + resolution * np.array(reachable_mask.shape)
    
#     workspace = Workspace(robot, bounds, resolution)
    
#     reachable_indexes = np.argwhere(reachable_mask == 1)
    
#     workspace.mask_shape = reachable_mask.shape
#     for index in reachable_indexes:
#         workspace.reachable_index[workspace.calc_grid_index_with_index(index)] = index
    
#     return workspace


def ellipse_in_workspace(ellips: Ellipse, workspace: Workspace, strong_check = True, verbose=0):
    
    if verbose > 0:
        grid_points = workspace.points
        plt.plot(grid_points[:,0],grid_points[:,1], "rx")
        reach_grid_points = workspace.reachable_points
        plt.plot(reach_grid_points[:,0],reach_grid_points[:,1], "gx")
    
    ellips_in_ws = False
    points_on_ellps = ellips.get_points(np.min(workspace.resolution)).T
    
    ellips_in_ws = workspace.check_points_in_ws(points_on_ellps)
    
    if verbose > 0:
        plt.plot(points_on_ellps[:,0], points_on_ellps[:,1], "c")    
    
    if ellips_in_ws and strong_check:
        reach_ws_points = workspace.points
        mask_ws_n_ellps = check_points_in_ellips(reach_ws_points, ellips, np.max(workspace.resolution)*15)
        ellips_in_ws = ellips_in_ws and workspace.check_points_in_ws(reach_ws_points[mask_ws_n_ellps,:])

        if verbose > 0:
            plt.plot(reach_ws_points[mask_ws_n_ellps,:][:,0], reach_ws_points[mask_ws_n_ellps,:][:,1], "xc")
    
    return ellips_in_ws


def build_graphs(graphs, builder, func, *args, **kwargs):
    for graph in graphs:
        fixed_robot, free_robot = func(graph, builder, *args, **kwargs)
        yield fixed_robot, free_robot