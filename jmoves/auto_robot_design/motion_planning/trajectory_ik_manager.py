from functools import partial
import time
import meshcat
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from auto_robot_design.motion_planning.ik_calculator import (
    closed_loop_ik_pseudo_inverse, open_loop_ik, closedLoopInverseKinematicsProximal)

IK_METHODS = {"Open_Loop": open_loop_ik,
              "Closed_Loop_PI": closed_loop_ik_pseudo_inverse,
              "Closed_Loop_Proximal": closedLoopInverseKinematicsProximal}


class TrajectoryIKManager():
    def __init__(self) -> None:
        self.model = None
        self.constraint_models = None
        self.solver = None
        self.visual_model = None
        self.default_name = "Closed_Loop_PI"
        # self.default_name = "Closed_Loop_Proximal"
        self.frame_name = "EE"

    def register_model(self, model, constraint_models, visual_model=None):
        """The function to register a model of a mechanism

        Args:
            model (_type_): pinocchio model of a mechanism
            constraint_models (_type_): model of constraints
        """
        self.model = model
        self.constraint_models = constraint_models
        if visual_model:
            self.visual_model = visual_model

    def set_solver(self, name: str, **params):
        """Set the IK solver for trajectory following.

            Function uses names of the solvers from the dictionary IK_METHODS. 
        Args:
            name (str): name of the IK solver algorithm
        """
        # try to set the solver and warn if the setting process failed, ib case of a fail set the default solver
        try:
            self.solver = partial(IK_METHODS[name], **params)
        except KeyError:
            print(
                f'Cannot set solver - wrong name: {name}. Solver set to default value: {self.default_name}')
            self.solver = partial(IK_METHODS[self.default_name], {})
        except TypeError:
            print(
                f"Cannot set solver - wrong parameters for solver: {name}. Solver set to default value: {self.default_name}")
            self.solver = partial(IK_METHODS[self.default_name], {})

    def follow_trajectory(self, trajectory: np.ndarray, q_start: np.ndarray = None, viz=None):
        """The function to follow a trajectory.

        Args:
            trajectory (np.array): trajectory which should be followed
            q_start (np.array, optional): initial point in configuration space. Defaults to None.

        Raises:
            Exception: raise an exception if the solver is not set

        Returns:
            results of trajectory following: ee positions, configuration space points, constraint errors, reach array
        """
        if self.solver:
            ik_solver = self.solver
        else:
            raise ValueError(
                "set a solver before an attempt to follow a trajectory")

        frame_id = self.model.getFrameId(self.frame_name)
        # create a copy of a registered model
        model = pin.Model(self.model)
        data = model.createData()
        if q_start is not None:
            q = q_start
        else:
            q = pin.neutral(self.model)

        # We initialize all arrays to have the length of the trajectory to have all results of the same shape
        # 3D coordinates of the following frame, TODO: consider a way to specify what kind of positioning we need
        poses = np.zeros((len(trajectory), 3))
        # reach mask
        reach_array = np.zeros(len(trajectory))
        # calculated positions in configuration space
        q_array = np.zeros((len(trajectory), len(q)))
        # final error for each point
        constraint_errors = np.zeros((len(trajectory), 1))
        for idx, point in enumerate(trajectory):
            q, min_feas, is_reach = ik_solver(
                model,
                self.constraint_models,
                point,
                frame_id,
                q_start=q,
            )
            # if the point is not reachable, we stop the trajectory following
            if not is_reach:
                break
            if viz:
                viz.display(q)
                time.sleep(0.03)

            # if the point is reachable, we store the values in corresponding arrays
            pin.framesForwardKinematics(model, data, q)
            poses[idx] = data.oMf[frame_id].translation
            q_array[idx] = q
            constraint_errors[idx] = min_feas
            reach_array[idx] = is_reach

        return poses, q_array, constraint_errors, reach_array
