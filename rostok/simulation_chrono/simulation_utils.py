from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pychrono as chrono

from rostok.block_builder_chrono.block_classes import ChronoEasyShapeObject
from rostok.virtual_experiment.sensors import DataStorage


def calculate_covering_sphere(obj: ChronoEasyShapeObject):
    v1 = chrono.ChVectorD(0, 0, 0)
    v2 = chrono.ChVectorD(0, 0, 0)
    obj.body.GetTotalAABB(bbmin=v1, bbmax=v2)
    local_center = (v1 + v2) * 0.5
    radius = ((v2 - v1).Length()) * 0.5
    visual = chrono.ChSphereShape(radius)
    visual.SetOpacity(0.3)
    obj.body.AddVisualShape(visual, chrono.ChFrameD(local_center))
    if isinstance(obj.body, chrono.ChBodyAuxRef):
        cog_center = obj.body.GetFrame_REF_to_COG().TransformPointLocalToParent(local_center)
    else:
        cog_center = local_center

    return cog_center, radius


def calculate_covering_ellipsoid(obj: ChronoEasyShapeObject):
    v_1 = chrono.ChVectorD(0, 0, 0)
    v_2 = chrono.ChVectorD(0, 0, 0)
    obj.body.GetTotalAABB(bbmin=v_1, bbmax=v_2)
    local_center = (v_1 + v_2) * 0.5
    axis_x = v_2.x - v_1.x
    axis_y = v_2.y - v_1.y
    axis_z = v_2.z - v_1.z
    visual = chrono.ChEllipsoidShape(axis_x, axis_y, axis_z)
    visual.SetOpacity(0.3)
    obj.body.AddVisualShape(visual, chrono.ChFrameD(local_center))
    if isinstance(obj.body, chrono.ChBodyAuxRef):
        cog_center = obj.body.GetFrame_REF_to_COG().TransformPointLocalToParent(local_center)
    else:
        cog_center = local_center

    return cog_center, (axis_x, axis_y, axis_z)


def set_covering_sphere_based_position(obj: ChronoEasyShapeObject,
                                       reference_point: chrono.ChVectorD = chrono.ChVectorD(
                                           0, 0, 0),
                                       direction: chrono.ChVectorD = chrono.ChVectorD(0, 1, 0)):
    center, radius = calculate_covering_sphere(obj)
    direction.Normalize()
    desired_position = reference_point + direction * radius
    shift = desired_position - obj.body.GetCoord().TransformPointLocalToParent(center)
    current_cog_pos = obj.body.GetPos()
    obj.body.SetPos(current_cog_pos + shift)


def set_covering_ellipsoid_based_position(obj: ChronoEasyShapeObject,
                                          reference_point: chrono.ChVectorD = chrono.ChVectorD(
                                              0, 0, 0),
                                          direction: chrono.ChVectorD = chrono.ChVectorD(0, 1, 0)):
    center, axis = calculate_covering_ellipsoid(obj)
    direction.Normalize()
    desired_position = reference_point + direction * axis[1] / 2
    shift = desired_position - obj.body.GetCoord().TransformPointLocalToParent(center)
    current_cog_pos = obj.body.GetPos()
    obj.body.SetPos(current_cog_pos + shift)


def set_covering_ellipsoid_based_position(obj: ChronoEasyShapeObject,
                                          reference_point: chrono.ChVectorD = chrono.ChVectorD(
                                              0, 0, 0),
                                          direction: chrono.ChVectorD = chrono.ChVectorD(0, 1, 0)):
    center, axis = calculate_covering_ellipsoid(obj)
    direction.Normalize()
    desired_position = reference_point + direction * axis[1] / 2
    shift = desired_position - obj.body.GetCoord().TransformPointLocalToParent(center)
    current_cog_pos = obj.body.GetPos()
    obj.body.SetPos(current_cog_pos + shift)


@dataclass
class SimulationResult:
    """Data class to aggregate the output of the simulation.
    
        Attributes:
            time (float): the total simulation time
            time_vector (List[float]): the vector of time steps
            n_steps (int): the maximum possible number of steps
            robot_final_ds (Optional[DataStorage]): final data store of the robot
            environment_final_ds (Optional[DataStorage]): final data store of the environment"""
    time: float = 0
    time_vector: List[float] = field(default_factory=list)
    n_steps = 0
    robot_final_ds: Optional[DataStorage] = None
    environment_final_ds: Optional[DataStorage] = None

    def reduce_ending(self, step_n):
        if self.robot_final_ds:
            storage = self.robot_final_ds.main_storage
            for key in storage:
                key_storage = storage[key]
                for key_2 in key_storage:
                    value = key_storage[key_2]
                    new_value = value[:step_n + 2:]
                    key_storage[key_2] = new_value

        if self.environment_final_ds:
            storage = self.environment_final_ds.main_storage
            for key in storage:
                key_storage = storage[key]
                for key_2 in key_storage:
                    value = key_storage[key_2]
                    new_value = value[:step_n + 2:]
                    key_storage[key_2] = new_value

    def reduce_nan(self):
        if self.robot_final_ds:
            storage = self.robot_final_ds.main_storage
            for key in storage:
                key_storage = storage[key]
                for key_2 in key_storage:
                    value = key_storage[key_2]
                    new_value = [x for x in value if np.logical_not(np.isnan(x).all())]
                    key_storage[key_2] = new_value

        if self.environment_final_ds:
            storage = self.environment_final_ds.main_storage
            for key in storage:
                key_storage = storage[key]
                for key_2 in key_storage:
                    value = key_storage[key_2]
                    new_value = [x for x in value if np.logical_not(np.isnan(x).all())]
                    key_storage[key_2] = new_value