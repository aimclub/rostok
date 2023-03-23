import numpy as np
import pychrono as chrono

from rostok.block_builder_chrono.block_classes import (BuildingBody, ChronoEasyShapeObject,
                                                       ChronoRevolveJoint)
from rostok.virtual_experiment.robot import Robot


class RobotSensor:
    """Sensor based on a robot (for chrono simulation)

    Returns:
        optional: Values of the sensor
    """

    # FIXME: Change to correct method
    @staticmethod
    def mean_center(in_robot: Robot) -> chrono.ChVectorD:
        """Mean center of the robot. Line center of the robot's mass

        Args:
            in_robot (Robot): Robot to measure

        Returns:
            chrono.ChVectorD: Coordinate of the center of the robot
        """
        blocks = in_robot.block_map.values()
        body_block = filter(lambda x: isinstance(x, BuildingBody), blocks)
        sum_cog_coord = chrono.ChVectorD(0, 0, 0)
        bodies = list(body_block)
        for body in bodies:
            sum_cog_coord += body.body.GetFrame_COG_to_abs().GetPos()
        mean_center: chrono.ChVectorD = sum_cog_coord / len(bodies)
        return mean_center

    @staticmethod
    def sum_contact_forces_blocks(in_robot: Robot):
        """Sensor of sum contact forces blocks of robot

        Args:
            in_robot (Robot): Robot to measure sum of contact forces

        Returns:
            dict[int, float]: Dictionary which keys are id blocks of robot bodies and values
            are sum of contact forces
        """
        blocks = in_robot.block_map
        body_block = filter(lambda x: isinstance(x[1], BuildingBody), blocks.items())
        contact_force_blocks = map(lambda x: (x[0], sum(x[1].list_n_forces)), body_block)
        return dict(contact_force_blocks)

    @staticmethod
    def abs_coord_COG_blocks(in_robot: Robot) -> dict[int, chrono.ChVectorD]:
        """Sensor of absolute coordinates of the robot boides

        Args:
            in_robot (Robot): Robot to measure

        Returns:
            dict[int, chrono.ChVectorD]: Dictionary which keys are id blocks of robot
            boides and values are coordinates COG
        """
        blocks = in_robot.block_map
        body_block = filter(lambda x: isinstance(x[1], BuildingBody), blocks.items())

        def cog_from_tuple(tupled):
            pos = tupled[1].body.GetPos()
            return (tupled[0], [pos.x, pos.y, pos.z])

        coord_COG_blocks = map(cog_from_tuple, body_block)
        return dict(coord_COG_blocks)

    # FIXME: Current method return bool of contact, not number of contact surfaces
    @staticmethod
    def amount_contact_surfaces_blocks(in_robot: Robot) -> dict[int, int]:
        """Sensors of amount of contact surfaces robot bodies

        Args:
            in_robot (Robot): Robot to measure

        Returns:
            dict[int, int]: Dictionary which keys are id blocks of robot
            bodies and values are number of contact surfaces
        """
        blocks = in_robot.block_map
        body_block = filter(lambda x: isinstance(x[1], BuildingBody), blocks.items())

        num_contact_surfaces = map(lambda x: (x[0], int(not (sum(x[1].list_n_forces) == 0))),
                                   body_block)

        return dict(num_contact_surfaces)

    @staticmethod
    def joints_angle(in_robot: Robot):
        """Sensor angle joint of the robot

        Args:
            in_robot (Robot): Robot to measure joint angles

        Returns:
            dict[int, float]: Dictionary which keys are id blocks of robot
            joints and values are radians of angle joint
        """
        blocks = in_robot.block_map
        joint_blocks = filter(lambda x: isinstance(x[1], ChronoRevolveJoint), blocks.items())
        joints_angle_block = map(lambda x: (x[0], x[1].joint.GetMotorRot()), joint_blocks)
        return dict(joints_angle_block)

    @staticmethod
    def std_contact_forces_object(obj):
        """Sensor of standart deviation of contact forces that affect on object

        Args:
            in_robot (Robot): Robot to measure sum of contact forces

        Returns:
            dict[int, float]: Dictionary which keys are id object and values of standart deviation of contact forces
        """

        if np.size(obj.list_n_forces) > 0:
            contact_force_obj = np.std(obj.list_n_forces)
            return dict([(-1, contact_force_obj)])
        else:
            return None

    @staticmethod
    def amount_contact_forces_object(obj):
        """Sensor of standart deviation of contact forces that affect on object

        Args:
            in_robot (Robot): Robot to measure sum of contact forces

        Returns:
            dict[int, float]: Dictionary which keys are id object and values of standart deviation of contact forces
        """

        if np.size(obj.list_n_forces) > 0:
            amount_contact_force_obj = np.size(obj.list_n_forces)
            return dict([(-1, amount_contact_force_obj)])
        else:
            return None

    @staticmethod
    def contact_coord(obj: ChronoEasyShapeObject):
        """Sensor of COG of contact points
        Args:
            obj (ChronoBodyEnv): Grasp object
        Returns:
            dict[int, float]: Dictionary which keys are id of object and values of COG of contact point volume in XYZ format
        """
        if np.size(obj.list_c_coord) > 0:
            coordinates = []
            coord_x = 0
            coord_y = 0
            coord_z = 0
            for coord in obj.list_c_coord:
                coord_x += coord[0]
                coord_y += coord[1]
                coord_z += coord[2]
            coordinates.append([
                coord_x / len(obj.list_c_coord), coord_y / len(obj.list_c_coord),
                coord_z / len(obj.list_c_coord)
            ])
            return dict([(-1, [
                coord_x / len(obj.list_c_coord), coord_y / len(obj.list_c_coord),
                coord_z / len(obj.list_c_coord)
            ])])
        else:
            return None

    @staticmethod
    def abs_coord_COG_obj(obj: ChronoEasyShapeObject):
        """Sensor of absolute coordinates of grasp object
        Args:
            obj (ChronoBodyEnv): Grasp object
        Returns:
            dict[int, chrono.ChVectorD]: Dictionary which keys are id of object 
            and value of object COG in XYZ format
        """
        if np.size(obj.list_c_coord) > 0:
            return dict([(-1, [obj.body.GetPos().x, obj.body.GetPos().y, obj.body.GetPos().z])])