from dataclasses import dataclass

import pychrono as chrono
import pychrono.irrlicht as chronoirr

import rostok.block_builder.control as control
from rostok.block_builder.node_render import ChronoRevolveJoint, RobotBody
from rostok.criterion.flags_simualtions import (ConditionStopSimulation,
                                                FlagStopSimualtions)
from rostok.graph_grammar.node import BlockWrapper, GraphGrammar
from rostok.virtual_experiment.auxilarity_sensors import RobotSensor
from rostok.virtual_experiment.robot import Robot


# Immutable classes with output simulation data for robot block
@dataclass(frozen=True)
class SimulationDataBlock:
    """Immutable class with output simulation data for robot block.

        Attr:
            id_block (int): id of robot block
            time (list[float]): list of time of simulation
    """
    id_block: int
    time: list[float]


@dataclass(frozen=True)
class DataJointBlock(SimulationDataBlock):
    """Immutable class with output simulation data for robot joint block (:py:class:`ChronoRevoluteJoint`).

        Attr:
            id_block (int): id of robot block
            time (list[float]): list of time of simulation
            angle_list (list[float]): list of angle of robot joint block
    """
    angle_list: list[float]


@dataclass(frozen=True)
class DataBodyBlock(SimulationDataBlock):
    """Immutable class with output simulation data for robot body block (:py:class:`rostok.block_builder.node_render.BoxChronoBody`)

        Attr:
            id_block (int): id of robot block
            time (list[float]): list of time of simulation
            sum_contact_forces (list[float]): list of contact forces sum
            abs_coord_COG (list[float]): list of absolute coordinates of block
            amount_contact_surfaces (list[int]): list of number of contact surfaces
    """
    sum_contact_forces: list[float]
    abs_coord_COG: list[list[float]]
    amount_contact_surfaces: list[int]


"""Type for output simulation. Store trajectory and block id"""


@dataclass(frozen=True)
class DataObjectBlock(SimulationDataBlock):
    """Immutable class with output simulation data for object body (:py:class:`rostok.block_builder.node_render.ChronoBodyEnv`)

        Attr:
            id_block (int): id of object block. Object id is less zero
            time (list[float]): time array of simulation
            sum_contact_forces (list[float]): list of summary contact forces
            abs_coord_COG (list[float]): list of absolute coordinates of block
            amount_contact_surfaces (list[int]): list of number of contact surfaces
    """
    obj_contact_forces: list[float]
    obj_amount_surf_forces: list[float]


"""Type for output simulation. Store trajectory and block id"""
SimOut = dict[int, SimulationDataBlock]

# Class for simulation system in loop optimization control


class SimulationStepOptimization:
    """Wrapper class of simulation robot with control on `pychrono <https://projectchrono.org/pychrono/>`_ physical engine.

    Before starting the simulation with the :py:meth:`SimulationStepOptimization.simulate_system` method, recommended set stopping flag
        of simulation with :py:meth:`SimulationStepOptimization.set_flags_stop_simulation`.
    Otherwise, the simulation time has no limits and lasts until the end of the program or until
    the visualization window is closed, if it is set.
    There are flags in module :py:mod:`flags_simualtions`

    Args:
        control_trajectory: Array arries of control trajectory for each joints.
            Control trajectory of one joint have to have format [[time, value], ...].
            Array must be same shape as array of joints.
        graph_mechanism (GraphGrammar): Graph of the robot to be simulated
        grasp_object (BlockWrapper): Wrapper of :py:class:`ChronoBodyEnv`.
            This is the object that the robot grabs.
    """

    def __init__(self, control_trajectory, graph_mechanism: GraphGrammar,
                 grasp_object: BlockWrapper):
        self.control_trajectory = control_trajectory
        self.graph_mechanism = graph_mechanism
        self.controller_joints = []

        # Create instance of chrono system and robot: grab mechanism
        self.chrono_system = chrono.ChSystemNSC()
        self.chrono_system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.chrono_system.SetSolverMaxIterations(100)
        self.chrono_system.SetSolverForceTolerance(1e-6)
        self.chrono_system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)

        self.grasp_object = grasp_object.create_block(self.chrono_system)

        self.grab_robot = Robot(self.graph_mechanism, self.chrono_system)

        # Add grasp object in system and set system without gravity
        self.chrono_system.Set_G_acc(chrono.ChVectorD(0, 0, 0))

        self.bind_trajectory(self.control_trajectory)
        self.fix_robot_base()

    def fix_robot_base(self):
        # Fixation palm of grab mechanism
        ids_blocks = list(self.grab_robot.block_map.keys())
        base_id = self.graph_mechanism.closest_node_to_root(ids_blocks)
        self.grab_robot.block_map[base_id].body.SetBodyFixed(True)

    def bind_trajectory(self, control_trajectory):
        """Create the controller joint from the control trajectory

        Args:
            control_trajectory (list): Array arries of control trajectory for each joints. Control
                trajectory of one joint have to have format [[time, value], ...].
                Array must be same shape as array of joints.

        Raises:
            IndexError: Exception in the case of a mismatch between the control and the joints
        """
        try:
            for id_finger, finger in enumerate(self.grab_robot.get_joints):
                finger_controller = []
                for id_joint, joint in enumerate(finger):
                    finger_controller.append(control.TrackingControl(joint))
                    finger_controller[-1].set_des_positions(control_trajectory[id_finger][id_joint])
                self.controller_joints.append(finger_controller)
        except IndexError:
            raise IndexError("Arries control and joints aren't same shape")

    # Setter flags of stop simulation
    def set_flags_stop_simulation(self, flags_stop_simulation: list[FlagStopSimualtions]):
        """Setter flags of stop simulation

        Args:
            flags_stop_simulation (list[FlagStopSimualtions]): List of desired checking flags.
                You can see flags in module :py:mod:`flags_simualtions`.
                Or create yours, that subclasses :py:class:`FlagStopSimualtions`
        """
        self.condion_stop_simulation = ConditionStopSimulation(self.chrono_system, self.grab_robot,
                                                               self.grasp_object,
                                                               flags_stop_simulation)

    # Add peculiar parameters of chrono system. Like that {"Set_G_acc":chrono.ChVectorD(0,0,0)}
    def change_config_system(self, dict_config: dict):
        """The method is for changing parameters of `pychrono <https://projectchrono.org/pychrono/>`_ simulation system.

        Args:
            dict_config (dict): The dictionary which contains the configuration parameters
                simulation system. The format is {"<name_of_method>": <method_arguments>}.
                For example, {"Set_G_acc":chrono.ChVectorD(0,0,0)}

        Raises:
            AttributeError: Exception in case the system does not have a specified method
        """
        for str_method, input in dict_config.items():
            try:
                metod_system = getattr(self.chrono_system, str_method)
                metod_system(input)
            except AttributeError:
                raise AttributeError(
                    "Chrono system doesn't have method {0}".format(str_method))


    # Run simulation
    def simulate_system(self, time_step, visualize=False) -> SimOut:
        """Start the simulation and return data from it

        Args:
            time_step (float): Width of time step the simulation
            visualize (bool, optional): Flag to enable visualization. Defaults to False.

        Returns:
            SimOut(dict[int, SimulationDataBlock]): Dictionary of simulation data of each block.
            The keys are block id, and the values dataclasses with data.
        """

        def append_arr_in_dict(x, y):
            if x[0] == y[0]:
                return (y[0], y[1] + [x[1]])

        FRAME_STEP = 1 / 30
        if visualize:
            # 30 fps

            vis = chronoirr.ChVisualSystemIrrlicht()
            vis.AttachSystem(self.chrono_system)
            vis.SetWindowSize(1024, 768)
            vis.SetWindowTitle('Grab demo')
            vis.Initialize()
            vis.AddCamera(chrono.ChVectorD(1.5, 3, -2))
            vis.AddTypicalLights()
            vis.EnableCollisionShapeDrawing(True)

        # Initilize temporarily dictionary of arries output data
        arrays_simulation_data_time = []
        arrays_simulation_data_joint_angle = map(
            lambda x: (x[0], []),
            filter(lambda x: isinstance(x[1], ChronoRevolveJoint),
                   self.grab_robot.block_map.items()))

        arrays_simulation_data_sum_contact_forces = map(
            lambda x: (x[0], []),
            filter(lambda x: isinstance(x[1], RobotBody), self.grab_robot.block_map.items()))

        arrays_simulation_data_abs_coord_COG = map(
            lambda x: (x[0], []),
            filter(lambda x: isinstance(x[1], RobotBody), self.grab_robot.block_map.items()))

        arrays_simulation_data_amount_contact_surfaces = map(
            lambda x: (x[0], []),
            filter(lambda x: isinstance(x[1], RobotBody), self.grab_robot.block_map.items()))

        arrays_simulation_data_obj_force = [(-1, [])]
        arrays_simulation_data_amount_obj_contact_surfaces = [(-1, [])]

        # Loop of simulation
        while not self.condion_stop_simulation.flag_stop_simulation():
            # while vis.Run():
            self.chrono_system.Update()
            self.chrono_system.DoStepDynamics(time_step)
            # Realtime for fixed step

            if self.chrono_system.GetStepcount() % int(FRAME_STEP / time_step) == 0:
                if visualize:

                    vis.Run()
                    vis.BeginScene(True, True, chrono.ChColor(0.1, 0.1, 0.1))
                    vis.Render()
                    vis.EndScene()

            arrays_simulation_data_time.append(self.chrono_system.GetChTime())

            # Get current variables from robot blocks
            current_data_joint_angle = RobotSensor.joints_angle(self.grab_robot)
            current_data_amount_contact_surfaces = RobotSensor.amount_contact_surfaces_blocks(
                self.grab_robot)
            current_data_sum_contact_forces = RobotSensor.sum_contact_forces_blocks(self.grab_robot)
            current_data_abs_coord_COG = RobotSensor.abs_coord_COG_blocks(self.grab_robot)
            current_data_std_obj_force = RobotSensor.std_contact_forces_object(self.grasp_object)

            current_data_amount_obj_contact_surfaces = dict([
                (-1, len([item for item in self.grasp_object.list_n_forces if item != 0]))
            ])
            # Append current data in output arries
            arrays_simulation_data_joint_angle = list(
                map(append_arr_in_dict, current_data_joint_angle.items(),
                    arrays_simulation_data_joint_angle))

            arrays_simulation_data_sum_contact_forces = list(
                map(append_arr_in_dict, current_data_sum_contact_forces.items(),
                    arrays_simulation_data_sum_contact_forces))

            arrays_simulation_data_abs_coord_COG = list(
                map(append_arr_in_dict, current_data_abs_coord_COG.items(),
                    arrays_simulation_data_abs_coord_COG))

            arrays_simulation_data_amount_contact_surfaces = list(
                map(append_arr_in_dict, current_data_amount_contact_surfaces.items(),
                    arrays_simulation_data_amount_contact_surfaces))

            if current_data_std_obj_force is not None:
                arrays_simulation_data_obj_force = map(append_arr_in_dict,
                                                       current_data_std_obj_force.items(),
                                                       arrays_simulation_data_obj_force)
            arrays_simulation_data_amount_obj_contact_surfaces = map(
                append_arr_in_dict, current_data_amount_obj_contact_surfaces.items(),
                arrays_simulation_data_amount_obj_contact_surfaces)

        if visualize:
            vis.GetDevice().closeDevice()

        # Create instance output data and add in dictionary
        simulation_data_joint_angle: dict[int, DataJointBlock] = dict(
            map(lambda x: (x[0], DataJointBlock(x[0], arrays_simulation_data_time, x[1])),
                arrays_simulation_data_joint_angle))
        simulation_data_body: dict[int, DataBodyBlock] = dict(
            map(
                lambda x, y, z:
                (x[0], DataBodyBlock(x[0], arrays_simulation_data_time, x[1], y[1], z[1])),
                arrays_simulation_data_sum_contact_forces, arrays_simulation_data_abs_coord_COG,
                arrays_simulation_data_amount_contact_surfaces))

        simulation_data_object: dict[int, DataObjectBlock] = dict(
            map(lambda x, y: (x[0], DataObjectBlock(x[0], arrays_simulation_data_time, x[1], y[1])),
                arrays_simulation_data_obj_force,
                arrays_simulation_data_amount_obj_contact_surfaces))
        simulation_data_joint_angle.update(simulation_data_body)
        simulation_data_joint_angle.update(simulation_data_object)

        return simulation_data_joint_angle
