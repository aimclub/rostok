from copy import deepcopy
from dataclasses import dataclass
from engine.node import GraphGrammar
from engine.node_render import ChronoBody, ChronoRevolveJoint
from utils.auxilarity_sensors import RobotSensor
from utils.blocks_utils import make_collide, CollisionGroup
from utils.flags_simualtions import ConditionStopSimulation, FlagStopSimualtions
import pychrono as chrono
import pychrono.irrlicht as chronoirr
from engine.robot import Robot
import engine.control as control



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
    """Immutable class with output simulation data for robot joint block.

        Attr:
            id_block (int): id of robot block
            time (list[float]): list of time of simulation
            angle_list (list[float]): list of angle of robot joint block
    """
    angle_list: list[float]


@dataclass(frozen=True)
class DataBodyBlock(SimulationDataBlock):
    """Immutable class with output simulation data for robot body block.

        Attr:
            id_block (int): id of robot block
            time (list[float]): list of time of simulation
            sum_contact_forces (list[float]): list of contact forces sum
            abs_coord_COG (list[float]): list of absolute coordinates of block
            amount_contact_surfaces (list[int]): list of number of contact surfaces
    """
    sum_contact_forces: list[float]
    abs_coord_COG: list [list[float]]
    amount_contact_surfaces: list[int]

# Class for simulation system in loop optimization control

# TODO: Bind traj into separate method
# TODO: Update data container into separate method
# TODO: Optional base fixation 
# TODO: Move methods to utils

class SimulationStepOptimization:
    def __init__(self, control_trajectory, graph_mechanism: GraphGrammar, grasp_object: chrono.ChBody):
        self.control_trajectory = control_trajectory
        self.graph_mechanism = graph_mechanism
        self.grasp_object = grasp_object
        self.controller_joints = []


        # Create instance of chrono system and robot: grab mechanism
        self.chrono_system = chrono.ChSystemNSC()
        self.grab_robot = Robot(self.graph_mechanism, self.chrono_system)

    
        # Create familry collision for robot
        blocks = self.grab_robot.block_map.values()
        body_block = filter(lambda x: isinstance(x, ChronoBody), blocks)
        make_collide(body_block, CollisionGroup.Robot)


        # Add grasp object in system and set system without gravity
        self.chrono_system.Add(self.grasp_object)
        self.chrono_system.Set_G_acc(chrono.ChVectorD(0, 0, 0))


        self.bind_trajectory(self.control_trajectory)
        self.fix_robot_base()

    def fix_robot_base(self):
        # Fixation palm of grab mechanism
        ids_blocks = list(self.grab_robot.block_map.keys())
        base_id = self.graph_mechanism.closest_node_to_root(ids_blocks)
        self.grab_robot.block_map[base_id].body.SetBodyFixed(True)

    def bind_trajectory(self, control_trajectory):
        # Create the controller joint from the control trajectory
        try:
            for id_finger, finger in enumerate(self.grab_robot.get_joints):
                for id_joint, joint in enumerate(finger):
                    self.controller_joints.append(
                        control.TrackingControl(joint))
                    self.controller_joints[-1].set_des_positions(
                        control_trajectory[id_finger][id_joint])
        except IndexError:
            raise IndexError("Arries control and joints aren't same shape")

    # Setter flags of stop simulation
    def set_flags_stop_simulation(self, flags_stop_simulation: list[FlagStopSimualtions]):

        self.condion_stop_simulation = ConditionStopSimulation(self.chrono_system,
                                                               self.grab_robot,
                                                               self.grasp_object,
                                                               flags_stop_simulation)

    # Add peculiar parameters of chrono system. Like that {"Set_G_acc":chrono.ChVectorD(0,0,0)}
    def change_config_system(self, dict_config: dict):
        for str_method, input in dict_config.items():
            try:
                metod_system = getattr(self.chrono_system, str_method)
                metod_system(input)
            except AttributeError:
                raise AttributeError(
                    "Chrono system don't have method {0}".format(str_method))

    # Run simulation
    def simulate_system(self, time_step, visualize=False) -> dict[int, SimulationDataBlock]:
        # Function appending arraies in map
        def append_arr_in_dict(x, y):
            if x[0] == y[0]:
                return (y[0], y[1] + [x[1]])

        if visualize:
            vis = chronoirr.ChVisualSystemIrrlicht()
            vis.AttachSystem(self.chrono_system)
            vis.SetWindowSize(1024, 768)
            vis.SetWindowTitle('Grab demo')
            vis.Initialize()
            vis.AddCamera(chrono.ChVectorD(3, 3, -3))
            vis.AddTypicalLights()

        # Initilize temporarily dictionary of arries output data
        arrays_simulation_data_time = []
        arrays_simulation_data_joint_angle = map(lambda x: (x[0], []),
                                                 filter(lambda x: isinstance(x[1], ChronoRevolveJoint),
                                                        self.grab_robot.block_map.items()))

        arrays_simulation_data_sum_contact_forces = map(lambda x: (x[0], []),
                                                        filter(lambda x: isinstance(x[1], ChronoBody),
                                                               self.grab_robot.block_map.items()))

        arrays_simulation_data_abs_coord_COG = map(lambda x: (x[0], []),
                                                   filter(lambda x: isinstance(x[1], ChronoBody),
                                                          self.grab_robot.block_map.items()))

        arrays_simulation_data_amount_contact_surfaces = map(lambda x: (x[0], []),
                                                             filter(lambda x: isinstance(x[1], ChronoBody),
                                                                    self.grab_robot.block_map.items()))

        # Loop of simulation
        while not self.condion_stop_simulation.flag_stop_simulation():
            self.chrono_system.Update()
            self.chrono_system.DoStepDynamics(time_step)

            if visualize:
                vis.BeginScene(True, True, chrono.ChColor(0.2, 0.2, 0.3))
                vis.Render()
                vis.EndScene()

            arrays_simulation_data_time.append(self.chrono_system.GetChTime())

            # Get current variables from robot blocks
            current_data_joint_angle = RobotSensor.joints_angle(
                self.grab_robot)
            current_data_amount_contact_surfaces = RobotSensor.amount_contact_surfaces_blocks(
                self.grab_robot)
            current_data_sum_contact_forces = RobotSensor.sum_contact_forces_blocks(
                self.grab_robot)
            current_data_abs_coord_COG = RobotSensor.abs_coord_COG_blocks(
                self.grab_robot)

            # Append current data in output arries
            arrays_simulation_data_joint_angle = map(append_arr_in_dict,
                                                     current_data_joint_angle.items(),
                                                     arrays_simulation_data_joint_angle)

            arrays_simulation_data_sum_contact_forces = map(append_arr_in_dict,
                                                            current_data_sum_contact_forces.items(),
                                                            arrays_simulation_data_sum_contact_forces)

            arrays_simulation_data_abs_coord_COG = map(append_arr_in_dict,
                                                       current_data_abs_coord_COG.items(),
                                                       arrays_simulation_data_abs_coord_COG)

            arrays_simulation_data_amount_contact_surfaces = map(append_arr_in_dict,
                                                                 current_data_amount_contact_surfaces.items(),
                                                                 arrays_simulation_data_amount_contact_surfaces)
        if visualize:
            vis.GetDevice().closeDevice()

        # Create instance output data and add in dictionary
        simulation_data_joint_angle: dict[int, DataJointBlock] = dict(map(lambda x: (x[0], DataJointBlock(x[0], arrays_simulation_data_time, x[1])),
                                                                          arrays_simulation_data_joint_angle))
        simulation_data_body: dict[int, DataBodyBlock] = dict(map(lambda x, y, z: (x[0], DataBodyBlock(x[0], arrays_simulation_data_time, x[1], y[1], z[1])),
                                                                  arrays_simulation_data_sum_contact_forces,
                                                                  arrays_simulation_data_abs_coord_COG,
                                                                  arrays_simulation_data_amount_contact_surfaces))
        simulation_data_joint_angle.update(simulation_data_body)

        return simulation_data_joint_angle
