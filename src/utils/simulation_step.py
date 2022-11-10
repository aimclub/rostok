import sys
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
from utils.nodes_division import *
sys.path.append("...app")
import reward_grab_mechanism

import numpy as np



# Immutable classes with output simulation data for robot block
@dataclass(frozen=True)
class SimulationDataBlock:
    id_block: int
    time: list[float]


@dataclass(frozen=True)
class DataJointBlock(SimulationDataBlock):
    angle_list: list[float]


@dataclass(frozen=True)
class DataBodyBlock(SimulationDataBlock):
    sum_contact_forces: list[float]
    abs_coord_COG: list [list[float]]
    amount_contact_surfaces: list[int]

# Class for simulation system in loop optimization control

# TODO: Bind traj into separate method
# TODO: Update data container into separate method
# TODO: Optional base fixation 
# TODO: Move methods to utils

class SimulationStepOptimization:
    def __init__(self, graph_mechanism: GraphGrammar, grasp_object: chrono.ChBody):
        self.control_trajectory = None
        self.graph_mechanism = graph_mechanism
        self.grasp_object = grasp_object
        self.controller_joints = []


        # Create instance of chrono system and robot: grab mechanism
        self.chrono_system = chrono.ChSystemNSC()
        self.chrono_system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.chrono_system.SetSolverMaxIterations(100)
        self.chrono_system.SetSolverForceTolerance(1e-6)
        self.chrono_system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        
        self.grab_robot = Robot(self.graph_mechanism, self.chrono_system)

        # self.chrono_system = chrono.ChSystemSMC()
        # self.chrono_system.SetSolverType(chrono.ChSolver.Type_MINRES)
        # self.chrono_system.SetSolverForceTolerance(1e-10)
        # self.chrono_system.SetSolverMaxIterations(100)
        # timestepper = chrono.ChTimestepperHHT(self.chrono_system)
        # self.chrono_system.SetTimestepper(timestepper)
        # timestepper.SetAbsTolerances(1e-5)
        # timestepper.SetScaling(True)
        # timestepper.SetStepControl(True)
        # timestepper.SetMinStepSize(1e-4)
        # timestepper.SetAlpha(-0.2)
        # timestepper.SetMaxiters(5)

    
        self.grab_robot = Robot(self.graph_mechanism, self.chrono_system)   
        joints = np.array(self.grab_robot.get_joints)   
        
        list_J = reward_grab_mechanism.list_J
        list_RM = reward_grab_mechanism.list_RM
        list_LM = reward_grab_mechanism.list_LM
        list_B = reward_grab_mechanism.list_B
        list_Palm = reward_grab_mechanism.list_Palm
        
        self.J_NODES_NEW = nodes_division(self.grab_robot, list_J)
        self.B_NODES_NEW = nodes_division(self.grab_robot, list_B)
        self.RB_NODES_NEW = sort_left_right(self.grab_robot, list_RM, list_B)
        self.LB_NODES_NEW = sort_left_right(self.grab_robot, list_LM, list_B)
        self.RJ_NODES_NEW = sort_left_right(self.grab_robot, list_RM, list_J)
        self.LJ_NODES_NEW = sort_left_right(self.grab_robot, list_LM, list_J)
        
        RB_blocks = [self.B_NODES_NEW[0].block]
        LB_blocks = [self.B_NODES_NEW[0].block]
        RJ_blocks = []
        
        PALM_blocks = self.B_NODES_NEW[0].block
        
        for i in range(len(self.RB_NODES_NEW)):
            for j in range(len(self.RB_NODES_NEW[i])):
                RB_blocks.append(self.RB_NODES_NEW[i][j].block)

        for i in range(len(self.LB_NODES_NEW)):
            for j in range(len(self.LB_NODES_NEW[i])):
                LB_blocks.append(self.LB_NODES_NEW[i][j].block)

        for i in range(len(self.RJ_NODES_NEW)):
            for j in range(len(self.RJ_NODES_NEW[i])):
                RJ_blocks.append(self.RJ_NODES_NEW[i][j].block)
        
        for m in range(6):
            if m == 0:
                traj_controller = np.array(np.mat('0 0.3 0.6 0.9 1.2 2; 0.5 0.5 0.5 0.5 0.5 0.5')) #Format: [Time; Value].
                traj_controller_inv = np.array(np.mat('0 0.3 0.6 0.9 1.2 2; -0.5 -0.5 -0.5 -0.5 -0.5 -0.5')) #Format: [Time; Value].
            elif m != 1:
                traj_controller[1,:] *=2
                traj_controller_inv[1,:] *=2
                print(traj_controller)
                
            arr_traj = []
            for ind, finger in enumerate(joints):
                arr_finger_traj = []
                for i, joint in enumerate(finger):
                    if joint in RJ_blocks:
                        arr_finger_traj.append(traj_controller)
                    else:
                        arr_finger_traj.append(traj_controller_inv)
                arr_traj.append(arr_finger_traj)
        
        self.control_trajectory = arr_traj
        
        ids_blocks = list(self.grab_robot.block_map.keys())
        base_id = graph_mechanism.closest_node_to_root(ids_blocks)
        self.grab_robot.block_map[base_id].body.SetBodyFixed(True)
        
        # blocks = self.grab_robot.block_map.values()
        # body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
        # make_collide(body_block, CollisionGroup.ROBOT)
        make_collide(RB_blocks, CollisionGroup.RIGHT_SIDE_PALM, disable_gproup=[CollisionGroup.PALM])
        make_collide(LB_blocks, CollisionGroup.LEFT_SIDE_PALM, disable_gproup=[CollisionGroup.PALM])
        
        self.chrono_system.Add(self.grasp_object)
        self.chrono_system.Set_G_acc(chrono.ChVectorD(0,0,0))
        
        self.controller_joints = []
        try:
            for id_finger, finger in enumerate(self.grab_robot.get_joints):
                for id_joint, joint in enumerate(finger):
                    self.controller_joints.append(
                        control.TrackingControl(joint))
                    self.controller_joints[-1].set_des_positions(
                    self.control_trajectory[id_finger][id_joint])
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
        # while not self.condion_stop_simulation.flag_stop_simulation():
        while self.chrono_system.GetChTime() < 5:
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
