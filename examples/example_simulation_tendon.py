import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pychrono as chrono
import pychrono.core as chrono
from scipy.spatial import distance
from example_vocabulary import (get_terminal_graph_three_finger, get_terminal_graph_two_finger,
                                get_terminal_graph_no_joints)
from example_vocabulary import get_terminal_graph_two_finger
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from rostok.virtual_experiment.sensors import (SensorCalls, SensorObjectClassification)
from rostok.block_builder_api.block_parameters import Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.easy_body_shapes import Box, Sphere 
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono
from rostok.block_builder_chrono.block_builder_chrono_api import ChronoBlockCreatorInterface as creator
from rostok.library.rule_sets.simple_designs import get_three_link_one_finger
import pychrono.irrlicht as chronoirr
import pychrono as chrono
import time
from rostok.graph_grammar.graph_utils import plot_graph, plot_graph_ids
from rostok.control_chrono.controller import ConstController, ForceControllerTemplate, ForceTorque, SinControllerChrono, YaxisShaker


class TendonForce(ForceControllerTemplate):
    def __init__(self, pos: list) -> None:
        super().__init__()
        self.set_vector_in_local_cord()
        self.pos = pos

    def get_force_torque(self, time: float, data) -> ForceTorque:
        force = data["Force"]
        angle = data["Angle"]
        impact = ForceTorque()
        x_force = -2 * np.sin(angle+0.001) * force 
        if angle < 0:
            x_force = 0
        impact.force = (x_force, 0, 0)
        return impact
        
    def set_vector_in_local_cord(self):
        self.force_maker_chrono.SetAlign(chrono.ChForce.BODY_DIR)
        self.torque_maker_chrono.SetAlign(chrono.ChForce.BODY_DIR)
    
    def bind_body(self, body: chrono.ChBody):
        super().bind_body(body)
        self.force_maker_chrono.SetVrelpoint(chrono.ChVectorD(*self.pos))
        sph_1 = chrono.ChSphereShape(0.005)
        sph_1.SetColor(chrono.ChColor(1, 0, 0))
        body.AddVisualShape(sph_1, chrono.ChFrameD(chrono.ChVectorD(*self.pos)))
        body.GetVisualShape(0).SetOpacity(0.6)
    

class TendonForceTip(ForceControllerTemplate):
    def __init__(self, pos: list) -> None:
        super().__init__()
        self.set_vector_in_local_cord()
        self.pos = pos

    def get_force_torque(self, time: float, data) -> ForceTorque:
        angle = data["Angle"]
        force = data["Force"]
        impact = ForceTorque()
        ANGLE = 30
        y_force = -force * np.cos(ANGLE * np.pi / 180 )
        x_force = -force *  np.sin(ANGLE * np.pi / 180 )
        impact.force = (x_force, y_force, 0)
        return impact
        
    def set_vector_in_local_cord(self):
        self.force_maker_chrono.SetAlign(chrono.ChForce.BODY_DIR)
        self.torque_maker_chrono.SetAlign(chrono.ChForce.BODY_DIR)
    
    def bind_body(self, body: chrono.ChBody):
        super().bind_body(body)
 
        self.force_maker_chrono.SetVrelpoint(chrono.ChVectorD(*self.pos))
        sph_1 = chrono.ChSphereShape(0.005)
        sph_1.SetColor(chrono.ChColor(1, 0, 0))
        body.AddVisualShape(sph_1, chrono.ChFrameD(chrono.ChVectorD(*self.pos)))
        body.GetVisualShape(0).SetOpacity(0.6)
 


def get_pure_angles(body: chrono.ChBody):
    quat = body.GetRot()
    angles = chrono.Q_to_Euler123(quat)
    return angles

def convert_angle(angle: float):
    return -angle


def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]


graph = get_three_link_one_finger()

print(get_joint_vector_from_graph(graph))

controll_parameters = {"initial_value": [0, 0, 0]}
#controll_parameters = {"initial_value": [0.1, 0.1, 0.1]}
print(controll_parameters)
sim = RobotSimulationChrono([])
 
mat = Material()
mat.Friction = 0.65
mat.DampingF = 0.65

obj = EnvironmentBodyBlueprint(shape=Sphere(0.05),
                               material=mat,
                               pos=FrameTransform([0.02, 0.22, 0], [1, 0, 0, 0]))

cnorono_obj = creator.init_block_from_blueprint(obj)


sim.add_object(cnorono_obj, is_fixed=True)

sim.add_design(graph,
               controll_parameters,
               control_cls=ConstController,
               Frame=FrameTransform([0, 0, 0], rotation_x(0)),
               is_fixed=True)
#


telo1 = sim.robot.get_graph().body_map_ordered[20].body
telo2 = sim.robot.get_graph().body_map_ordered[21].body
telo3 = sim.robot.get_graph().body_map_ordered[22].body


j1 = sim.robot.get_graph().joint_map_ordered[23].joint
j2 = sim.robot.get_graph().joint_map_ordered[24].joint
j3 = sim.robot.get_graph().joint_map_ordered[25].joint
STEPS = 1000*1
VIS = False
TENDON_FORCE = 0.5
VINOS = -0.04
force_tendon11 = TendonForce([VINOS, -0.04, 0])
force_tendon12 = TendonForce([VINOS, 0.04, 0])
force_tendon21 = TendonForce([VINOS, -0.04, 0])
force_tendon22 = TendonForce([VINOS, 0.04, 0])
force_tendon31 = TendonForce([VINOS, -0.04, 0])
force_tip = TendonForceTip([VINOS, 0.04, 0 ])


force_tendon11.bind_body(telo1)
force_tendon12.bind_body(telo1)
force_tendon21.bind_body(telo2)
force_tendon22.bind_body(telo2)
force_tendon31.bind_body(telo3)
force_tip.bind_body(telo3)

forces = [force_tip, force_tendon11, force_tendon12, force_tendon21,
          force_tendon22, force_tendon31, force_tip]
 

 
env_data_dict = {
    "pos": (SensorCalls.BODY_TRAJECTORY, SensorObjectClassification.BODY),
    "j_pos": (SensorCalls.JOINT_TRAJECTORY, SensorObjectClassification.JOINT),
    "contact": (SensorCalls.AMOUNT_FORCE, SensorObjectClassification.BODY)
}

#sim.add_env_data_type_dict(env_data_dict)
sim.add_robot_data_type_dict(env_data_dict)
sim.initialize(STEPS)
sim.chrono_system.Set_G_acc(chrono.ChVectorD(0, -9.8, 0))
#simot = sim.simulate(10000 * 4, 0.0005, 1, None, False)
 

if VIS:
    vis = chronoirr.ChVisualSystemIrrlicht()
    vis.AttachSystem(sim.chrono_system)
    vis.SetWindowSize(1024, 768)
    vis.SetWindowTitle('Grab demo')
    vis.Initialize()
    vis.AddCamera(chrono.ChVectorD(-0.51, -0.51, -0.51))
    vis.AddTypicalLights()
    vis.EnableCollisionShapeDrawing(True)
    
     
for i in range(STEPS):
    current_time = sim.chrono_system.GetChTime()
    sim.simulate_step(0.0005, current_time, i)
    
    angle_z_1 = j1.GetMotorRot()
    angle_z_2 = j2.GetMotorRot()
    angle_z_3 = j3.GetMotorRot()
    tendon_ang_11 = angle_z_1 / 2
    tendon_ang_22 = angle_z_2 / 2
    tendon_ang_3 = angle_z_3 / 2
 
    data_11 = {"Angle" : tendon_ang_11, "Force" : TENDON_FORCE}
    data_12 = {"Angle" : tendon_ang_22, "Force" : TENDON_FORCE}
    data_21 = {"Angle" : tendon_ang_22, "Force" : TENDON_FORCE}
    data_22 = {"Angle" : tendon_ang_3, "Force" : TENDON_FORCE}
    data_31 = {"Angle" : tendon_ang_3, "Force" : TENDON_FORCE}
    data_tip = {"Angle" : tendon_ang_3, "Force" : TENDON_FORCE}
    all_data = [data_11, data_12, data_21, data_22, data_31, data_tip]
    for a, b  in zip(forces, all_data):
        a.update(0,b)

    if VIS:
        vis.Run()
        vis.BeginScene(True, True, chrono.ChColor(0.1, 0.1, 0.1))
        vis.Render()
        vis.EndScene()
    
sim.result.environment_final_ds = sim.data_storage
sim.result.robot_final_ds = sim.robot.data_storage
sim.result.time = sim.chrono_system.GetChTime()
sim.n_steps = STEPS
sim.result.reduce_nan()


if VIS:
    vis.GetDevice().closeDevice()

pos1 = sim.result.robot_final_ds.main_storage["pos"][20]
pos2 = sim.result.robot_final_ds.main_storage["pos"][21]
pos3 = sim.result.robot_final_ds.main_storage["pos"][22]

j_pos1 = np.array(sim.result.robot_final_ds.main_storage["j_pos"][23])
j_pos2 = np.array(sim.result.robot_final_ds.main_storage["j_pos"][24])
j_pos3 = np.array(sim.result.robot_final_ds.main_storage["j_pos"][25])

contact1 = np.array(sim.result.robot_final_ds.main_storage["contact"][20])
contact2 = np.array(sim.result.robot_final_ds.main_storage["contact"][21])
contact3 = np.array(sim.result.robot_final_ds.main_storage["contact"][22])

all_contact = contact1 + contact2 + contact3



np_pos1 = np.array(pos1)[:, 0:2]
np_pos2 = np.array(pos2)[:, 0:2]
np_pos3 = np.array(pos3)[:, 0:2]

 
plt.figure()
plt.plot(np_pos1[:, 0], np_pos1[:, 1])
plt.plot(np_pos2[:, 0], np_pos2[:, 1])
plt.plot(np_pos3[:, 0], np_pos3[:, 1])

plt.figure()
plt.plot(-j_pos1*180/np.pi)
plt.plot(j_pos2*180/np.pi)
plt.plot(j_pos3*180/np.pi)

plt.figure()
plt.plot(all_contact)
plt.show()
pass