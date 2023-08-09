import numpy as np
import matplotlib.pyplot as plt
import pychrono as chrono
import pychrono.core as chrono
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph
from rostok.virtual_experiment.sensors import (SensorCalls, SensorObjectClassification)
from rostok.block_builder_api.block_parameters import Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.easy_body_shapes import Box, Sphere
from rostok.simulation_chrono.basic_simulation import RobotSimulationChrono
from rostok.block_builder_chrono.block_builder_chrono_api import ChronoBlockCreatorInterface as creator
import pychrono.irrlicht as chronoirr
import pychrono as chrono
from rostok.control_chrono.controller import ConstController, ForceControllerTemplate, ForceTorque
from rostok.block_builder_api.block_blueprints import TransformBlueprint, PrimitiveBodyBlueprint, RevolveJointBlueprint
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.block_builder_api.block_parameters import JointInputType
from rostok.graph_grammar.node import GraphGrammar, Node

STEPS = 1000 * 1
VIS = True

TENDON_FORCE = 0
VINOS = -0.03
PULLEY_POS = 0.03
CABEL_MOUNT_POS = 0.04

LINK_LENGTH = 0.1
WIDTH = 0.02
HEIGHT = 0.02
DENSITY = 1300 * 0.5
STARTING_ANGLE = -45
NOT_STIFNESS = False

def create_one_finger():
    mat = Material()

    palm = PrimitiveBodyBlueprint(Box(0.3, 0.02, 0.3), material=mat)
    finger_offset = TransformBlueprint(
        transform=FrameTransform(position=[0.09, 0, 0], rotation=[1, 0, 0, 0]))
    link_1 = PrimitiveBodyBlueprint(shape=Box(width_x=WIDTH, length_y=LINK_LENGTH, height_z=HEIGHT),
                                    density=DENSITY,
                                    material=mat)
    link_2 = PrimitiveBodyBlueprint(shape=Box(width_x=WIDTH, length_y=LINK_LENGTH, height_z=HEIGHT),
                                    density=DENSITY,
                                    material=mat)
    link_3 = PrimitiveBodyBlueprint(shape=Box(width_x=WIDTH, length_y=LINK_LENGTH, height_z=HEIGHT),
                                    density=DENSITY,
                                    material=mat)
    revolve1 = RevolveJointBlueprint(JointInputType.TORQUE,
                                     0.02,
                                     HEIGHT,
                                     material=mat,
                                     density=10,
                                     stiffness=0.12 * 4,
                                     damping=0.001,
                                     starting_angle=STARTING_ANGLE,
                                     equilibrium_position=-0.785398 / 4)
    revolve2 = RevolveJointBlueprint(JointInputType.TORQUE,
                                     0.02,
                                     HEIGHT,
                                     material=mat,
                                     density=10,
                                     stiffness=0.05 * 4,
                                     damping=0.001,
                                     equilibrium_position=(-0.785398 / 2) / 4)
    revolve3 = RevolveJointBlueprint(JointInputType.TORQUE,
                                     0.02,
                                     HEIGHT,
                                     material=mat,
                                     density=10,
                                     stiffness=0.05 * 4,
                                     damping=0.001,
                                     equilibrium_position=(-0.785398 / 2) / 4)
    
    if NOT_STIFNESS:
        revolve1 = RevolveJointBlueprint(JointInputType.TORQUE,
                                    0.02,
                                    HEIGHT,
                                    material=mat,
                                    density=10,                                 
                                    damping=0.001,
                                    starting_angle=STARTING_ANGLE)
        revolve2 = RevolveJointBlueprint(JointInputType.TORQUE,
                                    0.02,
                                    HEIGHT,
                                    material=mat,
                                    density=10,                                 
                                    damping=0.001)
        revolve3 = RevolveJointBlueprint(JointInputType.TORQUE,
                                    0.02,
                                    HEIGHT,
                                    material=mat,
                                    density=10,                                 
                                    damping=0.001)

    palm_node = Node("P1", True, palm)
    link_node_1 = Node("L1", True, link_1)
    link_node_2 = Node("L2", True, link_2)
    link_node_3 = Node("L3", True, link_3)
    revolve_node_1 = Node("J1", True, revolve1)
    revolve_node_2 = Node("J2", True, revolve2)
    revolve_node_3 = Node("J3", True, revolve3)
    finger_offset_node = Node("T1", True, finger_offset)

    mech_graph = GraphGrammar()
    mech_graph.remove_node(0)
    mech_graph.add_node(1, Node=palm_node)
    mech_graph.add_node(2, Node=finger_offset_node)
    mech_graph.add_node(3, Node=revolve_node_1)
    mech_graph.add_node(4, Node=link_node_1)
    mech_graph.add_node(5, Node=revolve_node_2)
    mech_graph.add_node(6, Node=link_node_2)
    mech_graph.add_node(7, Node=revolve_node_3)
    mech_graph.add_node(8, Node=link_node_3)
    mech_graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)])
    return mech_graph


class TendonForce(ForceControllerTemplate):

    def __init__(self, pos: list) -> None:
        super().__init__()
        self.set_vector_in_local_cord()
        self.pos = pos

    def get_force_torque(self, time: float, data) -> ForceTorque:
        force = data["Force"]
        angle = data["Angle"]
        impact = ForceTorque()
        x_force = -2 * np.sin(angle + 0.001) * force
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
        # y_force = -force * np.cos(ANGLE * np.pi / 180 )
        # x_force = -force *  np.sin(ANGLE * np.pi / 180 )
        # impact.force = (x_force, y_force, 0)
        y_force = -force
        impact.force = (0, y_force, 0)
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
        pass


def convert_angle(angle: float):
    return -angle


def rotation_x(alpha):
    quat_X_ang_alpha = chrono.Q_from_AngX(np.deg2rad(alpha))
    return [quat_X_ang_alpha.e0, quat_X_ang_alpha.e1, quat_X_ang_alpha.e2, quat_X_ang_alpha.e3]


graph = create_one_finger()

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

telo0 = sim.robot.get_graph().body_map_ordered[1].body
telo1 = sim.robot.get_graph().body_map_ordered[4].body
telo2 = sim.robot.get_graph().body_map_ordered[6].body
telo3 = sim.robot.get_graph().body_map_ordered[8].body
 
weighted_m = 0.05 * telo1.GetMass() + 0.15 * telo2.GetMass() + 0.2 * telo3.GetMass()
total_m = telo1.GetMass() + telo2.GetMass() + telo3.GetMass()
CoG_finger = weighted_m / total_m
torque_on_base_j = 9.8 * weighted_m
stifness = torque_on_base_j / np.deg2rad(45)

j1 = sim.robot.get_graph().joint_map_ordered[3].joint
j2 = sim.robot.get_graph().joint_map_ordered[5].joint
j3 = sim.robot.get_graph().joint_map_ordered[7].joint

force_tendon11 = TendonForce([VINOS, -PULLEY_POS, 0])
force_tendon12 = TendonForce([VINOS, PULLEY_POS, 0])
force_tendon21 = TendonForce([VINOS, -PULLEY_POS, 0])
force_tendon22 = TendonForce([VINOS, PULLEY_POS, 0])
force_tendon31 = TendonForce([VINOS, -PULLEY_POS, 0])
force_tip = TendonForceTip([VINOS, CABEL_MOUNT_POS, 0])

force_tendon11.bind_body(telo1)
force_tendon12.bind_body(telo1)
force_tendon21.bind_body(telo2)
force_tendon22.bind_body(telo2)
force_tendon31.bind_body(telo3)
force_tip.bind_body(telo3)

forces = [
    force_tip, force_tendon11, force_tendon12, force_tendon21, force_tendon22, force_tendon31,
    force_tip
]

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
    vis.AddCamera(chrono.ChVectorD(-0.51, 0.51, -0.51))
    vis.AddTypicalLights()
    #vis.EnableCollisionShapeDrawing(True)

for i in range(STEPS):
    current_time = sim.chrono_system.GetChTime()
    sim.simulate_step(0.0005, current_time, i)

    angle_z_1 = j1.GetMotorRot()
    angle_z_2 = j2.GetMotorRot()
    angle_z_3 = j3.GetMotorRot()
    tendon_ang_11 = angle_z_1 / 2
    tendon_ang_22 = angle_z_2 / 2
    tendon_ang_3 = angle_z_3 / 2

    data_11 = {"Angle": (angle_z_1 + STARTING_ANGLE * np.pi / 180) / 2  , "Force": TENDON_FORCE}
    data_12 = {"Angle": tendon_ang_22, "Force": TENDON_FORCE}
    data_21 = {"Angle": tendon_ang_22, "Force": TENDON_FORCE}
    data_22 = {"Angle": tendon_ang_3, "Force": TENDON_FORCE}
    data_31 = {"Angle": tendon_ang_3, "Force": TENDON_FORCE}
    data_tip = {"Angle": tendon_ang_3, "Force": TENDON_FORCE}
    all_data = [data_11, data_12, data_21, data_22, data_31, data_tip]
    for a, b in zip(forces, all_data):
        a.update(0, b)
        pass

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

pos1 = sim.result.robot_final_ds.main_storage["pos"][4]
pos2 = sim.result.robot_final_ds.main_storage["pos"][6]
pos3 = sim.result.robot_final_ds.main_storage["pos"][8]

j_pos1 = np.array(sim.result.robot_final_ds.main_storage["j_pos"][3])
j_pos2 = np.array(sim.result.robot_final_ds.main_storage["j_pos"][5])
j_pos3 = np.array(sim.result.robot_final_ds.main_storage["j_pos"][7])

contact1 = np.array(sim.result.robot_final_ds.main_storage["contact"][4])
contact2 = np.array(sim.result.robot_final_ds.main_storage["contact"][6])
contact3 = np.array(sim.result.robot_final_ds.main_storage["contact"][8])

all_contact = contact1 + contact2 + contact3

np_pos1 = np.array(pos1)[:, 0:2]
np_pos2 = np.array(pos2)[:, 0:2]
np_pos3 = np.array(pos3)[:, 0:2]

plt.figure()
plt.plot(np_pos1[:, 0], np_pos1[:, 1])
plt.plot(np_pos2[:, 0], np_pos2[:, 1])
plt.plot(np_pos3[:, 0], np_pos3[:, 1])

plt.figure()
plt.plot(j_pos1 * 180 / np.pi)
plt.plot(j_pos2 * 180 / np.pi)
plt.plot(j_pos3 * 180 / np.pi)

plt.figure()
plt.plot(all_contact)
plt.show()
pass