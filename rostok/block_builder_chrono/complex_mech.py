from rostok.graph_grammar.node import GraphGrammar, Node
from rostok.block_builder_api import block_blueprints
from rostok.simulation_chrono import basic_simulation
from rostok.block_builder_api.block_blueprints import easy_body_shapes
from rostok.block_builder_api.block_parameters import FrameTransform, JointInputType, Material
from rostok.block_builder_api.block_parameters import JointInputType
from rostok.graph_grammar.graph_utils import plot_graph

robot_material = Material()
WHEEL_WIDTH = 0.07
WHEEL_R = 0.13

RIGHT_WHEEL_T = FrameTransform([0, 0, WHEEL_WIDTH], [0.707, 0.707, 0, 0])
LEFT_WHEEL_T = FrameTransform([0, 0, -WHEEL_WIDTH], [0.707, -0.707, 0, 0])

SIDE = 0.41
RIGHT_SIDE = FrameTransform([0, -0.2, SIDE], [1, 0, 0, 0])
LEFT_SIDE = FrameTransform([0, -0.2, -SIDE], [1, 0, 0, 0])
BACK_HALF = FrameTransform([-0.3, 0, 0], [1, 0, 0, 0])
FRONT_HALF = FrameTransform([0.1, 0, 0], [1, 0, 0, 0])

KNEE_LENGTH = 0.3
FRONT_KNEE_LENGTH_HALF = FrameTransform([-KNEE_LENGTH / 2, 0, 0], [0.906, 0, 0, 0.423])
BACK_KNEE_LENGTH_HALF = FrameTransform([KNEE_LENGTH / 2, 0, 0], [0.906, 0, 0, -0.423])

long_link_bp = block_blueprints.PrimitiveBodyBlueprint(easy_body_shapes.Box(0.1, 0.5, 0.1),
                                                       color=[10, 150, 10])
long_link_front_bp = block_blueprints.PrimitiveBodyBlueprint(easy_body_shapes.Box(0.1, 0.6, 0.1),
                                                             color=[10, 150, 10])
body_bp = block_blueprints.PrimitiveBodyBlueprint(easy_body_shapes.Box(1, 0.2, 0.6),
                                                  color=[10, 80, 10])
knee_bp = block_blueprints.PrimitiveBodyBlueprint(easy_body_shapes.Box(KNEE_LENGTH, 0.1, 0.1),
                                                  color=[10, 150, 10])
wheel_bp = block_blueprints.PrimitiveBodyBlueprint(easy_body_shapes.Cylinder(WHEEL_R, WHEEL_WIDTH),
                                                   color=[1, 1, 1])

body_accum_bp = block_blueprints.PrimitiveBodyBlueprint(easy_body_shapes.Box(0.2, 0.1, 0.3),
                                                  density=50 ,color=[1, 1, 100])

transform_rwt_bp = block_blueprints.TransformBlueprint(RIGHT_WHEEL_T)
transform_lwt_bp = block_blueprints.TransformBlueprint(LEFT_WHEEL_T)

transform_left_side_bp = block_blueprints.TransformBlueprint(LEFT_SIDE)
transform_right_side_bp = block_blueprints.TransformBlueprint(RIGHT_SIDE)
transform_front_bp = block_blueprints.TransformBlueprint(FRONT_HALF)

transform_knee_front_bp = block_blueprints.TransformBlueprint(FRONT_KNEE_LENGTH_HALF)
transform_knee_back_bp = block_blueprints.TransformBlueprint(BACK_KNEE_LENGTH_HALF)

pasive_joint_bp = block_blueprints.RevolveJointBlueprint(JointInputType.UNCONTROL,
                                                         radius=0.08,
                                                         length=0.1,
                                                         starting_angle=-200)
pasive_joint_knee_bp = block_blueprints.RevolveJointBlueprint(JointInputType.UNCONTROL,
                                                              radius=0.06,
                                                              length=0.1,
                                                              starting_angle=70)
motor_joint_wheel_bp = block_blueprints.RevolveJointBlueprint(JointInputType.UNCONTROL,
                                                              radius=0.01,
                                                              length=0.01)

long_link = Node("L1", True, long_link_bp)
long_link_front = Node("L2", True, long_link_front_bp)
wheel = Node("W1", True, wheel_bp)
stiffness_joint = Node("JP1", True, pasive_joint_bp)
stiffness_knee_joint = Node("JP2", True, pasive_joint_knee_bp)
motor_joint_wheel = Node("JM1", True, motor_joint_wheel_bp)

body = Node("B1", True, body_bp)
knee = Node("BK1", True, knee_bp)
accum = Node("BA1", True, body_accum_bp)

transform_rwt = Node("TRWT1", True, transform_rwt_bp)
transform_lwt = Node("TLWT1", True, transform_lwt_bp)

transform_left_side = Node("TLS1", True, transform_left_side_bp)
transform_right_side = Node("TRS1", True, transform_right_side_bp)

transform_front = Node("TFR1", True, transform_front_bp)
transform_knee_front = Node("TKF1", True, transform_knee_front_bp)
transform_knee_back = Node("TKB1", True, transform_knee_back_bp)

mech_graph = GraphGrammar()
mech_graph.remove_node(0)

#
mech_graph.add_node(1, Node=body)
#
mech_graph.add_node(2, Node=transform_left_side)
mech_graph.add_node(3, Node=transform_front)
mech_graph.add_node(4, Node=stiffness_joint)
mech_graph.add_node(5, Node=knee)

upper_section_suspension = [(1, 2), (2, 3), (3, 4), (4, 5)]
mech_graph.add_edges_from(upper_section_suspension)

#
mech_graph.add_node(6, Node=transform_knee_front)
mech_graph.add_node(7, Node=transform_knee_back)
mech_graph.add_node(8, Node=long_link_front)
mech_graph.add_node(9, Node=long_link)

knee_section = [(5, 6), (5, 7), (6, 8), (7, 9)]
mech_graph.add_edges_from(knee_section)


mech_graph.add_node(10, Node=stiffness_knee_joint)
mech_graph.add_node(11, Node=knee)
mech_graph.add_node(12, Node=transform_knee_front)
mech_graph.add_node(13, Node=transform_knee_back)
mech_graph.add_node(14, Node=long_link)
mech_graph.add_node(15, Node=long_link)

bottom_section_suspension = [(9, 10), (10, 11), (11, 12), (11, 13), (12, 14), (13, 15)]
mech_graph.add_edges_from(bottom_section_suspension)
#
mount_ids = [x for x in mech_graph.nodes() if mech_graph.out_degree(x) == 0]
for num, ids in enumerate(mount_ids):
    id_transform = len(mech_graph) + 1
    id_joint = len(mech_graph) + 3
    id_wheel = len(mech_graph) + 2
    mech_graph.add_node(id_transform, Node=transform_lwt)
    mech_graph.add_node(id_wheel, Node=wheel)
    mech_graph.add_node(id_joint, Node=motor_joint_wheel)
    mech_graph.add_edges_from([(ids, id_transform), (id_transform, id_joint), (id_joint, id_wheel)])

id_offset = 100
mech_graph.add_node(id_offset + 2, Node=transform_right_side)
mech_graph.add_node(id_offset + 3, Node=transform_front)
mech_graph.add_node(id_offset + 4, Node=stiffness_joint)
mech_graph.add_node(id_offset + 5, Node=knee)
mech_graph.add_node(id_offset + 6, Node=transform_knee_front)
mech_graph.add_node(id_offset + 7, Node=transform_knee_back)
mech_graph.add_node(id_offset + 8, Node=long_link_front)
mech_graph.add_node(id_offset + 9, Node=long_link)
#
mech_graph.add_node(id_offset + 10, Node=stiffness_knee_joint)
mech_graph.add_node(id_offset + 11, Node=knee)
mech_graph.add_node(id_offset + 12, Node=transform_knee_front)
mech_graph.add_node(id_offset + 13, Node=transform_knee_back)
mech_graph.add_node(id_offset + 14, Node=long_link)
mech_graph.add_node(id_offset + 15, Node=long_link)

#
upper_section_suspension_right = [
    (x[0] + id_offset, x[1] + id_offset) for x in upper_section_suspension
]
upper_section_suspension_right[0] = (1, upper_section_suspension_right[0][1])
mech_graph.add_edges_from(upper_section_suspension_right)

#
knee_section_right = [(x[0] + id_offset, x[1] + id_offset) for x in knee_section]
mech_graph.add_edges_from(knee_section_right)
#
bottom_section_suspension_right = [
    (x[0] + id_offset, x[1] + id_offset) for x in bottom_section_suspension
]
mech_graph.add_edges_from(bottom_section_suspension_right)

mount_ids = [x for x in mech_graph.nodes() if mech_graph.out_degree(x) == 0 and x > id_offset]
for num, ids in enumerate(mount_ids):
    id_transform = len(mech_graph) + 1
    id_joint = len(mech_graph) + 3
    id_wheel = len(mech_graph) + 2
    mech_graph.add_node(id_transform, Node=transform_rwt)
    mech_graph.add_node(id_wheel, Node=wheel)
    mech_graph.add_node(id_joint, Node=motor_joint_wheel)
    mech_graph.add_edges_from([(ids, id_transform), (id_transform, id_joint), (id_joint, id_wheel)])

mech_graph.add_node(59, Node=transform_front)
mech_graph.add_node(60, Node=accum)
mech_graph.add_edges_from([(1, 59), (59, 60)])

from rostok.block_builder_chrono.block_classes import ChronoEasyShapeObject
plot_graph(mech_graph)
sim_preview = basic_simulation.RobotSimulationChrono([(ChronoEasyShapeObject(), True)])
sim_preview.add_design(mech_graph, [])
sim_preview.simulate(10000, 0.01, 10, None, True)