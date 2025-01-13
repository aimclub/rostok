from networkx import Graph
import numpy as np
import matplotlib.pyplot as plt

from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator, visualize_constrains

from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.optimization.problems import get_optimizing_joints
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_vertical_trajectory, create_simple_step_trajectory, get_workspace_trajectory
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, MIT_CHEETAH_PARAMS_DICT


def get_graph_and_traj(graph_number: int) -> tuple[Graph, dict, ParametrizedBuilder, list[np.ndarray], list[np.ndarray], np.ndarray]:
    anlge = np.deg2rad(-45)
    l1 = 0.21
    l2 = 0.18

    x_knee = l1 * np.sin(anlge)
    y_knee = -l1 * np.cos(anlge)

    y_ee = -y_knee + l2 * np.cos(anlge)
    print(x_knee, y_knee, y_ee)

    generator = TwoLinkGenerator()
    all_graphs = generator.get_standard_set(-0.105, shift=-0.10)
    graph, constrain_dict = all_graphs[graph_number]

    thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]/2
    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    density = MIT_CHEETAH_PARAMS_DICT["density"]
    body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]

    builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE,
                                  density={"default": density,
                                           "G": body_density},
                                  thickness={
                                      "default": thickness, "EE": 0.033/2},
                                  actuator={"default": actuator},
                                  size_ground=np.array(
                                      MIT_CHEETAH_PARAMS_DICT["size_ground"]),
                                  offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"]
                                  )

    workspace_trajectory = convert_x_y_to_6d_traj_xz(
        *get_workspace_trajectory([-0.1, -0.29], 0.07, 0.2, 10, 20))

    optimizing_joints = get_optimizing_joints(graph, constrain_dict)

    return graph, optimizing_joints, constrain_dict,  builder, workspace_trajectory


if __name__ == "__main__":
    graph, optimizing_joints, constrain_dict, builder, workspace_trajectory = get_graph_and_traj(
        0)

    plt.figure()

    plt.scatter(workspace_trajectory[:, 0],
                workspace_trajectory[:, 2], marker="1")

    draw_joint_point(graph)
    plt.figure()
    draw_joint_point(graph)
    visualize_constrains(graph, constrain_dict)
