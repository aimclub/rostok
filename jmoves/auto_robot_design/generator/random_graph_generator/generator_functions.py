"""Functions for generating random graph for a mechanism with two DoF

    The link lengths and joint positions are randomly sampled from the parametrized ranges.
"""
import networkx as nx
import numpy as np
from typing import Tuple, List

from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch
from auto_robot_design.description.utils import draw_joint_point

pi = np.pi

def build_main_branch(
    n_link_options: Tuple[float] = (2, 3, 4),
    length_range: Tuple[float] = (0.2, 1.4),
    angle_range: Tuple[float] = (-pi / 3, pi / 3),
    ender_fixed: bool = True,
):
    """Create the list of joints that represents the main branch of the mechanism

    Each link and angle are randomly sampled from uniform distribution within parametrized boundaries.

    Args:
        n_link_options (Tuple[float], optional): options for amount of links in the branch. Defaults to (2, 3, 4).
        length_range (Tuple[float], optional): range of link lengths. Defaults to (0.2, 1.4).
        angle_range (Tuple[float], optional): range of default joint angles. Defaults to (-pi / 3, pi / 3).
        ender_fixed (bool, optional): option that determined if the end-effector is directly under the ground joint. Defaults to True.

    Returns:
        Tuple[List[Joint_Point], int]: returns the list of joints to construct the new branch in the graph and the number of DoF
    """
    # required for calculating DoF
    body_counter = 0
    joint_counter = 0
    # sample the number of links in the main branch
    main_branch_links = np.random.choice(n_link_options)
    # create ground point and branch
    ground_joint = JointPoint(
        r=np.zeros(3), # main ground is always in the origin of coordinate system
        w=np.array([0, 1, 0]),
        attach_ground=True,
        active=True, 
        name="Ground_main",
    )
    joint_counter += 1
    main_branch = [ground_joint]
    for i in range(main_branch_links - 1):
        # sample length and angle for new joint placement
        sampled_length = np.random.uniform(*length_range)
        sampled_angle = np.random.uniform(*angle_range)
        x = sampled_length * np.sin(sampled_angle)
        y = 0
        z = -sampled_length * np.cos(sampled_angle)
        # the position is calculated relative to the previous joint
        new_coordenates = main_branch[-1].r + np.array([x, y, z])
        main_branch.append(
            JointPoint(r=new_coordenates, w=np.array([0, 1, 0]), name=f"J{i}m")
        )
        joint_counter += 1
        body_counter += 1

    # the last point in the branch is the end-effector
    sampled_length = np.random.uniform(*length_range)
    if ender_fixed:
        x = -main_branch[-1].r[0]# set the ee to the 0 position in x
        y = 0
        if sampled_length > abs(main_branch[-1].r[0]):
            # the z coordinate can be calculated according to sampled length
            z = -((sampled_length**2 - main_branch[-1].r[0] ** 2) ** 0.5)
        else:
            # the sampled length is too short, the z shift is set to be the min_length/2 (just an arbitrary number to not set it zero)
            z = -length_range[0]/2
        new_coordenates = main_branch[-1].r + np.array([x, y, z])
        main_branch.append(
            JointPoint(
                r=new_coordenates,
                w=np.array([0, 1, 0]),
                attach_endeffector=True,
                name=f"EEJoint",
            )
        )
        body_counter += 1
    else:
        sampled_angle = np.random.uniform(*angle_range)
        x = sampled_length * np.sin(sampled_angle)
        y = 0
        z = -sampled_length * np.cos(sampled_angle)
        new_coordenates = main_branch[-1].r + np.array([x, y, z])
        main_branch.append(
            JointPoint(
                r=new_coordenates,
                w=np.array([0, 1, 0]),
                attach_endeffector=True,
                name=f"EEJoint",
            )
        )
        body_counter += 1

    return main_branch, 3 * body_counter - 2 * joint_counter


def find_connect_point(
    first_point: np.array,
    second_point: np.array,
    length_factor: float = 0.8,
    width_factor: float = 0.5,
) -> np.array:
    center = (first_point + second_point) / 2
    half_length = np.linalg.norm(second_point - first_point) / 2
    direction = second_point - center
    ort_direction = np.cross(direction, np.array([0, 1, 0]))
    sample_length_shift = np.random.uniform(-length_factor, length_factor)
    sample_width_shift = np.random.uniform(-width_factor, width_factor)
    total_shift = (
        direction * sample_length_shift
        + ort_direction
        / np.linalg.norm(ort_direction)
        * (length_factor - abs(sample_length_shift))
        * sample_width_shift
        * half_length
    )
    pos = center + total_shift
    return pos


def sample_secondary_branch(
    graph,
    main_branch,
    length_range: Tuple[float] = (0.3, 0.6),
    dof_reduction: int = 0,
    branch_id=0,
):
    local_joints = 0
    local_bodies = 0
    length_constrains = length_range
    # chose first body to connect joint. Secondary branch always starts from some body.
    # each branch can be attached to a body from second branch only once
    idx_set = set(range(len(main_branch)))
    idx_sample = np.random.choice(list(idx_set))
    idx_set.difference_update({idx_sample})  # remove sampled element from index list
    secondary_branch = []  # initialize secondary branch
    if idx_sample != 0:
        # get positions of first and second joints
        first_joint = main_branch[idx_sample - 1].r
        second_joint = main_branch[idx_sample].r
        pos = find_connect_point(first_joint, second_joint)
        new_joint = JointPoint(r=pos, w=np.array([0, 1, 0]), name=f"J0_{branch_id}")
        local_joints += 1
        secondary_branch = [
            [main_branch[idx_sample - 1], main_branch[idx_sample]],
            new_joint,
        ]
    else:
        # idx == 0 means attached to ground
        pos = np.array([np.random.uniform(-1, 1), 0, 0])
        new_joint = JointPoint(
            r=pos, w=np.array([0, 1, 0]), attach_ground=True, name=f"J0_{branch_id}"
        )
        local_joints += 1
        secondary_branch.append(new_joint)

    # idx from the main branch that shows the attachment to a body from main branch
    new_joint.attached = idx_sample
    current_joint = new_joint
    attach = False
    i = 0
    while not attach > 0:
        i += 1
        if i < 3:
            attach = np.random.choice(
                [True, False]
            )  # randomly choose if the new joint is to be attached to main branch
        else:
            attach = True
        if attach:
            # the joint is to be attached to a body from the main branch
            if current_joint.attached != -1:
                # the current is attached to body and the new to be attached, hence the bodies shouldn't be adjacent
                attached_idx = current_joint.attached
                # removes the links between adjacent bodies
                new_set = idx_set.difference(
                    set([attached_idx - 1, attached_idx, attached_idx + 1])
                )
                if len(new_set) == 0:
                    return False  # there is no way to connect the branch
                idx_sample = np.random.choice(list(new_set))
            else:
                idx_sample = np.random.choice(list(idx_set))

            idx_set.difference_update({idx_sample})  # drop used index from the set
            # find a point to attach
            if idx_sample != 0:
                # get positions of first and second joints
                first_joint = main_branch[idx_sample - 1].r
                second_joint = main_branch[idx_sample].r
                pos = find_connect_point(first_joint, second_joint)

                new_joint = JointPoint(
                    r=pos,
                    w=np.array([0, 1, 0]),
                    attach_endeffector=False,
                    name=f"J{i}_{branch_id}",
                )
                local_joints += 1
                local_bodies += 1
                secondary_branch += [
                    new_joint,
                    [main_branch[idx_sample - 1], main_branch[idx_sample]],
                ]
            else:
                pos = np.array([np.random.uniform(-1, 1), 0, 0])
                new_joint = JointPoint(
                    r=pos,
                    w=np.array([0, 1, 0]),
                    active=False,
                    attach_ground=True,
                    name=f"J{i}_{branch_id}",
                )
                secondary_branch.append(new_joint)
                local_joints += 1
                local_bodies += 1
            new_joint.attached = idx_sample
        else:
            # we don't need a point to be attached on bodies, hence we only sample a random point, but it must be below z = 0
            new_pos = np.array([0, 0, 0])
            while new_pos[2] > -0.1:
                sampled_length = np.random.uniform(*length_constrains)
                sampled_angle = np.random.uniform(0, 2 * pi)
                x = sampled_length * np.cos(sampled_angle)
                y = 0
                z = sampled_length * np.sin(sampled_angle)
                new_pos = current_joint.r + np.array([x, y, z])

            new_joint = JointPoint(
                r=new_pos,
                w=np.array([0, 1, 0]),
                active=False,
                attach_endeffector=False,
                name=f"J{i}_{branch_id}",
            )
            new_joint.attached = -1
            secondary_branch.append(new_joint)
            local_joints += 1
            local_bodies += 1
            current_joint = new_joint

    # if the initial secondary branch reduces dof too hard the building is failed
    delta_dof = 3 * local_bodies - 2 * local_joints
    if delta_dof < dof_reduction:
        return False
    # get joint from secondary branch
    triangle_list = [x for x in secondary_branch if type(x) is JointPoint]
    triangle_idx = set(range(1, len(triangle_list)))
    if delta_dof - dof_reduction > min(len(triangle_idx), len(idx_set)):
        return False

    add_branch(graph, secondary_branch)
    j = 0
    while delta_dof > dof_reduction:
        j += 1
        i += 1
        # if len(triangle_idx)==0 or len(idx_set)==0: return False
        triangle_sample = np.random.choice(list(triangle_idx))
        triangle_idx.difference_update({triangle_sample})
        idx_sample = np.random.choice(list(idx_set))
        idx_set.difference_update({idx_sample})
        first_joint_triangle = triangle_list[triangle_sample - 1].r
        second_joint_triangle = triangle_list[triangle_sample].r
        new_pos = find_connect_point(first_joint_triangle, second_joint_triangle)
        new_joint_triangle = JointPoint(
            r=new_pos,
            w=np.array([0, 1, 0]),
            active=False,
            attach_endeffector=False,
            name=f"J{i}_{j}_{branch_id}",
        )
        i += 1
        if idx_sample != 0:
            first_joint = main_branch[idx_sample - 1].r
            second_joint = main_branch[idx_sample].r
            new_pos = find_connect_point(first_joint, second_joint)
            new_joint = JointPoint(
                r=new_pos,
                w=np.array([0, 1, 0]),
                active=False,
                attach_endeffector=False,
                name=f"J{i}_{j}_{branch_id}",
            )
            new_branch = [
                [triangle_list[triangle_sample - 1], triangle_list[triangle_sample]],
                new_joint_triangle,
                new_joint,
                [main_branch[idx_sample - 1], main_branch[idx_sample]],
            ]
        else:
            pos = np.array([np.random.uniform(-1, 1), 0, 0])
            new_joint = JointPoint(
                r=pos,
                w=np.array([0, 1, 0]),
                active=False,
                attach_ground=True,
                name=f"J{i}_{branch_id}",
            )
            new_branch = [
                [triangle_list[triangle_sample - 1], triangle_list[triangle_sample]],
                new_joint_triangle,
                new_joint,
            ]
        local_bodies += 1
        local_joints += 2
        add_branch(graph, new_branch)
        delta_dof = 3 * local_bodies - 2 * local_joints

    return graph


def generate_graph():
    graph = nx.Graph()
    main_branch, dof = build_main_branch(angle_range=(-pi / 12, pi / 12))
    add_branch(graph, main_branch)
    # dof = body_counter * 3 - 2 * joint_counter
    # # print(dof)
    zero_reduction = True
    b_idx = 0
    while dof > 2 or zero_reduction:
        if zero_reduction:
            sample_dof_reduction = np.random.randint(0, dof - 1)
            zero_reduction = False
        else:
            sample_dof_reduction = np.random.randint(1, dof - 1)
        i = 0
        while not sample_secondary_branch(
            graph, main_branch, branch_id=b_idx, dof_reduction=-sample_dof_reduction
        ):
            if i > 50:
                return False
            i += 1
        dof -= sample_dof_reduction
        b_idx += 1

    return graph


if __name__ == "__main__":
    for i in range(1000):
        body_counter = 0
        joint_counter = 0
        graph = generate_graph()

    if graph:
        draw_joint_point(graph)
    else:
        print("Fail!")
