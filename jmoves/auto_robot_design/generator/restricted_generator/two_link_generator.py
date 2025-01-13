import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List
import networkx as nx
from copy import deepcopy
import networkx

from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch
from auto_robot_design.description.utils import draw_joint_point
import itertools
from auto_robot_design.generator.restricted_generator.utilities import set_circle_points


class TwoLinkGenerator():
    """Generates all possible graphs with two links in main branch
    """

    def __init__(self) -> None:
        self.variants = list(range(7))
        self.constrain_dict = {}  # should be updated after creating each joint
        self.current_main_branch = []
        self.graph = nx.Graph()
        self.ground_x_movement = (-0.1, 0.1)
        self.ground_z_movement = (-0.01,0.1)
        self.free_x_movement = (-0.15, 0.15)
        self.free_z_movement = (-0.15, 0.15)
        self.bound_x_movement = (-0.1, 0.1)
        self.bound_z_movement = (-0.1, 0.1)

    def reset(self):
        """Reset the graph builder."""
        self.constrain_dict = {}  # should be updated after creating each joint
        self.current_main_branch = []
        self.graph = nx.Graph()

    #def build_standard_two_linker(self, knee_pos: float = -0.148, nominal_length=0.27577, right_shift=-0.148): old version
    def build_standard_two_linker(self, knee_pos: float = -0.15, nominal_length=0.3, right_shift=0):
        ground_joint = JointPoint(
            r=np.zeros(3),
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=True,
            name="Main_ground"
        )
        # graph_dict = {"TL_ground": ground_joint}
        self.constrain_dict[ground_joint.name] = {'optim': False,
                                                  'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        self.current_main_branch.append(ground_joint)

        knee_joint_pos = np.array([right_shift, 0, knee_pos])
        knee_joint = JointPoint(
            r=knee_joint_pos, w=np.array([0, 1, 0]), name="Main_knee")
        self.constrain_dict[knee_joint.name] = {
            'optim': True, 'x_range': (-0.1, 0.1), 'z_range': (-0.1, 0.1)}
        self.current_main_branch.append(knee_joint)
        ee = JointPoint(
            r=np.array([0, 0, -nominal_length]),
            w=np.array([0, 1, 0]),
            attach_endeffector=True,
            name="Main_ee"
        )
        self.current_main_branch.append(ee)
        self.constrain_dict[ee.name] = {
            'optim': False, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}

        add_branch(self.graph, self.current_main_branch)

    def add_2l_branch(self, inner: bool = True, shift=0.25, ground: bool = True):

        # we have several possible connection points and create a joint for each at the start
        if inner:
            ground_connection = np.array([-shift, 0, 0])
        else:
            ground_connection = np.array([shift, 0, 0])

        link_connection_points = [
            (self.current_main_branch[i-1].r + self.current_main_branch[i].r)/2 for i in range(1, len(self.current_main_branch))]

        # create joints for each possible point, not all of them will be used in the final graph
        # ground is always active
        ground_joint = JointPoint(r=ground_connection,
                                  w=np.array([0, 1, 0]),
                                  attach_ground=True,
                                  active=True,
                                  name=f"2L_ground")
        self.constrain_dict[ground_joint.name] = {
            'optim': True, 'x_range': self.ground_x_movement, 'z_range': self.ground_z_movement}
        # create connection dict
        connection_joints = {ground_joint: []}

        top_link_joint = JointPoint(r=link_connection_points[0], w=np.array([
                                    0, 1, 0]), active=True, name=f'2L_top')
        self.constrain_dict[top_link_joint.name] = {
            'optim': True, 'x_range': self.bound_x_movement, 'z_range': self.bound_z_movement}
        connection_joints[top_link_joint] = [
            [self.current_main_branch[0], self.current_main_branch[1]]]

        bot_link_joint = JointPoint(
            r=link_connection_points[1], w=np.array([0, 1, 0]), name=f'2L_bot')
        self.constrain_dict[bot_link_joint.name] = {
            'optim': True, 'x_range': self.bound_x_movement, 'z_range': self.bound_z_movement}
        connection_joints[bot_link_joint] = [
            [self.current_main_branch[1], self.current_main_branch[2]]]

        branch = []
        if ground:
            top_joint: JointPoint= ground_joint
        else:
            top_joint: JointPoint = top_link_joint
        bot_joint: JointPoint = bot_link_joint

        # create joints in the branch, currently depends on the type of the branch
        if inner:
            knee_point = (top_joint.r + bot_joint.r) / \
                2 + np.array([-shift, 0, 0])
        else:
            knee_point = (top_joint.r + bot_joint.r) / \
                2 + np.array([shift, 0, 0])
        branch_knee_joint = JointPoint(r=knee_point, w=np.array(
            [0, 1, 0]), name=f"2L_knee")
        self.constrain_dict[branch_knee_joint.name] = {
            'optim': True, 'x_range': self.free_x_movement, 'z_range': self.free_z_movement}

        branch += connection_joints[top_joint]
        branch.append(top_joint)
        branch.append(branch_knee_joint)
        branch.append(bot_joint)
        branch += connection_joints[bot_joint]
        top_joint.active = True

        add_branch(self.graph, branch)

    def add_4l_branch(self, inner: bool = True, shift=0.5, variant=0):
        # we have several possible connection points and create a joint for each at the start
        if inner:
            ground_connection = np.array([-shift, 0, 0])
        else:
            ground_connection = np.array([shift, 0, 0])

        link_connection_points = [
            (self.current_main_branch[i-1].r + self.current_main_branch[i].r)/2 for i in range(1, len(self.current_main_branch))]

        # create joints for each possible point, not all of them will be used in the final graph
        # ground is always active
        if variant not in [1,3]:
            ground_joint = JointPoint(r=ground_connection,
                                        w=np.array([0, 1, 0]),
                                        attach_ground=True,
                                        active=True,
                                        name="4L_ground")
        else:
            ground_joint = JointPoint(r=ground_connection,
                                        w=np.array([0, 1, 0]),
                                        attach_ground=True,
                                        active=False,
                                        name="4L_ground")
        self.constrain_dict[ground_joint.name] = {
            'optim': True, 'x_range': self.ground_x_movement, 'z_range': self.ground_z_movement}
        # create connection dict
        connection_joints = {ground_joint: []}
        top_link_joint = JointPoint(
            r=link_connection_points[0], w=np.array([0, 1, 0]), name=f'4L_top')
        self.constrain_dict[top_link_joint.name] = {
            'optim': True, 'x_range': self.bound_x_movement, 'z_range': self.bound_z_movement}
        connection_joints[top_link_joint] = [
            [self.current_main_branch[0], self.current_main_branch[1]]]

        bot_link_joint = JointPoint(
            r=link_connection_points[1], w=np.array([0, 1, 0]), name=f'4L_bot')
        self.constrain_dict[bot_link_joint.name] = {
            'optim': True, 'x_range': self.bound_x_movement, 'z_range': self.bound_z_movement}
        connection_joints[bot_link_joint] = [
            [self.current_main_branch[1], self.current_main_branch[2]]]

        # triangle with 3 connections
        if variant == 0:
            pos_1 = np.array([ground_joint.r[0], 0, bot_link_joint.r[2]])
            pos_2 = np.array([ground_joint.r[0], 0, bot_link_joint.r[2]/2])
            j1 = JointPoint(r=pos_1,
                            w=np.array([0, 1, 0]),
                            name="4LT1_triplet_bot")
            self.constrain_dict[j1.name] = {
                'optim': True, 'x_range':self.free_x_movement, 'z_range': self.free_z_movement}
            j2 = JointPoint(r=pos_2,
                            w=np.array([0, 1, 0]),
                            name="4LT1_triplet_top")
            self.constrain_dict[j2.name] = {
                'optim': True, 'x_range':self.free_x_movement, 'z_range': self.free_z_movement}

            branch = []
            branch += connection_joints[bot_link_joint]
            branch.append(bot_link_joint)
            branch.append(j1)
            branch.append(j2)
            branch.append(ground_joint)
            branch += connection_joints[ground_joint]
            add_branch(self.graph, branch)

            j3 = JointPoint(r=(pos_1+pos_2)/2,
                            w=np.array([0, 1, 0]),
                            name="4LT1_triplet_mid")
            self.constrain_dict[j3.name] = {
                'optim': True, 'x_range':self.free_x_movement, 'z_range': self.free_z_movement}

            secondary_branch = []
            secondary_branch += connection_joints[top_link_joint]
            secondary_branch.append(top_link_joint)
            secondary_branch.append(j3)
            secondary_branch.append([j1, j2])

            add_branch(self.graph, secondary_branch)

        else:
            new_joints = [ground_joint, top_link_joint, bot_link_joint]
            permutation = list(itertools.permutations(new_joints))[variant-1]

            new_joint_pos = set_circle_points(
                permutation[0].r, permutation[2].r, permutation[1].r, 4)

            branch = []
            branch += connection_joints[permutation[0]]
            branch.append(permutation[0])
            triangle_joints = []
            for i, pos in enumerate(new_joint_pos):
                flag = False
                if i == 1 and variant in [1,3]:
                    flag = True
                joint = JointPoint(r=pos, w=np.array(
                    [0, 1, 0]), active=flag,name=f"4LT2_j{i}")
                self.constrain_dict[joint.name] = {
                    'optim': True, 'x_range':self.free_x_movement, 'z_range': self.free_z_movement}
                branch.append(joint)
                if i < 2:
                    triangle_joints.append(joint)

            branch.append(permutation[2])
            branch += connection_joints[permutation[2]]
            add_branch(self.graph, branch)

            secondary_branch = [triangle_joints, permutation[1]
                                ] + connection_joints[permutation[1]]

            add_branch(self.graph, secondary_branch)

    def filter_constrain_dict(self):
        list_names = list(map(lambda x: x.name, self.graph.nodes))
        self.constrain_dict = dict(filter(lambda x:x[0] in list_names, self.constrain_dict.items()))

    def get_standard_set(self, knee_pos=-0.148, shift=0.1):
        result_list = []
        for inner in [True, False]:
            for ground in [True, False]:
                self.reset()
                self.build_standard_two_linker(knee_pos=knee_pos)
                self.add_2l_branch(inner=inner, ground=ground, shift=shift)
                self.filter_constrain_dict()
                result_list.append((self.graph, self.constrain_dict))
            for i in self.variants:
                self.reset()
                self.build_standard_two_linker(knee_pos=knee_pos)
                self.add_4l_branch(inner=inner, variant=i, shift=shift)
                self.filter_constrain_dict()
                result_list.append((self.graph, self.constrain_dict))
        return result_list


def get_constrain_space(constrain_dict: dict):
    space = []
    for key in constrain_dict:
        item = constrain_dict[key]
        if item['optim']:
            space.append(item.get('x_range'))
            space.append(item.get('z_range'))
    space = [x for x in space if x is not None]
    space = np.array(space)
    return space


def get_changed_graph(graph, constrain_dict, change_vector):
    new_graph: networkx.Graph = deepcopy(graph)
    vector_dict = {}
    i = 0
    for key in constrain_dict:
        if constrain_dict[key]['optim']:
            vector = np.zeros(3)
            if constrain_dict[key].get('x_range'):
                vector[0] = change_vector[i]
                i += 1
            if constrain_dict[key].get('z_range'):
                vector[2] = change_vector[i]
                i += 1
            vector_dict[key] = vector

    for node in new_graph.nodes:
        if node.name in vector_dict:
            node.r = node.r + vector_dict[node.name]

    return new_graph

def visualize_constrains(graph, constrain_dict):
    #draw_joint_point(graph)
    name2coord = dict(map(lambda x: (x.name, (x.r[0],x.r[2])), graph.nodes()))
    optimizing_joints = dict(
        filter(lambda x: x[1]["optim"], constrain_dict.items()))
    for key, value in optimizing_joints.items():
        x, z = name2coord.get(key)
        plt.plot(x, z, marker="o", markeredgecolor="red", markerfacecolor="green")
        width = value.get('x_range',[-0.01,0.01])
        x = x + width[0]
        width = abs(width[0]-width[1])
        height = value.get('z_range',[-0.01,0.01])
        z = z + height[0]
        height = abs(height[0]-height[1])
        rect = patches.Rectangle((x, z), width = width, height= height, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.show()


if __name__ == '__main__':
    gen = TwoLinkGenerator()
    graph, constrain_dict = gen.get_standard_set()[0]
    space = get_constrain_space(constrain_dict)
    random_vector = np.zeros(len(space))
    for j, r in enumerate(space):
        random_vector[j] = np.random.uniform(low=r[0], high=r[1])

    get_changed_graph(graph, constrain_dict, random_vector)
    draw_joint_point(graph)
