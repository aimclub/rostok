import numpy as np
from typing import Tuple, List
import networkx as nx

from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch
from auto_robot_design.description.utils import draw_joint_point
import itertools

from auto_robot_design.generator.restricted_generator.utilities import set_circle_points
class ThreeLinkGenerator():
    def __init__(self) -> None:
        self.variants_2l = 5
        self.variants_4l_t1 = 4
        self.variants_4l_t2 = 24
        self.total_variants = self.variants_2l + \
            self.variants_4l_t1 + self.variants_4l_t2
        self.constrain_dict = {}  # should be updated after creating each joint
        self.current_main_branch = []
        self.graph = nx.Graph()

    def reset(self):
        """Reset the graph builder."""
        self.constrain_dict = {}  # should be updated after creating each joint
        self.current_main_branch = []
        self.graph = nx.Graph()

    def build_standard_threelink(self, middle_length=0.4, middle_pos=0.5, nominal_length=1, right_shift=0.2):
        """Create a two-link branch to connect last or middle link with some link above or ground.

        Args:
            middle_length (float, optional): length of middle link. Defaults to 0.4.
            middle_pos (float, optional): center of middle link. Defaults to 0.5.
            nominal_length (int, optional): length from ground to end-effector. Defaults to 1.
            right_shift (float, optional): shift to the right of the middle link. Defaults to 0.2.
        """
        ground_joint = JointPoint(
            r=np.zeros(3),
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=True,
            name="Main_ground"
        )
        self.constrain_dict[ground_joint.name] = {
            'optim': False, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        self.current_main_branch.append(ground_joint)
        # works only for three-links main branch
        top_joint_pos = np.array(
            [right_shift, 0, -(middle_pos-middle_length*0.5)])
        bot_joint_pos = np.array(
            [right_shift, 0, -(middle_pos+middle_length*0.5)])
        top_joint = JointPoint(
            r=top_joint_pos, w=np.array([0, 1, 0]), name="Main_top")
        self.current_main_branch.append(top_joint)
        self.constrain_dict[top_joint.name] = {
            'optim': False, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        bot_joint = JointPoint(
            r=bot_joint_pos, w=np.array([0, 1, 0]), name="Main_bot")
        self.current_main_branch.append(bot_joint)
        self.constrain_dict[bot_joint.name] = {
            'optim': False, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        ee = JointPoint(
            r=np.array([0, 0, -nominal_length]),
            w=np.array([0, 1, 0]),
            attach_endeffector=True,
            name="Main_ee"
        )
        self.constrain_dict[ee.name] = {
            'optim': False, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        self.current_main_branch.append(ee)

        add_branch(self.graph, self.current_main_branch)

    def add_2l_branch(self,  inner: bool = True, shift=0.25, variant=0, branch_idx=0):
        """Adds a two-link chain top the main chain

        Args:
            inner (bool, optional): inner or outer chain. Defaults to True.
            shift (float, optional): shift in x direction. Defaults to 0.25.
            variant (int, optional): the variant of connection. Defaults to 0.
            branch_idx (int, optional): index of the branch for naming. Defaults to 0.
        """
        # we have several possible connection points and create a joint for each at the start
        if inner:
            ground_connection = np.array([-shift, 0, 0])
        else:
            ground_connection = np.array([shift, 0, 0])

        link_connection_points = [
            (self.current_main_branch[i-1].r + self.current_main_branch[i].r)/2 for i in range(1, len(self.current_main_branch))]

        # create joints for each possible point, not all of them will be used in the final graph
        # ground is always active
        ground = JointPoint(r=ground_connection,
                            w=np.array([0, 1, 0]),
                            attach_ground=True,
                            active=True,
                            name=f"2L_{branch_idx}_ground")
        self.constrain_dict[ground.name] = {
            'optim': True, 'x_range': (-0.2, 0.2)}
        # create connection dict
        connection_joints = {ground: []}
        for i, point in enumerate(link_connection_points):
            new_joint = JointPoint(r=point, w=np.array(
                [0, 1, 0]), name=f'2L_{branch_idx}_connection_{i}')
            self.constrain_dict[new_joint.name] = {
                'optim': True, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
            connection_joints[new_joint] = [
                [self.current_main_branch[i], self.current_main_branch[i+1]]]

        # create all ordered pairs of indexes
        pairs = list(itertools.combinations(
            list(range(len(connection_joints))), 2))
        # only works for three-links, remove the first pair to remove connection between ground and first link
        pairs.pop(0)
        self.variants_2l = len(pairs)
        if variant >= self.variants_2l:
            raise Exception('variant for two-link branch is out of range')
        branch = []
        pair = pairs[variant]  # actual pair for the variant
        top_joint: JointPoint = list(connection_joints)[pair[0]]
        bot_joint: JointPoint = list(connection_joints)[pair[1]]
        # create joints in the branch, currently depends on the type of the branch
        if inner:
            knee_point = (top_joint.r + bot_joint.r) / \
                2 + np.array([-shift, 0, 0])
        else:
            knee_point = (top_joint.r + bot_joint.r) / \
                2 + np.array([shift, 0, 0])
        branch_knee_joint = JointPoint(r=knee_point, w=np.array(
            [0, 1, 0]), name=f"branch_{branch_idx}_knee")
        self.constrain_dict[branch_knee_joint.name] = {
            'optim': True, 'x_range': (-0.4, 0.4), 'z_range': (-0.4, 0.4)}

        branch += connection_joints[top_joint]
        branch.append(top_joint)
        branch.append(branch_knee_joint)
        branch.append(bot_joint)
        branch += connection_joints[bot_joint]
        top_joint.active = True

        add_branch(self.graph, branch)

        return pair[0], pair[1]

    def add_4l_branch_type1(self, inner: bool = True, shift=0.5, variant=0, branch_idx=0):
        """Adds four-link with triangle in center and three links connected to the main branch.

        It is a symmetrical branch. 

        Args:
            inner (bool, optional): inner or outer chain. Defaults to True.
            shift (float, optional): shift in x direction. Defaults to 0.25.
            variant (int, optional): the variant of connection. Defaults to 0.
            branch_idx (int, optional): index of the branch for naming. Defaults to 0.

        """
        # we have several possible connection points and create a joint for each at the start
        if inner:
            ground_connection = np.array([-shift, 0, 0])
        else:
            ground_connection = np.array([shift, 0, 0])

        link_connection_points = [
            (self.current_main_branch[i-1].r + self.current_main_branch[i].r)/2 for i in range(1, len(self.current_main_branch))]

        # create joints for each possible point, not all of them will be used in the final graph
        # ground is always active
        ground = JointPoint(r=ground_connection,
                            w=np.array([0, 1, 0]),
                            attach_ground=True,
                            active=True,
                            name=f"4LT1_{branch_idx}_ground")
        self.constrain_dict[ground.name] = {
            'optim': True, 'x_range': (-0.2, 0.2)}
        # create connection dict
        connection_joints = {ground: []}
        for i, point in enumerate(link_connection_points):
            new_joint = JointPoint(r=point, w=np.array(
                [0, 1, 0]), name=f'4LT1_{branch_idx}_connection_{i}')
            self.constrain_dict[new_joint.name] = {
                'optim': True, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
            connection_joints[new_joint] = [
                [self.current_main_branch[i], self.current_main_branch[i+1]]]

        # create branch joints
        branch = []
        if inner:
            triplet_top = JointPoint(r=np.array([-shift, 0, self.current_main_branch[1].r[2]]), w=np.array(
                [0, 1, 0]), name=f"4LT1_{branch_idx}_triplet_top")
            self.constrain_dict[triplet_top.name] = {
                'optim': True, 'x_range': (-0.3, 0.3), 'z_range': (-0.3, 0.3)}
            triplet_mid = JointPoint(r=np.array([-shift, 0, (self.current_main_branch[1].r[2]+self.current_main_branch[2].r[2])/2]), w=np.array(
                [0, 1, 0]), name=f"4LT1_{branch_idx}_triplet_mid")
            self.constrain_dict[triplet_mid.name] = {'optim': True,
                                                     'x_range': (-0.3, 0.3), 'z_range': (-0.3, 0.3)}
            triplet_bot = JointPoint(r=np.array([-shift, 0, self.current_main_branch[2].r[2]]), w=np.array(
                [0, 1, 0]), name=f"4LT1_{branch_idx}_triplet_bot")
            self.constrain_dict[triplet_bot.name] = {'optim': True,
                                                     'x_range': (-0.3, 0.3), 'z_range': (-0.3, 0.3)}
        else:
            triplet_top = JointPoint(r=np.array([shift, 0, self.current_main_branch[1].r[2]]), w=np.array(
                [0, 1, 0]), name=f"4LT1_{branch_idx}_triplet_top")
            self.constrain_dict[triplet_top.name] = {'optim': True,
                                                     'x_range': (-0.3, 0.3), 'z_range': (-0.3, 0.3)}
            triplet_mid = JointPoint(r=np.array([shift, 0, (self.current_main_branch[1].r[2]+self.current_main_branch[2].r[2])/2]), w=np.array(
                [0, 1, 0]), name=f"4LT1_{branch_idx}_triplet_mid")
            self.constrain_dict[triplet_mid.name] = {'optim': True,
                                                     'x_range': (-0.3, 0.3), 'z_range': (-0.3, 0.3)}
            triplet_bot = JointPoint(r=np.array([shift, 0, self.current_main_branch[2].r[2]]), w=np.array(
                [0, 1, 0]), name=f"4LT1_{branch_idx}_triplet_bot")
            self.constrain_dict[triplet_bot.name] = {'optim': True,
                                                     'x_range': (-0.3, 0.3), 'z_range': (-0.3, 0.3)}
        triplets = list(itertools.combinations(
            list(range(len(connection_joints))), 3))
        self.variants_4l_t1 = len(triplets)
        if variant >= self.variants_4l_t1:
            raise Exception(
                'variant for four-link type_1 branch is out of range')
        triplet = triplets[variant]

        top_connection: JointPoint = list(connection_joints)[triplet[0]]
        top_connection.active = True
        mid_connection: JointPoint = list(connection_joints)[triplet[1]]
        bot_connection: JointPoint = list(connection_joints)[triplet[2]]

        branch += connection_joints[top_connection]
        branch.append(top_connection)
        branch.append(triplet_top)
        branch.append(triplet_bot)
        branch.append(bot_connection)
        branch += connection_joints[bot_connection]

        add_branch(self.graph, branch)

        secondary_branch = []
        secondary_branch += connection_joints[mid_connection]
        secondary_branch.append(mid_connection)
        secondary_branch.append(triplet_mid)
        secondary_branch.append([triplet_top, triplet_bot])

        add_branch(self.graph, secondary_branch)

        return triplet[0], triplet[2]

    def add_4l_branch_type2(self,  inner: bool = True, shift=0.5, variant=0, branch_idx=0):
        """Adds four-link with triangle in center and three links connected to the main branch.

        It is an asymmetrical branch. 

        Args:
            inner (bool, optional): inner or outer chain. Defaults to True.
            shift (float, optional): shift in x direction. Defaults to 0.25.
            variant (int, optional): the variant of connection. Defaults to 0.
            branch_idx (int, optional): index of the branch for naming. Defaults to 0.

        """
        # we have several possible connection points and create a joint for each at the start
        if inner:
            ground_connection = np.array([-shift, 0, 0])
        else:
            ground_connection = np.array([shift, 0, 0])

        link_connection_points = [
            (self.current_main_branch[i-1].r + self.current_main_branch[i].r)/2 for i in range(1, len(self.current_main_branch))]

        # create joints for each possible point, not all of them will be used in the final graph
        # ground is always active
        ground = JointPoint(r=ground_connection,
                            w=np.array([0, 1, 0]),
                            attach_ground=True,
                            active=True,
                            name=f"4LT2_{branch_idx}_ground")
        self.constrain_dict[ground.name] = {
            'optim': True, 'x_range': (-0.2, 0.2)}
        # create connection dict
        connection_joints = {ground: []}
        for i, point in enumerate(link_connection_points):
            new_joint = JointPoint(r=point, w=np.array(
                [0, 1, 0]), name=f'4LT2_{branch_idx}_connection_{i}')
            self.constrain_dict[new_joint.name] = {
                'optim': True, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
            connection_joints[new_joint] = [
                [self.current_main_branch[i], self.current_main_branch[i+1]]]

        branch = []
        triplets = list(itertools.combinations(
            list(range(len(connection_joints))), 3))
        # the branch is not symmetric, therefore for each triplet of connection points we have 6 different ways to connect
        variants = []
        for triplet in triplets:
            triplet_variants = list(itertools.permutations(triplet))
            variants += triplet_variants

        self.variants_4l_t2 = len(variants)
        if variant >= self.variants_4l_t2:
            raise Exception(
                'variant for four-link type_2 branch is out of range')
        triplet = variants[variant]

        first_connection: JointPoint = list(connection_joints)[triplet[0]]
        second_connection: JointPoint = list(connection_joints)[triplet[1]]
        third_connection: JointPoint = list(connection_joints)[triplet[2]]

        top_idx = min(triplet)
        list(connection_joints)[top_idx].active = True

        # connections = [connection_joints.keys[x] for x in triplet]

        new_pos_list = set_circle_points(
            first_connection.r, third_connection.r, second_connection.r, 4)
        branch += connection_joints[first_connection]
        branch.append(first_connection)
        triangle_joints = []
        for i, pos in enumerate(new_pos_list):
            joint = JointPoint(r=pos, w=np.array(
                [0, 1, 0]), name=f"4LT2_{branch_idx}_j{i}")
            self.constrain_dict[joint.name] = {
                'optim': True, 'x_range': (-0.4, 0.4), 'z_range': (-0.4, 0.4)}
            branch.append(joint)
            if i == 0 or i == 1:
                triangle_joints.append(joint)

        branch.append(third_connection)
        branch+=connection_joints[third_connection]
        add_branch(self.graph, branch)

        secondary_branch = []
        secondary_branch += connection_joints[second_connection]
        secondary_branch.append(second_connection)
        secondary_branch.append(triangle_joints)
        add_branch(self.graph, secondary_branch)

        return triplet[0], triplet[2]

    def get_graph(self, inner_variant=0, outer_variant=10, inner_shift=0.5, outer_shift=0.5):
        self.reset()
        self.build_standard_threelink()
        # inner branch
        if inner_variant < self.variants_2l:
            self.add_2l_branch(inner=True, variant=inner_variant,
                               shift=inner_shift, branch_idx=0)
        elif inner_variant < self.variants_2l+self.variants_4l_t1:
            self.add_4l_branch_type1(
                inner=True, variant=inner_variant-self.variants_2l, shift=inner_shift, branch_idx=0)
        else:
            self.add_4l_branch_type2(inner=True, variant=inner_variant-self.variants_2l -
                                     self.variants_4l_t1, shift=inner_shift, branch_idx=0)

        # outer branch
        if outer_variant < self.variants_2l:
            self.add_2l_branch(inner=False, variant=outer_variant,
                               shift=outer_shift, branch_idx=1)
        elif outer_variant < self.variants_2l+self.variants_4l_t1:
            self.add_4l_branch_type1(
                inner=False, variant=outer_variant-self.variants_2l, shift=outer_shift, branch_idx=1)
        else:
            self.add_4l_branch_type2(inner=False, variant=outer_variant -
                                     self.variants_2l-self.variants_4l_t1, shift=outer_shift, branch_idx=1)
        return self.graph, self.constrain_dict

    def filter_constrain_dict(self):
        list_names = list(map(lambda x: x.name, self.graph.nodes))
        self.constrain_dict = dict(filter(lambda x:x[0] in list_names, self.constrain_dict.items()))

    def get_all_topologies(self, main_shift=0.2, main_length=0.4, main_pos=0.5,inner_shift=0.5,outer_shift=0.5, nominal_length = 1):
        result = []
        for inner_variant in range(self.total_variants):
            for outer_variant in range(self.total_variants):
                self.reset()
                self.build_standard_threelink(right_shift=main_shift,middle_length=main_length,middle_pos=main_pos, nominal_length=nominal_length)
                if inner_variant < self.variants_2l:
                    top_in, bot_in = self.add_2l_branch(inner=True, variant=inner_variant,
                                    shift=inner_shift, branch_idx=0)
                elif inner_variant < self.variants_2l+self.variants_4l_t1:
                    top_in, bot_in = self.add_4l_branch_type1(
                        inner=True, variant=inner_variant-self.variants_2l, shift=inner_shift, branch_idx=0)
                else:
                    top_in, bot_in = self.add_4l_branch_type2(inner=True, variant=inner_variant-self.variants_2l -
                                            self.variants_4l_t1, shift=inner_shift, branch_idx=0)

                if outer_variant < self.variants_2l:
                    top_out, bot_out =self.add_2l_branch(inner=False, variant=outer_variant,
                                    shift=outer_shift, branch_idx=1)
                elif outer_variant < self.variants_2l+self.variants_4l_t1:
                    top_out, bot_out =self.add_4l_branch_type1(
                        inner=False, variant=outer_variant-self.variants_2l, shift=outer_shift, branch_idx=1)
                else:
                    top_out, bot_out =self.add_4l_branch_type2(inner=False, variant=outer_variant -
                                            self.variants_2l-self.variants_4l_t1, shift=outer_shift, branch_idx=1)
                
                if top_in!=top_out and bot_in!=bot_out:
                    self.filter_constrain_dict()
                    result.append((self.graph, self.constrain_dict))

        return result
                    
        
