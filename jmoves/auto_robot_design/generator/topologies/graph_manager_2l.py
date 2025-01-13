"""This module contains the class for the five bar mechanism topology manager."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch


class MutationType(Enum):
    """Enumerate for mutation types."""
    #UNMOVABLE = 0  # Unmovable joint that is not used for optimization
    ABSOLUTE = 1  # The movement of the joint are in the absolute coordinate system and are relative to the initial position
    RELATIVE = 2  # The movement of the joint are relative to some other joint or joints and doesn't have an initial position
    # The movement of the joint are relative to some other joint or joints and doesn't have an initial position. The movement is in percentage of the distance between the joints.
    RELATIVE_PERCENTAGE = 3


@dataclass
class GeneratorInfo:
    """Information for node of a generator."""
    mutation_type: int = MutationType.ABSOLUTE
    initial_coordinate: np.ndarray = np.zeros(3)
    mutation_range: List[Optional[Tuple]] = field(
        default_factory=lambda: [None, None, None])
    relative_to: Optional[Union[JointPoint, List[JointPoint]]] = None
    freeze_pos: List[Optional[Tuple]] = field(
        default_factory=lambda: [None, 0, None])


@dataclass
class ConnectionInfo:
    """Description of a point for branch connection."""
    connection_jp: JointPoint
    jp_connection_to_main: List[JointPoint]
    relative_mutation_range: List[Optional[Tuple]] # this parameter is used to set the mutation range of the branch joints


class GraphManager2L():
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.generator_dict = {}
        self.current_main_branch = []
        self.main_connections: List[ConnectionInfo] = []
        self.mutation_ranges = {}
        self.name = "Default"

    def reset(self):
        """Reset the graph builder."""
        self.generator_dict = {}
        self.current_main_branch = []
        self.graph = nx.Graph()
        self.mutation_ranges = {}

    def get_node_by_name(self, name:str):
        for node in self.graph.nodes:
            if node.name == name:
                return node
        return None

    def build_main(self, length: float, fully_actuated: bool = False):
        """Builds the main branch and create nodes for the connections.

        Args:
            length (float): length of the main branch that we use as a reference for all sizes.
        """
        ground_joint = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=True,
            name="Main_ground"
        )
        self.current_main_branch.append(ground_joint)
        self.generator_dict[ground_joint] = GeneratorInfo(freeze_pos=[0, 0, 0])

        ground_connection_jp = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=False,  # initial value is false, it should be changed in branch attachment process
            name="Ground_connection"
        )

        self.generator_dict[ground_connection_jp] = GeneratorInfo(mutation_type= MutationType.ABSOLUTE, initial_coordinate=np.array(
            [0, 0, 0.001]), mutation_range=[(-0.2, 0.), None, (-0.03, 0.07)])

        ground_connection_description = ConnectionInfo(
            ground_connection_jp, [], [(-0.05, 0.1), None, (-0.3, -0.1)])
        self.main_connections.append(ground_connection_description)

        knee_joint_pos = np.array([0.03, 0, -length/2])
        knee_joint = JointPoint(
            r=None, w=np.array([0, 1, 0]), active=fully_actuated, name="Main_knee")
        self.current_main_branch.append(knee_joint)
        
        self.generator_dict[knee_joint] = GeneratorInfo(
            MutationType.ABSOLUTE, initial_coordinate=knee_joint_pos.copy(), mutation_range=[None, None, (-0.1, 0.1)],freeze_pos=[0.03,0,None])

        first_connection = JointPoint(r=None, w=np.array([
                                      0, 1, 0]), name="Main_connection_1")
        # self.generator_dict[first_connection] = GeneratorInfo(MutationType.RELATIVE, None, mutation_range=[
        #                                                       (-0.05, 0.05), None, (-0.15, 0.15)], relative_to=[ground_joint, knee_joint])
        self.generator_dict[first_connection] = GeneratorInfo(MutationType.RELATIVE_PERCENTAGE, None, mutation_range=[
                                                              (-0.2, 0.2), None, (-0.6, 0.3)], relative_to=[ground_joint, knee_joint])
        first_connection_description = ConnectionInfo(
            first_connection, [ground_joint, knee_joint], [(-0.1, 0.0), None, (-0.1, 0.1)])
        self.main_connections.append(first_connection_description)

        ee = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            attach_endeffector=True,
            name="Main_ee"
        )
        self.current_main_branch.append(ee)
        self.generator_dict[ee] = GeneratorInfo(
            initial_coordinate=np.array([0, 0, -length]), freeze_pos=[0,0,-length])

        second_connection = JointPoint(r=None, w=np.array([0, 1, 0]), name="Main_connection_2")
        # self.generator_dict[second_connection] = GeneratorInfo(MutationType.RELATIVE, None, mutation_range=[
        #                                                        (-0.05, 0.05), None, (-0.15, 0.15)], relative_to=[knee_joint, ee])
        self.generator_dict[second_connection] = GeneratorInfo(MutationType.RELATIVE_PERCENTAGE, None, mutation_range=[(-0.2, 0.2), None, (-0.6, 0.3)], relative_to=[knee_joint, ee])
        second_connection_description = ConnectionInfo(
            second_connection, [knee_joint, ee], [(-0.1, 0), None, (0, 0.1)])
        self.main_connections.append(second_connection_description)

        add_branch(self.graph, self.current_main_branch)

    def build_3n2p_branch(self, connection_list: List[int]):
        """Generate a trivial branch that only have one node.

        Args:
            connection_list (List[int]): List of connection point indexes that we want to connect the branch to.
        """
        branch_joint = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            name="branch_1"
        )
        self.generator_dict[branch_joint] = GeneratorInfo(MutationType.RELATIVE, None,
                                                          mutation_range=self.main_connections[
                                                              connection_list[0]].relative_mutation_range,
                                                          relative_to=self.main_connections[connection_list[0]].connection_jp)

        for connection in connection_list:
            connection_description = self.main_connections[connection]
            jp = connection_description.connection_jp
            jp_connection_to_main = connection_description.jp_connection_to_main
            if len(jp_connection_to_main) == 0:
                # if the connection_description is empty, it means that the connection is directly to the ground
                self.graph.add_edge(jp, branch_joint)

            else:
                self.graph.add_edge(jp, branch_joint)
                for cd in jp_connection_to_main:
                    self.graph.add_edge(cd, jp)

            if connection == min(connection_list):
                jp.active = True

    def build_6n4p_symmetric(self, connection_list: List[int]):
        branch_jp_counter = 0
        branch_joints = []
        for connection in connection_list:
            connection_description = self.main_connections[connection]
            jp = connection_description.connection_jp
            jp_connection_to_main = connection_description.jp_connection_to_main
            branch_jp = JointPoint(
                r=None,
                w=np.array([0, 1, 0]),
                name=f"branch_{branch_jp_counter}"
            )
            branch_joints.append(branch_jp)
            branch_jp_counter += 1
            self.generator_dict[branch_jp] = GeneratorInfo(MutationType.RELATIVE, None,
                                                           mutation_range=self.main_connections[
                                                               connection].relative_mutation_range,
                                                           relative_to=self.main_connections[connection].connection_jp)
            if len(jp_connection_to_main) == 0:
                self.graph.add_edge(jp, branch_jp)
                jp.active = True

            elif len(jp_connection_to_main) == 2:
                self.graph.add_edge(jp, branch_jp)
                for cd in jp_connection_to_main:
                    self.graph.add_edge(cd, jp)

        self.graph.add_edge(branch_joints[0], branch_joints[1])
        self.graph.add_edge(branch_joints[1], branch_joints[2])
        self.graph.add_edge(branch_joints[2], branch_joints[0])

    def build_6n4p_asymmetric(self, connection_list: float):
        """Connects the 4l asymmetric branch to the main branch

        Args:
            connection_list (float): list of connecting points indexes for branch connection to main. Linkage chain of the largest length between the first and the third indices
        """
        if connection_list[0]+connection_list[1] == 1:
            branch_1_active = True
        else:
            branch_1_active = False

        connection_description = self.main_connections[connection_list[0]]
        jp = connection_description.connection_jp
        jp_connection_to_main = connection_description.jp_connection_to_main
        branch_jp_0 = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            name="branch_0"
        )
        self.generator_dict[branch_jp_0] = GeneratorInfo(MutationType.RELATIVE, None,
                                                         mutation_range=connection_description.relative_mutation_range,
                                                         relative_to=jp)
        if len(jp_connection_to_main) == 0:
            self.graph.add_edge(jp, branch_jp_0)
            jp.active = not branch_1_active
        else:
            self.graph.add_edge(jp, branch_jp_0)
            for cd in jp_connection_to_main:
                self.graph.add_edge(cd, jp)

        connection_description = self.main_connections[connection_list[1]]
        jp = connection_description.connection_jp
        jp_connection_to_main = connection_description.jp_connection_to_main

        branch_jp_1 = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            name="branch_1"
        )
        branch_jp_1.active = branch_1_active
        self.generator_dict[branch_jp_1] = GeneratorInfo(MutationType.RELATIVE, None,
                                                         mutation_range=connection_description.relative_mutation_range,
                                                         relative_to=jp)
        if len(jp_connection_to_main) == 0:
            self.graph.add_edge(jp, branch_jp_1)
            jp.active = not branch_1_active
        else:
            self.graph.add_edge(jp, branch_jp_1)
            for cd in jp_connection_to_main:
                self.graph.add_edge(cd, jp)
        self.graph.add_edge(branch_jp_0, branch_jp_1)
        self.graph.add_edge(branch_jp_0, jp)
        connection_description = self.main_connections[connection_list[2]]
        jp = connection_description.connection_jp
        jp_connection_to_main = connection_description.jp_connection_to_main
        branch_jp_2 = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            name="branch_2"
        )
        self.generator_dict[branch_jp_2] = GeneratorInfo(MutationType.RELATIVE, None,
                                                         mutation_range=connection_description.relative_mutation_range,
                                                         relative_to=jp)
        if len(jp_connection_to_main) == 0:
            self.graph.add_edge(jp, branch_jp_2)
            if not branch_1_active:
                jp.active = True
        else:
            self.graph.add_edge(jp, branch_jp_2)
            for cd in jp_connection_to_main:
                self.graph.add_edge(cd, jp)
        self.graph.add_edge(branch_jp_2, branch_jp_1)

    def freeze_joint(self, joint: JointPoint, freeze_pos: List[Optional[Tuple]]):
        """Freeze the position of the joint.

        Args:
            joint (JointPoint): the joint to be frozen.
            freeze_pos (List[Optional[Tuple]]): the position to be frozen.
        """
        self.generator_dict[joint].freeze_pos = freeze_pos

    def set_mutation_ranges(self):
        """Traverse the generator_dict to get all mutable parameters and their ranges.
        """
        self.mutation_ranges = {}
        # remove all auxiliary joint points from generator_dict
        keys = list(self.generator_dict)
        for key in keys:
            if key not in self.graph.nodes:
                del self.generator_dict[key]
        axes = ['x', 'y','z']
        for idx, pare in enumerate(self.generator_dict.items()):
            key, value = pare
            if value.mutation_type == MutationType.RELATIVE or value.mutation_type == MutationType.RELATIVE_PERCENTAGE:
                for i, r in enumerate(value.mutation_range):
                    if r is not None and value.freeze_pos[i] is None:
                            self.mutation_ranges[(key, axes[i])] = r

            elif value.mutation_type == MutationType.ABSOLUTE:
                for i, r in enumerate(value.mutation_range):
                    if r is not None and value.freeze_pos[i] is None:
                        self.mutation_ranges[(key, axes[i])] = (
                                r[0]+value.initial_coordinate[i], r[1]+value.initial_coordinate[i])

    def generate_random_from_mutation_range(self):
        """Sample random values from the mutation ranges.

        Returns:
            List[float]: a vector of parameters that are sampled from the mutation ranges.
        """
        result = []
        for _, value in self.mutation_ranges.items():
            result.append(np.random.uniform(value[0], value[1]))
        return result

    def generate_central_from_mutation_range(self):
        """Return values from center of the mutation ranges.

        Returns:
            List[float]: a vector of parameters that are centered on the mutation ranges.
        """
        result = []
        for _, value in self.mutation_ranges.items():
            result.append((value[0]+value[1])/2)
        return result

    def get_graph(self, parameters: List[float]):
        """Produce a graph of the set topology from the given parameters.

        Args:
            parameters List[float]: list of mutations.

        Raises:
            Exception: raise an exception if the number of parameters is not equal to the number of mutation ranges.

        Returns:
            nx.Graph: the graph of a mechanism with the given parameters.
        """
        if len(parameters) != len(list(self.mutation_ranges.keys())):
            raise ValueError(
                'Wrong number of parameters for graph specification!')

        parameter_counter = 0
        for jp, gi in self.generator_dict.items():
            if jp.r is None:
                jp.r = np.full(3, np.nan)

            if gi.mutation_type == MutationType.ABSOLUTE:
                for i, r in enumerate(gi.mutation_range):
                    if gi.freeze_pos[i] is not None:
                        jp.r[i] = gi.freeze_pos[i]
                    elif r is not None:
                        jp.r[i] = parameters[parameter_counter]
                        parameter_counter += 1
                    elif gi.initial_coordinate[i] is not None:
                        jp.r[i] = gi.initial_coordinate[i]
                    else:
                        raise ValueError(f"Failed to assign value for Joint Point {jp.name} coordinate {i}")

            elif gi.mutation_type == MutationType.RELATIVE:
                if isinstance(gi.relative_to, list) and len(gi.relative_to) == 2:
                    jp.r = (gi.relative_to[0].r + gi.relative_to[1].r)/2

                for i, r in enumerate(gi.mutation_range):
                    if gi.freeze_pos[i] is not None:
                        parameter = gi.freeze_pos[i]
                    elif r is not None:
                        parameter =parameters[parameter_counter]
                        parameter_counter += 1
                    else:
                        raise ValueError(f"Failed to assign value for Joint Point {jp.name} coordinate {i}")

                    if isinstance(gi.relative_to, JointPoint):
                        # if relative point is relative to a single joint we just add parameter to the coordinate of this joint
                        jp.r[i] = gi.relative_to.r[i] + parameter
                    else:
                        if len(gi.relative_to) == 2:
                            # TODO: this part would fail in 3D case
                            # if relative point is relative to two joints we calculate the direction of the link between these joints 
                            # and use its direction as z axis and the orthogonal direction as the x axis. Then we add the parameter to the center of the link.
                            link_direction = gi.relative_to[0].r - \
                                gi.relative_to[1].r
                            link_ortogonal = np.array(
                                [-link_direction[2], link_direction[1], link_direction[0]])
                            link_length = np.linalg.norm(link_direction)
                            if i == 0:
                                jp.r += parameter * link_ortogonal/link_length
                            if i == 2:
                                jp.r += parameter *  link_direction/link_length
                        #     jp.r += parameters[parameter_counter]*link_direction/link_length
                        #     jp.r += parameters[parameter_counter]*np.array([-link_direction[2],link_direction[1],link_direction[0]])/link_length

            elif gi.mutation_type == MutationType.RELATIVE_PERCENTAGE:
                if isinstance(gi.relative_to, list) and len(gi.relative_to) == 2:
                    jp.r = (gi.relative_to[0].r + gi.relative_to[1].r)/2

                for i, r in enumerate(gi.mutation_range):
                    if gi.freeze_pos[i] is not None:
                        parameter = gi.freeze_pos[i]
                    elif r is not None:
                        parameter =parameters[parameter_counter]
                        parameter_counter += 1
                    else:
                        raise ValueError(f"Failed to assign value for Joint Point {jp.name} coordinate {i}")

                    if isinstance(gi.relative_to, JointPoint):
                        raise ValueError(
                            'Relative percentage mutation type should have a list of joints as relative_to')
                    else:
                        #TODO: this part would fail in 3D case
                        if len(gi.relative_to) == 2:
                            link_direction = gi.relative_to[0].r - \
                                gi.relative_to[1].r
                            link_ortogonal = np.array(
                                [link_direction[2], link_direction[1], -link_direction[0]])
                            link_length = np.linalg.norm(link_direction)
                            if i == 0:
                                jp.r += parameter * link_ortogonal
                            if i == 2:
                                jp.r += parameter * link_direction

        return self.graph

from matplotlib import patches
def plot_one_jp_bounds(graph_manager, jp):
    jp = graph_manager.get_node_by_name(jp)
    info:GeneratorInfo = graph_manager.generator_dict[jp]
    if info.mutation_type == MutationType.ABSOLUTE:
        if graph_manager.mutation_ranges.get((jp, 'x')) is None:
            x_range = (info.freeze_pos[0]-0.001, info.freeze_pos[0]+0.001)
        elif graph_manager.mutation_ranges[(jp, 'x')][0] == graph_manager.mutation_ranges[(jp, 'x')][1]:
            x_range = (graph_manager.mutation_ranges[(jp, 'x')][0]-0.01, graph_manager.mutation_ranges[(jp, 'x')][1]+0.01)
        else:
            x_range = graph_manager.mutation_ranges[(jp, 'x')]
        if graph_manager.mutation_ranges.get((jp, 'z')) is None:
            z_range = (info.freeze_pos[2]-0.001, info.freeze_pos[2]+0.001)
        elif graph_manager.mutation_ranges[(jp, 'z')][0] == graph_manager.mutation_ranges[(jp, 'z')][1]:
            z_range = (graph_manager.mutation_ranges[(jp, 'z')][0]-0.01, graph_manager.mutation_ranges[(jp, 'z')][1]+0.01)
        else:
            z_range = graph_manager.mutation_ranges[(jp, 'z')]
        
        rect = patches.Rectangle(
            (x_range[0], z_range[0]),
            width=x_range[1]-x_range[0],
            height=z_range[1]-z_range[0],
            angle=0,
            linewidth=1,
            edgecolor='r',
            facecolor="none",
        )
        plt.gca().add_patch(rect)

    if info.mutation_type == MutationType.RELATIVE:
        if graph_manager.mutation_ranges.get((jp, 'x')) is None:
            x_range = (info.freeze_pos[0]-0.001+info.relative_to.r[0], info.freeze_pos[0]+0.001+info.relative_to.r[0])
        elif graph_manager.mutation_ranges[(jp, 'x')][0] == graph_manager.mutation_ranges[(jp, 'x')][1]:
            x_range = (graph_manager.mutation_ranges[(jp, 'x')][0]-0.001+info.relative_to.r[0], graph_manager.mutation_ranges[(jp, 'x')][0]+0.001+info.relative_to.r[0])
        else:
            x_range = (graph_manager.mutation_ranges[(jp, 'x')][0]+info.relative_to.r[0], info.relative_to.r[0]+graph_manager.mutation_ranges[(jp, 'x')][1])
        if graph_manager.mutation_ranges.get((jp, 'z')) is None:
            z_range = (info.freeze_pos[2]-0.001+ info.relative_to.r[2], info.freeze_pos[2]+0.001+ info.relative_to.r[2])
        elif graph_manager.mutation_ranges[(jp, 'z')][0] == graph_manager.mutation_ranges[(jp, 'z')][1]:
            z_range = (graph_manager.mutation_ranges[(jp, 'z')][0]-0.001+ info.relative_to.r[2], graph_manager.mutation_ranges[(jp, 'z')][1]+0.001+ info.relative_to.r[2])
        else:
            z_range = (graph_manager.mutation_ranges[(jp, 'z')][0]+ info.relative_to.r[2], info.relative_to.r[2]+graph_manager.mutation_ranges[(jp, 'z')][1])
        
        rect = patches.Rectangle(
            (x_range[0], z_range[0]),
            width=x_range[1]-x_range[0],
            height=z_range[1]-z_range[0],
            angle=0,
            linewidth=1,
            edgecolor='b',
            facecolor="none",
        )
        plt.gca().add_patch(rect)

    if info.mutation_type == MutationType.RELATIVE_PERCENTAGE:
        link_vector = info.relative_to[0].r-info.relative_to[1].r
        link_length = np.linalg.norm(link_vector)
        link_cener = (info.relative_to[1].r+info.relative_to[0].r)/2
        if graph_manager.mutation_ranges.get((jp, 'x')) is None:
            x_range = (info.freeze_pos[0] * link_length -0.001+link_cener[0], info.freeze_pos[0]*link_length+0.001+link_cener[0])
        elif graph_manager.mutation_ranges[(jp, 'x')][0] == graph_manager.mutation_ranges[(jp, 'x')][1]:
            x_range = (graph_manager.mutation_ranges[(jp, 'x')][0]* link_length-0.001+link_cener[0], graph_manager.mutation_ranges[(jp, 'x')][0]* link_length+0.001+link_cener[0])
        else:
            x_range = (graph_manager.mutation_ranges[(jp, 'x')][0]* link_length+link_cener[0], link_cener[0]+graph_manager.mutation_ranges[(jp, 'x')][1]* link_length)
        if graph_manager.mutation_ranges.get((jp, 'z')) is None:
            z_range = (info.freeze_pos[2]* link_length-0.001+link_cener[2], info.freeze_pos[2]* link_length+0.001+link_cener[2])
        elif graph_manager.mutation_ranges[(jp, 'z')][0] == graph_manager.mutation_ranges[(jp, 'z')][1]:
            z_range = (graph_manager.mutation_ranges[(jp, 'z')][0]* link_length-0.001+link_cener[2], graph_manager.mutation_ranges[(jp, 'z')][0]* link_length+0.001+link_cener[2])
        else:
            z_range = (graph_manager.mutation_ranges[(jp, 'z')][0]* link_length+link_cener[2], link_cener[2]+graph_manager.mutation_ranges[(jp, 'z')][1]* link_length)
        
        u = np.array([0, 0, 1])
        v = link_vector/link_length
        angle_rad = np.arctan2(u[0]*v[2] - u[2]*v[0], np.dot(u, v))  # atan2(det, dot)
        angle_deg = np.degrees(angle_rad)
        # angle = np.arccos(np.inner(link_vector, np.array([0, 0, 1]))/link_length)
        rect = patches.Rectangle(
            (x_range[0], z_range[0]),
            width=x_range[1]-x_range[0],
            height=z_range[1]-z_range[0],
            angle=angle_deg,
            rotation_point = (link_cener[0], link_cener[2]),
            linewidth=1,
            edgecolor='g',
            facecolor="none",
        )
        plt.gca().add_patch(rect)

def plot_2d_bounds(graph_manager):
    """
    Plot 2D bounds for each joint points in the graph manager. Different colors are used for different types of mutations.
    Absolute mutations are red, relative mutations are blue, and relative percentage mutations are green.

    Args:
        graph_manager (GraphManager): The graph manager object containing generator information.

    Returns:
        None
    """
    for jp, gen_info in graph_manager.generator_dict.items():
        # if gen_info.mutation_type == MutationType.UNMOVABLE:
        #     continue
        ez = np.array([1, 0, 0])
        x_bound = (-0.001,
                   0.001) if gen_info.mutation_range[0] is None else gen_info.mutation_range[0]
        z_bound = (-0.001,
                   0.001) if gen_info.mutation_range[2] is None else gen_info.mutation_range[2]
        bound = np.array([x_bound, z_bound])

        if gen_info.mutation_type == MutationType.ABSOLUTE:
            pos_initial = np.array(
                [gen_info.initial_coordinate[0], gen_info.initial_coordinate[2]])
            xz_rect_start = pos_initial + bound[:, 0]
            wh_rect = bound[:, 1] - bound[:, 0]
            angle = 0
            rot_point = np.zeros(2)
            color = "r"

        elif gen_info.mutation_type == MutationType.RELATIVE:

            if isinstance(gen_info.relative_to, JointPoint):
                rel_jp_xz = np.array(
                    [gen_info.relative_to.r[0], gen_info.relative_to.r[2]])
                xz_rect_start = rel_jp_xz + bound[:, 0]
                wh_rect = bound[:, 1] - bound[:, 0]
                angle = 0
                rot_point = np.zeros(2)
            else:
                if len(gen_info.relative_to) == 2:
                    xz_rect_start = (
                        gen_info.relative_to[0].r + gen_info.relative_to[1].r)/2
                    link_direction = gen_info.relative_to[0].r - \
                        gen_info.relative_to[1].r
                    link_ortogonal = np.array(
                        [-link_direction[2], link_direction[1], link_direction[0]])
                    link_length = np.linalg.norm(link_direction)
                    angle = np.arccos(np.inner(ez, link_ortogonal/link_length) /
                                      la.norm(link_ortogonal/link_length) /
                                      la.norm(ez))

                    xz_rect_start[0] += (np.array([bound[0, 0], 0, 0]) *
                                         link_ortogonal/link_length)[0]

                    xz_rect_start[1] += (np.array([0, 0, bound[1, 0]]) *
                                         link_direction/link_length)[2]

                    wh_rect = bound[:, 1] - bound[:, 0]
                    rot_point = (
                        gen_info.relative_to[1].r + gen_info.relative_to[0].r)/2
            color = "b"

        elif gen_info.mutation_type == MutationType.RELATIVE_PERCENTAGE:

            if len(gen_info.relative_to) == 2:
                xz_rect_start = (
                    gen_info.relative_to[1].r + gen_info.relative_to[0].r)[[0, 2]]/2

                link_direction = gen_info.relative_to[0].r - \
                    gen_info.relative_to[1].r
                link_ortogonal = np.array(
                    [-link_direction[2], link_direction[1], link_direction[0]])
                link_length = np.linalg.norm(link_direction)
                angle = np.arccos(np.inner(ez, link_ortogonal/link_length) /
                                  la.norm(link_ortogonal/link_length) /
                                  la.norm(ez))

                bound = bound * link_length

                if np.isclose(abs(angle), np.pi):
                    angle = 0

                # rot = R.from_rotvec(axis * angle)

                # start_rect_pos = rot.as_matrix() @ np.array([bound[0,0], 0, bound[1,0]])

                xz_rect_start[0] += bound[0, 0]  # start_rect_pos[0]
                xz_rect_start[1] += bound[1, 0]  # start_rect_pos[2]

                wh_rect = np.abs(bound[:, 1] - bound[:, 0])
                rot_point = (
                    gen_info.relative_to[1].r + gen_info.relative_to[0].r)[[0, 2]]/2
                color = "g"

        rect = patches.Rectangle(
            (xz_rect_start[0], xz_rect_start[1]),
            width=wh_rect[0],
            height=wh_rect[1],
            angle=-np.rad2deg(angle),
            rotation_point=(rot_point[0], rot_point[1]),
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        plt.gca().add_patch(rect)


def get_preset_by_index(idx: int):
    if idx == -1:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4, fully_actuated=True)
        gm.set_mutation_ranges()
        return gm

    if idx == 0:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_3n2p_branch([0, 2])
        gm.set_mutation_ranges()
        return gm

    if idx == 1:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_3n2p_branch([1, 2])
        gm.set_mutation_ranges()
        return gm

    if idx == 2:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_symmetric([0, 1, 2])
        gm.set_mutation_ranges()
        return gm

    if idx == 3:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([0, 1, 2])
        gm.set_mutation_ranges()
        return gm

    if idx == 4:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([0, 2, 1])
        gm.set_mutation_ranges()
        return gm

    if idx == 5:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([1, 0, 2])
        gm.set_mutation_ranges()
        return gm

    if idx == 6:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([1, 2, 0])
        gm.set_mutation_ranges()
        return gm

    if idx == 7:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([2, 0, 1])
        gm.set_mutation_ranges()
        return gm

    if idx == 8:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([2, 1, 0])
        gm.set_mutation_ranges()
        return gm

def scale_jp_graph(graph, scale):
    for jp in graph.nodes:
        jp.r = jp.r*scale
    return graph

def scale_graph_manager(graph_manager, scale):
    for jp in graph_manager.graph.nodes:
        generator_info:GeneratorInfo = graph_manager.generator_dict[jp]
        if generator_info.initial_coordinate is not None:
            generator_info.initial_coordinate = np.array(generator_info.initial_coordinate)*scale
        if generator_info.mutation_type != MutationType.RELATIVE_PERCENTAGE:
            for i, r in enumerate(generator_info.mutation_range):
                if r is not None:
                    generator_info.mutation_range[i] = (r[0]*scale, r[1]*scale)
            for i, r in enumerate(generator_info.freeze_pos):
                if r is not None:
                    generator_info.freeze_pos[i] = r*scale
    return graph_manager