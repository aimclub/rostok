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
from auto_robot_design.description.utils import draw_joint_point


class MutationType(Enum):
    """Enumerate for mutation types."""
    # UNMOVABLE = 0  # Unmovable joint that is not used for optimization
    ABSOLUTE = 1  # The movement of the joint are in the absolute coordinate system and are relative to the initial position
    RELATIVE = 2  # The movement of the joint are relative to some other joint or joints and doesn't have an initial position
    # The movement of the joint are relative to some other joint or joints and doesn't have an initial position. The movement is in percentage of the distance between the joints.
    RELATIVE_PERCENTAGE = 3


@dataclass
class GeneratorInfo:
    """Information for node of a generator."""
    mutation_type: int = MutationType.ABSOLUTE
    initial_coordinates: np.ndarray = np.zeros(3)# this is the position for calculation of range using mutation_range
    mutation_range: List[Optional[Tuple]] = field(
        default_factory=lambda: [None, None, None])
    relative_to: Optional[Union[JointPoint, List[JointPoint]]] = None
    freeze_pos: List[Optional[Tuple]] = field(
        default_factory=lambda: [None, 0, None])
    vis_pos: Optional[np.ndarray] = None


@dataclass
class ConnectionInfo:
    """Description of a point for branch connection."""
    connection_jp: JointPoint  # joint point that is used for connection
    connection_base: Optional[List[JointPoint]]
    # this parameter is used to set the mutation range of the branch joints
    relative_initial_coordinates: np.ndarray = np.zeros(3)
    relative_mutation_range: List[Optional[Tuple]] = field(default_factory=lambda: [None, None, None])
    freeze_pos: List[Optional[Tuple]] = field(default_factory=lambda: [None, 0, None])


class TopologyManager2D():
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.mutation_ranges = {}
        self.branch_ends = []
        self.edges = []
        self.generator_dict = {}
        self.connections = []
        self.connection_buffer = []
        self.ground_connection_counter = 0
        self.connection_counter = 0

    def add_absolute_node(self, jp: JointPoint, initial_coordinates, mutation_range, freeze_pos=[None, 0, None], parent_branch_idx=None, vis_pos=None):
        """Adds a joint point to the graph with absolute mutation type.

        Absolute mutation requires initial coordinates and mutation range. Final coordinates are calculated as initial_coordinates + value from mutation_range.
        Joint properties are set by JointPoint object.
        Args:
            jp (JointPoint): joint to be added to the mechanism.
            initial_coordinates (_type_): reference coordinates to calculate the final position.
            mutation_range (_type_): range of possible joint positions.
            freeze_pos (list, optional): override mutation range and sets the corresponding position to the frozen value. Defaults to [None,0,None].
            parent_branch_idx (_type_, optional): The joint must become the end of a branch.If None the new branch is created. Defaults to None.

        Raises:
            ValueError: JP names must be unique.
        """
        if jp.name in self.mutation_ranges:
            raise ValueError(
                f"Joint point {jp.name} already exists in the graph.")

        if parent_branch_idx is None:
            self.branch_ends.append([jp, jp])
            self.graph.add_node(jp)
        else:
            self.graph.add_edge(self.branch_ends[parent_branch_idx][1], jp)
            #self.edges.append((self.branch_ends[parent_branch_idx][1], jp))
            self.connection_buffer = [self.branch_ends[parent_branch_idx][1], jp]
            self.branch_ends[parent_branch_idx][1] = jp
        self.generator_dict[jp] = GeneratorInfo(
            mutation_type=MutationType.ABSOLUTE, initial_coordinates=initial_coordinates, mutation_range=mutation_range, freeze_pos=freeze_pos)

        # if vis_pos is not None:
        #     self.generator_dict[jp].vis_pos = vis_pos
        # else:
        #     self.generator_dict[jp].vis_pos = freeze_pos
        #     for i, pos in enumerate(freeze_pos):
        #         if pos is None:
        #             self.generator_dict[jp].vis_pos[i] = initial_coordinates[i]+(mutation_range[i][1]-mutation_range[i][0])/2

    def add_connection(self, self_mutation_range: np.ndarray, dependent_mutation_range:np.ndarray, relative_initial_coordinates=np.zeros(3), self_freeze_pos:list=[None, 0, None], dependent_freeze_pos:list=[None, 0, None], ground=True, self_vis_pos=None):
        """Create a connection point on either ground or the last added link.
            
            A connection point has two mutation ranges: one for connection point on the link and another for the dependent joint on the branch.
            The connection can be used to attach some independent branch, in that case dependent range is not used.

        Args:
            self_mutation_range (np.ndarray): mutation range for the connection point.
            dependent_mutation_range (np.ndarray): 
            ground (bool, optional): If true adds connection to ground, else to the last added link. Defaults to True.
        """

        if ground:
            ground_connection_jp = JointPoint(
                r=None,
                w=np.array([0, 1, 0]),
                attach_ground=True,
                active=True,
                name=f"Ground_connection_{self.ground_connection_counter}"
            )
            self.ground_connection_counter += 1
            self.connections.append(ConnectionInfo(
                connection_jp=ground_connection_jp, relative_initial_coordinates=relative_initial_coordinates,connection_base=None, relative_mutation_range=dependent_mutation_range, freeze_pos=dependent_freeze_pos))
            self.generator_dict[ground_connection_jp] = GeneratorInfo(mutation_type=MutationType.ABSOLUTE, initial_coordinates=np.zeros(3), mutation_range=self_mutation_range, freeze_pos=self_freeze_pos)
        else:
            connection_jp = JointPoint(r=None, w=np.array(
                [0, 1, 0]), attach_ground=False, active=False, name=f"Connection_{self.connection_counter}")
            self.connection_counter += 1
            self.connections.append(ConnectionInfo(
                connection_jp=connection_jp, connection_base=self.connection_buffer, relative_mutation_range=dependent_mutation_range, relative_initial_coordinates=relative_initial_coordinates,freeze_pos=dependent_freeze_pos))
            self.generator_dict[connection_jp] = GeneratorInfo(mutation_type=MutationType.RELATIVE_PERCENTAGE,relative_to=self.connection_buffer, initial_coordinates=None, mutation_range=self_mutation_range, freeze_pos=self_freeze_pos)

    def add_relative_node(self, jp: JointPoint, initial_coordinates=np.zeros(3), mutation_range=None, parent_branch_idx=None, freeze_pos=[None, 0, None]):
        """Add a joint that has position relative to another joint.

        Args:
            jp (JointPoint): joint to be added to the mechanism.
            mutation_range (_type_, optional): mutation range. None value means the node is to be linked to a connection. Defaults to None.
            parent_branch_idx (_type_, optional): If None the node starts a new branch, else connects to the corresponding branch. Defaults to None.
            freeze_pos (list, optional): override mutation range if needed. Defaults to [None, 0, None].

        Raises:
            ValueError: JP names must be unique.
        """
        if jp.name in self.mutation_ranges:
            raise ValueError(
                f"Joint point {jp.name} already exists in the graph.")
        # if parent branch is None we add a new branch
        if parent_branch_idx is None:
            self.branch_ends.append([jp, jp])
            self.graph.add_node(jp)
            self.generator_dict[jp] = GeneratorInfo(
                mutation_type=MutationType.RELATIVE, relative_to=None, mutation_range=mutation_range, freeze_pos=freeze_pos)
        else:
            parent_jp = self.branch_ends[parent_branch_idx][1]
            self.graph.add_edge(parent_jp, jp)
            self.branch_ends[parent_branch_idx][1] = jp
            self.generator_dict[jp] = GeneratorInfo(
                mutation_type=MutationType.RELATIVE, relative_to=parent_jp, mutation_range=mutation_range, freeze_pos=freeze_pos)
            self.connection_buffer = [parent_jp, jp]

    def add_dependent_connection(self, connection_idx, branch_idx, connect_head=True):
        connection = self.connections[connection_idx]
        jp = connection.connection_jp
        # if the connection is used in topology we add its edges to the graph
        link_jp = connection.connection_base
        if link_jp is not None:
            for parent_jp in link_jp:
                self.graph.add_edge(parent_jp, jp)

        if connect_head:
            connected_jp = self.branch_ends[branch_idx][0]
            self.generator_dict[connected_jp].relative_to = jp
            self.generator_dict[connected_jp].mutation_range = connection.relative_mutation_range

        else:
            connected_jp = self.branch_ends[branch_idx][1]
            self.generator_dict[connected_jp].relative_to = jp
            self.generator_dict[connected_jp].mutation_range = connection.relative_mutation_range

        self.graph.add_edge(jp, connected_jp)

    def add_independent_connection(self, node_1, node_2):
        for connection in self.connections:
            if connection.connection_jp == node_1 and not connection.connection_base is None:
                for parent_jp in connection.connection_base:
                    self.graph.add_edge(parent_jp, node_1)
            if connection.connection_jp == node_2 and not connection.connection_base is None:
                for parent_jp in connection.connection_base:
                    self.graph.add_edge(parent_jp, node_2)

        self.graph.add_edge(node_1, node_2)

    # def add_connection_node(self,jp,mutation_range, parent_pair, freeze_pos=[None,0,None]):

    #     self.generator_dict[jp.name] = GeneratorInfo(mutation_type=MutationType.RELATIVE, relative_to=parent_pair, mutation_range=mutation_range,freeze_pos=freeze_pos)

    def get_pos(self):
        """Return the dictionary of type {label: [x_coordinate, z_coordinate]} for the JP graph

        Args:
            G (nx.Graph): a graph with JP nodes

        Returns:
            dict: dictionary of type {node: [x_coordinate, z_coordinate]}
        """
        pos = {}
        for node in self.graph:
            pos[node] = [node.r[0], node.r[2]]

        return pos

    def visualize(self, draw_labels=True):
        'Visualize the current graph'
        self.set_mutation_ranges()
        self.graph = self.get_graph(
            self.generate_central_from_mutation_range())
        draw_joint_point(self.graph, draw_labels=draw_labels)

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

        for key, value in self.generator_dict.items():
            if value.mutation_type == MutationType.RELATIVE or value.mutation_type == MutationType.RELATIVE_PERCENTAGE:
                for i, r in enumerate(value.mutation_range):
                    if r is not None and value.freeze_pos[i] is None:
                        self.mutation_ranges[key.name+'_'+str(i)] = r
            elif value.mutation_type == MutationType.ABSOLUTE:
                for i, r in enumerate(value.mutation_range):
                    if r is not None and value.freeze_pos[i] is None:
                        self.mutation_ranges[key.name+'_'+str(i)] = (
                            r[0]+value.initial_coordinates[i], r[1]+value.initial_coordinates[i])

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
                        jp.r[i] = gi.initial_coordinates[i]
                    else:
                        raise ValueError(
                            f"Failed to assign value for Joint Point {jp.name} coordinate {i}")

            elif gi.mutation_type == MutationType.RELATIVE:
                if isinstance(gi.relative_to, list) and len(gi.relative_to) == 2:
                    jp.r = (gi.relative_to[0].r + gi.relative_to[1].r)/2

                for i, r in enumerate(gi.mutation_range):
                    if gi.freeze_pos[i] is not None:
                        parameter = gi.freeze_pos[i]
                    elif r is not None:
                        parameter = parameters[parameter_counter]
                        parameter_counter += 1
                    else:
                        raise ValueError(
                            f"Failed to assign value for Joint Point {jp.name} coordinate {i}")

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
                                jp.r += parameter * link_direction/link_length
                        #     jp.r += parameters[parameter_counter]*link_direction/link_length
                        #     jp.r += parameters[parameter_counter]*np.array([-link_direction[2],link_direction[1],link_direction[0]])/link_length

            elif gi.mutation_type == MutationType.RELATIVE_PERCENTAGE:
                if isinstance(gi.relative_to, list) and len(gi.relative_to) == 2:
                    jp.r = (gi.relative_to[0].r + gi.relative_to[1].r)/2

                for i, r in enumerate(gi.mutation_range):
                    if gi.freeze_pos[i] is not None:
                        parameter = gi.freeze_pos[i]
                    elif r is not None:
                        parameter = parameters[parameter_counter]
                        parameter_counter += 1
                    else:
                        raise ValueError(
                            f"Failed to assign value for Joint Point {jp.name} coordinate {i}")

                    if isinstance(gi.relative_to, JointPoint):
                        raise ValueError(
                            'Relative percentage mutation type should have a list of joints as relative_to')
                    else:
                        # TODO: this part would fail in 3D case
                        if len(gi.relative_to) == 2:
                            link_direction = gi.relative_to[0].r - \
                                gi.relative_to[1].r
                            link_ortogonal = np.array(
                                [-link_direction[2], link_direction[1], link_direction[0]])
                            link_length = np.linalg.norm(link_direction)
                            if i == 0:
                                jp.r += parameter * link_ortogonal
                            if i == 2:
                                jp.r += parameter * link_direction

        return self.graph
