from collections import deque
from itertools import combinations
from copy import deepcopy
from shlex import join
from typing import Optional
from matplotlib import pyplot as plt
from mediapy import set_ffmpeg

import numpy as np
import numpy.linalg as la

from scipy.spatial.transform import Rotation as R

import modern_robotics as mr
from modern_robotics import VecToso3
import networkx as nx

from auto_robot_design.description.kinematics import (
    Joint,
    Link,
    get_ground_joints,
    get_endeffector_joints,
)
from auto_robot_design.description.utils import (
    calc_weight_for_span,
    draw_joint_point,
    get_pos,
    weight_by_dist_active,
)


class KinematicGraph(nx.Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.EE: Optional[Link] = None
        self.G: Optional[Link] = None
        self.main_branch: nx.Graph = nx.Graph()
        self.kinematic_tree: nx.Graph = nx.Graph()
        self.joint_graph: nx.Graph = nx.Graph()
        self.jps_graph: nx.Graph = nx.Graph()

    @property
    def name2joint(self):
        return {j.jp.name: j for j in self.joint_graph.nodes()}
    
    @property
    def name2jp(self):
        return {j.jp.name: j.jp for j in self.jps_graph.nodes()}
    
    @property
    def name2link(self):
        return {l.name: l for l in self.nodes()}
    
    @property
    def active_joints(self):
        return set(map(lambda x: x, filter(lambda j: j.jp.active, self.joint_graph.nodes())))
    
    def define_main_branch(self):
        ground_joints = sorted(
            list(get_ground_joints(self.joint_graph)),
            key=lambda x: la.norm(x.jp.r),
        )
        main_G_j = ground_joints[0]
        ee_joints = sorted(
            list(get_endeffector_joints(self.joint_graph)),
            key=lambda x: la.norm(x.jp.r - main_G_j.jp.r),
        )
        main_EE_j = ee_joints[0]
        
        j_in_m_branch = nx.shortest_path(
            self.joint_graph, main_G_j, main_EE_j, weight=weight_by_dist_active
        )
        main_branch = [self.G]
        # print([(j.jp.name, l.name) for j in j_in_m_branch for l in j.links])
        for i in range(0, len(j_in_m_branch)):
            main_branch.append(
                (j_in_m_branch[i].links - set(main_branch)).pop()
            )
        self.main_branch = self.subgraph(main_branch)
        return self.main_branch

    def define_span_tree(self, main_branch=None):
        if main_branch:
            self.main_branch = self.subgraph(main_branch)
        main_branch = self.main_branch

        for edge in self.edges(data=True):
            weight = calc_weight_for_span(edge, self)
            self[edge[0]][edge[1]]["weight"] = weight

        for m_edge in main_branch.edges():
            self[m_edge[0]][m_edge[1]]["weight"] = (
                self[m_edge[0]][m_edge[1]]["weight"] + 1000
            )

        self.kinematic_tree = nx.maximum_spanning_tree(self, algorithm="prim")
        return self.kinematic_tree

    @property
    def joint2edge(self):
        edges = self.edges(data=True)
        j2edge = {j: set() for j in self.joint_graph.nodes()}
        for data in edges:
            j2edge[data[2]["joint"]] = set((data[0], data[1]))
        return j2edge

    # def get_next_link(self, joint: Joint, prev_link):
    #     joint.link_in = prev_link
    #     joint.link_out = (self.joint2edge[joint] - set([prev_link])).pop()
    #     return joint.link_out
    
    def get_in_joint(self, prev_link, next_link):
        in_joint: Joint = self[prev_link][next_link]["joint"]
        in_joint.link_out = next_link
        in_joint.link_in = prev_link
        return in_joint
    
    def set_link_frame_by_joints(self,link, in_j, out_j):
            ez = np.array([0,0,1])
            v_w = out_j.jp.r - in_j.jp.r
            angle = np.arccos(np.inner(ez, v_w) / 
                            la.norm(v_w) / 
                            la.norm(ez))
            
            axis = mr.VecToso3(ez) @ v_w
            if np.sum(axis) == 0 and angle in (0.0, np.pi):
                axis = in_j.jp.w
            else:
                axis /= la.norm(axis)
            
            
            rot = R.from_rotvec(axis * angle)
            pos = in_j.jp.r
            
            link.frame = mr.RpToTrans(rot.as_matrix(), 
                                        pos)
            
            pos_link_joints = [j.jp.r for j in link.joints]
            mean_pos = np.mean(pos_link_joints, axis=0)
            link.inertial_frame[:,3] = np.round(mr.TransInv(link.frame) @ np.r_[mean_pos, 1], 5)

    def define_link_frames(self):
        links = self.nodes() - set([self.G])
        
        path_from_G = nx.shortest_path(self.kinematic_tree, self.G)
        path_main_branch: list = nx.shortest_path(self.kinematic_tree, self.G, self.EE)
        
        for link in links:
            path_G_link = path_from_G[link]
            prev_link = path_G_link[-2]
            if len(link.joints) == 2:
                close_j_to_G: Joint = self.get_in_joint(prev_link,link)
                out_joint = (link.joints - set([close_j_to_G])).pop()
                if out_joint.link_in is None:
                    out_joint.link_in = link
                elif out_joint.link_out is None:
                    out_joint.link_out = link
                self.set_link_frame_by_joints(link, close_j_to_G, out_joint)

            elif len(link.joints) > 2:
                if link in path_main_branch:
                    num = path_main_branch.index(link)
                    prev_link = path_main_branch[num-1]

                in_joint = self.get_in_joint(prev_link,link)
                out_joints = link.joints - set([in_joint])
                j2edge = self.joint2edge
                
                joint_tree = set(filter(lambda j: tuple(j2edge[j]) in self.kinematic_tree.edges(),out_joints))
                joint_main = set(filter(lambda j: tuple(j2edge[j]) in self.main_branch.edges(),joint_tree))
                    
                if joint_main:
                    out_joint = joint_main.pop()
                elif joint_tree:
                    out_joint = sorted(list(joint_tree),
                                    key=lambda out_j: la.norm(out_j.jp.r - in_joint.jp.r),
                                    reverse=True)[0]
                else:
                    out_joint = sorted(list(out_joints),
                                    key=lambda out_j: la.norm(out_j.jp.r - in_joint.jp.r),
                                    reverse=True)[0]
                self.set_link_frame_by_joints(link, in_joint, out_joint)
                # out_joint.link_in = link
                # other_out_joint = out_joints - set([out_joint])
                for j in out_joints:
                    if j.link_in is None:
                        j.link_in = link
                    elif j.link_out is None:
                        j.link_out = link
            else:
                in_joint = self.get_in_joint(prev_link,link)
                link.frame[:3,3] = in_joint.jp.r
        
        for edges in self.kinematic_tree.edges(data=True):
            joint: Joint = self[edges[0]][edges[1]]["joint"]
            prev_link = joint.link_in
            next_link = joint.link_out
            joint.frame = mr.TransInv(prev_link.frame) @ next_link.frame
            
        for edges in self.edges() - self.kinematic_tree.edges():
            
            joint: Joint = self[edges[0]][edges[1]]["joint"]
            prev_link = joint.link_in
            next_link = joint.link_out
            joint.is_constraint = True
            # print(prev_link.name, joint.jp.name)#, next_link.name)
            prev_in_joint = list(filter(lambda j: j.link_in and j.link_in == prev_link, prev_link.joints))[0]
            
            rot, __ = mr.TransToRp(prev_in_joint.frame)
            pos = mr.TransInv(prev_link.frame) @ np.r_[joint.jp.r, 1]

            joint.frame = mr.RpToTrans(rot, pos[:3])
            
    def set_random_actuators(self, actuators: list):
        """
        Sets random actuators for the active joints in the mechanism.

        Parameters:
        - actuators (list): A list of available actuators.

        Returns:
        - None

        Example usage:
        >>> from auto_robot_design.description.mechanism import KinematicGraph
        >>> from auto_robot_design.description.actuators import t_motor_actuators
        >>> mechanism = KinematicGraph()
        >>> mechanism.set_random_actuators(t_motor_actuators)
        """
        active_joints = [j for j in self.joint_graph.nodes() if j.jp.active]
        list_actuators = np.random.choice(actuators, len(active_joints))
        
        for joint, actuator in zip(active_joints, list_actuators):
            joint.actuator = actuator
        
    def set_actuator_to_all_joints(self, actuator):
        """
        Sets the actuator for all active joints in the mechanism.

        Parameters:
        - actuator: The actuator object to be set for all joints.

        Returns:
        None
        """
        active_joints = [j for j in self.joint_graph.nodes() if j.jp.active]
        for joint in active_joints:
            joint.actuator = actuator
    
    def set_joint2actuator(self, joint2actuator):
        """
        Sets the actuator for each joint in the mechanism.

        Parameters:
        - joint2actuator (dict): A dictionary with the joint name as the key and the actuator as the value.

        Returns:
        None
        """
        if isinstance(joint2actuator, dict):
            for joint, actuator in joint2actuator.items():
                self.name2joint[joint].actuator = actuator
        elif isinstance(joint2actuator, (tuple, list)):
            for joint, actuator in joint2actuator:
                self.name2joint[joint].actuator = actuator
        else:
            raise ValueError("joint2actuator must be a dictionary or a tuple(list) of tuples(lists).")


def JointPoint2KinematicGraph(jp_graph: nx.Graph):
    """
    Converts a joint point graph to a kinematic graph.

    Args:
        jp_graph (nx.Graph): The joint point graph to convert.

    Returns:
        KinematicGraph: The converted kinematic graph.
    """
    
    # Change JP nodes to external nodes with kinematic and dynamic properties
    JP2Joint = {}
    for jp in jp_graph.nodes():
        JP2Joint[jp] = Joint(jp)
    jps_graph = deepcopy(jp_graph)
    joint_graph: nx.Graph = nx.relabel_nodes(jp_graph, JP2Joint)
    
    # Create ground and end-effector links
    ground_joints = set([JP2Joint[jp] for jp in get_ground_joints(jp_graph)])
    ee_joints = set([JP2Joint[jp] for jp in get_endeffector_joints(jp_graph)])

    ground_link = Link(ground_joints, "G")
    ee_link = Link(ee_joints, "EE")

    for joint in ground_joints:
        joint.link_in = ground_link
    for joint in ee_joints:
        joint.link_out = ee_link

    # Create stack of joints and add ground joints
    stack_joints: deque[Joint] = deque(maxlen=len(JP2Joint.values()))
    stack_joints += list(ground_joints)

    # Create expedited set of joints
    exped_j = set()
    # Create list of links
    links: list[Link] = [ee_link, ground_link]
    
    while stack_joints:
        # Get the current joint
        current_joint = stack_joints.pop()
        # current_joint = JP2Joint[curr_jp]
        # Get the link that the current joint is connected to
        L = next(iter(current_joint.links))
        # Add the current joint to the expedited set
        exped_j.add(current_joint)
        L1 = joint_graph.subgraph(L.joints)
        # Get the neighbors of the current joint that are not in the link
        N = set(joint_graph.neighbors(current_joint)) - L.joints
        nextN = {}
        lenNN = {}
        # Get the neighors of the neighbors of the current joint.
        # And calculate the number of neighbors that are in the link
        for n in N:
            nextN[n] = set(joint_graph.neighbors(n))
            lenNN[n] = len(nextN[n] & L.joints)
        if len(L.joints) <= 2: # If the link has less than or equal to 2 joints
            # Create a new link with the current joint and the neighbors
            L2 = Link(joints=(N | set([current_joint])))
            for j in L2.joints:
                j.links.add(L2)
        # If the link has more than 2 joints and number of neighbors is 1
        elif len(N) == 1:
            N = N.pop()
            if lenNN[N] == 1:
                L2 = Link(joints=set([N, current_joint]))
                for j in L2.joints:
                    j.links.add(L2)
            else:
                L.joints.add(N)
                N.links.add(L)
                continue
        # Otherwise
        else:
            more_one_adj_L1 = set(filter(lambda n: lenNN[n] > 1, N))
            for n in more_one_adj_L1:
                L.joints.add(n)
                n.links.add(L)
            less_one_adj_L1 = N - more_one_adj_L1
            if len(less_one_adj_L1) > 1:
                N = less_one_adj_L1
                L2 = Link(joints=(N | set([current_joint])))
                for j in L2.joints:
                    j.links.add(L2)
            else:
                N = list(less_one_adj_L1)[0]
                L2 = Link(joints=set([N, current_joint]))
                N.links.add(L2)
        links.append(L2)
        # Add the neighbors to the stack
        if isinstance(N, set):
            intersting_joints = set(filter(lambda n: len(n.links) < 2, N))
            stack_joints += list(intersting_joints)
        else:
            intersting_joints = N if len(N.links) < 2 else set()
            stack_joints.append(N)
        stack_joints = deque(filter(lambda j: len(j.links) < 2, stack_joints))

    kin_graph = KinematicGraph()
    kin_graph.EE = ee_link
    kin_graph.G = ground_link
    kin_graph.joint_graph = joint_graph
    kin_graph.jps_graph = jps_graph
    # Add edges to the kinematic graph
    for joint in joint_graph.nodes():
        connected_links = list(joint.links)
        if len(connected_links) == 2:
            kin_graph.add_edge(connected_links[0], connected_links[1], joint=joint)
    Link.instance_counter = 0
    return kin_graph


def define_link_frames(
    graph,
    span_tree,
    init_link="G",
    in_joint=None,
    main_branch=[],
    all_joints=set(),
    **kwargs
):
    if init_link == "G" and in_joint is None:
        kwargs = {}
        kwargs["ez"] = np.array([0, 0, 1, 0])
        kwargs["joint2edge"] = {
            data[2]["joint"]: set((data[0], data[1]))
            for data in span_tree.edges(data=True)
        }

        kwargs["get_next_link"] = lambda joint, prev_link: (
            (kwargs["joint2edge"][joint] - set([prev_link])).pop()
        )

        graph.nodes()["EE"]["frame_geom"] = (
            np.array([0, 0, 0]),
            np.array([0, 0, 0, 1]),
        )

        graph.nodes()["G"]["frame"] = (np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
        graph.nodes()["G"]["frame_geom"] = (np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
        graph.nodes()["G"]["H_w_l"] = mr.RpToTrans(np.eye(3), np.zeros(3))
        graph.nodes()["G"]["m_out"] = (
            span_tree[main_branch[0]][main_branch[1]]["joint"],
            main_branch[1],
        )
        graph.nodes()["G"]["out"] = {
            j: kwargs["get_next_link"](j, "G")
            for j in graph.nodes()["G"]["link"].joints
        }
        for j in graph.nodes()["G"]["out"]:
            define_link_frames(
                graph, span_tree, "G", j, main_branch, all_joints, **kwargs
            )
        return graph

    data_prev_link = graph.nodes()[init_link]
    link = kwargs["get_next_link"](in_joint, init_link)

    graph.nodes()[link]["in"] = (in_joint, init_link)
    sorted_out_jj = sorted(
        list(
            graph.nodes()[link]["link"].joints
            & set(kwargs["joint2edge"].keys()) - set([in_joint])
        ),
        key=lambda x: la.norm(x.r - in_joint.r),
        reverse=True,
    )

    H_w_L1 = data_prev_link["H_w_l"]
    if sorted_out_jj:
        if link in main_branch:
            i = np.argwhere(np.array(main_branch) == link).squeeze()
            graph.nodes()[link]["m_out"] = (
                span_tree[main_branch[i]][main_branch[i + 1]]["joint"],
                main_branch[i + 1],
            )
        else:
            graph.nodes()[link]["m_out"] = (
                sorted_out_jj[0],
                kwargs["get_next_link"](sorted_out_jj[0], link),
            )
        graph.nodes()[link]["out"] = {
            j: kwargs["get_next_link"](j, link) for j in sorted_out_jj
        }
        ee_jj = graph.nodes()[link]["m_out"][0].r
        v_w = graph.nodes()[link]["m_out"][0].r - in_joint.r
    else:
        if link == "EE":
            ee_jj = all_joints - set(
                map(lambda x: x[2]["joint"], graph.edges(data=True))
            )
        else:
            ee_jj = (all_joints - set(kwargs["joint2edge"].keys())) & graph.nodes()[
                link
            ]["link"].joints
        if ee_jj:
            # G.nodes()[link]["out"] = {j for j in ee_jj}
            ee_jj = sorted(
                list(ee_jj),
                key=lambda x: la.norm(x.r - in_joint.r),
                reverse=True,
            )
            graph.nodes()[link]["m_out"] = (ee_jj[0],)
            ee_jj = ee_jj[0].r
            v_w = ee_jj - in_joint.r
        else:
            ee_jj = in_joint.r
            v_w = np.array([0, 0, 1])
    ez_l_w = H_w_L1 @ kwargs["ez"]
    angle = np.arccos(np.inner(ez_l_w[:3], v_w) / la.norm(v_w) / la.norm(ez_l_w[:3]))
    axis = mr.VecToso3(ez_l_w[:3]) @ v_w
    axis /= la.norm(axis)

    pos = mr.TransInv(H_w_L1) @ np.array([*in_joint.r.tolist(), 1])
    pos = np.round(pos, 15)
    rot = R.from_rotvec(axis * angle)
    H_w_L2 = H_w_L1 @ mr.RpToTrans(rot.as_matrix(), pos[:3])
    graph.nodes()[link]["H_w_l"] = H_w_L2
    graph.nodes()[link]["frame"] = (pos[:3], rot.as_quat())
    graph.nodes()[link]["frame_geom"] = (
        ((mr.TransInv(H_w_L2) @ np.array([*ee_jj.tolist(), 1])) / 2)[:3],
        np.array([0, 0, 0, 1]),
    )
    if link == "EE":
        return graph
    if graph.nodes()[link].get("out", {}):
        for jj_out in graph.nodes()[link]["out"]:
            if jj_out in kwargs["joint2edge"].keys():
                define_link_frames(
                    graph, span_tree, link, jj_out, main_branch, all_joints, **kwargs
                )
    return graph
