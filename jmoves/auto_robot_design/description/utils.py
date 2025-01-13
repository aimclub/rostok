from itertools import permutations
import networkx as nx
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from auto_robot_design.description.kinematics import Link
from trimesh import Trimesh
from scipy.spatial.transform import Rotation as R
import modern_robotics as mr

# from auto_robot_design.description.mechanism import KinematicGraph

def all_combinations_active_joints_n_actuator(graph: nx.Graph, actuators):
    """
    Generates all possible combinations of active joints and actuators.

    Args:
        graph (nx.Graph): The graph representing the robot design.
        actuators (list): List of available actuators.

    Returns:
        list: List of tuples representing pairs of name of active joints and actuators.
    """
    try:
        active_joints = [j.jp.name for j in graph.active_joints]
    except AttributeError:
        active_joints = [j.name for j in graph.nodes() if j.active]

    combination_actuator = permutations(actuators, len(active_joints))
    pairs_joint_actuator = []
    for combination in combination_actuator:
        pairs_joint_actuator.append(tuple(zip(active_joints, combination)))
    return pairs_joint_actuator


def trans2_xyz_rpy(trans: np.ndarray) -> tuple[list[float]]:
    rot, pos = mr.TransToRp(trans)
    return (pos.tolist(), R.from_matrix(rot).as_euler("xyz"))

def trans2_xyz_quat(trans: np.ndarray) -> tuple[list[float]]:
    rot, pos = mr.TransToRp(trans)
    return (pos.tolist(), R.from_matrix(rot).as_quat("xyz"))

def tensor_inertia_sphere(density, r):
    mass = 4/3 * np.pi * r**3 * density
    central_inertia =  2/5 * mass * r**2
    
    tensor_inertia = np.diag([central_inertia for __ in range(3)])
    
    return mass, tensor_inertia

def tensor_inertia_sphere_by_mass(mass, r):
    central_inertia =  2/5 * mass * r**2
    
    tensor_inertia = np.diag([central_inertia for __ in range(3)])
    
    return tensor_inertia

def tensor_inertia_box(density, x, y, z):
    mass = x*y*z*density
    inertia = lambda a1, a2:  1/12 * mass * (a1**2 + a2**2)
    
    inertia_xx = inertia(y, z)
    inertia_yy = inertia(x, z)
    inertia_zz = inertia(x, y)
    
    tensor_inertia = np.diag([inertia_xx, inertia_yy, inertia_zz])
    
    return mass, tensor_inertia

def tensor_inertia_mesh(density, mesh: Trimesh):
    mesh.density = density
    return mesh.mass, mesh.moment_inertia

def weight_by_dist_active(e1, e2, d):
    dist = la.norm(e1.jp.r - e2.jp.r)
    w = dist * 2 if any([j.jp.active for j in [e1, e2]]) else dist
    return np.round(1/w, 3)

def calc_weight_for_span(edge, graph: nx.Graph):
    length_to_EE = [nx.shortest_path_length(graph, graph.EE, target=e) for e in edge[:2]]
    edge_min_length = np.argmin(length_to_EE)
    min_length_to_EE = min(length_to_EE)
    next_joints_link = edge[edge_min_length].joints - set([edge[-1]["joint"]])
    if next_joints_link:
        length_next_j_to_j = max([la.norm(edge[-1]["joint"].jp.r - next_j.jp.r) for next_j in next_joints_link])
    else:
        length_next_j_to_j = 0
    if edge[-1]["joint"].jp.active:
        weight = np.round(len(graph.nodes()) * 100 + min_length_to_EE * 10 + length_next_j_to_j/10, 3)
    else:
        weight = np.round(min_length_to_EE * 10 + length_next_j_to_j/10, 3)
    # print(edge[0].name, edge[1].name, weight)
    return weight


def calc_weight_for_main_branch(edge, graph: nx.Graph):
    pass


def get_pos(G: nx.Graph):
    """Return the dictionary of type {node: [x_coordinate, z_coordinate]} for the JP graph

    Args:
        G (nx.Graph): a graph with JP nodes

    Returns:
        dict: dictionary of type {node: [x_coordinate, z_coordinate]}
    """
    pos = {}
    for node in G:
        pos[node] = [node.r[0], node.r[2]]

    return pos


def plot_link(L: Link, graph: nx.Graph, color):
    sub_g_l = graph.subgraph(L.joints)
    pos = get_pos(sub_g_l)
    nx.draw(
        sub_g_l,
        pos,
        node_color=color,
        linewidths=1.5,
        edge_color=color,
        node_shape="o",
        node_size=100,
        width=5,
        with_labels=False,
    )

def draw_links(kinematic_graph, JP_graph: nx.Graph):
    links = kinematic_graph.nodes()
    EE_joint = next(iter(kinematic_graph.EE.joints))
    colors = range(len(links))
    draw_joint_point(JP_graph) 
    for link, color in zip(links, colors):
        sub_graph_l = JP_graph.subgraph(set([j.jp for j in link.joints]))
        name_link = link.name
        options = {
            "node_color": "orange",
            "edge_color": "orange",
            "alpha": color/len(links),
            "width": 5,
            "edge_cmap": plt.cm.Blues,
            "linewidths": 1.5,
            "node_shape": "o",
            "node_size": 100,
            "with_labels": False,
        }
        pos = get_pos(sub_graph_l)
        list_pos = [p for p in pos.values()]
        if len(list_pos) == 1:
            pos_name = np.array(list_pos).squeeze() + np.ones(2) * 0.2 * la.norm(EE_joint.jp.r)/5
        else:
            pos_name = np.mean([p for p in pos.values()], axis=0)
        nx.draw(sub_graph_l, pos, **options)
        plt.text(pos_name[0],pos_name[1], name_link, fontsize=15)

def draw_joint_point(graph: nx.Graph, labels=0, draw_legend=True, draw_lines=False, **kwargs):
    pos = get_pos(graph)
    pos_list = [p for p in pos.values()]
    pos_matrix = np.array(pos_list)
    min_x, min_y = np.round(np.min(pos_matrix, axis=0),2)
    max_x, max_y = np.round(np.max(pos_matrix, axis=0),2)
    for key, value in pos.items():
        value
    G_pos = np.array(
        list(
        map(
            lambda n: [n.r[0], n.r[2]],
            filter(lambda n: n.attach_ground, graph),
        )
        )
    )
    EE_pos = np.array(
        list(
        map(
            lambda n: [n.r[0], n.r[2]],
            filter(lambda n: n.attach_endeffector, graph),
        )
        )
    )
    active_j_pos = np.array(
        list(
        map(
            lambda n: [n.r[0], n.r[2]],
            filter(lambda n: n.active, graph),
        )
        )
    )
    if labels==0:
        labels = {n:n.name for n in graph.nodes()}
    elif labels==1:
        labels = {n:i for i,n in enumerate(graph.nodes())}
    else:
        labels = {n:str() for n in graph.nodes()}
    nx.draw(
        graph,
        pos,
        node_color="w",
        linewidths=3,
        edgecolors="k",
        node_shape="o",
        node_size=150,
        with_labels=False,
        width=2,
    )
    #pos_labels = {g:np.array(p) + np.array([-0.2, 0.2])*la.norm(EE_pos)/5 for g, p in pos.items()}
    pos_labels = {}
    coef = 1000
    pos_additions = [np.array([0.2, 0.2])*la.norm(EE_pos)/coef, np.array([0.2, -0.2])*la.norm(EE_pos)/coef, 
                     np.array([0.2,-0.2])*la.norm(EE_pos)/coef, np.array([-0.2, -0.2])*la.norm(EE_pos)/coef]
    for g,p in pos.items():
        pos_flag = False
        for pos_addition in pos_additions:
            new_pos = np.array(p) + pos_addition
            if all([la.norm(new_pos-op)>la.norm(EE_pos)/5 for op in pos_labels.values()]):
                pos_labels[g] = new_pos
                pos_flag = True
                break
        if not pos_flag:
            pos_labels[g] = np.array(p)
    nx.draw_networkx_labels(
        graph,
        pos_labels,
        labels,
        font_color = "#ff5A00",
        font_family = "monospace",
        font_size=20
    )

    #"#fe8a18"
    if nx.is_weighted(graph):
        edge_labels = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels,
            font_color = "c",
            font_family = "monospace"

        )
    plt.plot(G_pos[:,0], G_pos[:,1], "ok", label="Ground")
    plt.axis("equal")
    
    import matplotlib.ticker as ticker
    if draw_lines:
        ax = plt.gca()
        ax.set_axis_on()
        # ax.set_title('JP graph')
        ax.set_ylabel('z [м]')
        ax.set_xlabel('x [м]')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # ax.set_xticks(np.arange(min_x,max_x+0.1,0.1))
        # ax.set_yticks(np.arange(min_y-0.1,max_y+0.1,0.1))
        ax.set_xlim(min_x-kwargs.get("offset_lim", 0.1), max_x+kwargs.get("offset_lim", 0.1))
        ax.set_ylim(min_y-kwargs.get("offset_lim", 0.1), max_y+kwargs.get("offset_lim", 0.1))
        pass


    # plt.axis('on')
    if EE_pos.size != 0:
        plt.plot(EE_pos[:,0], EE_pos[:,1], "ob", label="EndEffector")
    plt.plot(active_j_pos[:,0], active_j_pos[:,1], "og",
             markersize=20, 
             fillstyle="none", label="Active")
    if draw_legend: plt.legend()


def draw_joint_point_widjet(graph: nx.Graph, labels=0, draw_legend=True, draw_lines=False):
    pos = get_pos(graph)
    pos_list = [p for p in pos.values()]
    pos_matrix = np.array(pos_list)
    min_x, min_y = np.min(pos_matrix, axis=0)
    max_x, max_y = np.max(pos_matrix, axis=0)
    for key, value in pos.items():
        value
    G_pos = np.array(
        list(
        map(
            lambda n: [n.r[0], n.r[2]],
            filter(lambda n: n.attach_ground, graph),
        )
        )
    )
    EE_pos = np.array(
        list(
        map(
            lambda n: [n.r[0], n.r[2]],
            filter(lambda n: n.attach_endeffector, graph),
        )
        )
    )
    active_j_pos = np.array(
        list(
        map(
            lambda n: [n.r[0], n.r[2]],
            filter(lambda n: n.active, graph),
        )
        )
    )
    if labels==0:
        labels = {n:n.name for n in graph.nodes()}
    elif labels==1:
        labels = {n:i for i,n in enumerate(graph.nodes())}
    else:
        labels = {n:str() for n in graph.nodes()}
    nx.draw(
        graph,
        pos,
        node_color="w",
        linewidths=3,
        edgecolors="k",
        node_shape="o",
        node_size=250,
        with_labels=False,
        width=2,
    )
    #pos_labels = {g:np.array(p) + np.array([-0.2, 0.2])*la.norm(EE_pos)/5 for g, p in pos.items()}
    pos_labels = {}
    coef = 1000
    pos_additions = [np.array([0.2, 0.2])*la.norm(EE_pos)/coef, np.array([0.2, -0.2])*la.norm(EE_pos)/coef, 
                     np.array([0.2,-0.2])*la.norm(EE_pos)/coef, np.array([-0.2, -0.2])*la.norm(EE_pos)/coef]
    for g,p in pos.items():
        pos_flag = False
        for pos_addition in pos_additions:
            new_pos = np.array(p) + pos_addition
            if all([la.norm(new_pos-op)>la.norm(EE_pos)/5 for op in pos_labels.values()]):
                pos_labels[g] = new_pos
                pos_flag = True
                break
        if not pos_flag:
            pos_labels[g] = np.array(p)
    nx.draw_networkx_labels(
        graph,
        pos_labels,
        labels,
        font_color = "#ff5A00",
        font_family = "monospace",
        font_size=12
    )

    #"#fe8a18"
    if nx.is_weighted(graph):
        edge_labels = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels,
            font_color = "c",
            font_family = "monospace"

        )
    plt.plot(G_pos[:,0], G_pos[:,1], "o", label="Прикреплён к базе", color="silver", ms=10)
    plt.axis("equal")
    
    import matplotlib.ticker as ticker
    if draw_lines:
        ax = plt.gca()
        ax.set_axis_on()
        # ax.set_title('JP graph')
        ax.set_ylabel('z [м]')
        ax.set_xlabel('x [м]')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # ax.set_xticks(np.arange(min_x-0.1,max_x+0.1,0.1))
        # ax.set_yticks(np.arange(min_y-0.1,max_y+0.1,0.1))
        ax.set_xlim(min_x-0.1, max_x+0.1)
        ax.set_ylim(min_y-0.1, max_y+0.1)
        pass


    # plt.axis('on')
    if EE_pos.size != 0:
        plt.plot(EE_pos[:,0], EE_pos[:,1], "o", label="Рабочий инструмент", ms=10, color="lightsteelblue")
    plt.plot(active_j_pos[:,0], active_j_pos[:,1], "og",
             markersize=22, 
             fillstyle="none", label="Актуирован")
    if draw_legend: plt.legend()

def draw_kinematic_graph(graph: nx.Graph):
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["joint"].jp.active]
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if not d["joint"].jp.active]
    labels = {l:l.name for l in graph.nodes()}
    pos = nx.planar_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=700)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    nx.draw_networkx_labels(graph, pos, labels, font_size=20, font_family="sans-serif")

    # edge_labels = nx.get_edge_attributes(graph, "weight")
    edge_labels = {(u, v):d["joint"].jp.name for (u, v, d) in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()


def draw_link_frames(kinematic_graph: nx.Graph):
    ex = np.array([1, 0, 0, 0])
    ez = np.array([0, 0, 1, 0])
    p = np.array([0, 0, 0, 1])
    H = np.eye(4)
    max_length = np.max([la.norm(n.inertial_frame[:3,3]) for n in  kinematic_graph.nodes()])
    scale = max_length/4
    plt.figure(figsize=(15,15))
    for link in kinematic_graph.nodes():
        H_w_l = link.frame 
        Hg = link.inertial_frame
        H = H_w_l
        ex_l = H @ ex
        ez_l = H @ ez
        p_l = H @ p

        ex_g_l = (
            H @ Hg @ ex
        )
        ez_g_l = (
            H @ Hg @ ez
        )
        p_g_l = H @ Hg @ p

        plt.arrow(p_l[0], p_l[2], ex_l[0] * scale, ex_l[2] * scale, color="r")
        plt.arrow(p_l[0], p_l[2], ez_l[0] * scale, ez_l[2] * scale, color="b")
        plt.arrow(p_g_l[0], p_g_l[2], ex_g_l[0] * scale, ex_g_l[2] * scale, color="g")
        plt.arrow(p_g_l[0], p_g_l[2], ez_g_l[0] * scale, ez_g_l[2] * scale, color="c")


def draw_joint_frames(kinematic_graph: nx.Graph):
    ex = np.array([1, 0, 0, 0])
    ez = np.array([0, 0, 1, 0])
    p = np.array([0, 0, 0, 1])
    H = np.eye(4)
    max_length = np.max([la.norm(n.inertial_frame[:3,3]) for n in  kinematic_graph.nodes()])
    scale = max_length/4
    plt.figure(figsize=(15,15))
    for edges in kinematic_graph.edges(data=True):
        joint = edges[2]["joint"]
        H_l_j = joint.frame
        H_w_l = joint.link_in.frame
        H = H_w_l @ H_l_j
        ex_l = H @ ex
        ez_l = H @ ez
        p_l = H @ p


        plt.arrow(p_l[0], p_l[2], ex_l[0] * scale, ex_l[2] * scale, color="r")
        plt.arrow(p_l[0], p_l[2], ez_l[0] * scale, ez_l[2] * scale, color="b")

def calculate_inertia(length):
    Ixx = 1 / 12 * 1 * (0.001**2 * length**2)
    Iyy = 1 / 12 * 1 * (0.001**2 * length**2)
    Izz = 1 / 12 * 1 * (0.001**2 * 0.001**2)
    return {"ixx": Ixx, "ixy": 0, "ixz": 0, "iyy": Iyy, "iyz": 0, "izz": Izz}