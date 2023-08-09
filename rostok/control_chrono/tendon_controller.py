from collections import namedtuple, UserDict
from rostok.control_chrono.controller import ForceControllerTemplate, ForceTorque
from dataclasses import dataclass
import numpy as np
import pychrono as chrono
from rostok.graph_grammar.node import GraphGrammar
from collections import defaultdict
from typing import Any, NamedTuple, Optional, TypedDict, Union
from rostok.graph_grammar.node_block_typing import NodeFeatures
from rostok.library.rule_sets.simple_designs import get_one_link_four_finger
import networkx as nx
from rostok.virtual_experiment.built_graph_chrono import BuiltGraphChrono
from rostok.control_chrono.controller import RobotControllerChrono
from rostok.virtual_experiment.sensors import Sensor


@dataclass
class PylleyParams:
    """ pos -- along planhx
        offset --
    """
    pos: float = 0
    offset: float = 0


class PulleyForce(ForceControllerTemplate):

    def __init__(self, pos: list) -> None:
        super().__init__(is_local=True)
        #self.set_vector_in_local_cord()
        self.pos = pos

    def get_force_torque(self, time: float, data) -> ForceTorque:
        force = data["Force"]
        angle = data["Angle"]
        impact = ForceTorque()
        x_force = -2 * np.sin(angle + 0.0005) * force
        #if angle < 0:
            #x_force = 0
        impact.force = (x_force, 0, 0)
        return impact

    def bind_body(self, body: chrono.ChBody):
        super().bind_body(body)
        self.__body = body
        self.force_maker_chrono.SetVrelpoint(chrono.ChVectorD(*self.pos))

    def add_visual_pulley(self):
        sph_1 = chrono.ChSphereShape(0.005)
        sph_1.SetColor(chrono.ChColor(1, 0, 0))
        self.__body.AddVisualShape(sph_1, chrono.ChFrameD(chrono.ChVectorD(*self.pos)))
        self.__body.GetVisualShape(0).SetOpacity(0.6)


class TipForce(PulleyForce):

    def get_force_torque(self, time: float, data) -> ForceTorque:
        force = data["Force"]
        impact = ForceTorque()
        ANGLE = 30
        # y_force = -force * np.cos(ANGLE * np.pi / 180 )
        # x_force = -force *  np.sin(ANGLE * np.pi / 180 )
        # impact.force = (x_force, y_force, 0)
        y_force = -force
        impact.force = (0, y_force, 0)
        return impact


class PulleyKey(NamedTuple):
    """
        finger_id: int -- column in GraphGrammar.get_sorted_root_based_paths
        body_id: int -- id from GraphGrammar
        pulley_number: int -- from bootom to top pos
    """
    finger_id: int
    body_id: int
    pulley_number: int


class PulleyParamsFinger_2p(NamedTuple):

    finger_id: int
    pos_bottom: tuple[float, float, float]
    pos_top: tuple[float, float, float]


"Pulleys container [finger_num by uniq repr][id_body][lower/upper]"


def is_star_topology(graph: nx.DiGraph):
    degree = dict(graph.degree())
    root = [n for n, d in graph.in_degree() if d == 0][0]
    del degree[root]
    return all(value <= 2 for value in degree.values())


def create_pulley_dict_n(graph: GraphGrammar, n=1) -> dict[PulleyKey, Any]:
    if not is_star_topology(graph):
        raise Exception("Graph should be star topology")
    branches = graph.get_sorted_root_based_paths()
    root = [n for n, d in graph.in_degree() if d == 0][0]
    # Remove root id
    branches_no_root = [[item for item in sublist if item != root] for sublist in branches]
    is_body_id = lambda id: NodeFeatures.is_body(graph.get_node_by_id(id))
    branches_body = [[item for item in sublist if is_body_id(item)] for sublist in branches_no_root]
    pulley_dict = {}
    for finger_num, branch in enumerate(branches_body):
        for body_id in branch:
            pulley_keys = [
                PulleyKey(finger_num, body_id, pulley_number) for pulley_number in range(n)
            ]
            pulley_dict_buff = dict.fromkeys(pulley_keys, None)
            pulley_dict.update(pulley_dict_buff)

    return pulley_dict


def create_pulley_params_same(mech_graph: GraphGrammar, settings_for_pulleys, amount_pulley):
    """Same params on all pulleys

    Args:
        mech_graph (GraphGrammar): _description_
        settings_for_pulleys (_type_): _description_
    """
    pulley_dict = create_pulley_dict_n(mech_graph, amount_pulley)
    for p_k in pulley_dict.keys():
        pulley_dict[p_k] = settings_for_pulleys
    return pulley_dict


def create_pulley_params_finger_2p(mech_graph: GraphGrammar,
                                   settings_for_pulleys: list[PulleyParamsFinger_2p]):
    """Same params on finger

    Args:
        mech_graph (GraphGrammar): _description_
        settings_for_pulleys (_type_): _description_
    """
    AMOUNT_PULLEY_PHLANX = 2
    pulley_dict = create_pulley_dict_n(mech_graph, AMOUNT_PULLEY_PHLANX)
    ret = {}
    for pulley_p in settings_for_pulleys:
        pulleys_key_bottom = [
            i for i in pulley_dict if i.finger_id == pulley_p.finger_id and i.pulley_number == 0
        ]
        pulleys_key_top = [
            i for i in pulley_dict if i.finger_id == pulley_p.finger_id and i.pulley_number == 1
        ]
        finger_dict_bottom = dict.fromkeys(pulleys_key_bottom, pulley_p.pos_bottom)
        finger_dict_top = dict.fromkeys(pulleys_key_top, pulley_p.pos_top)
        ret.update(finger_dict_bottom)
        ret.update(finger_dict_top)
    return ret

@dataclass
class RelativeSetting_2p:
    bottom_percent: float 
    top_percent: float
    lever_percent: float

def create_pulley_params_relative_finger_2p(mech_graph: GraphGrammar,
                                   settings_for_pulleys: RelativeSetting_2p):
 
    AMOUNT_PULLEY_PHLANX = 2
    pulley_dict = create_pulley_dict_n(mech_graph, AMOUNT_PULLEY_PHLANX)
    ret = {}
    
 

    for i in pulley_dict:
        body_node = mech_graph.get_node_by_id(i.body_id)
        length = body_node.block_blueprint.shape.length_y
        height = body_node.block_blueprint.shape.height_z
        if i.pulley_number == 0:
            offset = -length / 2 * settings_for_pulleys.bottom_percent
        elif i.pulley_number == 1:
            offset = length / 2 * settings_for_pulleys.top_percent
            
        pulley_dict[i] = (settings_for_pulleys.lever_percent*height, offset, 0)

 
    return pulley_dict

def get_leaf_body_id(graph: GraphGrammar) -> list[int]:
    leaf_nodes = [
        node for node in graph.nodes() if graph.in_degree(node) != 0 and graph.out_degree(node) == 0
    ]
    return leaf_nodes


def init_pulley_and_tip_force(graph: GraphGrammar, pulley_dict: dict[PulleyKey, tuple[float, float,
                                                                                      float]]):
    ret_dict: dict[PulleyKey, Union[TipForce, PulleyForce]] = {}
    tips_dict = get_tips_elememt(graph, pulley_dict)
    for key, value in pulley_dict.items():
        if key in tips_dict.keys():
            ret_dict[key] = TipForce(list(value))
        else:
            ret_dict[key] = PulleyForce(list(value))
    return ret_dict


def get_tips_elememt(graph: GraphGrammar, pulley_dict: dict[PulleyKey,
                                                            Any]) -> dict[PulleyKey, Any]:
    tip_bodies_id = get_leaf_body_id(graph)
    tip_keys = []
    for tip_b in tip_bodies_id:
        pulleys = [i for i in pulley_dict if i.body_id == tip_b]
        tip = max(pulleys, key=lambda x: x.pulley_number)
        tip_keys.append(tip)
    return {x: pulley_dict[x] for x in tip_keys}


def nearest_joint(mech_graph: GraphGrammar, start_find_id: int, is_before: bool) -> Optional[int]:
    is_joint_id = lambda id: NodeFeatures.is_joint(mech_graph.get_node_by_id(id))
    branches = mech_graph.get_sorted_root_based_paths()
    cord = []
    for col, branch in enumerate(branches):
        if start_find_id in branch:
            row = branch.index(start_find_id)
            cord = [col, row]
            break
    if len(cord) == 0:
        raise Exception("Body id not find")
    target_finger = branches[cord[0]]

    if is_before:
        find_list = list(reversed(target_finger[:cord[1]]))
    else:
        find_list = target_finger[cord[1]:]

    for el in find_list:
        if is_joint_id(el):
            return el
    return None


def create_map_joint_2p(mech_graph: GraphGrammar,
                        pulley_dict_shape: dict[PulleyKey, Any]) -> dict[PulleyKey, int]:
    map_joint = {}
    for key in pulley_dict_shape.keys():
        if (key.pulley_number == 0):
            map_joint[key] = nearest_joint(mech_graph, key.body_id, is_before=True)
        elif (key.pulley_number == 1):
            map_joint[key] = nearest_joint(mech_graph, key.body_id, is_before=False)
        else:
            raise Exception("Pulley number should less 2")

    return map_joint

def create_map_joint_tip_2p(mech_graph: GraphGrammar,
                        pulley_dict_shape: dict[PulleyKey, Any]) -> dict[PulleyKey, int]:
    tips_dict = get_tips_elememt(mech_graph, pulley_dict_shape)
    map_joint_tips = {}
    for key in tips_dict.keys():
        map_joint_tips[key] = nearest_joint(mech_graph, key.body_id, is_before=True)
    return map_joint_tips
         

def bind_pulleys(built_graph: BuiltGraphChrono,
                 pulley_dict: dict[PulleyKey, Union[PulleyForce, TipForce]],
                 is_draw=True):

    for key in pulley_dict.keys():
        body = built_graph.body_map_ordered[key.body_id].body
        pulley_dict[key].bind_body(body)
        if is_draw:
            pulley_dict[key].add_visual_pulley()


class TendonController_2p(RobotControllerChrono):

    def __init__(self, graph: BuiltGraphChrono, control_parameters):
        pulley_params_dict: dict[PulleyKey, Any] = control_parameters["pulley_params_dict"]
        self.force_finger_dict: dict = control_parameters["force_finger_dict"]

        self.robot_graph = graph.graph
        self.pulley_params_dict = pulley_params_dict
        map_joint_tip = create_map_joint_tip_2p(self.robot_graph, self.pulley_params_dict)
        self.map_joint_id_pulley = create_map_joint_2p(self.robot_graph, self.pulley_params_dict)
        self.map_joint_id_pulley.update(map_joint_tip)
        self.pulley_and_tip_dict_obj = init_pulley_and_tip_force(self.robot_graph, pulley_params_dict)
        bind_pulleys(graph, self.pulley_and_tip_dict_obj)

    def update_functions(self, time, robot_data: Sensor, environment_data):
        angle_joint_dict = robot_data.get_joint_z_trajectory_point()
        for key in self.pulley_and_tip_dict_obj.keys():
            force = self.force_finger_dict[key.finger_id]
            joint_id = self.map_joint_id_pulley[key]
            angle = angle_joint_dict[joint_id]
            data = {"Force" : force, "Angle": angle}
            self.pulley_and_tip_dict_obj[key].update(time, data)

