from collections import namedtuple, UserDict
from typing import Any, Dict, List, Tuple
from rostok.control_chrono.controller import ForceControllerTemplate, ForceTorque
from dataclasses import dataclass, field
import numpy as np
import pychrono as chrono
from rostok.graph_grammar.node import GraphGrammar
from collections import defaultdict
from typing import Any, NamedTuple, Optional, TypedDict, Union
from rostok.graph_grammar.node_block_typing import NodeFeatures

import networkx as nx
from rostok.virtual_experiment.built_graph_chrono import BuiltGraphChrono
from rostok.control_chrono.controller import RobotControllerChrono
from rostok.virtual_experiment.sensors import Sensor
from rostok.graph_grammar.graph_comprehension import is_star_topology, get_tip_ids

class ForceType(Enum):
    PULLEY = 0
    TIP = 1
    POINT = 2
class PulleyForce(ForceControllerTemplate):

    def __init__(self, pos: list, name = 'default') -> None:
        super().__init__(is_local=False)
        #self.set_vector_in_local_cord()
        self.pos = pos
        self.name = name
        # with open(f"{self.name}.dat",'w') as file:
        #     pass

    def get_force_torque(self, time: float, data) -> ForceTorque:
        impact = ForceTorque()
        pre_point = data[0]
        point = data[1]
        post_point = data[2]
        tension = data[3]
        force_v = ((post_point-point).GetNormalized() + (pre_point-point).GetNormalized())*tension
        impact.force = (force_v.x, force_v.y, force_v.z)
        #with open(f"self.name_force_{round(self.pos[0],5)}_{round(self.pos[1],5)}.dat",'a') as file:
        # with open(f"{self.name}.dat",'a') as file:
        #     file.write(f'{round(force_v.x, 6)} {round(force_v.y,6)} {round(time, 5)} \n')
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
        impact = ForceTorque()
        pre_point = data[0]
        point = data[1]
        tension = data[2]
        force_v = (pre_point - point).GetNormalized()*tension
        impact.force = (force_v.x, force_v.y, force_v.z)
        # with open(f"{self.name}.dat",'a') as file:
        #     file.write(f'{round(force_v.x, 6)} {round(force_v.y,6)} {round(time, 5)} \n')
        return impact

@dataclass
class TendonControllerParameters:
    amount_pulley_in_body:int = 2
    pulley_parameters_for_body: Dict[int, List[float]] = field(default_factory=dict)
    #pulley_parameters_for_body: Dict[int, List[float]] = {i:[0,0,0] for i in range(amount_pulley_in_body)}
    tip: bool = True
    tip_parameters: List[float] = field(default_factory=list)
    forces:List[float] = field(default_factory=list)
    starting_point_parameters: List[float] = field(default_factory=list)

    def check_parameters_length(self):
        if self.amount_pulley_in_body != len(self.pulley_parameters_for_body):
            raise Exception("Invalid parameters for pulleys...")

@dataclass
class PulleyParameters():
    """
        finger_id: int -- column in GraphGrammar.get_sorted_root_based_paths
        body_id: int -- id from GraphGrammar
        pulley_number: int -- from bootom to top pos
    """
    finger_id: int = 0
    body_id: int = 0
    pulley_number: int = 0
    position: List[float] = field(default_factory=list)
    force_type: ForceType = ForceType.POINT 

def create_pulley_lines(graph: GraphGrammar, pulleys_in_phalanx=2, finger_base = True):
    if not is_star_topology(graph):
        raise Exception("Graph should be star topology")
    tip_ids = get_tip_ids(graph)
    branches = graph.get_sorted_root_based_paths()
    pulley_lines = []
    for finger_n, path in enumerate(branches):
        # find bodies from root to tip, w/o root 
        path.pop(0)
        line = []
        for idx in path:
            if  not NodeFeatures.is_body(self.graph.get_node_by_id(idx)):
                continue

            if len(line)==0:
                pulley_parameters = PulleyParameters(finger_id=finger_n, body_id=idx)
            else:
                for i in range(pulleys_in_phalanx):

                
            



class TendonController_2p(RobotControllerChrono):

    def __init__(self, graph: BuiltGraphChrono, control_parameters:TendonControllerParameters):
        super().__init__(graph, control_parameters)
        self.pulley_lists =[] 
        self.create_force_points()

        # pulley_params_dict: dict[PulleyKey, Any] = control_parameters["pulley_params_dict"]
        # self.force_finger_dict: dict = control_parameters["force_finger_dict"]

        # self.robot_graph = graph.graph
        # self.pulley_params_dict = pulley_params_dict
        # map_joint_tip = create_map_joint_tip_2p(self.robot_graph, self.pulley_params_dict)
        # self.map_joint_id_pulley = create_map_joint_2p(self.robot_graph, self.pulley_params_dict)
        # self.map_joint_id_pulley.update(map_joint_tip)
        # self.pulley_and_tip_dict_obj = init_pulley_and_tip_force(self.robot_graph, pulley_params_dict)
        # bind_pulleys(graph, self.pulley_and_tip_dict_obj)

    def create_force_points(self):
        paths = self.graph.get_sorted_root_based_paths()
        # for each finger
        for path in paths:
            # find bodies from root to tip, w/o root 
            path.pop(0)
            self.pulley_lists.append([])
            i = 0
            while True:
                tip_id = path[-1-i]
                if NodeFeatures.is_body(self.graph.get_node_by_id(tip_id)):
                    break
            first_body = True
            for id in path:
                if id in self.built_graph.body_map_ordered:
                    body = self.built_graph.body_map_ordered[id]
                    node = self.graph.get_node_by_id(id)
                    x = node.block_blueprint.shape.width_x
                    y = node.block_blueprint.shape.length_y
                    z = node.block_blueprint.shape.height_z
                    if first_body:
                        parameters = self.parameters.starting_point_parameters
                        pos = [i[0]*(i[1]*0.5) for i in zip(parameters,[x,y,z])]
                        force_point = PulleyForce(pos, f'bottom_force_{id}')
                        force_point.bind_body(body.body)
                        force_point.add_visual_pulley()
                        self.pulley_lists[-1].append(force_point)
                        force_point.force_maker_chrono.SetNameString("Bottom_force")
                        first_body = False
                    else:
                        for i in range(self.parameters.amount_pulley_in_body):
                            node = self.graph.get_node_by_id(id)
                            x = node.block_blueprint.shape.width_x
                            y = node.block_blueprint.shape.length_y
                            z = node.block_blueprint.shape.height_z
                            parameters = self.parameters.pulley_parameters_for_body[i]
                            #pos = [i[0]*(i[1]*0.5) for i in zip(parameters,[x,y,z])]
                            pos = [parameters[0]*0.5*x, 0.5*y*(i*2-1)+parameters[1], parameters[1]*0.5*z]
                            force_point = PulleyForce(pos, f'pulley_force_{id}_{i}')
                            force_point.bind_body(body.body)
                            force_point.add_visual_pulley()
                            force_point.force_maker_chrono.SetNameString(f"Pulley_force {i}")
                            self.pulley_lists[-1].append(force_point)
            # do tip
            body = self.built_graph.body_map_ordered[tip_id]
            node = self.graph.get_node_by_id(tip_id)
            x = node.block_blueprint.shape.width_x
            y = node.block_blueprint.shape.length_y
            z = node.block_blueprint.shape.height_z
            parameters = self.parameters.tip_parameters
            pos = [i[0]*(i[1]*0.5) for i in zip(parameters,[x,y,z])]
            force_point = TipForce(pos, f'tip_force_{id}')
            force_point.bind_body(body.body)
            force_point.add_visual_pulley()
            force_point.force_maker_chrono.SetNameString("Tip_force")
            self.pulley_lists[-1].append(force_point)


    def update_functions(self, time, robot_data: Sensor, environment_data):
        for i, finger in enumerate(self.pulley_lists):
            tension = self.parameters.forces[i]
            for j in range(1,len(finger)-1):
                pre_point = finger[j-1].force_maker_chrono.GetVpoint()
                point = finger[j].force_maker_chrono.GetVpoint()
                post_point = finger[j+1].force_maker_chrono.GetVpoint()
                finger[j].update(time, [pre_point, point, post_point, tension])
            pre_point = finger[-2].force_maker_chrono.GetVpoint()
            point = finger[-1].force_maker_chrono.GetVpoint()
            finger[-1].update(time, [pre_point, point, tension])