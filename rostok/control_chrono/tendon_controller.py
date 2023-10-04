from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Union

import pychrono as chrono

from rostok.control_chrono.controller import RobotControllerChrono
from rostok.control_chrono.external_force import (ForceControllerTemplate,
                                                  ForceTorque)
from rostok.graph_grammar.graph_comprehension import (get_tip_ids,
                                                      is_star_topology)
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_block_typing import NodeFeatures
from rostok.virtual_experiment.built_graph_chrono import BuiltGraphChrono
from rostok.virtual_experiment.sensors import Sensor


class ForceType(Enum):
    PULLEY = 0
    TIP = 1
    BASE_CONNECTION = 2


class PulleyForce(ForceControllerTemplate):

    def get_force_torque(self, time: float, data) -> ForceTorque:
        impact = ForceTorque()
        pre_point = data[0]
        point = data[1]
        post_point = data[2]
        tension = data[3]
        force_v = ((post_point - point).GetNormalized() +
                   (pre_point - point).GetNormalized()) * tension
        impact.force = (force_v.x, force_v.y, force_v.z)
        if self.path:
            with open(self.path, 'a') as file:
                file.write(
                    f'{self.name} {round(time, 5)} {round(point.x,5)} {round(point.y,5)} {round(point.z,5)} {round(force_v.x, 6)} {round(force_v.y,6)} {round(force_v.z,6)}\n'
                )
        return impact


class TipForce(ForceControllerTemplate):

    def get_force_torque(self, time: float, data) -> ForceTorque:
        impact = ForceTorque()
        pre_point = data[0]
        point = data[1]
        tension = data[2]
        force_v = (pre_point - point).GetNormalized() * tension
        impact.force = (force_v.x, force_v.y, force_v.z)
        if self.path:
            with open(self.path, 'a') as file:
                file.write(
                    f'{self.name} {round(time, 5)} {round(point.x,5)} {round(point.y,5)} {round(point.z,5)} {round(force_v.x, 6)} {round(force_v.y,6)} {round(force_v.z,6)}\n'
                )
        return impact


@dataclass
class TendonControllerParameters:
    amount_pulley_in_body: int = 2
    pulley_parameters_for_body: Dict[int, List[float]] = field(default_factory=dict)
    tip: bool = True
    tip_parameters: List[float] = field(default_factory=list)
    forces: List[float] = field(default_factory=list)
    starting_point_parameters: List[float] = field(default_factory=list)
    create_pulley_data_file: bool = False

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
    force_type: ForceType = ForceType.BASE_CONNECTION


def create_pulley_lines_2p(graph: GraphGrammar, pulleys_in_phalanx=2, finger_base=True):
    if not is_star_topology(graph):
        raise Exception("Graph should be star topology")

    tip_ids = get_tip_ids(graph)
    branches = graph.get_sorted_root_based_paths()
    is_joint_id = lambda id: NodeFeatures.is_joint(graph.get_node_by_id(id))

    pulley_lines = []
    for finger_n, path in enumerate(branches):
        # find bodies from root to tip, w/o root
        if finger_base:
            path.pop(0)
        line = []
        for idx in path:
            if not NodeFeatures.is_body(graph.get_node_by_id(idx)):
                continue

            if len(line) == 0:
                pulley_parameters = PulleyParameters(finger_id=finger_n, body_id=idx)
                line.append([pulley_parameters, None])
            else:
                if idx in tip_ids:
                    for i in range(pulleys_in_phalanx - 1):
                        pulley_parameters = PulleyParameters(finger_id=finger_n,
                                                             body_id=idx,
                                                             pulley_number=i,
                                                             force_type=ForceType.PULLEY)
                        line.append([pulley_parameters, None])

                    pulley_parameters = PulleyParameters(finger_id=finger_n,
                                                         body_id=idx,
                                                         pulley_number=i + 1,
                                                         force_type=ForceType.TIP)
                    line.append([pulley_parameters, None])
                else:
                    for i in range(pulleys_in_phalanx):
                        pulley_parameters = PulleyParameters(finger_id=finger_n,
                                                             body_id=idx,
                                                             pulley_number=i,
                                                             force_type=ForceType.PULLEY)
                        line.append([pulley_parameters, None])

        if len(line) > 0:
            pulley_lines.append(line)

    return pulley_lines

PULLEY_LINES_TYPE = list[list[list[PulleyParameters, Union[PulleyForce, TipForce]]]]

class TendonController_2p(RobotControllerChrono):

    def __init__(self, graph: BuiltGraphChrono, control_parameters: TendonControllerParameters):
        super().__init__(graph, control_parameters)
        self.pulley_lines: PULLEY_LINES_TYPE = []
        self.create_force_points()

    def set_pulley_positions(self, tendon_lines: PULLEY_LINES_TYPE):
        for line in tendon_lines:
            for force_point in line:
                idx = force_point[0].body_id
                node = self.graph.get_node_by_id(idx)
                x = node.block_blueprint.shape.width_x
                y = node.block_blueprint.shape.length_y
                z = node.block_blueprint.shape.height_z
                if force_point[0].force_type == ForceType.BASE_CONNECTION:
                    parameters = self.parameters.starting_point_parameters
                    pos_x = parameters[0].get_offset(0.5 * x)
                    pos_y = parameters[1].get_offset(0.5 * y)
                    pos_z = parameters[2].get_offset(0.5 * z)
                    force_point[0].position = [pos_x, pos_y, pos_z]

                elif force_point[0].force_type == ForceType.TIP:
                    parameters = self.parameters.tip_parameters
                    pos_x = parameters[0].get_offset(0.5 * x)
                    pos_y = parameters[1].get_offset(0.5 * y)
                    pos_z = parameters[2].get_offset(0.5 * z)
                    force_point[0].position = [pos_x, pos_y, pos_z]
                else:
                    parameters = self.parameters.pulley_parameters_for_body[
                        force_point[0].pulley_number]
                    pos_x = parameters[0].get_offset(0.5 * x)
                    pos_y = parameters[1].get_offset(y * (-0.5 + force_point[0].pulley_number))
                    pos_z = parameters[2].get_offset(0.5 * z)
                    force_point[0].position = [pos_x, pos_y, pos_z]

    def set_forces_to_pulley_line(self, tendon_lines: PULLEY_LINES_TYPE):
        for line in tendon_lines:
            for force_point in line:
                idx = force_point[0].body_id
                body = self.built_graph.body_map_ordered[idx]
                if force_point[0].force_type == ForceType.PULLEY:
                    force_point[1] = PulleyForce(pos=list(force_point[0].position),
                                                 name=f'{idx}_p_{force_point[0].pulley_number}')
                    force_point[1].bind_body(body.body)
                    force_point[1].visualize_application_point()

                elif force_point[0].force_type == ForceType.TIP:
                    force_point[1] = TipForce(pos=list(force_point[0].position), name=f'{idx}_t')
                    force_point[1].bind_body(body.body)
                    force_point[1].visualize_application_point()

                elif force_point[0].force_type == ForceType.BASE_CONNECTION:
                    force_point[1] = TipForce(pos=list(force_point[0].position))
                    force_point[1].bind_body(body.body)
                    force_point[1].visualize_application_point()

    def create_force_points(self):
        self.pulley_lines: List[List[PulleyParameters,
                                     Union[PulleyForce,
                                           TipForce]]] = create_pulley_lines_2p(self.graph)
        self.set_pulley_positions(self.pulley_lines)
        self.set_forces_to_pulley_line(self.pulley_lines)

    def update_functions(self, time, robot_data: Sensor, environment_data):
        for i, line in enumerate(self.pulley_lines):
            tension = self.parameters.forces[i]
            for j, _ in enumerate(line):
                if line[j][0].force_type == ForceType.PULLEY:
                    pre_point = line[j - 1][1].force_maker_chrono.GetVpoint()
                    point = line[j][1].force_maker_chrono.GetVpoint()
                    post_point = line[j + 1][1].force_maker_chrono.GetVpoint()
                    line[j][1].update(time, [pre_point, point, post_point, tension])
                elif line[j][0].force_type == ForceType.TIP:
                    pre_point = line[j - 1][1].force_maker_chrono.GetVpoint()
                    point = line[j][1].force_maker_chrono.GetVpoint()
                    line[j][1].update(time, [pre_point, point, tension])
