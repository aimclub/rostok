from dataclasses import dataclass, field

from rostok.control_chrono.external_force import ForceChronoWrapper
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_block_typing import get_joint_vector_from_graph


@dataclass
class ForceTorqueContainer:
    controller_list: list[ForceChronoWrapper] = field(default_factory=list)

    def update_all(self, time: float, data=None):
        for i in self.controller_list:
            i.update(time, data)

    def add(self, controller: ForceChronoWrapper):
        if controller.is_bound:
            self.controller_list.append(controller)
        else:
            raise Exception("Force controller should be bound to body, before use")


def build_control_graph_from_joint(graph: GraphGrammar, joint_dict: dict):
    """Build control parametrs based on joint_dict and graph structure

    Args:
        graph (GraphGrammar): _description_
        joint_dict (dict): maps joint to value

    Returns:
        _type_: _description_
    """
    joints = get_joint_vector_from_graph(graph)
    control_sequence = []
    for idx in joints:
        node = graph.get_node_by_id(idx)
        control_sequence.append(joint_dict[node])
    return control_sequence
