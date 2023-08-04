from collections import namedtuple, UserDict
from rostok.control_chrono.controller import ForceControllerTemplate, ForceTorque
from dataclasses import dataclass
import numpy as np
import pychrono as chrono
from rostok.graph_grammar.node import GraphGrammar
from collections import defaultdict
@dataclass
class PylleyParams:
    """ pos -- along planhx
        offset --
    """
    pos: float = 0
    offset: float = 0

class PulleyForce(ForceControllerTemplate):

    def __init__(self, pos: list) -> None:
        super().__init__(is_local= True)
        #self.set_vector_in_local_cord()
        self.pos = pos

    def get_force_torque(self, time: float, data) -> ForceTorque:
        force = data["Force"]
        angle = data["Angle"]
        impact = ForceTorque()
        x_force = -2 * np.sin(angle + 0.001) * force
        if angle < 0:
            x_force = 0
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

"""
_summary_
"""

 


"Pulleys container [finger_num by uniq repr][id_body][lower/upper]"



def magic_mapping_joint(finger, body, lower_upper) -> int:  # joint id from graph
    pass

def magic_mapping_finger(finger, body, lower_upper) -> int:  # number of finger from uniq repr 
    pass

def magik_update_all_pulley(dict_angle, finger_force_dict):
    for finger, body, lb in self.magik_pulley_container.keys:
        pulley = self.magik_pulley_container[finger, body, lb]
        id_force = magic_mapping_finger(finger, body, lb)
        id_joint = magic_mapping_joint(finger, body, lb)
        force = finger_force_dict[id_force]
        angle = dict_angle[id_joint]
        data = {"Force" : force, "Angle" : angle}
        #pulley.update(time=0, data)

def magik_create_default_shape_2_pulley(mech_graph: GraphGrammar): # -> magik_dict[finger][body][LB]{}:
    pass

def magik_create_pulley_params(mech_graph: GraphGrammar, settings_for_pulleys):
    pass
def magik_ini_pulleys_force_matrix(pulley_params): # -> magik_dict[finger][body][LB] {pulley_forcer}:
    pass
# mech_graph with init blocks needed for bind
def magik_init(settings_for_pulleys, force_finger, mech_graph: GraphGrammar):
    self.force_finger = force_finger
    pulley_prams = magik_create_pulley_params(mech_graph, settings_for_pulleys)
    magik_ini_pulleys_force_matrix(pulley_prams)


d = dict()

d[(0, 1, 1)] = 10
d[(0, 1, 2)] = 1
d[(1, 1, 1)] = 2
d[(1, 1, 2)] = 2

for i, j, k in d.keys():
    print(i, j, k )