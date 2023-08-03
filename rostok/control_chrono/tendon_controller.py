from collections import namedtuple, UserDict
from rostok.control_chrono.controller import ForceControllerTemplate, ForceTorque
from dataclasses import dataclass
import numpy as np
import pychrono as chrono

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

def update_finger(dict_id_angles):
    prepare_angle(dict_id_angles)
    finger_force
    pass

"Pulleys container [finger_num by uniq repr][id_body][botom/upper]"

def create_mapping_angles_pulley():
    """ create stuff  input: pulley id -> output: joint id"""
    pass