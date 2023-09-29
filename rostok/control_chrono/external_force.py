from abc import abstractmethod
from dataclasses import dataclass
from math import sin
from typing import Any, Callable

import pychrono.core as chrono


@dataclass
class ForceTorque:
    """Forces and torques are given in the xyz convention
    """
    force: tuple[float, float, float] = (0, 0, 0)
    torque: tuple[float, float, float] = (0, 0, 0)


CALLBACK_TYPE = Callable[[float, Any], ForceTorque]


class ForceControllerTemplate():
    """Base class for creating force and moment actions.

    To use it, you need to implement the get_force_torque method, 
    which determines the force and torque at time t.
    """

    def __init__(self,
                 is_local: bool = False,
                 name: str = "unnamed_force",
                 pos: list[float] = [0, 0, 0]) -> None:
        self.pos = pos
        self.name = name
        self.path = None

        self.x_force_chrono = chrono.ChFunction_Const(0)
        self.y_force_chrono = chrono.ChFunction_Const(0)
        self.z_force_chrono = chrono.ChFunction_Const(0)

        self.x_torque_chrono = chrono.ChFunction_Const(0)
        self.y_torque_chrono = chrono.ChFunction_Const(0)
        self.z_torque_chrono = chrono.ChFunction_Const(0)

        self.force_vector_chrono = [self.x_force_chrono, self.y_force_chrono, self.z_force_chrono]
        self.torque_vector_chrono = [
            self.x_torque_chrono, self.y_torque_chrono, self.z_torque_chrono
        ]
        self.force_maker_chrono = chrono.ChForce()
        self.force_maker_chrono.SetNameString(name + "_force")
        self.torque_maker_chrono = chrono.ChForce()
        self.torque_maker_chrono.SetNameString(name + "_torque")
        self.is_bound = False
        self.is_local = is_local
        self.setup_makers()

    @abstractmethod
    def get_force_torque(self, time: float, data) -> ForceTorque:
        pass

    def update(self, time: float, data=None):
        force_torque = self.get_force_torque(time, data)
        for val, functor in zip(force_torque.force + force_torque.torque,
                                self.force_vector_chrono + self.torque_vector_chrono):
            functor.Set_yconst(val)

    def setup_makers(self):
        self.force_maker_chrono.SetMode(chrono.ChForce.FORCE)
        self.torque_maker_chrono.SetMode(chrono.ChForce.TORQUE)
        if self.is_local:
            self.force_maker_chrono.SetAlign(chrono.ChForce.BODY_DIR)
            self.torque_maker_chrono.SetAlign(chrono.ChForce.BODY_DIR)
        else:
            self.force_maker_chrono.SetAlign(chrono.ChForce.WORLD_DIR)
            self.torque_maker_chrono.SetAlign(chrono.ChForce.WORLD_DIR)

        self.force_maker_chrono.SetF_x(self.x_force_chrono)
        self.force_maker_chrono.SetF_y(self.y_force_chrono)
        self.force_maker_chrono.SetF_z(self.z_force_chrono)

        self.torque_maker_chrono.SetF_x(self.x_torque_chrono)
        self.torque_maker_chrono.SetF_y(self.y_torque_chrono)
        self.torque_maker_chrono.SetF_z(self.z_torque_chrono)

    def bind_body(self, body: chrono.ChBody):
        body.AddForce(self.force_maker_chrono)
        body.AddForce(self.torque_maker_chrono)
        self.__body = body
        self.force_maker_chrono.SetVrelpoint(chrono.ChVectorD(*self.pos))
        self.torque_maker_chrono.SetVrelpoint(chrono.ChVectorD(*self.pos))
        self.is_bound = True

    def visualize_application_point(self,
                                    size=0.005,
                                    color=chrono.ChColor(1, 0, 0),
                                    body_opacity=0.6):
        sph_1 = chrono.ChSphereShape(size)
        sph_1.SetColor(color)
        self.__body.AddVisualShape(sph_1, chrono.ChFrameD(chrono.ChVectorD(*self.pos)))

        self.__body.GetVisualShape(0).SetOpacity(body_opacity)

    @property
    def body(self):
        return self.__body

    def enable_data_dump(self, path):
        self.path = path
        with open(path, 'w') as file:
            file.write('Data for external action:',self.name)


class ForceControllerOnCallback(ForceControllerTemplate):

    def __init__(self, callback: CALLBACK_TYPE) -> None:
        super().__init__()
        self.callback = callback

    def get_force_torque(self, time: float, data) -> ForceTorque:
        return self.callback(time, data)


class YaxisShaker(ForceControllerTemplate):

    def __init__(self,
                 name: str = 'unnamed',
                 pos: list[float] = [0, 0, 0],
                 is_local=False,
                 amp: float = 5,
                 amp_offset: float = 1,
                 freq: float = 5,
                 start_time: float = 0.0) -> None:
        super().__init__(is_local=is_local, name=name, pos=pos)
        self.amp = amp
        self.amp_offset = amp_offset
        self.freq = freq
        self.start_time = start_time

    def get_force_torque(self, time: float, data) -> ForceTorque:
        impact = ForceTorque()
        y_force = 0
        if time >= self.start_time:
            y_force = self.amp * sin(self.freq * (time - self.start_time)) + self.amp_offset
        impact.force = (0, y_force, 0)
        return impact
