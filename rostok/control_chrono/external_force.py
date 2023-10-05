from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any, Callable, List

import pychrono.core as chrono


def random_3d_vector(amp):
    """Calculate random 3d vector with given amplitude (uniform distribution on sphere)

    Args:
        amp (float): amplitude of vector

    Returns:
        tuple: x, y, z components of vector
    """
    phi = np.random.uniform(0, 2 * np.pi)
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    z_force = amp * cos_theta
    y_force = amp * sin_theta * np.sin(phi)
    x_force = amp * sin_theta * np.cos(phi)
    return x_force, y_force, z_force


def random_2d_vector(amp, angle: float = 0):
    """Calculate random 2d vector with given amplitude (uniform distribution on circle)

    Args:
        amp (float): amplitude of vector
        angle (float, optional): angle along axis z of vector. Defaults to 0.

    Returns:
        tuple: x, y, z components of vector
    """
    angle = np.random.uniform(0, 2 * np.pi)

    el1 = np.cos(angle) * amp
    el2 = np.sin(angle) * amp

    v1 = chrono.ChVectorD(el1, el2, 0)

    q1 = chrono.Q_from_AngZ(angle)
    v1 = q1.Rotate(v1)

    return v1.x, v1.y, v1.z


@dataclass
class ForceTorque:
    """Forces and torques are given in the xyz convention
    """
    force: tuple[float, float, float] = (0, 0, 0)
    torque: tuple[float, float, float] = (0, 0, 0)


CALLBACK_TYPE = Callable[[float, Any], ForceTorque]


class ForceTemplate(ABC):

    def __init__(self,
                 name: str = "unnamed_force",
                 start_time: float = 0.0,
                 pos: np.ndarray = np.zeros(3)) -> None:
        self.path = None
        self.name = name
        self.pos = pos
        self.start_time = start_time

    @abstractmethod
    def calculate_spatial_force(self, time, data) -> np.ndarray:
        return np.zeros(6)

    def enable_data_dump(self, path):
        self.path = path
        with open(path, 'w') as file:
            file.write('Data for external action:')
            file.write(self.__dict__)


class ForceChronoWrapper():
    """Base class for creating force and moment actions.

    To use it, you need to implement the get_force_torque method, 
    which determines the force and torque at time t.
    """

    def __init__(self, force: ForceTemplate, is_local: bool = False) -> None:
        self.path = None
        self.force = force

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

        name = force.name.split("_")[0]
        self.force_maker_chrono.SetNameString(name + "_force")
        self.torque_maker_chrono = chrono.ChForce()
        self.torque_maker_chrono.SetNameString(name + "_torque")
        self.is_bound = False
        self.is_local = is_local
        self.setup_makers()

    def get_force_torque(self, time: float, data) -> ForceTorque:
        impact = ForceTorque()
        spatial_force = self.force.calculate_spatial_force(time, data)
        impact.force = tuple(spatial_force[3:])
        impact.torque = tuple(spatial_force[:3])
        return impact

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
        self.force_maker_chrono.SetVrelpoint(chrono.ChVectorD(*self.force.pos.tolist()))
        self.torque_maker_chrono.SetVrelpoint(chrono.ChVectorD(*self.force.pos.tolist()))
        self.is_bound = True

    def visualize_application_point(self,
                                    size=0.005,
                                    color=chrono.ChColor(1, 0, 0),
                                    body_opacity=0.6):
        sph_1 = chrono.ChSphereShape(size)
        sph_1.SetColor(color)
        self.__body.AddVisualShape(sph_1,
                                   chrono.ChFrameD(chrono.ChVectorD(*self.force.pos.tolist())))

        self.__body.GetVisualShape(0).SetOpacity(body_opacity)

    @property
    def body(self):
        return self.__body

    def enable_data_dump(self, path):
        self.path = path
        with open(path, 'w') as file:
            file.write('Data for external action:', self.force.name)


class ForceControllerOnCallback(ForceTemplate):

    def __init__(self,
                 callback: CALLBACK_TYPE,
                 name: str = "callback_force",
                 start_time: float = 0.0,
                 pos: np.ndarray = np.zeros(3)) -> None:
        super().__init__(name, start_time, pos)
        self.callback = callback

    def calculate_spatial_force(self, time, data) -> np.ndarray:
        return self.callback(time, data)


class YaxisSin(ForceTemplate):

    def __init__(self,
                 amp: float = 5,
                 amp_offset: float = 1,
                 freq: float = 5,
                 start_time: float = 0.0,
                 pos: np.ndarray = np.zeros(3)) -> None:
        """Shake by sin along y axis

        Args:
            amp (float, optional): Amplitude of sin. Defaults to 5.
            amp_offset (float, optional): Amplitude offset of force. Defaults to 1.
            freq (float, optional): Frequency of sin. Defaults to 5.
            start_time (float, optional): Start time of force application. Defaults to 0.0.
        """
        super().__init__("y_sin_force", start_time, pos)
        self.amp = amp
        self.amp_offset = amp_offset
        self.freq = freq

    def calculate_spatial_force(self, time, data) -> np.ndarray:
        spatial_force = np.zeros(6)
        if time >= self.start_time:
            spatial_force[4] = self.amp * np.sin(self.freq *
                                                 (time - self.start_time)) + self.amp_offset
        return spatial_force


class NullGravity(ForceTemplate):

    def __init__(self, start_time: float = 0.0) -> None:
        """Apply force to compensate gravity

        Args:
            gravitry_force (float): gravity force of object
            start_time (float, optional): start time of force application. Defaults to 0.0.
        """
        super().__init__(name="null_gravity_force", start_time=start_time, pos=np.zeros(3))

    def calculate_spatial_force(self, time: float, data) -> np.ndarray:
        spatial_force = np.zeros(6)
        if time >= self.start_time:
            mass = data.body_map_ordered[0].body.GetMass()
            g = data.grav_acc
            spatial_force[3:] = -mass * g
        return spatial_force


class RandomForces(ForceTemplate):

    def __init__(self,
                 amp: float,
                 width_step: int = 20,
                 start_time: float = 0.0,
                 pos: np.ndarray = np.zeros(3),
                 dimension="3d",
                 angle=0.0) -> None:
        """Apply force with random direction and given amplitude

        Args:
            amp (float): amplitude of force
            start_time (float, optional): Start time of force application. Defaults to 0.0.
            width_step (int, optional): Number of steps between changes of force direction. Defaults to 20.
        """
        super().__init__(name="random_force", start_time=start_time, pos=pos)
        self.width_step = width_step
        self.amp = amp
        self.dim = dimension
        self.angle = angle

        self.counter = 0
        self.spatial_force = np.zeros(6)

    def calculate_spatial_force(self, time: float, data) -> np.ndarray:

        if time >= self.start_time:
            if self.counter % self.width_step == 0:
                if self.dim == '2d':
                    self.spatial_force[3:] = random_2d_vector(self.amp, self.angle)
                else:
                    self.spatial_force[3:] = random_3d_vector(self.amp)
            self.counter += 1
        return self.spatial_force


class ClockXZForces(ForceTemplate):

    def __init__(self,
                 amp: float,
                 angle_step: float = np.pi / 6,
                 width_step: int = 20,
                 start_time: float = 0.0,
                 pos: np.ndarray = np.zeros(3)) -> None:
        """Apply force with given amplitude in xz plane and rotate it with given angle step

        Args:
            amp (float): amplitude of force
            angle_step (float, optional): Size of angle for changing force direction. Defaults to np.pi/6.
            start_time (float, optional): Start time of force application. Defaults to 0.0.
            width_step (int, optional): _description_. Defaults to 20.
        """
        super().__init__(name="clock_xz_force", start_time=start_time, pos=pos)
        self.amp = amp
        self.width_step = width_step
        self.counter: int = 0
        self.angle: float = 0.0
        self.angle_step: float = angle_step

    def calculate_spatial_force(self, time: float, data) -> np.ndarray:
        spatial_force = np.zeros(6)
        if time >= self.start_time:
            if self.counter % self.width_step == 0:
                self.angle += self.angle_step
            self.counter += 1
            spatial_force[3] = np.cos(self.angle_step) * self.amp
            spatial_force[4] = np.sin(self.angle_step) * self.amp
        return spatial_force


class ExternalForces(ForceTemplate):

    def __init__(self, force_controller: ForceTemplate | List[ForceTemplate]) -> None:
        """Class for combining several external forces

        Args:
            force_controller (ForceTemplate | List[ForceTemplate]): Forces or list of forces
        """

        if isinstance(force_controller, list):
            positions = np.array([i.pos for i in force_controller])
            if np.all(positions != positions[0]):
                raise Exception("All forces should have the same position")

        super().__init__(name="external_forces", start_time=0.0, pos=np.zeros(3))
        self.force_controller = force_controller

    def add_force(self, force: ForceTemplate):
        if isinstance(self.force_controller, list):
            self.force_controller.append(force)
        else:
            self.force_controller = [self.force_controller, force]

    def calculate_spatial_force(self, time: float, data) -> ForceTorque:
        if isinstance(self.force_controller, list):
            v_forces = np.zeros(3)
            for controller in self.force_controller:
                v_forces += np.array(controller.calculate_spatial_force(time, data))
            impact = ForceTorque()
            impact.force = tuple(v_forces[3:])
            impact.torque = tuple(v_forces[:3])
            return impact
        else:
            return self.force_controller.calculate_spatial_force(time, data)