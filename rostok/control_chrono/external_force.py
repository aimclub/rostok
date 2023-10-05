from abc import abstractmethod
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
    phi = np.random.uniform(0, 2*np.pi)
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    z_force = amp*cos_theta
    y_force = amp*sin_theta*np.sin(phi)
    x_force = amp*sin_theta*np.cos(phi)
    return x_force, y_force, z_force

def random_2d_vector(amp, angle: float = 0):
    """Calculate random 2d vector with given amplitude (uniform distribution on circle)

    Args:
        amp (float): amplitude of vector
        angle (float, optional): angle along axis z of vector. Defaults to 0.

    Returns:
        tuple: x, y, z components of vector
    """
    angle = np.random.uniform(0, 2*np.pi)
    
    el1 = np.cos(angle)*amp
    el2 = np.sin(angle)*amp
    
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

    
    def get_force_torque(self, time: float, data) -> ForceTorque:
        impact = ForceTorque()
        impact.force = self.calculate_impact(time, data)
        return impact
    
    @abstractmethod
    def calculate_impact(self, time: float, data) -> tuple[float, float, float]:
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
            file.write('Data for external action:', self.name)


class ForceControllerOnCallback(ForceControllerTemplate):

    def __init__(self, callback: CALLBACK_TYPE) -> None:
        super().__init__(name="callback_force")
        self.callback = callback

    def get_force_torque(self, time: float, data) -> ForceTorque:
        return self.callback(time, data)


class YaxisSin(ForceControllerTemplate):

    def __init__(self,
                 amp: float = 5,
                 amp_offset: float = 1,
                 freq: float = 5,
                 start_time: float = 0.0) -> None:
        """Shake by sin along y axis

        Args:
            amp (float, optional): Amplitude of sin. Defaults to 5.
            amp_offset (float, optional): Amplitude offset of force. Defaults to 1.
            freq (float, optional): Frequency of sin. Defaults to 5.
            start_time (float, optional): Start time of force application. Defaults to 0.0.
        """
        super().__init__(name="y_sin_force")
        self.amp = amp
        self.amp_offset = amp_offset
        self.freq = freq
        self.start_time = start_time
    
    def calculate_impact(self, time, data) -> tuple[float, float, float]:
        y_force = 0
        if time >= self.start_time:
            y_force = self.amp * np.sin(self.freq * (time - self.start_time)) + self.amp_offset 
        return 0, y_force, 0

class NullGravity(ForceControllerTemplate):
    def __init__(self, start_time: float = 0.0) -> None:
        """Apply force to compensate gravity

        Args:
            gravitry_force (float): gravity force of object
            start_time (float, optional): start time of force application. Defaults to 0.0.
        """
        super().__init__(name="null_gravity_force")
        self.start_time = start_time
    
    def calculate_impact(self, time: float, data) -> tuple[float, float, float]:
        force_g = np.zeros(3)
        if time >= self.start_time:
            mass = data.body_map_ordered[0].body.GetMass()
            g = data.grav_acc
            force_g = -mass*g
        return force_g[0], force_g[1], force_g[2]

class RandomForces(ForceControllerTemplate):
    def __init__(self, amp: float, start_time: float = 0.0, width_step: int = 20, *args) -> None:
        """Apply force with random direction and given amplitude

        Args:
            amp (float): amplitude of force
            start_time (float, optional): Start time of force application. Defaults to 0.0.
            width_step (int, optional): Number of steps between changes of force direction. Defaults to 20.
        """
        super().__init__(name="random_force")
        self.start_time = start_time
        self.width_step = width_step
        self.amp = amp
        self.counter = 0
        self.y_force = 0
        self.x_force = 0
        self.z_force = 0
        self.args = args
    
    def calculate_impact(self, time: float, data) -> tuple[float, float, float]:
        if time >= self.start_time:
                if self.counter % self.width_step == 0:
                    if self.args and self.args[0] == '2d':
                        self.x_force, self.y_force, self.z_force = random_2d_vector(self.amp, self.args[1])
                    else:
                        self.x_force, self.y_force, self.z_force = random_3d_vector(self.amp)
                    self.x_force, self.y_force, self.z_force = random_3d_vector(self.amp)
                self.counter += 1
        return self.x_force, self.y_force, self.z_force

class ClockXZForces(ForceControllerTemplate):
    def __init__(self, amp: float, angle_step: float = np.pi/6,  start_time: float = 0.0,  width_step: int = 20) -> None:
        """Apply force with given amplitude in xz plane and rotate it with given angle step

        Args:
            amp (float): amplitude of force
            angle_step (float, optional): Size of angle for changing force direction. Defaults to np.pi/6.
            start_time (float, optional): Start time of force application. Defaults to 0.0.
            width_step (int, optional): _description_. Defaults to 20.
        """
        super().__init__(name="clock_xz_force")
        self.amp = amp
        self.start_time = start_time
        self.width_step = width_step
        self.counter: int = 0
        self.angle: float = 0.0
        self.angle_step: float = angle_step
    
    def calculate_impact(self, time: float, data) -> tuple[float, float, float]:
        y_force = 0
        x_force = 0
        z_force = 0
        if time >= self.start_time:
            if self.counter % self.width_step == 0:
                self.angle += self.angle_step
            self.counter += 1
            x_force = np.cos(self.angle_step)*self.amp
            z_force = np.sin(self.angle_step)*self.amp
        return x_force, y_force, z_force
    
class ExternalForces(ForceControllerTemplate):
    def __init__(self, force_controller: ForceControllerTemplate | List[ForceControllerTemplate]) -> None:
        """Class for combining several external forces

        Args:
            force_controller (ForceControllerTemplate | List[ForceControllerTemplate]): Forces or list of forces
        """
        super().__init__(name="external_forces")
        self.force_controller = force_controller
        
    def add_force(self, force: ForceControllerTemplate):
        if isinstance(self.force_controller, list):
            self.force_controller.append(force)
        else:
            self.force_controller = [self.force_controller, force]
        
    def get_force_torque(self, time: float, data) -> ForceTorque:
        if isinstance(self.force_controller, list):
            v_forces = np.zeros(3)
            for controller in self.force_controller:
                v_forces += np.array(controller.calculate_impact(time, data))
            impact = ForceTorque()
            impact.force = tuple(v_forces)
            return impact
        else:
            return self.force_controller.get_force_torque(time, data)