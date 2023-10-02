from typing import Any, Dict, List, Tuple
from abc import abstractmethod
from dataclasses import dataclass, field

import pychrono as chrono
import numpy as np
from matplotlib.pyplot import cla

from rostok.control_chrono.controller import ForceTorque, ForceControllerTemplate

def random_3d_vector(amp):
    phi = np.random.uniform(0, 2*np.pi)
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    z_force = amp*cos_theta
    y_force = amp*sin_theta*np.sin(phi)
    x_force = amp*sin_theta*np.cos(phi)
    return x_force, y_force, z_force

def random_2d_vector(amp, angle: float = 0):
    angle = np.random.uniform(0, 2*np.pi)
    
    el1 = np.cos(angle)*amp
    el2 = np.sin(angle)*amp
    
    v1 = chrono.ChVectorD(el1, el2, 0)
    
    q1 = chrono.Q_from_AngZ(angle)
    v1 = q1.Rotate(v1)
    
    return v1.x, v1.y, v1.z


class YaxisShaker(ForceControllerTemplate):

    def __init__(self,
                 amp: float = 5,
                 amp_offset: float = 1,
                 freq: float = 5,
                 start_time: float = 0.0) -> None:
        super().__init__()
        self.amp = amp
        self.amp_offset = amp_offset
        self.freq = freq
        self.start_time = start_time

    def get_force_torque(self, time: float, data) -> ForceTorque:
        impact = ForceTorque()
        y_force = 0
        if time >= self.start_time:
            y_force = self.amp * np.sin(self.freq * (time - self.start_time)) + self.amp_offset
        impact.force = (0, y_force, 0)
        return impact
    

class NullGravity(ForceControllerTemplate):
    def __init__(self, gravitry_force, start_time: float = 0.0) -> None:
        super().__init__()
        self.gravity = gravitry_force
        self.start_time = start_time
    
    def get_force_torque(self, time: float, data) -> ForceTorque:
        impact = ForceTorque()
        y_force = 0
        x_force = 0
        z_force = 0
        if time >= self.start_time:
            y_force = -self.gravity
        impact.force = (x_force, y_force, z_force)
        return impact

class RandomShaker(ForceControllerTemplate):
    def __init__(self, amp: float, start_time: float = 0.0, width_step: int = 20, *args) -> None:
        super().__init__()
        self.start_time = start_time
        self.width_step = width_step
        self.amp = amp
        self.counter = 0
        self.y_force = 0
        self.x_force = 0
        self.z_force = 0
        self.args = args
    
    def get_force_torque(self, time: float, data) -> ForceTorque:
        impact = ForceTorque()
        if time >= self.start_time:
                if self.counter % self.width_step == 0:
                    if self.args and self.args[0] == '2d':
                        self.x_force, self.y_force, self.z_force = random_2d_vector(self.amp, self.args[1])
                    else:
                        self.x_force, self.y_force, self.z_force = random_3d_vector(self.amp)
                    self.x_force, self.y_force, self.z_force = random_3d_vector(self.amp)
                self.counter += 1
        impact.force = (self.x_force, self.y_force, self.z_force)
        return impact

class ClockXZShaker(ForceControllerTemplate):
    def __init__(self, amp: float, angle_step: float = np.pi/6,  start_time: float = 0.0,  width_step: int = 20) -> None:
        super().__init__()
        self.amp = amp
        self.start_time = start_time
        self.width_step = width_step
        self.counter = 0
        self.angle = 0
        self.angle_step = angle_step
    
    def get_force_torque(self, time: float, data) -> ForceTorque:
        impact = ForceTorque()
        y_force = 0
        x_force = 0
        z_force = 0
        if time >= self.start_time:
                if self.counter % self.width_step == 0:
                    self.angle += self.angle_step
                self.counter += 1
            x_force = np.cos(self.angle_step)*self.amp
            z_force = np.sin(self.angle_step)*self.amp
        impact.force = (x_force, y_force, z_force)
        return impact