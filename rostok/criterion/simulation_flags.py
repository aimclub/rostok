from abc import ABC

import numpy as np
import pychrono.core as chrono
from rostok.virtual_experiment.sensors import Sensor, DataStorage


class FlagStopSimualtions(ABC):
    def __init__(self):
        self.state = False

    def reset_flag(self):
        self.state = False

    def update_state(self, current_time,robot_data, env_data):
        pass

class FlagFlyingApart(FlagStopSimualtions): 
    def __init__(self, max_distance:float):
        super().__init__()
        self.max_distance = max_distance

    def update_state(self, current_time, robot_data:Sensor, env_data:Sensor):
        base_position = robot_data.get_body_trajectory_point()[0][1]
        for block in robot_data.get_body_trajectory_point():
            position = block[1]
            if sum((np.array(base_position) - np.array(position))**2) > self.max_distance:
                self.state = True
                break

class FlagSlipout(FlagStopSimualtions):
    def __init__(self, ref_time):
        super().__init__()
        self.time_last_contact = None
        self.reference_time = ref_time

    def update_state(self, current_time, robot_data:Sensor, env_data:Sensor): 
        contact = len(env_data.get_amount_contacts()) > 0
        if contact:
            self.time_last_contact = current_time
            self.state = False
        else:
            if self.time_last_contact is None:
                self.state = False
            else: 
                if current_time - self.time_last_contact > self.reference_time:
                    self.state = True
                else:
                    self.state = False

class FlagContactTimeOut(FlagStopSimualtions):
    def __init__(self, ref_time):
        super().__init__()
        self.reference_time = ref_time
        self.contact = False

    def reset_flag(self):
        super().reset_flag()
        self.contact = False

    def update_state(self, current_time, robot_data:Sensor, env_data:Sensor): 
        if not self.contact:
            self.contact = len(env_data.get_amount_contacts()) > 0

        if not (self.contact or self.state):
            if current_time > self.reference_time:
                self.state = True