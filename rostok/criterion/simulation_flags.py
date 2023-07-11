from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import pychrono.core as chrono

from rostok.virtual_experiment.sensors import DataStorage, Sensor


class EventCommands(Enum):
    STOP = 0
    CONTINUE = 1
    ACTIVATE = 2


class SimulationSingleEvent(ABC):
    """ The abstract class for the event that can occur during the simulation only once.

        At each step of the simulation the event is checked using the current states of the sensors

        Attributes:
            state (bool): event occurrence flag 
            step_n (int): the step of the event occurrence
    """

    def __init__(self, verbosity=0):
        self.state = False
        self.step_n = None
        self.verbosity = verbosity

    def reset(self):
        """Reset the values of the attributes for the new simulation."""

        self.state = False
        self.step_n = None

    @abstractmethod
    def event_check(self, current_time: float, step_n: int, robot_data, env_data):
        if self.state:
            return EventCommands.CONTINUE


class EventContactTimeOut(SimulationSingleEvent):

    def __init__(self, ref_time):
        super().__init__()
        self.reference_time = ref_time
        self.contact = False

    def reset(self):
        super().reset()
        self.contact = False

    def event_check(self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor):
        if self.contact:
            return EventCommands.CONTINUE
        else:
            self.contact = env_data.get_amount_contacts()[0] > 0
            if self.contact:
                return EventCommands.CONTINUE
            else:
                if current_time > self.reference_time:
                    self.state = True
                    self.step_n = step_n
                    return EventCommands.STOP


class EventFlyingApart(SimulationSingleEvent):

    def __init__(self, max_distance: float):
        super().__init__()
        self.max_distance = max_distance

    def event_check(self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor):
        trajectory_points = robot_data.get_body_trajectory_point()
        # It takes the position of the first block in the list, that should be the base body
        base_position = trajectory_points[next(iter(trajectory_points))]
        for block in trajectory_points.values():
            position = block
            if np.linalg.norm(np.array(base_position) - np.array(position)) > self.max_distance:
                self.state = True
                self.step_n = step_n
                return EventCommands.STOP


class EventSlipOut(SimulationSingleEvent):

    def __init__(self, ref_time):
        super().__init__()
        self.time_last_contact = None
        self.reference_time = ref_time

    def reset(self):
        self.time_last_contact = None
        super().reset()

    def event_check(self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor):
        contact = env_data.get_amount_contacts()[0] > 0
        if contact:
            self.time_last_contact = current_time
            return EventCommands.CONTINUE
        else:
            if self.time_last_contact is None:
                return EventCommands.CONTINUE
            else:
                if current_time - self.time_last_contact > self.reference_time:
                    self.step_n = step_n
                    self.state = True
                    return EventCommands.STOP
                else:
                    return EventCommands.CONTINUE


class EventGrasp(SimulationSingleEvent):

    def __init__(self, activation_code, verbosity=0):
        super().__init__(verbosity)
        self.grasp_steps = 0
        self.contact = False
        self.activation_code = activation_code

    def reset(self):
        super().reset()
        self.grasp_steps = 0
        self.contact = False

    def event_check(self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor):
        if self.state:
            return EventCommands.CONTINUE
        if not self.contact:
            self.contact = env_data.get_amount_contacts()[0] > 0
        else:
            obj_velocity = np.linalg.norm(np.array(env_data.get_velocity()[0]))
            if obj_velocity <= 0.01 and env_data.get_amount_contacts()[0] >= 2:
                self.grasp_steps += 1
            else:
                self.grasp_steps = 0

            if self.grasp_steps == 10:
                self.state = True
                self.step_n = step_n
                if self.verbosity > 0:
                    print('Grasp event!')

                return EventCommands.ACTIVATE
