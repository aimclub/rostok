from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import types
import numpy as np
import pychrono.core as chrono

from rostok.virtual_experiment.sensors import DataStorage, Sensor


class EventCommands(Enum):
    """Commands available for the events. The simulation class handles these commands"""
    STOP = 0
    CONTINUE = 1
    ACTIVATE = 2


class SimulationSingleEvent(ABC):
    """ The abstract class for the event that can occur during the simulation only once.

        At each step of the simulation the event is checked using the current states of the sensors.
        step_n is a single value because the event can occur only once in simulation.

        Attributes:
            state (bool): event occurrence flag 
            step_n (int): the step of the event occurrence
            verbosity (int): the parameter that controls the console output of the class methods
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
        """Simulation calls that method each step to check if the event occurred.

        Args:
            current_time (float): time from the start of the simulation 
            step_n (int): step number of the simulation
            robot_data (_type_): current state of the robot sensors
            env_data (_type_): current state of the environment sensors
        """
        
    def __repr__(self) -> str:
        str_type = str(type(self))
        str_class = re.findall('\'([^\']*)\'', str_type)[0]
        self_attributes = dir(self)
        self_fields = list(filter(lambda x: not (x.startswith("__") or x.endswith("__")), self_attributes))
        self_fields = list(filter(lambda x: not isinstance(getattr(self, x), types.MethodType), self_fields))
        str_self = f"{str_class}:\n"
        for str_field in self_fields:
            str_self = str_self + f"    {str_field} = {getattr(self, str_field)} \n"
        return str_self


class EventContact(SimulationSingleEvent):
    """Event of contact between robot and object

    """

    def event_check(self, current_time: float, step_n: int, robot_data, env_data):
        if env_data.get_amount_contacts()[0] > 0:
            self.state = True
            self.step_n = step_n
            return EventCommands.CONTINUE

        return EventCommands.CONTINUE
    
    def __repr__(self) -> str:
        return f"EventContact(verbosity = {self.verbosity})"


class EventContactTimeOut(SimulationSingleEvent):
    """Event that occurs if the robot doesn't contact with body during the reference time from the start of the simulation.

    Attributes:
        reference_time (float): the moment of time where the simulation is interrupted 
            if there is no contact with the body
        contact (bool): the flag that determines if there was a contact with body
    """

    def __init__(self, ref_time: float, contact_event: EventContact):
        super().__init__()
        self.reference_time = ref_time
        self.contact_event: EventContact = contact_event

    def event_check(self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor):
        """Return STOP if the time exceeds the reference time and there was no contact with body.

        Returns:
            EventCommands: return a command for simulation
        """
        if self.contact_event.state:
            return EventCommands.CONTINUE

        if current_time > self.reference_time:
            self.state = True
            self.step_n = step_n
            return EventCommands.STOP

        return EventCommands.CONTINUE
    
    def __str__(self) -> str:
        return f"EventContactTimeOut(verbos)"


class EventFlyingApart(SimulationSingleEvent):
    """The event that stops simulation if the robot parts have flown apart.

    Attributes::
        max_distance (float): the max distance for robot parts 
    """

    def __init__(self, max_distance: float):
        super().__init__()
        self.max_distance = max_distance

    def event_check(self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor):
        """Return STOP if the current position of a part is max_distance away from the robot base body.

        Returns:
            EventCommands: return a command for simulation
        """
        trajectory_points = robot_data.get_body_trajectory_point()
        # It takes the position of the first block in the list, that should be the base body
        base_position = trajectory_points[next(iter(trajectory_points))]
        for block in trajectory_points.values():
            position = block
            if np.linalg.norm(np.array(base_position) - np.array(position)) > self.max_distance:
                self.state = True
                self.step_n = step_n
                return EventCommands.STOP

        return EventCommands.CONTINUE


class EventSlipOut(SimulationSingleEvent):
    """The event that stops simulation if the body slips out from the grasp after the contact.

    Attributes:
        time_last_contact (float): time of last contact of robot and body
        reference_time (float): time of contact loss until the stop of the simulation
    """

    def __init__(self, ref_time):
        super().__init__()
        self.time_last_contact = None
        self.reference_time = ref_time

    def reset(self):
        self.time_last_contact = None
        super().reset()

    def event_check(self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor):
        """Return STOP if the body and mech lose contact for the reference time.

        Returns:
            EventCommands: return a command for simulation
        """
        contact = env_data.get_amount_contacts()[0] > 0
        if contact:
            self.time_last_contact = current_time
            return EventCommands.CONTINUE

        if (not self.time_last_contact is None):
            if current_time - self.time_last_contact > self.reference_time:
                self.step_n = step_n
                self.state = True
                return EventCommands.STOP

        return EventCommands.CONTINUE


class EventGrasp(SimulationSingleEvent):
    """Event that activates the force if 

    Attributes:
        grasp_steps (int): the amount of consecutive steps body is not moving
        grasp_time (float): the moment of the grasp event
        contact (bool): the flag of body and object first contact
        activation_code (int): the activation code for ACTIVATE command
        grasp_limit_time (float): the time limit for the grasp event
        force_test_time (float): the time period of the force test of the grasp
    """

    def __init__(self, grasp_limit_time: float, contact_event: EventContact, verbosity: int = 0):
        super().__init__(verbosity)
        self.grasp_steps: int = 0
        self.grasp_time: Optional[float] = None
        self.grasp_limit_time = grasp_limit_time
        self.contact_event: EventContact = contact_event

    def reset(self):
        super().reset()
        self.grasp_steps = 0
        self.grasp_time = None

    def check_grasp_timeout(self, current_time):
        if current_time > self.grasp_limit_time:
            return True
        else:
            return False

    def check_grasp_current_step(self, env_data: Sensor):
        obj_velocity = np.linalg.norm(np.array(env_data.get_velocity()[0]))
        if obj_velocity <= 0.01 and env_data.get_amount_contacts()[0] >= 2:
            self.grasp_steps += 1
        else:
            self.grasp_steps = 0

    def event_check(self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor):
        """Return ACTIVATE if the body was in contact with the robot and after that at some 
        point doesn't move for at least 10 steps. Return STOP if the grasp didn't occur during grasp_limit_time. 
        

        Returns:
            EventCommands: return a command for simulation
        """

        if self.check_grasp_timeout(current_time):
            return EventCommands.STOP

        if self.contact_event.state:
            self.check_grasp_current_step(env_data)

            if self.grasp_steps == 10:
                self.state = True
                self.step_n = step_n
                self.grasp_time = current_time
                if self.verbosity > 0:
                    print('Grasp event!', current_time)

                return EventCommands.ACTIVATE

        return EventCommands.CONTINUE


class EventStopExternalForce(SimulationSingleEvent):

    def __init__(self, grasp_event: EventGrasp, force_test_time: float):
        super().__init__()
        self.grasp_event = grasp_event
        self.force_test_time = force_test_time

    def event_check(self, current_time: float, step_n: int, robot_data, env_data):
        """STOP simulation in force_test_time after the grasp."""

        # self.grasp_event.grasp_time is None until the grasp event have been occurred. 
        # Therefore we use nested if operators.
        if self.grasp_event.state:
            if current_time > self.force_test_time + self.grasp_event.grasp_time:
                self.state = True
                self.step_n = step_n
                return EventCommands.STOP

        return EventCommands.CONTINUE
