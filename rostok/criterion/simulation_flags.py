import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np

from rostok.utils.json_encoder import RostokJSONEncoder
from rostok.virtual_experiment.sensors import Sensor


class EventCommands(Enum):
    """Commands available for the events. The simulation class handles these commands"""

    STOP = 0
    CONTINUE = 1
    ACTIVATE = 2


class SimulationSingleEvent(ABC):
    """The abstract class for the event that can occur during the simulation only once.

    At each step of the simulation the event is checked using the current states of the sensors.
    step_n is a single value because the event can occur only once in simulation.

    Attributes:
        state (bool): event occurrence flag
        step_n (int): the step of the event occurrence
        verbosity (int): controls the console output of the event
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
    def event_check(
        self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor
    ):
        """Simulation calls that method each step to check if the event occurred.

        Args:
            current_time (float): time from the start of the simulation
            step_n (int): step number of the simulation
            robot_data (_type_): current state of the robot sensors
            env_data (_type_): current state of the environment sensors
        """

    def __repr__(self) -> str:
        json_data = json.dumps(self, cls=RostokJSONEncoder)
        return json_data

    def __str__(self) -> str:
        json_data = json.dumps(self, indent=4, cls=RostokJSONEncoder)
        return json_data


class EventContact(SimulationSingleEvent):
    """Event of contact between robot and object

    Attributes:
    from_body (bool): flag determines the source of contact information.
    """

    def __init__(self, take_from_body: bool = False):
        super().__init__()
        self.from_body = take_from_body

    def event_check(
        self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor
    ):
        if self.state:
            if __debug__:
                raise Exception(
                    "The EventContact.event_check is called after occurrence in simulation."
                )
            return EventCommands.CONTINUE

        if not self.from_body:
            # the contact information from the object
            if env_data.get_amount_contacts()[0] > 0:
                self.state = True
                self.step_n = step_n
        else:
            # the contact information from the robot
            robot_contacts = robot_data.get_amount_contacts()
            # it works only with current rule set, where the palm/flat always has the smallest index among the bodies
            flat_idx_ = list(robot_contacts.keys())[0]
            contacts = 0
            # we calculate only the amount of unique keys, therefore the amount of unique contacting bodies
            for key, value in robot_contacts.items():
                if key != flat_idx_ and value > 0:
                    contacts += 1

            if contacts > 0:
                self.state = True
                self.step_n = step_n

        return EventCommands.CONTINUE


class EventBuilder():
    def __init__(self, event_class):
        self.even_class = event_class

    def find_event(self, event_list: list[SimulationSingleEvent]):
        for event in event_list:
            if isinstance(event, self.even_class):
                return event
        return None

    @abstractmethod
    def build_event(self, event_list):
        pass


class EventContactBuilder(EventBuilder):
    def __init__(self, take_from_body: bool = False) -> None:
        super().__init__(event_class=EventContact)
        self.from_body = take_from_body

    def build_event(self, event_list) -> EventContact:
        event = self.find_event(event_list=event_list)
        if event is None:
            return event_list.append(EventContact(take_from_body=self.from_body))
        else:
            raise Exception(
                'Attempt to create two same events for a simulation: EventContact')


class EventContactTimeOut(SimulationSingleEvent):
    """
    Event that occurs if the robot doesn't contact with body during the reference time from the start of the simulation.

    Attributes:
        reference_time (float): the moment of time where the simulation is interrupted
            if there is no contact with the body
        contact (bool): the flag that determines if there was a contact with body
    """

    def __init__(self, ref_time: float, contact_event: EventContact):
        super().__init__()
        self.reference_time = ref_time
        self.contact_event: EventContact = contact_event

    def event_check(
        self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor
    ):
        """
        Return STOP if the time exceeds the reference time and there was no contact with body.

        Returns:
            EventCommands: return a command for simulation
        """
        if self.state:
            if __debug__:
                raise Exception(
                    "The EventContactTimeOut.event_check is called after occurrence in simulation."
                )
            return EventCommands.STOP

        # if the contact has already occurred in simulation, return CONTINUE
        if self.contact_event.state:
            return EventCommands.CONTINUE

        if current_time > self.reference_time:
            self.state = True
            self.step_n = step_n
            return EventCommands.STOP

        return EventCommands.CONTINUE


class EventContactTimeOutBuilder(EventBuilder):
    def __init__(self, reference_time, event_contact_builder: EventContactBuilder) -> None:
        super().__init__(event_class=EventContactTimeOut)
        self.reference_time = reference_time
        self.event_contact_builder = event_contact_builder

    def build_event(self, event_list) -> EventContactTimeOut:
        event = self.find_event(event_list=event_list)
        if event:
            raise Exception(
                'Attempt to create two same events for a simulation: EventContactTimeOut')
        contact_event = self.event_contact_builder.find_event(
            event_list=event_list)
        if contact_event is None:
            raise Exception(
                'Event requires another event prebuilt: EventContactTimeOut <- EventContact')
        return event_list.append(EventContactTimeOut(ref_time=self.reference_time, contact_event=contact_event))


class EventFlyingApart(SimulationSingleEvent):
    """
    The event that stops simulation if the robot parts have flown apart.

    Attributes:
        max_distance (float): the max distance for robot parts
    """

    def __init__(self, max_distance: float):
        super().__init__()
        self.max_distance = max_distance

    def event_check(
        self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor
    ):
        """Return STOP if the current position of a part is max_distance away from the robot base body.

        Returns:
            EventCommands: return a command for simulation
        """
        if self.state:
            if __debug__:
                raise Exception(
                    "The EventFlyingApart.event_check is called after occurrence in simulation."
                )
            return EventCommands.STOP

        trajectory_points = robot_data.get_body_trajectory_point()
        # It takes the position of the first block in the list, that should be the base body
        base_position = trajectory_points[next(iter(trajectory_points))]
        for position in trajectory_points.values():
            if (
                np.linalg.norm(np.array(base_position) - np.array(position))
                > self.max_distance
            ):
                self.state = True
                self.step_n = step_n
                return EventCommands.STOP

        return EventCommands.CONTINUE


class EventFlyingApartBuilder(EventBuilder):
    def __init__(self, max_distance) -> None:
        super().__init__(event_class=EventFlyingApart)
        self.max_distance = max_distance

    def build_event(self, event_list) -> EventFlyingApart:
        event = self.find_event(event_list=event_list)
        if event is None:
            return event_list.append(EventFlyingApart(max_distance=self.max_distance))
        else:
            raise Exception(
                'Attempt to create two same events for a simulation: EventFlyingApart')


class EventSlipOut(SimulationSingleEvent):
    """
    The event that stops simulation if the body slips out from the grasp after the contact.

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

    def event_check(
        self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor
    ):
        """
        Return STOP if the body and mech lose contact for the reference time.

        Returns:
            EventCommands: return a command for simulation
        """
        if self.state:
            if __debug__:
                raise Exception(
                    "The EventSlipOut.event_check is called after occurrence in simulation."
                )
            return EventCommands.STOP

        robot_contacts = robot_data.get_amount_contacts()
        # it works only with current rule set, where the palm/flat always has the smallest index among the bodies
        flat_idx_ = list(robot_contacts.keys())[0]
        contacts = 0
        # we calculate only the amount of unique keys, therefore the amount of unique contacting bodies
        for key, value in robot_contacts.items():
            if key != flat_idx_ and value > 0:
                contacts += 1

        contact = contacts > 0
        if contact:
            self.time_last_contact = current_time
            return EventCommands.CONTINUE

        if not (self.time_last_contact is None):
            if current_time - self.time_last_contact > self.reference_time:
                self.step_n = step_n
                self.state = True
                return EventCommands.STOP

        return EventCommands.CONTINUE


class EventSlipOutBuilder(EventBuilder):

    def __init__(self, ref_time):
        super().__init__(event_class=EventSlipOut)
        self.reference_time = ref_time

    def build_event(self, event_list) -> EventSlipOut:
        event = self.find_event(event_list=event_list)
        if event is None:
            return event_list.append(EventSlipOut(ref_time=self.reference_time))
        else:
            raise Exception(
                'Attempt to create two same events for a simulation: EventSlipOut')


class EventGrasp(SimulationSingleEvent):
    """
    Event that activates the force if

    Attributes:
        grasp_steps (int): the amount of consecutive steps body is not moving
        grasp_time (float): the moment of the grasp event
        contact (bool): the flag of body and object first contact
        activation_code (int): the activation code for ACTIVATE command
        grasp_limit_time (float): the time limit for the grasp event
        force_test_time (float): the time period of the force test of the grasp
    """

    def __init__(
        self,
        grasp_limit_time: float,
        contact_event: EventContact,
        verbosity: int = 0,
        simulation_stop: bool = False,
    ):
        super().__init__(verbosity=verbosity)
        self.grasp_steps: int = 0
        self.grasp_time: Optional[float] = None
        self.grasp_limit_time = grasp_limit_time
        self.contact_event: EventContact = contact_event
        self.simulation_stop = simulation_stop

    def reset(self):
        super().reset()
        self.grasp_steps = 0
        self.grasp_time = None

    def check_grasp_timeout(self, current_time):
        if current_time > self.grasp_limit_time:
            return True
        return False

    def check_grasp_current_step(self, env_data: Sensor, robot_data: Sensor):
        obj_velocity = np.linalg.norm(np.array(env_data.get_velocity()[0]))
        robot_contacts = robot_data.get_amount_contacts()
        # it works only with current rule set, where the palm/flat always has the smallest index among the bodies
        flat_idx_ = list(robot_contacts.keys())[0]
        contacts = 0
        # we calculate only the amount of unique keys, therefore the amount of unique contacting bodies
        for key, value in robot_contacts.items():
            if key != flat_idx_ and value > 0:
                contacts += 1

        if obj_velocity <= 0.01 and contacts >= 2:
            self.grasp_steps += 1
        else:
            self.grasp_steps = 0

    def event_check(
        self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor
    ):
        """
        Return ACTIVATE if the body was in contact with the robot and after that at some
        point doesn't move for at least 10 steps. Return STOP if the grasp didn't occur during grasp_limit_time.

        Returns:
            EventCommands: return a command for simulation
        """

        if self.state:
            if __debug__:
                raise Exception(
                    "The EventGrasp.event_check is called after occurrence in simulation."
                )
            return EventCommands.CONTINUE

        if self.check_grasp_timeout(current_time):
            return EventCommands.STOP

        if self.contact_event.state:
            self.check_grasp_current_step(env_data, robot_data)

            if self.grasp_steps == 10:
                self.state = True
                self.step_n = step_n
                self.grasp_time = current_time
                if self.verbosity > 0:
                    print("Grasp event!", current_time)
                if self.simulation_stop > 0:
                    input("press enter to continue")

                return EventCommands.ACTIVATE

        return EventCommands.CONTINUE


class EventGraspBuilder(EventBuilder):
    def __init__(
        self,
        event_contact_builder: EventContactBuilder,
        grasp_limit_time: float,
        verbosity: int = 0,
        simulation_stop: bool = False,
    ):
        super().__init__(event_class=EventGrasp)
        self.event_contact_builder = event_contact_builder
        self.verbosity = verbosity
        self.grasp_limit_time = grasp_limit_time
        self.simulation_stop = simulation_stop

    def build_event(self, event_list) -> EventGrasp:
        event = self.find_event(event_list=event_list)
        if event:
            raise Exception(
                'Attempt to create two same events for a simulation: EventGrasp')
        contact_event = self.event_contact_builder.find_event(
            event_list=event_list)
        if contact_event is None:
            raise Exception(
                'Event requires another event prebuilt: EventGrasp <- EventContact')
        return event_list.append(EventGrasp(verbosity=self.verbosity, grasp_limit_time=self.grasp_limit_time, simulation_stop=self.simulation_stop, contact_event=contact_event))


class EventStopExternalForce(SimulationSingleEvent):
    """
    Event that controls external force and stops simulation after reference time has passed.
    """

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


class EventStopExternalForceBuilder(EventBuilder):
    def __init__(self, force_test_time: float, event_grasp_builder: EventGraspBuilder):
        super().__init__(event_class=EventStopExternalForce)
        self.event_grasp_builder = event_grasp_builder
        self.force_test_time = force_test_time

    def build_event(self, event_list) -> EventStopExternalForce:
        event = self.find_event(event_list=event_list)
        if event:
            raise Exception(
                'Attempt to create two same events for a simulation: EventStopExternalForce')
        grasp_event = self.event_grasp_builder.find_event(
            event_list=event_list)
        if grasp_event is None:
            raise Exception(
                'Event requires another event prebuilt: EventStopExternalForce <- EventGrasp')
        return event_list.append(EventStopExternalForce(force_test_time=self.force_test_time, grasp_event=grasp_event))


class EventBodyTooLow(SimulationSingleEvent):
    def __init__(self, ref_height: float):
        super().__init__()
        self.ref_height = ref_height

    def event_check(self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor):
        robot_position = robot_data.get_body_trajectory_point()
        # it works only with current rule set, where the palm/flat always has the smallest index among the bodies
        main_body_pos = list(robot_position.items())[0][1]

        if main_body_pos[1] < self.ref_height:
            return EventCommands.STOP

        return EventCommands.CONTINUE


class EventContactInInitialPosition(SimulationSingleEvent):
    def __init__(self):
        super().__init__()

    def event_check(self, current_time: float, step_n: int, robot_data: Sensor, env_data: Sensor):
        if step_n <=3:
            n_contacts = env_data.get_amount_contacts()
            if n_contacts[0] > 0:
                return EventCommands.STOP

        return EventCommands.CONTINUE


class EventContactInInitialPositionBuilder(EventBuilder):
    def __init__(self):
        super().__init__(event_class=EventStopExternalForce)

    def build_event(self, event_list) -> EventContactInInitialPosition:
        event = self.find_event(event_list=event_list)
        if event:
            raise Exception(
                'Attempt to create two same events for a simulation: EventContactInInitialPosition')
        return event_list.append(EventContactInInitialPosition())


class EventBodyTooLowBuilder(EventBuilder):
    def __init__(self, ref_height: float):
        super().__init__(event_class=EventStopExternalForce)
        self.ref_height = ref_height

    def build_event(self, event_list) -> EventBodyTooLow:
        event = self.find_event(event_list=event_list)
        if event:
            raise Exception(
                'Attempt to create two same events for a simulation: EventBodyTooLow')
        return event_list.append(EventBodyTooLow(ref_height=self.ref_height))
