from abc import ABC
from copy import deepcopy
from functools import reduce
import rostok.virtual_experiment.robot as robot
import pychrono as chrono
from rostok.block_builder.node_render import RobotBody


class BuilderNotInitializedError(Exception):

    def __init__(self, *args):
        """Exception raised when the builder is not initialized
        """
        super().__init__(*args)
        self.message = args[0] if args else None

    def __str__(self):
        return_str = 'Builder Not Initialized: {0}!'.format(
            self.message) if self.message else 'Builder Not Initialized before use it!'
        return return_str


class FlagStopSimualtions(ABC):
    """Abstract class for stopping flags
    """

    def __init__(self):
        self.flag_state = False
        self.robot = None
        self.obj = None
        self.system = None

    def build(self, chrono_system: chrono.ChSystem, in_robot: robot.Robot, obj: chrono.ChBody):
        """Build flag on the chrono system, robot and object

        Args:
            chrono_system (chrono.ChSystem): chrono system to build
            in_robot (robot.Robot): robot to build
            obj (chrono.ChBody): object to build
        """
        self.INIT_BUILD = True
        self.robot = in_robot
        self.obj = obj
        self.system = chrono_system

    def _check_builder(self):
        if self.robot is None or self.obj is None or self.system is None:
            raise BuilderNotInitializedError("Flags builder must initialize for checking")

    def get_flag_state(self):
        """Getter flag state

        Returns:
            bool: Current state of flag
        """
        self._check_builder()
        return self.flag_state


class FlagFlyingApart(FlagStopSimualtions):

    def __init__(self, max_distance):
        super().__init__()
        self.max_distance = max_distance

    def get_flag_state(self):
        base_body = self.robot.get_base_body()
        base_cog_frame = base_body.body.GetFrame_COG_to_abs()

        blocks = self.robot.block_map.values()

        body_block = filter(lambda x: isinstance(x, RobotBody), blocks)
        abs_cog_frame_robot_bodies = map(lambda x: x.body.GetFrame_COG_to_abs(), body_block)
        rel_cog_frame_robot_bodies = map(lambda x: base_cog_frame * x, abs_cog_frame_robot_bodies)
        body_distance_to_base = list(map(lambda x: x.GetPos().Length2(), rel_cog_frame_robot_bodies))
        self.flag_state = max(body_distance_to_base) >= self.max_distance

        return self.flag_state


class FlagMaxTime(FlagStopSimualtions):
    """Flag to stop simulation in case of maximum time

    Args:
        max_time_simulation (float):Max seconds simulation
    """

    def __init__(self, max_time_simulation):
        super().__init__()
        self.max_time = max_time_simulation

    def get_flag_state(self):
        """Getter flag state

        Returns:
            bool: Current state of flag
        """
        self._check_builder()
        self.flag_state = self.system.GetChTime() > self.max_time
        return self.flag_state


class FlagWithContact(FlagStopSimualtions, ABC):
    """Abstract class of stop flag simulation base on contact with bodies
    """

    def __init__(self):
        super().__init__()

    def get_flag_state(self):
        """Getter flag state

        Returns:
            bool: Current state of flag
        """
        self._check_builder()
        self.flag_state = self.is_contact()
        return self.flag_state

    def is_contact(self):
        """Check state of contact bodies

        Returns:
            bool: True when contact is exsist
        """
        blocks = self.robot.block_map.values()
        body_block = filter(lambda x: isinstance(x, RobotBody), blocks)
        array_normal_forces = map(lambda x: x.list_n_forces, body_block)
        sum_contacts = reduce(lambda x, y: sum(x)
                              if isinstance(x, list) else x + sum(y), array_normal_forces)
        return not sum_contacts == 0


class FlagSlipout(FlagWithContact):
    """Class stop flag chrono simulation when an object slips out

    Args:
        time_to_contact (float): Max time from start simulation to contact. Defaults to 3..
        time_without_contact (float): Max time without contact. Defaults to 0.2.
    """

    def __init__(self, time_to_contact: float = 3., time_without_contact: float = 0.2):
        super().__init__()

        self.time_to_contact = time_to_contact
        self.time_out_contact = time_without_contact

        self.curr_time = 0.
        self.time_last_contact = float("inf")

    def get_flag_state(self):
        """Getter flag state

        Returns:
            bool: Current state of flag
        """
        self._check_builder()
        prev_time = self.curr_time
        current_time = self.system.GetChTime()

        self.time_last_contact = current_time if self.is_contact() else self.time_last_contact
        if current_time >= self.time_to_contact and not self.flag_state:
            self.flag_state = not self.is_contact(
            ) if current_time - self.time_last_contact >= self.time_out_contact else False

        self.curr_time = current_time
        return self.flag_state


class FlagNotContact(FlagWithContact):
    """Flag to stop chrono-modeling when the maximum time without contact from start is reached

    Args:
        time_to_contact (float): Max time without contact. Defaults to 3..
    """

    def __init__(self, time_to_contact: float = 3.):
        super().__init__()

        self.time_to_contact = time_to_contact

        self.curr_time = 0.
        self.time_last_contact = float("inf")
        self.time_first_contact = float("inf")

    def get_flag_state(self):
        self._check_builder()
        current_time = self.system.GetChTime()

        self.time_last_contact = current_time if self.is_contact() else self.time_last_contact
        self.time_first_contact = self.time_last_contact if self.time_first_contact == float(
            "inf") else self.time_first_contact
        if current_time >= self.time_to_contact and not self.flag_state:
            self.flag_state = False if self.time_first_contact <= self.time_to_contact else True

        self.curr_time = current_time
        return self.flag_state


class ConditionStopSimulation:
    """A class of flag-based chrono-modeling stopping conditions.

    Args:
        chrono_system (chrono.ChSystem): System which checking on condition
        in_robot (robot.Robot): Robot which checking on condition_description_
        obj (chrono.ChBody): Object which checking on condition_description_
        flags (list[FlagStopSimualtions]): Flag of the stopping simulation
    """

    def __init__(self, chrono_system: chrono.ChSystem, in_robot: robot.Robot, obj: chrono.ChBody,
                 flags: list[FlagStopSimualtions]):
        self.__stop_flag = False
        self.chrono_system = chrono_system
        self.in_robot = in_robot
        self.obj = obj
        self.flags = deepcopy(flags)

        for flag in self.flags:
            flag.build(self.chrono_system, self.in_robot, self.obj)

    def flag_stop_simulation(self):
        """Condition of stop simulation

        Returns:
            bool: True if simulation have to be stopped
        """
        state_flags = map(lambda x: x.get_flag_state(), self.flags)
        self.__stop_flag = reduce(lambda x, y: x or y, state_flags)
        return self.__stop_flag
