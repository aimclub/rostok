from abc import ABC
from functools import reduce
import engine.robot as robot
import pychrono as chrono
from engine.node_render import ChronoBody


class FlagStopSimualtions(ABC):
    def __init__(self, chrono_system: chrono.ChSystem, in_robot: robot.Robot, obj: chrono.ChBody):
        self.flag_state = False
        self.robot = in_robot
        self.obj = obj
        self.system = chrono_system
        
    def get_flag_state(self):
        return self.flag_state

class FlagWithContact(FlagStopSimualtions, ABC):
    def __init__(self, chrono_system, in_robot: robot.Robot, obj: chrono.ChBody):
        super().__init__(chrono_system, in_robot, obj)
        
    def get_flag_state(self):
        self.flag_state =self.is_contact()
        return self.flag_state
    
    def is_contact(self):
        blocks = self.robot.block_map.values()
        body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
        array_normal_forces = map(lambda x: x.list_n_forces, body_block)
        sum_contacts = reduce(lambda x, y: sum(x) if isinstance(x,list) else x + sum(y),
                              array_normal_forces)
        return not sum_contacts == 0
        
class FlagSlipout(FlagWithContact):
    def __init__(self, chrono_system, in_robot: robot.Robot, obj: chrono.ChBody,
                 time_to_contact: float = 3., time_without_contact: float = 0.2):
        super().__init__(chrono_system, in_robot, obj)
        
        self.time_to_contact = time_to_contact
        self.time_out_contact = time_without_contact
        
        self.curr_delta_center: chrono.ChVectorD = chrono.ChVectorD(0,0,0)
        self.curr_time = 0.
        self.time_last_contact = float("inf")
        
    def get_flag_state(self):
        prev_time = self.curr_time
        current_time = self.system.GetChTime()
        
        self.time_last_contact = current_time if self.is_contact() else self.time_last_contact
        if current_time >= self.time_to_contact and not self.flag_state:
            self.flag_state = not self.is_contact() if current_time - self.time_last_contact >= self.time_out_contact else False
        
        self.curr_time = current_time
        return self.flag_state

class FlagNotContact(FlagWithContact):
    def __init__(self, chrono_system, in_robot: robot.Robot, obj: chrono.ChBody,
                 time_to_contact: float = 3.):
        super().__init__(chrono_system, in_robot, obj)
        
        self.time_to_contact = time_to_contact
        
        self.curr_delta_center: chrono.ChVectorD = chrono.ChVectorD(0,0,0)
        self.curr_time = 0.
        self.time_last_contact = float("inf")
        self.time_first_contact = float("inf")
        
    def get_flag_state(self):
        prev_time = self.curr_time
        current_time = self.system.GetChTime()
        
        self.time_last_contact = current_time if self.is_contact() else self.time_last_contact
        self.time_first_contact = self.time_last_contact if self.time_first_contact == float("inf") else self.time_first_contact
        if current_time >= self.time_to_contact and not self.flag_state:
            self.flag_state = False if self.time_first_contact <= self.time_to_contact else True
        
        self.curr_time = current_time
        return self.flag_state
    

class ConditionStopSimulation:
    def __init__(self, chrono_system: chrono.ChSystem, in_robot: robot.Robot, obj: chrono.ChBody, flags: dict):
        self.__stop_flag = False
        self.chrono_system = chrono_system
        self.in_robot = in_robot
        self.obj = obj
        self.flags = {flag(chrono_system,in_robot,obj, *params) for flag, params in flags.items()}
        
    def flag_stop_simulation(self):
        state_flags = map(lambda x: x.get_flag_state(), self.flags)
        self.__stop_flag = reduce(lambda x,y: x or y, state_flags)
        return self.__stop_flag