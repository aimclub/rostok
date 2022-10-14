from functools import reduce
import context
import engine.robot as robot
import pychrono as chrono
from engine.node_render import ChronoBody

class FlagsSimualtions:
    def __init__(self, chrono_system: chrono.ChSystem, in_robot: robot.Robot, obj: chrono.ChBody):
        self.__flags_state = False
        self.robot = in_robot
        self.obj = obj
        self.system = chrono_system
        
    def get_flag_state(self):
        return self.__flags_state

class StopSimulation(FlagsSimualtions):
    def __init__(self, chrono_system, in_robot: robot.Robot, obj: chrono.ChBody,
                 time_to_contact: float, time_without_contact: float):
        super().__init__(chrono_system, in_robot, obj)
        
        self.time_to_contact = time_to_contact
        self.time_out_contact = time_without_contact
        
        self.curr_delta_center: chrono.ChVectorD = chrono.ChVectorD(0,0,0)
        self.curr_time = 0.
        self.time_last_contact = float("inf")
        self.time_first_contact = float("inf")
        
    def sim_stop(self):
        prev_time = self.curr_time
        current_time = self.system.GetChTime()
        
        self.flag_not_contact = False
        self.flag_slipout = False
        
        self.time_last_contact = current_time if self.__is_contact() else self.time_last_contact
        self.time_first_contact = self.time_last_contact if self.time_first_contact == float("inf") else self.time_first_contact
        if current_time >= self.time_to_contact:
            self.flag_not_contact = False if self.time_first_contact <= self.time_to_contact else True
            self.flag_slipout = not self.__is_contact() if current_time - self.time_last_contact >= self.time_out_contact and not self.flag_slipout else False
        
        self.curr_time = current_time
        return self.__diff_center2object()
    
    def __is_contact(self):
        blocks = self.robot.block_map.values()
        body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
        array_normal_forces = map(lambda x: x.list_n_forces, body_block)
        sum_contacts = reduce(lambda x, y: sum(x) if isinstance(x,list) else x + sum(y),
                              array_normal_forces)
        return not sum_contacts == 0

class SuccessSimulation(FlagsSimualtions):
    def __init__(self, chrono_system, in_robot: robot.Robot, obj: chrono.ChBody):
        super().__init__(chrono_system, in_robot, obj)

    def sim_stop(self):
        self.__number_contacts()
        return True
        
    def __number_contacts(self):
        blocks = self.robot.block_map.values()
        body_block = filter(lambda x: isinstance(x,ChronoBody),blocks)
        array_normal_forces = map(lambda x: x.list_n_forces, body_block)
        is_contact = map(lambda x: not (sum(x) == 0) or x , # if isinstance(x,list) else x
                              array_normal_forces)
        number_contact = sum(is_contact)
        return number_contact