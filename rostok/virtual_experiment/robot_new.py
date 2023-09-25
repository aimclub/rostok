import pychrono.core as chrono

from rostok.block_builder_api.block_parameters import (DefaultFrame, FrameTransform)
from rostok.control_chrono.controller import ConstController
from rostok.graph_grammar.node import GraphGrammar
from rostok.virtual_experiment.built_graph_chrono import BuiltGraphChrono
from rostok.virtual_experiment.sensors import DataStorage, Sensor


class RobotChrono:
    """Robot object consists of built graph, sensor and robot control.
    
        Attributes:
            built_graph (built_graph_chrono): the built graph
            sensor (Sensor): sensor set for collect data from all blocks of the robot
            controller : robot controller"""

    def __init__(self,
                 robot_graph: GraphGrammar,
                 system: chrono.ChSystem,
                 control_parameters,
                 control_cls=ConstController,
                 start_frame: FrameTransform = DefaultFrame,
                 starting_positions=[],
                 is_fixed=True):
        """Build mechanism into system and bind sensor to robot blocks.
        
            Args:
                robot_graph (GraphGrammar): graph representation of the robot
                system (chrono.ChSystem): system for current simulation
                control_parameters : list of parameters for controller
                control_trajectories : list of trajectories for joints
                start_frame: initial position of the base body"""

        self.__built_graph = BuiltGraphChrono(robot_graph, system, start_frame, starting_positions,
                                              is_fixed)
        self.sensor = Sensor(self.__built_graph.body_map_ordered,
                             self.__built_graph.joint_map_ordered)
        self.sensor.contact_reporter.reset_contact_dict()
        self.controller = control_cls(self.__built_graph, control_parameters)
        self.data_storage = DataStorage(self.sensor)

    def get_graph(self):
        return self.__built_graph

    def get_data(self):
        return self.data_storage
