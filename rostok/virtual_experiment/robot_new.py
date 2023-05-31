from copy import deepcopy
from typing import Dict, List, Tuple

import pychrono.core as chrono
from pychrono.core import ChQuaternionD, ChVectorD

from rostok.block_builder_api.block_parameters import (DefaultFrame,
                                                       FrameTransform)
from rostok.block_builder_chrono_alt.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.block_builder_chrono_alt.block_classes import (BLOCK_CLASS_TYPES,
                                                       ChronoRevolveJoint,
                                                       PrimitiveBody)
from rostok.block_builder_chrono_alt.block_connect import place_and_connect
from rostok.control_chrono.controller import ConstController
from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_block_typing import NodeFeatures
from rostok.virtual_experiment.sensors import Sensor, DataStorage


class BuiltGraphChrono:
    """The object of this class is a graph representation in Chrono engine.
    
    Attributes:
        __graph (GraphGrammar):a graph that contains blocks in the node attributes
        block_ids (List[int]): the collection of ids for all blocks in the building order
        block_vector (List[Tuple[int, PrimitiveBody]]): vector of pairs (id, PrimitiveBody)
        joint_ids (List[int]): the collection of ids for all joints in the building order
        joint_vector (List[Tuple[int, ChronoRevolveJoint]]): vector of pairs 
            (id, ChronoRevolveJoint)
        joint_link_map (Dict[int,Tuple[int, int]]): maps a pair of (previous block, next block) 
            to each joint
        """

    def __init__(self,
                 graph: GraphGrammar,
                 system: chrono.ChSystem,
                 initial_position: FrameTransform = DefaultFrame,
                 is_base_fixed=True):
        """Build graph into system and fill vectors.
        
            Args:
                graph (GraphGrammar): graph of the mechanism to build
                system (chrono.ChSystem): chrono system for mechanism simulation
                initial_position (FrameTransform): starting position of the base block
                is_base_fixed (bool): determines if the base is fixed in the simulation"""

        self.__graph: GraphGrammar = deepcopy(graph)
        self.body_map_ordered: Dict[int, PrimitiveBody] = {}
        self.joint_map_ordered: Dict[int, ChronoRevolveJoint] = {}
        self.joint_link_map: Dict[int, Tuple[int, int]] = {}
        self.build_into_system(system, initial_position)
        if is_base_fixed:
            self.fix_base()

        self.build_joint_link_map()

    def build_joint_link_map(self):
        """Build the joint_link_map"""
        paths = self.__graph.get_sorted_root_based_paths()
        for path in paths:
            pre_block = None
            current_joint = None
            for idx in path:
                if NodeFeatures.is_joint(self.__graph.nodes[idx]["Node"]):
                    current_joint = idx
                if NodeFeatures.is_body(self.__graph.nodes[idx]["Node"]):
                    if pre_block is None:
                        pre_block = idx
                    else:
                        if current_joint is None:
                            pre_block = idx
                        else:
                            self.joint_link_map[current_joint] = (pre_block, idx)
                            pre_block = None
                            current_joint = None

    def build_into_system(self,
                          system: chrono.ChSystem,
                          initial_position: FrameTransform = DefaultFrame):
        """Build the graph into system and add the built blocks into node attributes with 
            key `Blocks`.
        
            Args:
                system (chrono.ChSystem): chrono system for cuurent simulation
                initial_position (FrameTransform): starting position of the base block"""
        # builds the root based paths in the sorted root based paths order
        paths = self.__graph.get_sorted_root_based_paths()
        block_chains: List[List[BLOCK_CLASS_TYPES]] = []
        for path in paths:
            chain: List[BLOCK_CLASS_TYPES] = []
            for idx in path:
                # if the node hasn't been built yet
                if self.__graph.nodes[idx].get("Blocks", None) is None:
                    # build all objects relevant to the node and add them into graph
                    blueprint = self.__graph.nodes[idx]["Node"].block_blueprint
                    created_blocks = creator.init_block_from_blueprint(blueprint)
                    if NodeFeatures.is_joint(self.__graph.nodes[idx].get("Node", None)):
                        self.joint_map_ordered[idx] = created_blocks
                    elif NodeFeatures.is_body(self.__graph.nodes[idx].get("Node", None)):
                        self.body_map_ordered[idx] = created_blocks

                    chain.append(created_blocks)
                    self.__graph.nodes[idx]["Blocks"] = created_blocks
                else:
                    chain.append(self.__graph.nodes[idx].get("Blocks", None))

            block_chains.append(chain)

        chrono_vector_position = ChVectorD(*initial_position.position)
        chrono_quat_rotation = ChQuaternionD(*initial_position.rotation)
        base_id = self.__graph.get_root_id()
        base = self.__graph.nodes[base_id].get("Blocks", None)
        base.body.SetPos(chrono_vector_position)
        base.body.SetRot(chrono_quat_rotation)
        for line in block_chains:
            place_and_connect(line, system)

    def fix_base(self):
        """Set body fixed for the base body"""
        # Fixation palm of grab mechanism
        base_id = self.__graph.get_root_id()
        base = self.__graph.nodes[base_id].get("Blocks", None)
        base.body.SetBodyFixed(True)


class RobotChrono:
    """Robot object consists of built graph, sensor and robot control.
    
        Attributes:
            built_graph (BuiltGraphChrono): the built graph
            sensor (Sensor): sensor set for collect data from all blocks of the robot
            controller : robot controller"""

    def __init__(self,
                 robot_graph: GraphGrammar,
                 system: chrono.ChSystem,
                 control_parameters,
                 control_cls = ConstController,
                 start_frame: FrameTransform = DefaultFrame, is_fixed = True):
        """Build mechanism into system and bind sensor to robot blocks.
        
            Args:
                robot_graph (GraphGrammar): graph representation of the robot
                system (chrono.ChSystem): system for current simulation
                control_parameters : list of parameters for controller
                control_trajectories : list of trajectories for joints
                start_frame: initial position of the base body"""
        self.__built_graph = BuiltGraphChrono(robot_graph, system, start_frame, is_fixed)
        self.sensor = Sensor(self.__built_graph.body_map_ordered, self.__built_graph.joint_map_ordered)
        self.controller = control_cls(self.__built_graph.joint_map_ordered, control_parameters)
        self.data_storage = DataStorage()
    def get_graph(self):
        return self.__built_graph
    def get_data(self):
        return self.data_storage
