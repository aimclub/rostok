from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

import pychrono.core as chrono
from pychrono.core import ChQuaternionD, ChVectorD

from rostok.block_builder_api.block_parameters import (DefaultFrame,
                                                       FrameTransform)
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.block_builder_chrono.block_classes import (BLOCK_CLASS_TYPES,
                                                       ChronoRevolveJoint,
                                                       PrimitiveBody)
from rostok.block_builder_chrono.block_connect import place_and_connect
from rostok.graph_grammar.node import GraphGrammar, Node, UniqueBlueprint
from rostok.graph_grammar.node_block_typing import NodeFeatures
from rostok.virtual_experiment.sensors import ContactReporter


@dataclass
class RobotNode:
    id: int
    block: BLOCK_CLASS_TYPES
    node: Node


class BuiltGraph:

    def __init__(self,
                 graph,
                 system,
                 is_base_fixed=True,
                 initial_position: FrameTransform = DefaultFrame):
        self.__graph: GraphGrammar = deepcopy(graph)
        self.block_map: List[PrimitiveBody] = []
        self.block_vector: List[Tuple[int, PrimitiveBody]] = []
        self.joint_map: List[ChronoRevolveJoint] = []
        self.joint_vector: List[Tuple[int, ChronoRevolveJoint]] = []
        self.build_into_system(system, initial_position)
        if is_base_fixed:
            self.fix_base()
        self.fill_maps()

    def fill_maps(self):
        paths = self.__graph.get_sorted_root_based_paths()
        for path in paths:
            joint_path = []
            block_path = []
            for idx in path:
                if NodeFeatures.is_joint(self.__graph.nodes[idx]["Node"]):
                    joint_path.append(idx)
                if NodeFeatures.is_body(self.__graph.nodes[idx]["Node"]):
                    block_path.append(idx)
            self.block_map.append(block_path)
            self.joint_map.append(joint_path)

    def build_into_system(self,
                          system: chrono.ChSystem,
                          initial_position: FrameTransform = DefaultFrame):
        paths = self.__graph.get_sorted_root_based_paths()
        block_chains: List[BLOCK_CLASS_TYPES] = []
        for path in paths:
            chain: List[BLOCK_CLASS_TYPES] = []
            for idx in path:
                # if the node hasn't been built yet
                if self.__graph.nodes[idx].get("Blocks", None) is None:
                    # build all objects relevant to the node and add them into graph
                    blueprint = self.__graph.nodes[idx]["Node"].block_blueprint
                    created_blocks = creator.init_block_from_blueprint(blueprint)
                    if NodeFeatures.is_joint(self.__graph.nodes[idx].get("Node", None)):
                        self.joint_vector.append((idx, created_blocks))
                    elif NodeFeatures.is_body(self.__graph.nodes[idx].get("Node", None)):
                        self.block_vector.append((idx, created_blocks))

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
            #print(line)
            place_and_connect(line, system)

    def fix_base(self):
        # Fixation palm of grab mechanism
        base_id = self.__graph.get_root_id()
        base = self.__graph.nodes[base_id].get("Blocks", None)
        base.body.SetBodyFixed(True)




from rostok.control_chrono.controller import RobotControllerChrono
class Robot:
    def __init__(self,
                 robot_graph: GraphGrammar,
                 system,
                 control_parameters,
                 start_frame: FrameTransform = DefaultFrame):
        self.__built_graph = BuiltGraph(robot_graph, system, start_frame)
        self.contact_reporter = ContactReporter()
        self.contact_reporter.set_body_list(self.__built_graph.block_vector)

        self.controller = RobotControllerChrono(self.__built_graph.joint_vector, control_parameters)
        
        self.bridge_set: set[int] = set()
        unique_blueprint_array = self.__graph.build_unique_blueprint_array()
        # Map { id from graph : block }
        self.block_map = self.__build_robot(unique_blueprint_array, start_frame)
        self.__bind_blocks_to_graph()

    def __build_robot(self, unique_blueprint_array: list[list[UniqueBlueprint]],
                      start_frame: FrameTransform) -> dict[int, BLOCK_CLASS_TYPES]:
        blocks = []
        uniq_blocks = {}
        for unique_blueprint_line in unique_blueprint_array:
            block_line = []
            for unique_blueprint in unique_blueprint_line:

                id = unique_blueprint.id
                blueprint = unique_blueprint.block_blueprint

                if not (id in uniq_blocks.keys()):
                    block_buf = creator.init_block_from_blueprint(blueprint)
                    block_line.append(block_buf)
                    uniq_blocks[id] = block_buf
                else:
                    block_buf = uniq_blocks[id]
                    block_line.append(block_buf)

            blocks.append(block_line)

        chrono_vector_position = ChVectorD(*start_frame.position)

        chrono_quat_rotation = ChQuaternionD(*start_frame.rotation)

        ids_blocks = list(uniq_blocks)

        base_id = self.__graph.closest_node_to_root(ids_blocks)
        uniq_blocks[base_id].body.SetPos(chrono_vector_position)
        uniq_blocks[base_id].body.SetRot(chrono_quat_rotation)

        for line in blocks:
            place_and_connect(line, self.__simulation)

        return uniq_blocks

    def __bind_blocks_to_graph(self):
        for node_id, node in self.__graph.nodes.items():
            block = self.block_map[node_id]
            # Modify graph
            node["Block"] = block

    def fix_base(self):
        # Fixation palm of grab mechanism
        __fixed = True
        ids_blocks = list(self.block_map.keys())
        base_id = self.__graph.closest_node_to_root(ids_blocks)
        self.block_map[base_id].body.SetBodyFixed(True)

    def get_base_body(self):
        ids_blocks = list(self.block_map.keys())
        base_id = self.__graph.closest_node_to_root(ids_blocks)
        return self.block_map[base_id]

    @property
    def get_joints(self):
        """Create 2D-list joints from list of blocks. First index is the number
            partition graph, second index is the number of joint + create graph joint of robot

        """

        def is_joint(rbnode: RobotNode):
            return NodeFeatures.is_joint(rbnode.node)

        dfs_j = []
        dfs_rbnode = self.get_dfs_partiton()
        for branch in dfs_rbnode:
            branch_rb = list(filter(is_joint, branch))
            branch_block = list(map(lambda x: x.block, branch_rb))
            len_joints = len(branch_block)
            if len_joints != 0:
                dfs_j.append(branch_block)

        dfs_j.sort(key=len)
        return dfs_j

    def get_block_graph(self):
        return self.__graph

    def get_dfs_partiton(self) -> list[list[RobotNode]]:
        partition = self.__graph.graph_partition_dfs()

        def covert_to_robot_node(x):
            return RobotNode(x, self.__graph.nodes()[x]["Block"], self.__graph.nodes()[x]["Node"])

        partiton_graph = [list(map(covert_to_robot_node, x)) for x in partition]

        return partiton_graph
