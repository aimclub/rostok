from copy import deepcopy
from dataclasses import dataclass

from pychrono.core import ChQuaternionD, ChVectorD

from rostok.graph_grammar.node_block_typing import NodeFeatures
from rostok.block_builder_chrono.block_connect import place_and_connect
from rostok.block_builder_chrono.block_types import Block
from rostok.block_builder_chrono.block_classes import BLOCK_CLASS_TYPES
from rostok.block_builder_chrono.blocks_utils import (FrameTransform, DefaultFrame)
from rostok.graph_grammar.node import GraphGrammar, Node, UniqueBlueprint


@dataclass
class RobotNode:
    id: int
    block: BLOCK_CLASS_TYPES
    node: Node


class Robot:
    __fixed = False

    def __init__(self,
                 robot_graph: GraphGrammar,
                 simulation,
                 start_frame: FrameTransform = DefaultFrame):
        self.__graph = deepcopy(robot_graph)
        self.__simulation = simulation
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
                    block_buf = blueprint.create_block()
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
