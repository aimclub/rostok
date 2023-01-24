from copy import deepcopy
from dataclasses import dataclass

from pychrono.core import ChVectorD, ChQuaternionD

from rostok.block_builder.blocks_utils import NodeFeatures
from rostok.block_builder.node_render import (Block, connect_blocks)
from rostok.graph_grammar.node import GraphGrammar, Node, WrapperTuple
from rostok.block_builder.transform_srtucture import FrameTransform, OriginWorldFrame


@dataclass
class RobotNode:
    id: int
    block: Block
    node: Node


class Robot:
    __fixed = False

    def __init__(self,
                 robot_graph: GraphGrammar,
                 simulation,
                 start_frame: FrameTransform = OriginWorldFrame):
        self.__graph = deepcopy(robot_graph)
        self.__simulation = simulation
        wrapper_tuple_array = self.__graph.build_terminal_wrapper_array()
        # Map { id from graph : block }
        self.block_map = self.__build_robot(wrapper_tuple_array, start_frame)
        self.__bind_blocks_to_graph()

    def __build_robot(self, wrapper_tuple_array: list[list[WrapperTuple]],
                      start_frame: FrameTransform) -> dict[int, Block]:
        blocks = []
        uniq_blocks = {}
        for wrapper_tuple_line in wrapper_tuple_array:
            block_line = []
            for wrapper_tuple in wrapper_tuple_line:

                id = wrapper_tuple.id
                wrapper = wrapper_tuple.block_wrapper

                if not (id in uniq_blocks.keys()):
                    block_buf = wrapper.create_block(self.__simulation)
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
            connect_blocks(line)

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
