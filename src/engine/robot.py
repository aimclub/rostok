from engine.node import Node, BlockWrapper, WrapperTuple, GraphGrammar
from engine.node_render import Block, connect_blocks, ChronoRevolveJoint
from dataclasses import dataclass
import networkx as nx

class Robot:
    def __init__(self, robot_graph: GraphGrammar, simulation):
        self.graph = robot_graph
        self.__joint_graph = nx.DiGraph()
        wrapper_tuple_array = self.graph.build_terminal_wrapper_array()
        # Map { id from graph : block }
        self.block_map = self.__build_robot(simulation, wrapper_tuple_array)
        self.__bind_blocks_to_graph()

    def __build_robot(self, simulation, wrapper_tuple_array: list[list[WrapperTuple]]):
        blocks = []
        uniq_blocks = {}
        for wrapper_tuple_line in wrapper_tuple_array:
            block_line = []
            for wrapper_tuple in wrapper_tuple_line:

                id = wrapper_tuple.id
                wrapper = wrapper_tuple.block_wrapper

                if not (id in uniq_blocks.keys()):
                    wrapper.builder = simulation

                    block_buf = wrapper.create_block()
                    block_line.append(block_buf)
                    uniq_blocks[id] = block_buf
                else:
                    block_buf = uniq_blocks[id]
                    block_line.append(block_buf)
            blocks.append(block_line)

        for line in blocks:
            connect_blocks(line)

        return uniq_blocks
    
    def __bind_blocks_to_graph(self):
        for node_id, node in self.graph.nodes.items():
            block = self.block_map[node_id]
            # Modify graph
            node[Block] = block

    @property
    def get_joints(self):
        
        def is_joint(node: Node):
            if node.block_wrapper.block_cls == ChronoRevolveJoint:
                return True
            else:
                return False
        
        self.__joint_graph: nx.DiGraph = nx.DiGraph(self.graph)
        not_joints = []
        for raw_node in self.__joint_graph.nodes.items():
            node: Node = raw_node[1]["Node"]
            node_id = raw_node[0]
            in_edges = list(self.__joint_graph.in_edges(node_id))
            out_edges = list(self.__joint_graph.out_edges(node_id))
            if not is_joint(node):
                not_joints.append(node_id)
                if in_edges and out_edges:
                    for in_edge in in_edges:
                        for out_edge in out_edges:
                            self.__joint_graph.add_edge(in_edge[0],out_edge[1])
        self.__joint_graph.remove_nodes_from(not_joints)               
        joint_blocks = {node: self.block_map[node] for node in list(self.__joint_graph)}
        return joint_blocks
