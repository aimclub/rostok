from node import Node, BlockWrapper, WrapperTuple
from node_render import connect_blocks

class Robot:
    def __init__(self, wrapper_tuple_array: list[list[WrapperTuple]], simulation):
        self.wrapper_array = self.__build_robot(simulation,wrapper_tuple_array) 
        self.graph_representation 
    def __build_robot(self, simulation, wrapper_tuple_array: list[list[WrapperTuple]]) -> list[list[BlockWrapper]]:
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
        
        return blocks