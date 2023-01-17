
from rostok.block_builder.basic_node_block import SimpleBody
from rostok.launch.open_chain_gen import (OpenChainGen,
                                          create_generator_by_config)

model = create_generator_by_config("rostok/launch/config.ini")
model.set_grasp_object(SimpleBody.SPHERE)
graph_mechanism, trajectory, reward = model.run_generation()
