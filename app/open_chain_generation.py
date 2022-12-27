
from rostok.api import OpenChainGen, create_generator_by_config
from rostok.block_builder.basic_node_block import SimpleBody

model = create_generator_by_config("rostok/config.ini")
model.set_grasp_object(SimpleBody.SPHERE)
graph_mechanism, trajectory, reward = model.run_generation()
