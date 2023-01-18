
from rostok.block_builder.envbody_shapes import Sphere
from rostok.launch.open_chain_gen import (OpenChainGen,
                                          create_generator_by_config)

model = create_generator_by_config("rostok/launch/config.ini")
model.set_grasp_object(Sphere)
graph_mechanism, trajectory, reward = model.run_generation()
