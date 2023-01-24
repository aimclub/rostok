
from rostok.block_builder.envbody_shapes import Sphere
from rostok.launch.open_chain_gen import create_generator_by_config

model = create_generator_by_config("rostok/launch/config.ini")
model.set_grasp_object(Sphere())
reporter = model.run_generation()
model.save_result()
model.visualize_result()
