
from rostok.api import OpenChainGen, create_generator_by_config

model = create_generator_by_config("rostok/config.ini")
model.run_generation()
