from rostok.block_builder_chrono.block_builder_chrono_api import ChronoBlockCreatorInterface
 
def calc_volume_body(blueprint):
    body = ChronoBlockCreatorInterface.init_block_from_blueprint(blueprint)
    volume = body.body.GetMass() / body.body.GetDensity()
    return volume