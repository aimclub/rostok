from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.control_chrono.controller import SimpleKeyBoardController
from rostok.simulation_chrono.simulation_scenario import WalkingScenario
import pychrono as chrono
from rostok.utils.dataset_materials.material_dataclass_manipulating import DefaultChronoMaterialNSC
from wheels import get_stiff_wheels, get_wheels, get_stiff_wheels_ell, get_stiff_wheels_4
from rostok.graph_grammar.graph_utils import plot_graph
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface
from rostok.block_builder_api.easy_body_shapes import Box

def create_bump_track():
    def_mat = DefaultChronoMaterialNSC()
    floor = ChronoBlockCreatorInterface.create_environment_body(EnvironmentBodyBlueprint(Box(5, 0.05, 5), material=def_mat, color=[215, 255, 0]))
    chrono_material = chrono.ChMaterialSurfaceNSC()
    #chrono_material.SetFriction(0.67)
    mesh = chrono.ChBodyEasyMesh("Bump.obj", 8000, True, True, True, chrono_material, 0.002)
    floor.body = mesh
    floor.body.SetNameString("Floor")
    floor.body.SetPos(chrono.ChVectorD(1.5,-0.07,0))
    floor.body.GetVisualShape(0).SetTexture("./chess.png", 0.03, 0.03)
    floor.body.SetBodyFixed(True)
    return floor

def create_track():
    def_mat = DefaultChronoMaterialNSC()
    floor = ChronoBlockCreatorInterface.create_environment_body(EnvironmentBodyBlueprint(Box(5, 0.05, 5), material=def_mat, color=[215, 255, 0]))
    chrono_material = chrono.ChMaterialSurfaceNSC()
    #chrono_material.SetFriction(0.67)
    mesh = chrono.ChBodyEasyMesh("TRACKMANIA.obj", 8000, True, True, True, chrono_material, 0.002)
    floor.body = mesh
    floor.body.SetNameString("Floor")
    floor.body.SetPos(chrono.ChVectorD(6.6,-0.04,5.2))
    floor.body.GetVisualShape(0).SetTexture("./chess.png", 0.03, 0.03)
    floor.body.SetBodyFixed(True)
    return floor


floor = create_track()

scenario = WalkingScenario(0.001, 10000, SimpleKeyBoardController)
scenario.set_floor(floor)
graph = get_stiff_wheels_4()

parameters = {}
parameters["forward"] = 0.5
parameters["reverse"]= 0.5
parameters["forward_rotate"] = 0.5
parameters["reverse_rotate"] = 0.3

 

scenario.run_simulation(graph, parameters, starting_positions=[[-60,60,0], [60,-60,0], [-60,60,0], [60,-60,0]], vis = True, delay=True, is_follow_camera = True)

