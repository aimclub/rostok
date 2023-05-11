import rostok.criterion.criterion_calc as criterion
from rostok.block_builder_api.block_blueprints import TransformBlueprint, PrimitiveBodyBlueprint, \
EnvironmentBodyBlueprint, RevolveJointBlueprint
from rostok.block_builder_api.easy_body_shapes import Box
from rostok.block_builder_chrono.blocks_utils import FrameTransform
from rostok.graph_grammar.node import ROOT, GraphGrammar

from rostok.trajectory_optimizer.trajectory_generator import \
    create_torque_traj_from_x
from rostok.virtual_experiment.simulation_step import SimOut
from rostok.block_builder_api.block_parameters import DefaultFrame, Material, FrameTransform
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator


def get_object_to_grasp():
    matich = Material()
    matich.Friction = 0.65
    matich.DampingF = 0.65

    shape_box = Box(0.2, 0.2, 0.5)

    object_blueprint = EnvironmentBodyBlueprint(shape_box,
                                   material=matich,
                                   pos=FrameTransform([0, 0.5, 0], [0, -0.048, 0.706, 0.706]))
    obj = creator.init_block_from_blueprint(object_blueprint)
    return obj


def grab_crtitrion(sim_output: dict[int, SimOut], weight):

    return criterion.criterion_calc(sim_output, weight)


def create_grab_criterion_fun(weight):

    def fun(sim_output):
        return grab_crtitrion(sim_output, weight)

    return fun

