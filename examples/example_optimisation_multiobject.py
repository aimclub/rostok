from copy import deepcopy
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.block_builder_chrono.blocks_utils import frame_transform_to_chcoordsys
from rostok.criterion.criterion_calculation import (ForceCriterion, LateForceAmountCriterion,
                                                    LateForceCriterion, ObjectCOGCriterion,
                                                    SimulationReward, TimeCriterion)
from rostok.criterion.simulation_flags import (FlagContactTimeOut, FlagFlyingApart, FlagSlipout)
from rostok.simulation_chrono.simulation_scenario import ConstTorqueMultiGrasp, ConstTorqueGrasp
from rostok.trajectory_optimizer.control_optimizer import (CalculatorWithGraphOptimization,
                                                           CalculatorWithOptimizationDirect,
                                                           LinearControlOptimizationDirect, TendonLikeControlOptimization)

from rostok.block_builder_api.block_parameters import Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.easy_body_shapes import Box, Sphere

from rostok.library.rule_sets.simple_designs import  get_two_link_one_finger, get_one_link_one_finger, get_two_link_three_finger 


# def config_with_standard_multiobject(grasp_object_blueprint: list[EnvironmentBodyBlueprint]):
#     # configurate the simulation manager
#     simulation_manager = ConstTorqueMultiGrasp(0.001, 3)
    
#     object_callback = lambda id: creator.create_environment_body(grasp_object_blueprint[id])
#     simulation_manager.grasp_object_callback = object_callback
#     simulation_manager.add_flag(FlagContactTimeOut(1))
#     simulation_manager.add_flag(FlagFlyingApart(10))
    
#     simulation_manager.add_flag(FlagSlipout(0.8))
#     #create criterion manager
#     simulation_rewarder = SimulationReward()
#     #create criterions and add them to manager
#     simulation_rewarder.add_criterion(TimeCriterion(3), 1)
#     simulation_rewarder.add_criterion(ForceCriterion(), 1)
#     simulation_rewarder.add_criterion(ObjectCOGCriterion(), 1)
#     simulation_rewarder.add_criterion(LateForceCriterion(0.5, 3), 1)
#     simulation_rewarder.add_criterion(LateForceAmountCriterion(0.5), 1)

#     control_optimizer = TendonLikeControlMultiOptimization(simulation_manager, simulation_rewarder,
#                                                             (3, 15), 3, object_weights=[1/3,1/3,1/3])

#     return control_optimizer

def config_with_standard_multiobject(grasp_object_blueprint: list[EnvironmentBodyBlueprint]):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(0.001, 3)

    simulation_manager.add_flag(FlagContactTimeOut(1))
    simulation_manager.add_flag(FlagFlyingApart(10))
    
    simulation_manager.add_flag(FlagSlipout(0.8))
    simulation_managers = []
    object_callback = [(lambda obj=obj: creator.create_environment_body(obj)) for obj in grasp_object_blueprint]
    for k in range(len(grasp_object_blueprint)):
        simulation_managers.append(deepcopy(simulation_manager))
        simulation_managers[-1].grasp_object_callback = object_callback[k]

    #create criterion manager
    simulation_rewarder = SimulationReward()
    #create criterions and add them to manager
    simulation_rewarder.add_criterion(TimeCriterion(1), 1)
    simulation_rewarder.add_criterion(ForceCriterion(), 1)
    simulation_rewarder.add_criterion(ObjectCOGCriterion(), 1)
    simulation_rewarder.add_criterion(LateForceCriterion(0.5, 3), 1)
    simulation_rewarder.add_criterion(LateForceAmountCriterion(0.5), 1)

    control_optimizer = TendonLikeControlOptimization(simulation_managers, simulation_rewarder,
                                                            (3, 15), 1)

    return control_optimizer

mat = Material()
mat.Friction = 0.65
mat.DampingF = 0.65
objs = []
objs.append(EnvironmentBodyBlueprint(shape=Box(0.7, 0.4, 0.6),
                               material=mat,
                               pos=FrameTransform([0.1, 0.5, 0.15], [1, 0, 0, 0])))

objs.append(EnvironmentBodyBlueprint(shape=Sphere(),
                               material=mat,
                               pos=FrameTransform([0.1, 0.5, 0.15], [1, 0, 0, 0])))
objs.append(EnvironmentBodyBlueprint(shape=Box(),
                               material=mat,
                               pos=FrameTransform([0.1, 0.5, 0.15], [1, 0, 0, 0])))
optic_cfg = config_with_standard_multiobject(objs)
res = optic_cfg.calculate_reward(get_two_link_three_finger())
data = optic_cfg.optim_parameters2data_control(res[1], get_two_link_three_finger())
None