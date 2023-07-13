from copy import deepcopy

import hyperparameters as hp

from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.criterion.criterion_calculation import (
    ForceCriterion, InstantContactingLinkCriterion, LateForceAmountCriterion,
    LateForceCriterion, ObjectCOGCriterion, SimulationReward, TimeCriterion)
from rostok.criterion.simulation_flags import (FlagContactTimeOut,
                                               FlagFlyingApart, FlagSlipout)
from rostok.simulation_chrono.simulation_scenario import ConstTorqueGrasp
from rostok.trajectory_optimizer.control_optimizer import (
    CalculatorWithGraphOptimization, CalculatorWithOptimizationDirect,CalculatorWithOptimizationDirectList)


def config_with_standard(grasp_object_blueprint):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
        grasp_object_blueprint)
    simulation_manager.add_flag(FlagContactTimeOut(hp.FLAG_TIME_NO_CONTACT))
    simulation_manager.add_flag(FlagFlyingApart(hp.FLAG_FLYING_APART))
    simulation_manager.add_flag(FlagSlipout(hp.FLAG_TIME_SLIPOUT))
    #create criterion manager
    simulation_rewarder = SimulationReward(1)
    #create criterions and add them to manager
    simulation_rewarder.add_criterion(TimeCriterion(hp.TIME_SIMULATION), hp.TIME_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(ForceCriterion(), hp.FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(ObjectCOGCriterion(), hp.OBJECT_COG_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(LateForceCriterion(0.5, 3), hp.LATE_FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(LateForceAmountCriterion(0.5),
                                      hp.LATE_FORCE_AMOUNT_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(
        InstantContactingLinkCriterion(hp.CONTACTING_LINK_CALCULATION_TIME),
        hp.INSTANT_CONTACTING_LINK_CRITERION_WEIGHT)

    control_optimizer = CalculatorWithOptimizationDirect(simulation_manager, simulation_rewarder,
                                                         hp.CONTROL_OPTIMIZATION_BOUNDS,
                                                         hp.CONTROL_OPTIMIZATION_ITERATION)

    return control_optimizer


def config_with_standard_graph(grasp_object_blueprint, torque_dict):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
        grasp_object_blueprint)
    simulation_manager.add_flag(FlagContactTimeOut(hp.FLAG_TIME_NO_CONTACT))
    simulation_manager.add_flag(FlagFlyingApart(hp.FLAG_FLYING_APART))
    simulation_manager.add_flag(FlagSlipout(hp.FLAG_TIME_SLIPOUT))
    #create criterion manager
    simulation_rewarder = SimulationReward()
    #create criterions and add them to manager
    simulation_rewarder.add_criterion(TimeCriterion(hp.TIME_SIMULATION), hp.TIME_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(ForceCriterion(), hp.FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(ObjectCOGCriterion(), hp.OBJECT_COG_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(LateForceCriterion(0.5, 3), hp.LATE_FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(LateForceAmountCriterion(0.5),
                                      hp.LATE_FORCE_AMOUNT_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(
        InstantContactingLinkCriterion(hp.CONTACTING_LINK_CALCULATION_TIME),
        hp.INSTANT_CONTACTING_LINK_CRITERION_WEIGHT)

    control_optimizer = CalculatorWithGraphOptimization(simulation_manager, simulation_rewarder,
                                                        torque_dict)

    return control_optimizer

def config_with_standard_multiobject(grasp_object_blueprint):
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
    simulation_rewarder.add_criterion(TimeCriterion(3), 1)
    simulation_rewarder.add_criterion(ForceCriterion(), 1)
    simulation_rewarder.add_criterion(ObjectCOGCriterion(), 1)
    simulation_rewarder.add_criterion(LateForceCriterion(0.5, 3), 1)
    simulation_rewarder.add_criterion(LateForceAmountCriterion(0.5), 1)

    control_optimizer = CalculatorWithOptimizationDirect(simulation_managers, simulation_rewarder,
                                                            (3, 15), 1)

    return control_optimizer