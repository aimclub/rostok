import hyperparameters as hp

from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.criterion.criterion_calculation import (ForceCriterion, LateForceAmountCriterion,
                                                    LateForceCriterion, ObjectCOGCriterion,
                                                    SimulationReward, TimeCriterion, InstantContactingLinkCriterion)
from rostok.criterion.simulation_flags import (FlagContactTimeOut, FlagFlyingApart, FlagSlipout)
from rostok.simulation_chrono.simulation_scenario import ConstTorqueGrasp
from rostok.trajectory_optimizer.control_optimizer import (CalculatorWithGraphOptimization,
                                                           CalculatorWithOptimizationDirect)


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
    simulation_rewarder.add_criterion(LateForceCriterion(0.5, 3), hp.OBJECT_COG_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(LateForceAmountCriterion(0.5), hp.OBJECT_COG_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantContactingLinkCriterion(2), 1)

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
    simulation_rewarder.add_criterion(LateForceCriterion(0.5, 3), hp.OBJECT_COG_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(LateForceAmountCriterion(0.5), hp.OBJECT_COG_CRITERION_WEIGHT)

    control_optimizer = CalculatorWithGraphOptimization(simulation_manager, simulation_rewarder,
                                                        torque_dict)

    return control_optimizer