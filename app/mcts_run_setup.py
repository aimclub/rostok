import hyperparameters as hp

from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.criterion.criterion_calculation import (
    ForceCriterion, InstantContactingLinkCriterion,InstantForceCriterion, InstantObjectCOGCriterion,
    SimulationReward, TimeCriterion, FinalPositionCriterion, GraspTimeCriterion)
from rostok.criterion.simulation_flags import (EventContactTimeOut, EventFlyingApart, EventGrasp,EventSlipOut)
from rostok.simulation_chrono.simulation_scenario import ConstTorqueGrasp
from rostok.trajectory_optimizer.control_optimizer import (
    CalculatorWithGraphOptimization, CalculatorWithOptimizationDirect)


def config_with_standard(grasp_object_blueprint):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
        grasp_object_blueprint)
    event_timeout = EventContactTimeOut(hp.FLAG_TIME_NO_CONTACT)
    simulation_manager.add_event(event_timeout)
    event_flying_apart = EventFlyingApart(hp.FLAG_FLYING_APART)
    simulation_manager.add_event(event_flying_apart)
    event_slipout = EventSlipOut(hp.FLAG_TIME_SLIPOUT)
    simulation_manager.add_event(event_slipout)
    event_grasp = EventGrasp(activation_code=0, verbosity=1)
    simulation_manager.add_event(event_grasp)

    #create criterion manager
    simulation_rewarder = SimulationReward(1)
    #create criterions and add them to manager
    simulation_rewarder.add_criterion(TimeCriterion(hp.TIME_SIMULATION, event_timeout, event_grasp), hp.TIME_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(ForceCriterion(hp.TIME_SIMULATION, event_timeout), hp.FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantContactingLinkCriterion(event_grasp), hp.OBJECT_COG_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantForceCriterion(event_grasp), 1)
    simulation_rewarder.add_criterion(InstantObjectCOGCriterion(event_grasp),1)
    n_steps = int(hp.TIME_SIMULATION / hp.TIME_STEP_SIMULATION)
    print(n_steps)
    simulation_rewarder.add_criterion(
        GraspTimeCriterion(event_grasp, n_steps),
        hp.INSTANT_CONTACTING_LINK_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(FinalPositionCriterion(10, event_grasp, event_slipout), 10)

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