from copy import deepcopy

import hyperparameters as hp

from rostok.control_chrono.tendon_controller import TendonControllerParameters
from rostok.criterion.criterion_calculation import (
    FinalPositionCriterion, GraspTimeCriterion, InstantContactingLinkCriterion,
    InstantForceCriterion, InstantObjectCOGCriterion, SimulationReward,
    TimeCriterion)
from rostok.criterion.simulation_flags import (EventContact,
                                               EventContactTimeOut,
                                               EventFlyingApart, EventGrasp,
                                               EventSlipOut,
                                               EventStopExternalForce)
from rostok.simulation_chrono.simulation_scenario import GraspScenario
from rostok.trajectory_optimizer.control_optimizer import (
    CalculatorWithOptimizationDirect, TendonOptimizerCombinationForce)
from rostok.utils.numeric_utils import Offset
from rostok.trajectory_optimizer.control_optimizer import (CalculatorWithGraphOptimization,
                                                           CalculatorWithOptimizationDirect,
                                                           LinearCableControlOptimization,
                                                           TendonLikeControlOptimization,
                                                           LinearControlOptimizationDirect)


def config_independent_torque(grasp_object_blueprint):
    # configurate the simulation manager
    simulation_manager = GraspScenario(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION, tendon = False)
    # simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
    #     grasp_object_blueprint)
    simulation_manager.grasp_object_callback = grasp_object_blueprint
    event_contact = EventContact()
    simulation_manager.add_event(event_contact)
    event_timeout = EventContactTimeOut(hp.FLAG_TIME_NO_CONTACT, event_contact)
    simulation_manager.add_event(event_timeout)
    event_flying_apart = EventFlyingApart(hp.FLAG_FLYING_APART)
    simulation_manager.add_event(event_flying_apart)
    event_slipout = EventSlipOut(hp.FLAG_TIME_SLIPOUT)
    simulation_manager.add_event(event_slipout)
    event_grasp = EventGrasp(
        grasp_limit_time=hp.GRASP_TIME,
        contact_event=event_contact,
        verbosity=0,
    )
    simulation_manager.add_event(event_grasp)
    event_stop_external_force = EventStopExternalForce(grasp_event=event_grasp,
                                                       force_test_time=hp.FORCE_TEST_TIME)
    simulation_manager.add_event(event_stop_external_force)

    #create criterion manager
    simulation_rewarder = SimulationReward(verbosity=0)
    #create criterions and add them to manager
    simulation_rewarder.add_criterion(TimeCriterion(hp.GRASP_TIME, event_timeout, event_grasp),
                                      hp.TIME_CRITERION_WEIGHT)
    #simulation_rewarder.add_criterion(ForceCriterion(event_timeout), hp.FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantContactingLinkCriterion(event_grasp),
                                      hp.INSTANT_CONTACTING_LINK_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantForceCriterion(event_grasp),
                                      hp.INSTANT_FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantObjectCOGCriterion(event_grasp),
                                      hp.OBJECT_COG_CRITERION_WEIGHT)
    n_steps = int(hp.GRASP_TIME / hp.TIME_STEP_SIMULATION)
    print(n_steps)
    simulation_rewarder.add_criterion(GraspTimeCriterion(event_grasp, n_steps),
                                      hp.GRASP_TIME_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(
        FinalPositionCriterion(hp.REFERENCE_DISTANCE, event_grasp, event_slipout),
        hp.FINAL_POSITION_CRITERION_WEIGHT)

    control_optimizer = CalculatorWithOptimizationDirect(simulation_manager, simulation_rewarder,
                                                         hp.CONTROL_OPTIMIZATION_BOUNDS,
                                                         hp.CONTROL_OPTIMIZATION_ITERATION)
    return control_optimizer


def config_with_tendon(grasp_object_blueprint):
    # configurate the simulation manager

    simulation_manager = GraspScenario(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    simulation_manager.grasp_object_callback = grasp_object_blueprint #lambda: creator.create_environment_body(
        #grasp_object_blueprint)
    event_contact = EventContact()
    simulation_manager.add_event(event_contact)
    event_timeout = EventContactTimeOut(hp.FLAG_TIME_NO_CONTACT, event_contact)
    simulation_manager.add_event(event_timeout)
    event_flying_apart = EventFlyingApart(hp.FLAG_FLYING_APART)
    simulation_manager.add_event(event_flying_apart)
    event_slipout = EventSlipOut(hp.FLAG_TIME_SLIPOUT)
    simulation_manager.add_event(event_slipout)
    event_grasp = EventGrasp(grasp_limit_time=hp.GRASP_TIME,
                             contact_event=event_contact,
                             verbosity=0,
                             simulation_stop=1)
    simulation_manager.add_event(event_grasp)
    event_stop_external_force = EventStopExternalForce(grasp_event=event_grasp,
                                                       force_test_time=hp.FORCE_TEST_TIME)
    simulation_manager.add_event(event_stop_external_force)

    #create criterion manager
    simulation_rewarder = SimulationReward(verbosity=0)
    #create criterions and add them to manager
    simulation_rewarder.add_criterion(TimeCriterion(hp.GRASP_TIME, event_timeout, event_grasp),
                                      hp.TIME_CRITERION_WEIGHT)
    #simulation_rewarder.add_criterion(ForceCriterion(event_timeout), hp.FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantContactingLinkCriterion(event_grasp),
                                      hp.INSTANT_CONTACTING_LINK_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantForceCriterion(event_grasp),
                                      hp.INSTANT_FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantObjectCOGCriterion(event_grasp),
                                      hp.OBJECT_COG_CRITERION_WEIGHT)
    n_steps = int(hp.GRASP_TIME / hp.TIME_STEP_SIMULATION)
    print(n_steps)
    simulation_rewarder.add_criterion(GraspTimeCriterion(event_grasp, n_steps),
                                      hp.GRASP_TIME_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(
        FinalPositionCriterion(hp.REFERENCE_DISTANCE, event_grasp, event_slipout),
        hp.FINAL_POSITION_CRITERION_WEIGHT)

    data = TendonControllerParameters()
    data.amount_pulley_in_body = 2
    data.pulley_parameters_for_body = {
        0: [Offset(-0.14, True), Offset(0.005, False, True),
            Offset(0, True)],
        1: [Offset(-0.14, True), Offset(-0.005, False, True),
            Offset(0, True)]
    }
    data.starting_point_parameters = [Offset(-1., True), Offset(5 , True), Offset(0, True)]
    data.tip_parameters = [Offset(-0.3, True), Offset(-0.005, False, True), Offset(0, True)]

    control_optimizer = TendonOptimizerCombinationForce(simulation_scenario = simulation_manager, rewarder = simulation_rewarder,
                                            data = data,
                                            starting_finger_angles=-45,
                                            tendon_forces = hp.TENDON_DISCRETE_FORCES)

    return control_optimizer
