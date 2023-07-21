from copy import deepcopy

import hyperparameters as hp

from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.criterion.criterion_calculation import (ForceCriterion, InstantContactingLinkCriterion,
                                                    InstantForceCriterion,
                                                    InstantObjectCOGCriterion, SimulationReward,
                                                    TimeCriterion, FinalPositionCriterion,
                                                    GraspTimeCriterion)
from rostok.criterion.simulation_flags import (EventContact, EventContactTimeOut, EventFlyingApart,
                                               EventGrasp, EventSlipOut, EventStopExternalForce)
from rostok.simulation_chrono.simulation_scenario import ConstTorqueGrasp
from rostok.trajectory_optimizer.control_optimizer import (CalculatorWithGraphOptimization,
                                                           CalculatorWithOptimizationDirect, LinearCableControlOptimization, TendonLikeControlOptimization, LinearControlOptimizationDirect)


def config_with_standard(grasp_object_blueprint):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
        grasp_object_blueprint)
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


def config_with_standard_tendon(grasp_object_blueprint):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
        grasp_object_blueprint)
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

    control_optimizer = TendonLikeControlOptimization(simulation_manager, simulation_rewarder,
                                                         hp.CONTROL_OPTIMIZATION_BOUNDS_TENDON,
                                                         hp.CONTROL_OPTIMIZATION_ITERATION_TENDON, const_parameter=hp.TENDON_CONST)
    return control_optimizer

def config_with_standard_linear(grasp_object_blueprint):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
        grasp_object_blueprint)
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
    
    control_optimizer = LinearControlOptimizationDirect(simulation_manager, simulation_rewarder,
                                                         hp.CONTROL_OPTIMIZATION_BOUNDS_TENDON,
                                                         hp.CONTROL_OPTIMIZATION_ITERATION_TENDON,
                                                        const_parameter=hp.TENDON_CONST)
    return control_optimizer

def config_with_standard_graph(grasp_object_blueprint, torque_dict):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
        grasp_object_blueprint)
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
    simulation_rewarder.add_criterion(TimeCriterion(hp.TIME_SIMULATION, event_timeout, event_grasp),
                                      hp.TIME_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(ForceCriterion(event_timeout), hp.FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantContactingLinkCriterion(event_grasp),
                                      hp.INSTANT_CONTACTING_LINK_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantForceCriterion(event_grasp),
                                      hp.INSTANT_FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantObjectCOGCriterion(event_grasp),
                                      hp.OBJECT_COG_CRITERION_WEIGHT)
    n_steps = int(hp.TIME_SIMULATION / hp.TIME_STEP_SIMULATION)
    print(n_steps)
    simulation_rewarder.add_criterion(GraspTimeCriterion(event_grasp, n_steps),
                                      hp.GRASP_TIME_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(
        FinalPositionCriterion(hp.REFERENCE_DISTANCE, event_grasp, event_slipout),
        hp.FINAL_POSITION_CRITERION_WEIGHT)

    control_optimizer = CalculatorWithGraphOptimization(simulation_manager, simulation_rewarder,
                                                        torque_dict)

    return control_optimizer


def config_with_standard_multiobject(grasp_object_blueprint, weights):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    object_callback = [
        (lambda obj=obj: creator.create_environment_body(obj)) for obj in grasp_object_blueprint
    ]
    simulation_managers = []
    for k in range(len(grasp_object_blueprint)):
        simulation_managers.append((deepcopy(simulation_manager), weights[k]))
        simulation_managers[-1][0].grasp_object_callback = object_callback[k]

    event_contact = EventContact()
    event_timeout = EventContactTimeOut(hp.FLAG_TIME_NO_CONTACT, event_contact)
    event_flying_apart = EventFlyingApart(hp.FLAG_FLYING_APART)
    event_slipout = EventSlipOut(hp.FLAG_TIME_SLIPOUT)
    event_grasp = EventGrasp(
        grasp_limit_time=hp.GRASP_TIME,
        contact_event=event_contact,
        verbosity=0,
    )
    event_stop_external_force = EventStopExternalForce(grasp_event=event_grasp,
                                                       force_test_time=hp.FORCE_TEST_TIME)
    for manager in simulation_managers:
        manager[0].add_event(event_contact)
        manager[0].add_event(event_timeout)
        manager[0].add_event(event_flying_apart)
        manager[0].add_event(event_grasp)
        manager[0].add_event(event_stop_external_force)

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

    control_optimizer = CalculatorWithOptimizationDirect(simulation_managers, simulation_rewarder,
                                                         hp.CONTROL_OPTIMIZATION_BOUNDS,
                                                         hp.CONTROL_OPTIMIZATION_ITERATION_TENDON)

    return control_optimizer

def config_with_standard_tendon_multiobject(grasp_object_blueprint, weights):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    object_callback = [
        (lambda obj=obj: creator.create_environment_body(obj)) for obj in grasp_object_blueprint
    ]
    simulation_managers = []
    for k in range(len(grasp_object_blueprint)):
        simulation_managers.append((deepcopy(simulation_manager), weights[k]))
        simulation_managers[-1][0].grasp_object_callback = object_callback[k]

    event_contact = EventContact()
    event_timeout = EventContactTimeOut(hp.FLAG_TIME_NO_CONTACT, event_contact)
    event_flying_apart = EventFlyingApart(hp.FLAG_FLYING_APART)
    event_slipout = EventSlipOut(hp.FLAG_TIME_SLIPOUT)
    event_grasp = EventGrasp(
        grasp_limit_time=hp.GRASP_TIME,
        contact_event=event_contact,
        verbosity=0,
    )
    event_stop_external_force = EventStopExternalForce(grasp_event=event_grasp,
                                                       force_test_time=hp.FORCE_TEST_TIME)
    for manager in simulation_managers:
        manager[0].add_event(event_contact)
        manager[0].add_event(event_timeout)
        manager[0].add_event(event_flying_apart)
        manager[0].add_event(event_grasp)
        manager[0].add_event(event_stop_external_force)

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

    control_optimizer = LinearControlOptimizationDirect(simulation_managers, simulation_rewarder,
                                                         hp.CONTROL_OPTIMIZATION_BOUNDS_TENDON,
                                                         hp.CONTROL_OPTIMIZATION_ITERATION_TENDON, const_parameter = hp.TENDON_CONST)

    return control_optimizer

def config_with_standard_cable(grasp_object_blueprint):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
        grasp_object_blueprint)
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

    control_optimizer = LinearCableControlOptimization(simulation_manager, simulation_rewarder,
                                                         hp.CONTROL_OPTIMIZATION_BOUNDS_TENDON,
                                                         hp.CONTROL_OPTIMIZATION_ITERATION_TENDON, const_parameter=hp.TENDON_CONST)
    return control_optimizer