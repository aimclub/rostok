from copy import deepcopy

import hyperparameters as hp

from rostok.control_chrono.tendon_controller import TendonController_2p, TendonControllerParameters
from rostok.criterion.criterion_calculation import (FinalPositionCriterion, GraspTimeCriterion,
                                                    InstantContactingLinkCriterion,
                                                    InstantForceCriterion,
                                                    InstantObjectCOGCriterion, SimulationReward,
                                                    TimeCriterion)
from rostok.criterion.simulation_flags import (EventContactBuilder, EventContactTimeOutBuilder,
                                               EventFlyingApartBuilder, EventGraspBuilder,
                                               EventSlipOutBuilder, EventStopExternalForceBuilder)
from rostok.simulation_chrono.simulation_scenario import GraspScenario
from rostok.trajectory_optimizer.control_optimizer import BruteForceOptimisation1D, ConstTorqueOptiVar, GlobalOptimisationEachSim, TendonForceOptiVar
from rostok.utils.numeric_utils import Offset
import rostok.control_chrono.external_force as f_ext


def get_tendon_cfg():
    tendon_controller_cfg = TendonControllerParameters()
    tendon_controller_cfg.amount_pulley_in_body = 2
    tendon_controller_cfg.pulley_parameters_for_body = {
        0: [Offset(-0.14, True), Offset(0.005, False, True),
            Offset(0, True)],
        1: [Offset(-0.14, True), Offset(-0.005, False, True),
            Offset(0, True)]
    }
    tendon_controller_cfg.starting_point_parameters = [
        Offset(-0.02, False), Offset(0.025, False),
        Offset(0, True)
    ]
    tendon_controller_cfg.tip_parameters = [
        Offset(-0.3, True), Offset(-0.005, False, True),
        Offset(0, True)
    ]
    return tendon_controller_cfg


def config_independent_torque(grasp_object_blueprint):

    simulation_manager = GraspScenario(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)

    simulation_manager.grasp_object_callback = grasp_object_blueprint
    event_contact_builder = EventContactBuilder()
    simulation_manager.add_event_builder(event_contact_builder)
    event_timeout_builder = EventContactTimeOutBuilder(hp.FLAG_TIME_NO_CONTACT,
                                                       event_contact_builder)
    simulation_manager.add_event_builder(event_timeout_builder)
    event_flying_apart_builder = EventFlyingApartBuilder(hp.FLAG_FLYING_APART)
    simulation_manager.add_event_builder(event_flying_apart_builder)
    event_slipout_builder = EventSlipOutBuilder(hp.FLAG_TIME_SLIPOUT)
    simulation_manager.add_event_builder(event_slipout_builder)
    event_grasp_builder = EventGraspBuilder(
        grasp_limit_time=hp.GRASP_TIME,
        event_contact_builder=event_contact_builder,
        verbosity=0,
    )
    simulation_manager.add_event_builder(event_grasp_builder)
    event_stop_external_force = EventStopExternalForceBuilder(
        event_grasp_builder=event_grasp_builder, force_test_time=hp.FORCE_TEST_TIME)
    simulation_manager.add_event_builder(event_stop_external_force)

    #create criterion manager
    simulation_rewarder = SimulationReward(verbosity=0)
    #create criterions and add them to manager
    simulation_rewarder.add_criterion(
        TimeCriterion(hp.GRASP_TIME, event_timeout_builder, event_grasp_builder),
        hp.TIME_CRITERION_WEIGHT)
    #simulation_rewarder.add_criterion(ForceCriterion(event_timeout), hp.FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantContactingLinkCriterion(event_grasp_builder),
                                      hp.INSTANT_CONTACTING_LINK_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantForceCriterion(event_grasp_builder),
                                      hp.INSTANT_FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantObjectCOGCriterion(event_grasp_builder),
                                      hp.OBJECT_COG_CRITERION_WEIGHT)
    n_steps = int(hp.GRASP_TIME / hp.TIME_STEP_SIMULATION)
    print(n_steps)
    simulation_rewarder.add_criterion(GraspTimeCriterion(event_grasp_builder, n_steps),
                                      hp.GRASP_TIME_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(
        FinalPositionCriterion(hp.REFERENCE_DISTANCE, event_grasp_builder, event_slipout_builder),
        hp.FINAL_POSITION_CRITERION_WEIGHT)

    const_optivar = ConstTorqueOptiVar(simulation_rewarder, -45)
    direct_args = {"maxiter": 2}
    global_const = GlobalOptimisationEachSim([simulation_manager], const_optivar, (0, 10),
                                             direct_args)

    return global_const


def config_tendon(grasp_object_blueprint):

    simulation_manager = GraspScenario(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION, TendonController_2p)

    simulation_manager.grasp_object_callback = grasp_object_blueprint
    event_contact_builder = EventContactBuilder()
    simulation_manager.add_event_builder(event_contact_builder)
    event_timeout_builder = EventContactTimeOutBuilder(hp.FLAG_TIME_NO_CONTACT,
                                                       event_contact_builder)
    simulation_manager.add_event_builder(event_timeout_builder)
    event_flying_apart_builder = EventFlyingApartBuilder(hp.FLAG_FLYING_APART)
    simulation_manager.add_event_builder(event_flying_apart_builder)
    event_slipout_builder = EventSlipOutBuilder(hp.FLAG_TIME_SLIPOUT)
    simulation_manager.add_event_builder(event_slipout_builder)
    event_grasp_builder = EventGraspBuilder(
        grasp_limit_time=hp.GRASP_TIME,
        event_contact_builder=event_contact_builder,
        verbosity=0,
    )
    simulation_manager.add_event_builder(event_grasp_builder)
    event_stop_external_force = EventStopExternalForceBuilder(
        event_grasp_builder=event_grasp_builder, force_test_time=hp.FORCE_TEST_TIME)
    simulation_manager.add_event_builder(event_stop_external_force)

    #create criterion manager
    simulation_rewarder = SimulationReward(verbosity=0)
    #create criterions and add them to manager
    simulation_rewarder.add_criterion(
        TimeCriterion(hp.GRASP_TIME, event_timeout_builder, event_grasp_builder),
        hp.TIME_CRITERION_WEIGHT)
    #simulation_rewarder.add_criterion(ForceCriterion(event_timeout), hp.FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantContactingLinkCriterion(event_grasp_builder),
                                      hp.INSTANT_CONTACTING_LINK_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantForceCriterion(event_grasp_builder),
                                      hp.INSTANT_FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(InstantObjectCOGCriterion(event_grasp_builder),
                                      hp.OBJECT_COG_CRITERION_WEIGHT)
    n_steps = int(hp.GRASP_TIME / hp.TIME_STEP_SIMULATION)
    print(n_steps)
    simulation_rewarder.add_criterion(GraspTimeCriterion(event_grasp_builder, n_steps),
                                      hp.GRASP_TIME_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(
        FinalPositionCriterion(hp.REFERENCE_DISTANCE, event_grasp_builder, event_slipout_builder),
        hp.FINAL_POSITION_CRITERION_WEIGHT)
    tendon_controller_cfg = get_tendon_cfg()
    tendon_optivar = TendonForceOptiVar(tendon_controller_cfg, simulation_rewarder, -45)
    tendon_optivar.is_vis = False
    brute_tendon = BruteForceOptimisation1D(hp.TENDON_DISCRETE_FORCES, [simulation_manager],
                                            tendon_optivar,
                                            num_cpu_workers=1)

    return brute_tendon

""" 

def config_with_tendon(grasp_object_blueprint):
    # configurate the simulation manager

    obj_forces = []
    obj_forces.append(f_ext.NullGravity(0))
    obj_forces.append(f_ext.RandomForces(1e6, 100, 0))
    obj_forces = f_ext.ExternalForces(obj_forces)
    simulation_manager = GraspScenario(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION, TendonController_2p, obj_external_forces=obj_forces)
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



def config_combination_force_tendon_multiobject_parallel(grasp_object_blueprint, weights):
    # configurate the simulation manager
    simulation_manager = GraspScenario(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION, TendonController_2p)
    object_callback = grasp_object_blueprint
    #[
    #    (lambda obj=obj: creator.create_environment_body(obj)) for obj in grasp_object_blueprint
    #]
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

    data = TendonControllerParameters()
    data.amount_pulley_in_body = 2
    data.pulley_parameters_for_body = {
        0: [Offset(-0.14, True), Offset(0.005, False, True),
            Offset(0, True)],
        1: [Offset(-0.14, True), Offset(-0.005, False, True),
            Offset(0, True)]
    }
    data.starting_point_parameters = [Offset(-0.02, False), Offset(0.025, False), Offset(0, True)]
    data.tip_parameters = [Offset(-0.3, True), Offset(-0.005, False, True), Offset(0, True)]


    control_optimizer = ParralelOptimizerCombinationForce(simulation_managers,
                                                       simulation_rewarder, data, hp.TENDON_DISCRETE_FORCES, starting_finger_angles=-25)

    return control_optimizer

"""