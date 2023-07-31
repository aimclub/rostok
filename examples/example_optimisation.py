from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.block_builder_chrono.blocks_utils import frame_transform_to_chcoordsys
from rostok.criterion.criterion_calculation import (ForceCriterion, SimulationReward, TimeCriterion,
                                                    EventSlipOut, InstantContactingLinkCriterion,
                                                    InstantForceCriterion,
                                                    InstantObjectCOGCriterion, GraspTimeCriterion,
                                                    FinalPositionCriterion)
from rostok.criterion.simulation_flags import (EventContact, EventContactTimeOut, EventFlyingApart,
                                               EventStopExternalForce, EventGrasp)
from rostok.simulation_chrono.simulation_scenario import ConstTorqueGrasp
from rostok.trajectory_optimizer.control_optimizer import (CalculatorWithGraphOptimization,
                                                           CalculatorWithOptimizationDirect,
                                                           LinearControlOptimizationDirect,
                                                           TendonLikeControlOptimization)

from rostok.block_builder_api.block_parameters import Material, FrameTransform
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.block_builder_api.easy_body_shapes import Box

from rostok.library.rule_sets.simple_designs import get_two_link_one_finger, get_one_link_one_finger, get_two_link_three_finger


def config_with_standard(grasp_object_blueprint):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(0.001, 3)
    simulation_manager.grasp_object_callback = lambda: creator.create_environment_body(
        grasp_object_blueprint)
    event_contact = EventContact()
    simulation_manager.add_event(event_contact)
    event_timeout = EventContactTimeOut(2, event_contact)
    simulation_manager.add_event(event_timeout)
    event_flying_apart = EventFlyingApart(4)
    simulation_manager.add_event(event_flying_apart)
    event_slipout = EventSlipOut(0.5)
    simulation_manager.add_event(event_slipout)
    event_grasp = EventGrasp(
        grasp_limit_time=4,
        contact_event=event_contact,
        verbosity=0,
    )
    simulation_manager.add_event(event_grasp)
    event_stop_external_force = EventStopExternalForce(grasp_event=event_grasp, force_test_time=3)
    simulation_manager.add_event(event_stop_external_force)
    print(repr(event_stop_external_force))
    #create criterion manager
    simulation_rewarder = SimulationReward(verbosity=0)
    #create criterions and add them to manager
    simulation_rewarder.add_criterion(TimeCriterion(10, event_timeout, event_grasp), 1)
    simulation_rewarder.add_criterion(ForceCriterion(event_timeout), 1)
    simulation_rewarder.add_criterion(InstantContactingLinkCriterion(event_grasp), 1)
    simulation_rewarder.add_criterion(InstantForceCriterion(event_grasp), 1)
    simulation_rewarder.add_criterion(InstantObjectCOGCriterion(event_grasp), 1)
    n_steps = int(10 / 0.001)
    print(n_steps)
    simulation_rewarder.add_criterion(GraspTimeCriterion(event_grasp, n_steps), 1)
    simulation_rewarder.add_criterion(FinalPositionCriterion(20, event_grasp, event_slipout), 1)

    control_optimizer = TendonLikeControlOptimization(simulation_manager, simulation_rewarder,
                                                      (3, 15), 10)

    return control_optimizer


mat = Material()
mat.Friction = 0.65
mat.DampingF = 0.65
obj = EnvironmentBodyBlueprint(shape=Box(0.7, 0.4, 0.6),
                               material=mat,
                               pos=FrameTransform([0.1, 0.5, 0.15], [1, 0, 0, 0]))
optic_cfg = config_with_standard(obj)
optic_cfg.calculate_reward(get_two_link_three_finger())