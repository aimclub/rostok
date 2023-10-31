from copy import deepcopy
from dataclasses import dataclass, field
from typing import Type
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.control_chrono.controller import ConstController, RobotControllerChrono
from rostok.criterion.criterion_calculation import FinalPositionCriterion, GraspTimeCriterion, InstantContactingLinkCriterion, InstantForceCriterion, InstantObjectCOGCriterion, SimulationReward, TimeCriterion
from rostok.criterion.simulation_flags import EventContact, EventContactTimeOut, EventFlyingApart, EventSlipOut, \
EventGrasp, EventStopExternalForce
from rostok.simulation_chrono.simulation_scenario import GraspScenario, ParametrizedSimulation
import rostok.control_chrono.external_force as f_ext
from rostok.trajectory_optimizer.control_optimizer import BasePrepareOptiVar, ConstTorqueOptiVar


@dataclass
class SimulationConfig:
    time_step: float = 0.0001
    time_simulation: float = 10
    scenario_cls: Type[GraspScenario] = GraspScenario
    obj_disturbance_forces: list[f_ext.ABCForceCalculator] = field(default_factory=list)


# From tis stuff generates sim_manager


@dataclass
class GraspObjective:
    object_list: list[EnvironmentBodyBlueprint] = field(default_factory=list)
    weight_list: list[float] = field(default_factory=list)
    event_time_no_contact: float = 2
    event_flying_apart_time: float = 2
    event_slipout_time: float = 1
    event_grasp_time: float = 5
    event_force_test_time: float = 5

    time_criterion_weight: float = 1
    instant_contact_link_criterion_weight: float = 1
    instant_force_criterion_weight: float = 1
    instant_cog_criterion_weight: float = 1
    grasp_time_criterion_weight: float = 1
    final_pos_criterion_weight: float = 1


@dataclass
class BaseCalculateRewardCfg:
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)
    grasp_objective: GraspObjective = field(default_factory=GraspObjective)
    prepare_reward: BasePrepareOptiVar = ConstTorqueOptiVar(SimulationReward(), None)


@dataclass
class BruteForceRewardCfg(BaseCalculateRewardCfg):
    pass


def create_base_sim(contoller_cls, sim_cfg: SimulationConfig):
    """Base sim without object

    Args:
        contoller_cls (_type_): _description_
        sim_cfg (SimulationConfig): _description_

    Returns:
        _type_: _description_
    """
    return sim_cfg.scenario_cls(sim_cfg.time_step,
                                sim_cfg.time_simulation,
                                contoller_cls,
                                obj_external_forces=sim_cfg.obj_disturbance_forces)


def add_grasp_events_from_cfg(sim: GraspScenario, cfg: GraspObjective):
    event_contact = EventContact()
    sim.add_event(event_contact)
    event_timeout = EventContactTimeOut(cfg.event_time_no_contact, event_contact)
    sim.add_event(event_timeout)
    event_flying_apart = EventFlyingApart(cfg.event_flying_apart_time)
    sim.add_event(event_flying_apart)
    event_slipout = EventSlipOut(cfg.event_slipout_time)
    sim.add_event(event_slipout)
    event_grasp = EventGrasp(grasp_limit_time=cfg.event_grasp_time,
                             contact_event=event_contact,
                             verbosity=0,
                             simulation_stop=1)
    sim.add_event(event_grasp)
    event_stop_external_force = EventStopExternalForce(grasp_event=event_grasp,
                                                       force_test_time=cfg.event_force_test_time)
    sim.add_event(event_stop_external_force)
    return event_contact, event_timeout, event_flying_apart, event_slipout, event_stop_external_force


def create_sim_list(base_sim: GraspScenario, object_list: list[EnvironmentBodyBlueprint]):
    sim_list = []
    for obj_i in object_list:
        sim_i = deepcopy(base_sim)
        sim_i.grasp_object_callback = obj_i
        sim_list.append(sim_i)
    return sim_list


def prepare_simulation_scenario_list(contoller_cls, sim_cfg: SimulationConfig,
                                     grasp_objective_cfg: GraspObjective):

    base_simulation = create_base_sim(contoller_cls, sim_cfg)
    event_contact, event_timeout, event_flying_apart, event_slipout, event_stop_external_force = add_grasp_events_from_cfg(
        base_simulation, grasp_objective_cfg)
    sim_list = create_sim_list(base_simulation, grasp_objective_cfg.object_list)

    return sim_list


def create_rewarder(grasp_objective_cfg: GraspObjective):
    simulation_rewarder = SimulationReward(verbosity=0)

    simulation_rewarder.add_criterion(TimeCriterion(hp.GRASP_TIME, event_timeout, event_grasp),
                                      hp.TIME_CRITERION_WEIGHT)

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
