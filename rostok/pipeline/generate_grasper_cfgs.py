from copy import deepcopy
from dataclasses import dataclass, field
from typing import Type

from pyparsing import Any
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.control_chrono.controller import ConstController, RobotControllerChrono
from rostok.criterion.criterion_calculation import FinalPositionCriterion, GraspTimeCriterion, InstantContactingLinkCriterion, InstantForceCriterion, InstantObjectCOGCriterion, SimulationReward, TimeCriterion
from rostok.criterion.simulation_flags import EventContactBuilder, EventContactTimeOutBuilder, EventFlyingApartBuilder, EventSlipOutBuilder, \
EventGraspBuilder, EventStopExternalForceBuilder
from rostok.simulation_chrono.simulation_scenario import GraspScenario, ParametrizedSimulation
import rostok.control_chrono.external_force as f_ext
from rostok.trajectory_optimizer.control_optimizer import BasePrepareOptiVar, BruteForceOptimisation1D, ConstTorqueOptiVar, GlobalOptimisationEachSim
from scipy.optimize import direct


@dataclass
class SimulationConfig:
    time_step: float = 0.0001
    time_simulation: float = 10
    scenario_cls: Type[GraspScenario] = GraspScenario
    obj_disturbance_forces: list[f_ext.ABCForceCalculator] = field(default_factory=list)


@dataclass
class GraspObjective:
    object_list: list[EnvironmentBodyBlueprint] = field(default_factory=list)
    weight_list: list[float] = field(default_factory=list)
    event_time_no_contact_param: float = 2
    event_flying_apart_time_param: float = 2
    event_slipout_time_param: float = 1
    event_grasp_time_param: float = 5
    event_force_test_time_param: float = 5

    time_criterion_weight: float = 1
    instant_contact_link_criterion_weight: float = 1
    instant_force_criterion_weight: float = 1
    instant_cog_criterion_weight: float = 1
    grasp_time_criterion_weight: float = 1
    final_pos_criterion_weight: float = 1

    refernece_distance: float = 0.3


@dataclass
class BruteForceRewardCfg():
    variants: list[float] = field(default_factory=list)
    num_cpu_workers = 1
    timeout_parallel: float = 20
    chunksize = 'auto'


@dataclass
class GlobalOptimisationRewardCfg():
    optimisation_tool: Any = direct
    args_for_optimiser = None
    bound: tuple[float, float] = (0, 1)


@dataclass
class MCTSCfg():
    C: int = 5
    full_loop: int = 10
    base_iteration: int = 10
    max_number_rules: int = 13


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
    event_contact = EventContactBuilder()
    sim.add_event_builder(event_contact)
    event_timeout = EventContactTimeOutBuilder(cfg.event_time_no_contact_param, event_contact)
    sim.add_event_builder(event_timeout)
    event_flying_apart = EventFlyingApartBuilder(cfg.event_flying_apart_time_param)
    sim.add_event_builder(event_flying_apart)
    event_slipout = EventSlipOutBuilder(cfg.event_slipout_time_param)
    sim.add_event_builder(event_slipout)
    event_grasp = EventGraspBuilder(grasp_limit_time=cfg.event_grasp_time_param,
                                    event_contact_builder=event_contact,
                                    verbosity=0,
                                    simulation_stop=False)
    sim.add_event_builder(event_grasp)
    event_stop_external_force = EventStopExternalForceBuilder(
        event_grasp_builder=event_grasp, force_test_time=cfg.event_force_test_time_param)
    sim.add_event_builder(event_stop_external_force)
    return event_contact, event_timeout, event_flying_apart, event_slipout, event_stop_external_force, event_grasp


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
    # event_contact, event_timeout, event_flying_apart, event_slipout, event_stop_external_force, event_grasp = add_grasp_events_from_cfg(
    #     base_simulation, grasp_objective_cfg)
    sim_list = create_sim_list(base_simulation, grasp_objective_cfg.object_list)

    return sim_list


def create_rewarder(grasp_objective_cfg: GraspObjective, sim_cfg: SimulationConfig, event_timeout,
                    event_grasp, event_slipout):
    simulation_rewarder = SimulationReward(verbosity=0)

    simulation_rewarder.add_criterion(
        TimeCriterion(grasp_objective_cfg.event_grasp_time_param, event_timeout, event_grasp),
        grasp_objective_cfg.time_criterion_weight)

    simulation_rewarder.add_criterion(InstantContactingLinkCriterion(event_grasp),
                                      grasp_objective_cfg.instant_contact_link_criterion_weight)
    simulation_rewarder.add_criterion(InstantForceCriterion(event_grasp),
                                      grasp_objective_cfg.instant_contact_link_criterion_weight)
    simulation_rewarder.add_criterion(InstantObjectCOGCriterion(event_grasp),
                                      grasp_objective_cfg.instant_cog_criterion_weight)
    n_steps = int(grasp_objective_cfg.event_grasp_time_param / sim_cfg.time_step)

    simulation_rewarder.add_criterion(GraspTimeCriterion(event_grasp, n_steps),
                                      grasp_objective_cfg.grasp_time_criterion_weight)
    simulation_rewarder.add_criterion(
        FinalPositionCriterion(grasp_objective_cfg.refernece_distance, event_grasp, event_slipout),
        grasp_objective_cfg.final_pos_criterion_weight)

    return simulation_rewarder


def create_global_optimisation(sim_list: list[GraspScenario], preapare_reward: BasePrepareOptiVar,
                               glop_cfg: GlobalOptimisationRewardCfg):
    rew = GlobalOptimisationEachSim(sim_list, preapare_reward, glop_cfg.bound,
                                    glop_cfg.args_for_optimiser, glop_cfg.optimisation_tool)

    return rew


def create_bruteforce_optimisation(sim_list: list[GraspScenario],
                                   preapare_reward: BasePrepareOptiVar,
                                   grasp_objective: GraspObjective, brute_cfg: BruteForceRewardCfg):
    rew = BruteForceOptimisation1D(brute_cfg.variants, sim_list, preapare_reward,
                                   grasp_objective.weight_list, brute_cfg.num_cpu_workers,
                                   brute_cfg.chunksize, brute_cfg.timeout_parallel)

    return rew


def create_reward_calulator(sim_config: SimulationConfig, grasp_objective: GraspObjective,
                            prepare_reward: BasePrepareOptiVar,
                            optimisation_control_cgf: GlobalOptimisationRewardCfg |
                            BruteForceRewardCfg):

    simlist = prepare_simulation_scenario_list(prepare_reward.control_class, sim_config,
                                               grasp_objective)

    event_timeout = event_grasp = event_slipout = None

    for sim_i in simlist:
        _, event_timeout, _, event_slipout, _, event_grasp = add_grasp_events_from_cfg(
            sim_i, grasp_objective)

    rewarder = create_rewarder(grasp_objective, sim_config, event_timeout, event_grasp,
                               event_slipout)

    prepare_reward.set_reward_fun(rewarder)

    if isinstance(optimisation_control_cgf, GlobalOptimisationRewardCfg):
        return create_global_optimisation(simlist, prepare_reward, optimisation_control_cgf)
    elif isinstance(optimisation_control_cgf, BruteForceRewardCfg):
        return create_bruteforce_optimisation(simlist, prepare_reward, grasp_objective,
                                              optimisation_control_cgf)
    else:
        raise Exception("Wrong type of optimisation_control_cgf")
