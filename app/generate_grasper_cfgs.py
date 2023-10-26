from dataclasses import dataclass, field
from typing import Type
from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.simulation_chrono.simulation_scenario import GraspScenario, ParametrizedSimulation
import rostok.control_chrono.external_force as f_ext


@dataclass
class SimulationConfig:
    time_step: float = 0.0001
    time_simulation: float = 10
    scenario_cls: Type[ParametrizedSimulation] = GraspScenario
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
class ControlOptimizationParams:
    optimisation_bound: tuple = field(default_factory=tuple)
    optimisation_iter: int = field(default_factory=int)

@dataclass
class ControlBruteForceParams:
    force_list: list[float] = field(default_factory=list)
    staring_angle: float = field(default_factory=float)
   
