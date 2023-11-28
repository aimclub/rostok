from rostok.control_chrono.tendon_controller import TendonControllerParameters
from rostok.library.obj_grasp.objects import get_object_box, get_object_ellipsoid, get_object_parametrized_cuboctahedron, get_object_parametrized_dipyramid_3, get_object_parametrized_trapezohedron
from rostok.pipeline.generate_grasper_cfgs import MCTSCfg, SimulationConfig, GraspObjective, BruteForceRewardCfg
from rostok.simulation_chrono.simulation_scenario import GraspScenario
import rostok.control_chrono.external_force as f_ext
from rostok.trajectory_optimizer.control_optimizer import TendonForceOptiVar
from rostok.utils.numeric_utils import Offset


HARD_OBJECT_SET = [
    get_object_parametrized_cuboctahedron(0.05, mass=0.100),
    get_object_parametrized_dipyramid_3(0.05, mass=0.200)
]
HARD_OBJECT_SET_W = [1.0, 1.0]


NORMAL_OBJECT_SET = [
    get_object_box(0.14, 0.19, 0.28, 0, mass = 0.268),
    get_object_parametrized_trapezohedron(0.15, mass = 0.2),
    get_object_ellipsoid(0.14, 0.14, 0.22, 0, mass=0.188)
]
NORMAL_OBJECT_SET_W = [1.0, 1.0, 1.0]


def get_random_force_with_null_grav(amp):
    obj_forces = []
    obj_forces.append(f_ext.NullGravity(0))
    obj_forces.append(f_ext.RandomForces(amp, 100, 0))
    obj_forces = f_ext.ExternalForces(obj_forces)
    return obj_forces


def get_default_tendon_params():
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
    return data


rand_null_force = get_random_force_with_null_grav(10)
default_simulation_config = SimulationConfig(0.0005, 4.5, GraspScenario, rand_null_force)
fast_mock_simulation_config = SimulationConfig(0.001, 0.1, GraspScenario, rand_null_force)

default_grasp_objective = GraspObjective(
    # Objects setup
    NORMAL_OBJECT_SET,
    NORMAL_OBJECT_SET_W,
    # Event setup
    event_time_no_contact_param=0.5,
    event_flying_apart_time_param=10,
    event_slipout_time_param=0.4,
    event_grasp_time_param=1.5,
    event_force_test_time_param=3,
    # Weight setup
    time_criterion_weight=3,
    final_pos_criterion_weight=5,
    # Object pos setup
    refernece_distance=0.3)

brute_force_opti_default_cfg = BruteForceRewardCfg([10, 15, 20])

hyperparams_mcts_default = MCTSCfg()
hyperparams_mcts_default_max_r15 = MCTSCfg(max_number_rules = 15)