from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.control_chrono.controller import ConstController
from rostok.control_chrono.tendon_controller import TendonControllerParameters, TendonController_2p
from rostok.criterion.criterion_calculation import SimulationReward, TimeCriterion
from rostok.graph_grammar.graphgrammar_explorer import random_search_mechs_n_branch
from rostok.library.rule_sets.simple_designs import get_three_link_one_finger, get_two_link_three_finger
from rostok.simulation_chrono.simulation_scenario import GraspScenario
from rostok.trajectory_optimizer.control_optimizer import ConstTorqueOptiVar, GlobalOptimisationEachSim, MockOptiVar, BruteForceOptimisation1D, TendonForceOptiVar
from rostok.library.obj_grasp import objects
from rostok.utils.numeric_utils import Offset
from rostok.library.rule_sets.rulset_simple_fingers import create_rules


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


if __name__ == '__main__':
    VIS = False
    tendon_controller_cfg = get_tendon_cfg()

    gpesk_tendon1 = GraspScenario(0.002, 1, TendonController_2p)
    gpesk_tendon1.grasp_object_callback = objects.get_object_box(0.1, 0.1, 0.1, 0)

    gpesk_tendon2 = GraspScenario(0.002, 1, TendonController_2p)
    gpesk_tendon2.grasp_object_callback = objects.get_object_box(0.1, 0.1, 0.1, 1)

    gpesk_const1 = GraspScenario(0.002, 1, ConstController)
    gpesk_const1.grasp_object_callback = objects.get_object_box(0.1, 0.1, 0.1, 0)

    gpesk_const2 = GraspScenario(0.002, 1, ConstController)
    gpesk_const2.grasp_object_callback = objects.get_object_box(0.1, 0.1, 0.1, 1)

    simulation_rewarder = SimulationReward(verbosity=0)

    graph = get_two_link_three_finger()
    graph_finger = get_three_link_one_finger()

    tendon_optivar = TendonForceOptiVar(tendon_controller_cfg, simulation_rewarder, -45)
    tendon_optivar.is_vis = VIS

    const_optivar = ConstTorqueOptiVar(simulation_rewarder, -45)
    const_optivar.is_vis = VIS
    direct_args = {"maxiter": 2}

    global_const = GlobalOptimisationEachSim([gpesk_const1, gpesk_const2], const_optivar, (0, 10),
                                             direct_args)
    res = global_const.calculate_reward(graph_finger)

    brute_const = BruteForceOptimisation1D([0, 10], [gpesk_const1],
                                           const_optivar,
                                           num_cpu_workers=1)
    res = brute_const.calculate_reward(graph_finger)

    brute_tendon = BruteForceOptimisation1D([0, 10], [gpesk_tendon1, gpesk_tendon2],
                                            tendon_optivar,
                                            num_cpu_workers=1)
    res = brute_tendon.calculate_reward(graph)

    global_opti_tendon = GlobalOptimisationEachSim([gpesk_tendon1, gpesk_tendon2], tendon_optivar,
                                                   (0, 10), direct_args)
    res = global_opti_tendon.calculate_reward(graph)

    pass