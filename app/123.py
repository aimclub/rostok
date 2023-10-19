from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.control_chrono.tendon_controller import TendonControllerParameters, TendonController_2p
from rostok.criterion.criterion_calculation import SimulationReward, TimeCriterion
from rostok.library.rule_sets.simple_designs import get_three_link_one_finger, get_two_link_three_finger
from rostok.simulation_chrono.simulation_scenario import GraspScenario
from rostok.trajectory_optimizer.control_optimizer import MockOptiVar, BruteForceOptimisation1D, TendonForceOptiVar
from rostok.library.obj_grasp import objects
from rostok.utils.numeric_utils import Offset


if __name__ == '__main__':
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
    
    gpesk1 = GraspScenario(0.001, 1, TendonController_2p)
    gpesk1.grasp_object_callback = objects.get_object_box(0.1, 0.1, 0.1, 0)

    gpesk2 = GraspScenario(0.001, 1, TendonController_2p)
    gpesk2.grasp_object_callback = objects.get_object_box(0.1, 0.1, 0.1, 1)


    simulation_rewarder = SimulationReward(verbosity=0)
 


    graph = get_three_link_one_finger()
    ttt = TendonForceOptiVar(data, simulation_rewarder, -45)
     
    kukish = BruteForceOptimisation1D([10, 10, 11, 12, 13], [gpesk1, gpesk2], ttt, num_cpu_workers=1)
    sosik = kukish.parallel_calculate_reward(graph)
    pass