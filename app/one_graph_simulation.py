import tendon_graph_evaluators

from rostok.library.obj_grasp.objects import get_object_sphere
from rostok.library.rule_sets.simple_designs import (
    get_three_link_one_finger,
    get_three_link_one_finger_independent,
    get_two_link_three_finger,
)
if __name__ == "__main__":
    # create blueprint for object to grasp
    grasp_object_blueprint = get_object_sphere(0.05, mass=0.2)

    control_optimizer = tendon_graph_evaluators.evaluator_tendon_standart_parallel

    graph = get_three_link_one_finger_independent()
    graph = graph = get_two_link_three_finger()

    rewsss = control_optimizer.calculate_reward(graph)
 
    first_object = control_optimizer.prepare_reward.reward_one_sim_scenario(
        x=rewsss[1][0], graph=graph, sim=control_optimizer.simulation_scenario[0])
    print(rewsss)