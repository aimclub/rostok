from rostok.graph_grammar.nodes_division import *
from scipy.spatial import distance
import numpy as np


def app_v_2_l(final: list, val: list):
    """
    Function to append value of the list "val" to list "final"
    Args:
        final (list): list of the lists
        val (list): list of the values

    Returns:
        final (list)
    """
    myit = iter(val)

    for _, res in enumerate(final):
        it_n = next(myit)
        res.append(it_n)
    return res


def criterion_calc(sim_output, b_nodes, j_nodes, lb_nodes, rb_nodes, weights, gait) -> float:
    """
    Function that calculates reward for grasp device. It has four riterions. All of them should be maximized.
    1) Criterion of isotropy of contact forces, that affects on object
        Desciption: obj_contact_forces contains values of standart deviation of contact forces for each contact point of object.
                    If there are no contacts for object or median of contact surfaces is less than 6, criterion is equal 0.
        Otherwise, mean value of standart deviation of contact forces is calculated. We want to minimize this.
    2) Criterion of simulation time
        Desciption: algorithm has to maximize simulation time

    Args:
        sim_output (dict): simulation results
        b_nodes (list): list that contains nodes of Body type
        j_nodes (list): list that contains nodes of Joint type
        lb_nodes (list): list that contains lists of nodes of Body type from left fingers
        rb_nodes (list): list that contains lists of nodes of Body type from right fingers
        weights (list): list of weight coefficients

        gait (float): time value of grasping's gait period

    Returns:
        reward (float): Reward for grasping device
    """

    [_, j_nodes_sim, _,
     _] = traj_to_list(b_nodes, j_nodes, lb_nodes, rb_nodes, sim_output)

    #1) Force criterion
    if np.size(sim_output[-1].obj_contact_forces) == 0 or np.median(
            sim_output[-1].obj_amount_surf_forces) < 4:
        force_crit = 0
    else:
        force_crit = 1 / (1 + np.mean(sim_output[-1].obj_contact_forces))

    #2) Time criterion
    if np.size(j_nodes_sim) > 0:
        time_crit = np.square(j_nodes_sim[0]['time'][-1])
    else:
        time_crit = 0
        

    # 3) Obj COG coordinats
    dist_list = []
    if np.size(sim_output[-1].obj_cont_coord) > 0:
       for idx,_ in enumerate(sim_output[-1].obj_cont_coord):
            dist_list.append(distance.euclidean(sim_output[-1].obj_cont_coord[idx], sim_output[-1].obj_COG[idx]))
    if np.size(dist_list) > 0:
        cog_crit = 1/(1+np.mean(dist_list))
    else:
        cog_crit = 0


    # if np.size(sim_output[-1].obj_contact_forces) == 0:
    #     return 0
    if force_crit == 0:
        return 0.5*(-weights[0] * force_crit - weights[1] * time_crit- weights[2] * cog_crit)
    else:
        return -weights[0] * force_crit - weights[1] * time_crit- weights[2] * cog_crit
    # return -weights[0] * force_crit - weights[1] * time_crit- weights[2] * cog_crit
