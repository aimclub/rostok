import numpy as np
from scipy.spatial import distance


def criterion_calc(sim_output, weights) -> float:
    """
    Function that calculates reward for grasp device. It has four riterions. All of them should be maximized.
    1) Criterion of isotropy of contact forces, that affects on object
        Desciption: obj_contact_forces contains values of standart deviation of contact forces for each contact point of object.
                    If there are no contacts for object or median of contact surfaces is less than 6, criterion is equal 0.
        Otherwise, mean value of standart deviation of contact forces is calculated. We want to minimize this.
    2) Criterion of simulation time
        Desciption: algorithm has to maximize simulation time
    3) Criterion of fixtance between COGs
        Description: algoritm tries to minimize distance between COG of grasp object and centroid of grasp polygon

    Args:
        sim_output (dict): simulation results
        weights (list): list of weight coefficients

    Returns:
        reward (float): Reward for grasping device
    """

    #1) Force criterion
    if np.size(sim_output[-1].obj_contact_forces) == 0 or np.median(
            sim_output[-1].obj_amount_surf_forces) < 4:
        force_crit = 0
    else:
        force_crit = 1 / (1 + np.mean(sim_output[-1].obj_contact_forces))

    #2) Time criterion
    if np.size(sim_output[-1].time) > 0:
        time_crit = np.square(sim_output[-1].time[-1])
    else:
        time_crit = 0

    # 3) Obj COG coordinats
    dist_list = []
    if np.size(sim_output[-1].obj_cont_coord) > 0:
        for idx, _ in enumerate(sim_output[-1].obj_cont_coord):
            dist_list.append(
                distance.euclidean(sim_output[-1].obj_cont_coord[idx], sim_output[-1].obj_COG[idx]))
    if np.size(dist_list) > 0:
        cog_crit = 1 / (1 + np.mean(dist_list))
    else:
        cog_crit = 0

    reward = -weights[0] * force_crit - weights[1] * time_crit - weights[2] * cog_crit

    if force_crit == 0:
        return 0.25 * (reward)
    else:
        return reward
