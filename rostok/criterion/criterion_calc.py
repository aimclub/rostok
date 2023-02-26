from scipy.spatial import distance
import numpy as np


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
    list_surfaces_1 = []
    if np.size(sim_output[-1].obj_amount_surf_forces) > 0:
        greater_zero = np.where(np.array(sim_output[-1].obj_amount_surf_forces) > 0)[0]
        for ind in greater_zero:
            list_surfaces_1.append(sim_output[-1].obj_amount_surf_forces[ind])

    if np.size(sim_output[-1].obj_contact_forces) == 0 or np.median(
            list_surfaces_1) < 4:
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
       for idx,_ in enumerate(sim_output[-1].obj_cont_coord):
            dist_list.append(distance.euclidean(sim_output[-1].obj_cont_coord[idx], sim_output[-1].obj_COG[idx]))
    if np.size(dist_list) > 0:
        cog_crit = 1/(1+np.mean(dist_list))
    else:
        cog_crit = 0

    # 4) Time of force crit
    
    CUTOFF = 0.5
    THR_FORCE = 5
    """Crit belongs (0, 1)
    The percentage of time after the cutoff 
    when the force acting on the body is greater than the threshold.

    """
    obj_forces = sim_output[-1].obj_forces
    if np.size(obj_forces) > 0:
        len_array = np.size(obj_forces)
        checkpoint = int(len_array*CUTOFF)
        sliced_force =  np.array(obj_forces[checkpoint : -1])
        greater_thr_len = np.size(np.where(sliced_force > THR_FORCE)[0])
        if np.size(sliced_force) == 0:
            time_force_thr_crit = 0
        else:    
            time_force_thr_crit = greater_thr_len / np.size(sliced_force)
    else:
        time_force_thr_crit = 0
    
    # 5) Amount contact forces
    """Crit belongs (0, 1)
    The percentage of time after the cutoff 
    when the force acting on the body is greater than the threshold.

    """
    THR_FORCE2 = 0.1
    CUTOFF2 = 0.5
    obj_forces = sim_output[-1].obj_forces
    if np.size(obj_forces) > 0:
        len_array = np.size(obj_forces)
        checkpoint2 = int(len_array*CUTOFF2)
        sliced_force =  np.array(obj_forces[checkpoint2 : -1])
        list_surfaces = []
        greater_thr_ind = np.where(np.array(sliced_force) > THR_FORCE2)[0]
        for ind in greater_thr_ind:
            list_surfaces.append(sim_output[-1].obj_amount_surf_forces[ind])
        
        if len(list_surfaces) == 0:
            midle_contact_crit = 0
        else:
            midle_contact_crit = np.sum(list_surfaces) / len(sliced_force)
    
    reward = -weights[0] * force_crit - weights[1] * time_crit - weights[2] * cog_crit - weights[3] * time_force_thr_crit - weights[4] * midle_contact_crit

    if force_crit == 0:
        return 0.5*(reward)
    else:
        return reward
