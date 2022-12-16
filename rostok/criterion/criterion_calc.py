from rostok.graph_grammar.nodes_division import *
from scipy.spatial import distance
import numpy as np

def app_v_2_l(final: list, val: list ):
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
    2) Criterion of number of contact surfaces
        Desciption: cont_surf_crit is equal the ratio of median of contact surfaces (during simulation)
                    to the overall potentional number of contact surfaces.
    3) Criterion of mean values of distance between each of fingers
        Desciption: algorithm has to minimize mean values of the distance for each finger.
    4) Criterion of simulation time
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

    [b_nodes_sim, j_nodes_sim, lb_nodes_sim, rb_nodes_sim]  = traj_to_list(b_nodes, j_nodes, lb_nodes, rb_nodes, sim_output)

    #1) Force criterion
    if np.size(sim_output[-1].obj_contact_forces) == 0 or np.median(sim_output[-1].obj_amount_surf_forces)<6:
        force_crit = 0
    else:
        force_crit = 1/(1+np.mean(sim_output[-1].obj_contact_forces))
 
    #2) Contact surfaces criterion
    if np.size(b_nodes_sim) > 0:
        cont_surf_crit = np.median(sim_output[-1].obj_amount_surf_forces) / len(b_nodes_sim)
    else:
        cont_surf_crit = 0

    #3) Distances between fingers criterion
    if np.size(rb_nodes_sim) > 0 and np.size(lb_nodes_sim) > 0:
        r_sum_cog = []
        l_sum_cog = []
        step_n = 0
        temp_dist = []
        #While coordinates exist
        while step_n < len(rb_nodes_sim[0][0]['abs_coord_cog']):
            # list, which contains values of euclidean distances between right and left fingers
            euc_dist = []
            # Counting of the right fingers (choose i-th right finger)
            for finger in rb_nodes_sim:
                rb_temp_pos = [0, 0, 0]
                # Counting of the body blocks of the right finger
                for body in finger:
                    #step_n-th value of COG coord in [XYZ] format for j-th block of i-th right finger
                    temp_xyz = [body['abs_coord_cog'][step_n][0],
                                body['abs_coord_cog'][step_n][1],
                                body['abs_coord_cog'][step_n][2]]
                    #Element - by - element addition. Right i-th finger's summ of coordinates
                    rb_temp_pos = list(map(sum, zip(rb_temp_pos,temp_xyz)))
                #COG value of i-th right finger
                r_sum_cog.append([x/len(finger) for x in rb_temp_pos])
            
            # Counting of the left fingers (choose i-th left finger)
            for finger in lb_nodes_sim:
                lb_temp_pos = [0, 0, 0]
                # Counting of the body blocks of the left finger
                for body in finger:
                    #step_n-th value of COG coord in [XYZ] format for body block of left finger
                    temp_xyz = [body['abs_coord_cog'][step_n][0],
                                body['abs_coord_cog'][step_n][1],
                                body['abs_coord_cog'][step_n][2]]
                    #Element - by - element addition. Left i-th finger's summ of coordinates
                    lb_temp_pos = list(map(sum, zip(lb_temp_pos,temp_xyz)))
                #COG value of i-th left finger
                l_sum_cog.append([x/len(finger) for x in lb_temp_pos])

            #If number of fingers is more than 2 (at least 2 fingers on one side)
            if step_n == 0 and (len(r_sum_cog)*len(l_sum_cog))>1:
                for _ in range (len(r_sum_cog)*len(l_sum_cog)):
                    #If grasp has more than 2 fingers, then temp_dist is list of the lists.
                    #Temp dist has number of list is equal number of distances
                    temp_dist.append([])
            elif step_n == 0 and (len(r_sum_cog)*len(l_sum_cog))==1:
                temp_dist = []
            else:
                pass

            for r_cog_val in r_sum_cog:
                for l_cog_val in l_sum_cog:
                    #Euclidean distance is calculated for each step
                    euc_dist.append(distance.euclidean(r_cog_val,l_cog_val)) 
            #If grasp has more than 2 fingers
            if len(euc_dist)>1:
                #Add a distance value to the corresponding list
                app_v_2_l(temp_dist, euc_dist)
            else:
                temp_dist.extend(euc_dist)

            r_sum_cog = []
            l_sum_cog = []

            #Next iter
            step_n+=1

        #Calculation
        if len(euc_dist)>1:
            mean_euc_dist = []
            for val in temp_dist:
                mean_euc_dist.append(np.mean(val))
            distance_crit = 1/(1+(sum(mean_euc_dist)))
        else:
            distance_crit = 1/(1+(np.mean(temp_dist)))
    else:
        distance_crit = 0
   
    #4) Time criterion
    if np.size(j_nodes_sim) > 0:
        time_crit = j_nodes_sim[0]['time'][-1]/gait
    else:
        time_crit = 0
    
    return -weights[0]*force_crit - weights[1]*cont_surf_crit - weights[2]*distance_crit - weights[3]*time_crit
