from ast import Break
from time import sleep
import sys


from engine.node  import BlockWrapper, Node, Rule, GraphGrammar
from utils.blocks_utils import make_collide, CollisionGroup   
from engine.node_render import *
from utils.nodes_division import *
import engine.robot as robot
import engine.control as ctrl

from pychrono import ChCoordsysD, ChVectorD, ChQuaternionD
from pychrono import Q_ROTATE_Z_TO_Y, Q_ROTATE_Z_TO_X, \
    Q_ROTATE_Y_TO_X, Q_ROTATE_Y_TO_Z, \
    Q_ROTATE_X_TO_Y, Q_ROTATE_X_TO_Z
import pychrono as chrono
from scipy.spatial import distance
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def appV2L(final: list, val: list ):

   
    """
    Function to append value of the list "val" to list "final"
    Args:
        final (list): list of the lists
        val (list): list of the values

    Returns:
        final (list)
    """
    myit = iter(val)
    for i in range(len(final)):
        it = next(myit)
        final[i].append(it)
    return final

def criterion_calc(sim_output, B, J, LB, RB, W, gait) -> float:


    [B_NODES_NEW, J_NODES_NEW, LB_NODES_NEW, RB_NODES_NEW]  = traj_to_list(B, J, LB, RB, sim_output)
    """
    Function that calculates reward for grasp device. It has four (f1-f4) criterions. All of them should be maximized.
    1) Criterion of isotropy of contact forces
        Desciption: list cont contains mean values of contact force for each body element. If sum of cont is equal zero (no body is in contact), then criterion f1 = 0. 
        Otherwise, delta_u (standart deviation of contact forces) is calculated. We want to minimize deviation.
    2) Criterion of number of contact surfaces
        Desciption: f2 is equal the ratio of mean value of contact surfaces (during simulation) to the overall potentional number of contact surfaces.
    3) Criterion of mean values of distance between each of fingers
        Desciption: algorithm has to minimize mean values of the distance for each finger.
    4) Criterion of simulation time
        Desciption: algorithm has to maximize simulation time

    Args:
        sim (SimulationStepOptimization): instance of class
        sim_output (dict): simulation results
        W (list): list of weight coefficients 
        gait (float): time value of grasping's gait period

    Returns:
        reward (float): Reward for grasping device
    """
    
    cont = []


    #f1
    for i in range(len(B_NODES_NEW)):
        if sum(B_NODES_NEW[i]['sum_contact_forces'])>0:
            cont.append(np.mean(B_NODES_NEW[i]['sum_contact_forces'])) #All mean values of contact forces (for each body) 
    if sum(cont) == 0 or len(cont) < 2:
        f1 = 0
    else:
        delta_u = np.std(cont)
        f1 = 1/(1+delta_u)
    

    #f2
    if len(B_NODES_NEW) > 0:
        res = [0] * len(B_NODES_NEW[0]['amount_contact_surfaces']) 
        for i in range(len(B_NODES_NEW)):
            res = list(map(sum, zip(B_NODES_NEW[i]['amount_contact_surfaces'],res))) #Element - by - element addition
        f2 = np.mean(res) / len(B_NODES_NEW)
    else: 
        f2 = 0

    # Надо максимизировать
    if len(RB_NODES_NEW) > 0 and len(LB_NODES_NEW) > 0:
        Rsum_cog_coord = []
        Lsum_cog_coord = []
        z = 0  
        temp_dist = []  
        while z < len(RB_NODES_NEW[0][0]['abs_coord_cog']): #While coordinates exist
            euc_dist = [] # list, which contains values of euclidean distances between right and left fingers

            for i in range(len(RB_NODES_NEW)): # Counting of the right fingers (choose i-th right finger)
                RB_temp_pos = [0, 0, 0]
                for j in range(len(RB_NODES_NEW[i])): # Counting of the blocks of i-th right finger
                    temp_XYZ = [RB_NODES_NEW[i][j]['abs_coord_cog'][z][0],
                                RB_NODES_NEW[i][j]['abs_coord_cog'][z][1],
                                RB_NODES_NEW[i][j]['abs_coord_cog'][z][2]] #z-th value of COG coordinates in [XYZ] format for j-th block of i-th right finger
                    RB_temp_pos = list(map(sum, zip(RB_temp_pos,temp_XYZ))) #Element - by - element addition. Right i-th finger's summ of coordinates
                Rsum_cog_coord.append([x/len(RB_NODES_NEW[i]) for x in RB_temp_pos]) #COG value of i-th right finger

            for i in range(len(LB_NODES_NEW)): # Counting of the left fingers (choose i-th left finger)
                LB_temp_pos = [0, 0, 0]
                for j in range(len(LB_NODES_NEW[i])): # Counting of the blocks of i-th left finger
                    temp_XYZ = [LB_NODES_NEW[i][j]['abs_coord_cog'][z][0],
                                LB_NODES_NEW[i][j]['abs_coord_cog'][z][1],
                                LB_NODES_NEW[i][j]['abs_coord_cog'][z][2]] #z-th value of COG coordinates in [XYZ] format for j-th block of i-th left finger
                    LB_temp_pos = list(map(sum, zip(LB_temp_pos,temp_XYZ))) #Element - by - element addition. Left i-th finger's summ of coordinates
                Lsum_cog_coord.append([x/len(LB_NODES_NEW[i]) for x in LB_temp_pos]) #COG value of i-th left finger


            if z == 0 and (len(Rsum_cog_coord)*len(Lsum_cog_coord))>1: #If number of fingers is more than 2 (at least 2 fingers on one side)
              for i in range (len(Rsum_cog_coord)*len(Lsum_cog_coord)):
                  temp_dist.append([])                                 #If grasp has more than 2 fingers, then temp_dist is list of the lists. Temp dist has number of list is equal number of distances
            elif z == 0 and (len(Rsum_cog_coord)*len(Lsum_cog_coord))==1:
                temp_dist = []
            else: 
                pass 


            for i in range(len(Rsum_cog_coord)): 
                for j in range(len(Lsum_cog_coord)): 
                    euc_dist.append(distance.euclidean(Rsum_cog_coord[i],Lsum_cog_coord[j])) #Euclidean distance is calculated for each step 

            if len(euc_dist)>1: #If grasp has more than 2 fingers
                appV2L(temp_dist, euc_dist) #Add a distance value to the corresponding list
            else: 
                temp_dist.extend(euc_dist)

            Rsum_cog_coord = []
            Lsum_cog_coord = [] 

            z+=1 #Next iter

        #Calculation
        if len(euc_dist)>1:
            q3 = []
            for i in range(len(temp_dist)):
                q3.append(np.mean(temp_dist[i])) 
            f3 = 1/(1+(sum(q3)))
        else:
            f3 = 1/(1+(sum(np.mean(temp_dist))))
    else:
        f3 = 0
   
    #f4
    if len(J_NODES_NEW) > 0:
        f4 = J_NODES_NEW[0]['time'][-1]/gait
    else: 
        f4 = 0

    
    return -W[0]*f1 - W[1]*f2 - W[2]*f3 - W[3]*f4
    
 
    
    


    
