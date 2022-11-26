import rostok.virtual_experiment.robot as robot
from itertools import product 

def nodes_division(rob: robot, list_N):
    """Division all nodes by type (body, joint, etc.)
    Args:
        rob (robot): instance of the Robot class 
        list_N (list): desired type of joints
    Returns:
        NODES_NEW(list): list of nodes of the desired type from the robot
    """
    NODES_NEW = []
    partition_dfs = rob.get_dfs_partiton()
    for i in range(len(partition_dfs)):
        for j in range(len(partition_dfs[i])):
            if (partition_dfs[i][j].node in list_N) and partition_dfs[i][j] not in NODES_NEW:
                NODES_NEW.append(partition_dfs[i][j])   
    return NODES_NEW


def sort_left_right(rob: robot, list_S, list_B):
    """Sorting nodes of Body type into right/left (depending on finger's side) and splitting them by fingers 
    Args:
        rob (robot): instance of the robot class 
        list_S (list):side list that contains transition node (left/right)
        list_B (list): list of possible Body nodes
    Raises:
        StopIteration: If finger has "side" node from list_S
    Returns:
        NODES_NEW (list): list that contains lists of nodes of Body type with side from list_S; size 
                          size of list NODES_NEW is equal numbers of fingers from a relevant side
    """
    NODES_NEW = []
    partition_dfs = rob.get_dfs_partiton() 
    for i in range(len(partition_dfs)):
        try:  
            for j in range(len(partition_dfs[i])):
                if (partition_dfs[i][j].node in list_S):
                    raise StopIteration
        except StopIteration:
            continue
        passed_body = []
        for j in range(len(partition_dfs[i])):
               if (partition_dfs[i][j].node in list_B) and j != 0:
                   passed_body.append(partition_dfs[i][j])
        NODES_NEW.append(passed_body)

    return NODES_NEW



def traj_to_list(B_NODES, J_NODES, LB_NODES, RB_NODES, sim_out: dict):
    """Combines simulation results with the corresponding node
    Args:
        B_NODES (list): list that contains nodes of Body type 
        J_NODES (list): list that contains nodes of Joint type 
        LB_NODES (list): list that contains lists of nodes of Body type from left fingers
        RB_NODES (list): list that contains lists of nodes of Body type from right fingers
        sim_out (dict): results of simulation
    Returns:
        B: list of dictionaries that contains required information (id, number of contact surfaces, contact forces) about node of Body type after simulation
        J: list of dictionaries that contains required information (id, angle time-series values) about node of Joint type after simulation
        LB: list of lists that contain dictionaries with required information (id, COG) about node of Body type from left fingers after simulation
        RB: list of lists that contain dictionaries with required information (id, COG) about node of Body type from right fingers after simulation
    """
    b_temp = {}
    j_temp = {}
    lb_temp = {}
    rb_temp = {}
    B = []
    J = []
    LB = []
    RB = []

    for i in range(len(B_NODES)):
        if B_NODES[i].id == sim_out[B_NODES[i].id].id_block:
            b_temp = {'id':B_NODES[i].id, 'amount_contact_surfaces':sim_out[B_NODES[i].id].amount_contact_surfaces, 'sum_contact_forces': sim_out[B_NODES[i].id].sum_contact_forces}
            B.append(b_temp)

    for i in range(len(J_NODES)):
        if J_NODES[i].id == sim_out[J_NODES[i].id].id_block:
            j_temp = {'id':J_NODES[i].id, 'angle_list': sim_out[J_NODES[i].id].angle_list, 'time': sim_out[J_NODES[i].id].time}
            J.append(j_temp)

    for i in range(len(LB_NODES)):
        LB_temp = []
        for j in range(len(LB_NODES[i])):
            if LB_NODES[i][j].id == sim_out[LB_NODES[i][j].id].id_block:
                lb_temp = {'id':LB_NODES[i][j].id, 'abs_coord_cog':sim_out[LB_NODES[i][j].id].abs_coord_COG}
                LB_temp.append(lb_temp)
        LB.append(LB_temp)
        
    for i in range(len(RB_NODES)):
        RB_temp = []
        for j in range(len(RB_NODES[i])):
            if RB_NODES[i][j].id == sim_out[RB_NODES[i][j].id].id_block:
                rb_temp = {'id':RB_NODES[i][j].id, 'abs_coord_cog':sim_out[RB_NODES[i][j].id].abs_coord_COG}
                RB_temp.append(rb_temp)
        RB.append(RB_temp)
        
    return B, J, LB, RB



    