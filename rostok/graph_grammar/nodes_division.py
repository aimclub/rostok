from rostok.virtual_experiment.robot import Robot


def nodes_division(rob: Robot, type_node: list):
    """Division all nodes by type (body, joint, etc.)
    Args:
        rob (robot): instance of the Robot class
        type_node (list): desired type of joints
    Returns:
        nodes(list): list of nodes of the desired type from the robot
    """
    nodes = []
    partition_dfs = rob.get_dfs_partiton()

    for part in partition_dfs:
        for node in part:
            if (node.node in type_node) and node not in nodes:
                nodes.append(node)
    return nodes


def sort_left_right(rob: Robot, side_list: list, body_list: list):
    """Sorting nodes of Body type into right/left (depending on finger's side) and splitting them by fingers 
    Args:
        rob (robot): instance of the Robot class
        side_list (list):side list that contains transition node (left/right)
        body_list (list): list of possible Body nodes
    Raises:
        StopIteration: If finger has "side" node from side_list
    Returns:
        nodes (list): list that contains lists of nodes of Body type with side from side_list; size 
                          size of list nodes is equal numbers of fingers from a relevant side
    """
    nodes = []
    partition_dfs = rob.get_dfs_partiton()
    for part in partition_dfs:
        try:
            for node in part:
                if node.node in side_list:
                    raise StopIteration
        except StopIteration:
            continue
        passed_body = []

        for idx, node in enumerate(part):
            if (node.node in body_list) and idx != 0:
                passed_body.append(node)
        nodes.append(passed_body)
    return nodes


def traj_to_list(b_nodes, j_nodes, lb_nodes, rb_nodes, sim_out: dict):
    """Combines simulation results with the corresponding node
    Args:
        b_nodes (list): list that contains nodes of Body type
        j_nodes (list): list that contains nodes of Joint type
        lb_nodes (list): list that contains lists of nodes of Body type from left fingers
        rb_nodes (list): list that contains lists of nodes of Body type from right fingers
        sim_out (dict): results of simulation
    Returns:
        list_b: list of dictionaries that contains required information (id, number of contact surfaces, contact forces)
                about node of Body type after simulation
        list_j: list of dictionaries that contains required information (id, angle time-series values)
                about node of Joint type after simulation
        list_lb: list of lists that contain dictionaries with required information (id, COG)
                 about node of Body type from left fingers after simulation
        list_rb: list of lists that contain dictionaries with required information (id, COG)
                 about node of Body type from right fingers after simulation
    """
    dict_b_temp = {}
    dict_j_temp = {}
    dict_lb_temp = {}
    dict_rb_temp = {}
    list_b = []
    list_j = []
    list_lb = []
    list_rb = []

    for b_node in b_nodes:
        if b_node.id == sim_out[b_node.id].id_block:
            dict_b_temp = {
                'id': b_node.id,
                'amount_contact_surfaces': sim_out[b_node.id].amount_contact_surfaces,
                'sum_contact_forces': sim_out[b_node.id].sum_contact_forces
            }
            list_b.append(dict_b_temp)

    for j_node in j_nodes:
        if j_node.id == sim_out[j_node.id].id_block:
            dict_j_temp = {
                'id': j_node.id,
                'angle_list': sim_out[j_node.id].angle_list,
                'time': sim_out[j_node.id].time
            }
            list_j.append(dict_j_temp)

    for left_bodies in lb_nodes:
        list_dict = []
        for body in left_bodies:
            if body.id == sim_out[body.id].id_block:
                dict_lb_temp = {'id': body.id, 'abs_coord_cog': sim_out[body.id].abs_coord_COG}
                list_dict.append(dict_lb_temp)
        list_lb.append(list_dict)

    for right_bodies in rb_nodes:
        list_dict = []
        for body in right_bodies:
            if body.id == sim_out[body.id].id_block:
                dict_rb_temp = {'id': body.id, 'abs_coord_cog': sim_out[body.id].abs_coord_COG}
                list_dict.append(dict_rb_temp)
        list_rb.append(list_dict)

    return list_b, list_j, list_lb, list_rb
