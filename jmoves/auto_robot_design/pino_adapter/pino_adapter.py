def get_pino_description(ative_joints, constraints):

    joint_mot_description = {
        "name_mot": ative_joints,
        "joint_name": [],
        "joint_type": [],
    }
    loop_description = {
        "closed_loop": constraints,
        "type": ["6d"] * len(constraints)
    }
    return joint_mot_description, loop_description

def get_pino_description_3d_constraints(ative_joints, constraints):

    joint_mot_description = {
        "name_mot": ative_joints,
        "joint_name": [],
        "joint_type": [],
    }
    loop_description = {
        "closed_loop": constraints,
        "type": ["3d"] * len(constraints)
    }
    return joint_mot_description, loop_description
