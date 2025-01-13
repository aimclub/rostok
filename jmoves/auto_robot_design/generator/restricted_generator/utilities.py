import numpy as np

def set_circle_points(pos_1, pos_2, add_pos, n):
    center = (pos_1+pos_2)/2
    vec = pos_1-center
    if np.linalg.norm(add_pos-center) > np.linalg.norm(pos_1-center):
        pos_turn = center + np.array([vec[0]*np.cos(np.pi/n)-vec[2]*np.sin(
            np.pi/n), 0, vec[2]*np.cos(np.pi/n)+vec[0]*np.sin(np.pi/n)])
        neg_turn = center + np.array([vec[0]*np.cos(-np.pi/n)-vec[2]*np.sin(-np.pi/n),
                                     0, vec[2]*np.cos(-np.pi/n)+vec[0]*np.sin(-np.pi/n)])
        new_pos_list = []
        crit = int((-0.5+int(np.linalg.norm(pos_turn-add_pos)
                   < np.linalg.norm(neg_turn-add_pos)))*2)
        for i in range(crit*1, crit * n, crit):
            angle = i*np.pi/n
            new_pos_list.append(center + np.array([vec[0]*np.cos(angle)-vec[2]*np.sin(
                angle), 0, vec[2]*np.cos(angle)+vec[0]*np.sin(angle)]))
    else:
        new_pos_list = []
        for i in range(1, n):
            new_pos_list.append(pos_1 + (pos_2-pos_1)/n*i)
    return new_pos_list