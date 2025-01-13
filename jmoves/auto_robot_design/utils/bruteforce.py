import numpy as np


def get_n_dim_linspace(upper_bounds, lower_bounds, point_num = 5):
    ranges = np.array([lower_bounds, upper_bounds]).T

    linspaces = [np.linspace(start, stop, point_num)
                 for start, stop in ranges]
    meshgrids = np.meshgrid(*linspaces)
    vec = np.array([dim_i.flatten() for dim_i in meshgrids]).T
    return vec