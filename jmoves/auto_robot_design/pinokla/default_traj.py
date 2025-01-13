import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple


def convert_x_y_to_6d_traj(x: np.ndarray, y: np.ndarray):
    traj_6d = np.zeros((len(x), 6), dtype=np.float64)
    traj_6d[:, 0] = x
    traj_6d[:, 1] = y
    return traj_6d


def convert_x_y_to_6d_traj_xz(x: np.ndarray, y: np.ndarray):
    traj_6d = np.zeros((len(x), 6), dtype=np.float64)
    traj_6d[:, 0] = x
    traj_6d[:, 2] = y
    return traj_6d


def simple_traj_derivative(traj_6d: np.ndarray, dt: float = 0.001):
    traj_6d_v = np.zeros(traj_6d.shape)
    # (traj_6d[1:, :] - traj_6d[:-1, :])/dt
    traj_6d_v[1:, :] = np.diff(traj_6d, axis=0)/dt
    return traj_6d_v


def get_simple_spline():
    # Sample data points
    x = np.array([-0.5, 0, 0.5])
    y = np.array([-1.02, -0.8, -1.02])
    # y = y - 0.5
    # x = x + 0.4
    # Create the cubic spline interpolator
    cs = CubicSpline(x, y)

    # Create a dense set of points where we evaluate the spline
    x_traj_spline = np.linspace(x.min(), x.max(), 75)
    y_traj_spline = cs(x_traj_spline)

    # Plot the original data points
    # plt.plot(x, y, 'o', label='data points')

    # Plot the spline interpolation
    # plt.plot(x_traj_spline, y_traj_spline, label='cubic spline')

    # plt.legend()
    # plt.show()
    return (x_traj_spline, y_traj_spline)


def create_simple_step_trajectory(starting_point, step_height, step_width, n_points=75):
    x_start = starting_point[0]
    x_end = x_start + step_width
    x = np.array([x_start, (x_start+x_end)/2, x_end])
    y = [starting_point[1], starting_point[1]+step_height, starting_point[1]]
    cs = CubicSpline(x, y)
    x_traj_spline = np.linspace(x.min(), x.max(), n_points)
    y_traj_spline = cs(x_traj_spline)
    return (x_traj_spline, y_traj_spline)


def get_vertical_trajectory(starting_point, height, x_shift, n_points=50):
    x_trajectory = np.zeros(n_points)
    x_trajectory += x_shift
    y_trajectory = np.linspace(starting_point, starting_point+height, n_points)
    return (x_trajectory, y_trajectory)


def get_workspace_trajectory(starting_point, height, width, n_vertical, n_horizontal):
    vertical_step = height/(n_vertical-1)
    horizontal_step = width/(n_horizontal-1)
    current_point = starting_point
    x_list = []
    y_list = []
    x_list.append(current_point[0])
    y_list.append(current_point[1])
    for i in range(n_horizontal):
        for _ in range(n_vertical-1):
            current_point[1] += vertical_step*((0.5-i % 2)*2)
            x_list.append(current_point[0])
            y_list.append(current_point[1])
        current_point[0] += horizontal_step
        x_list.append(current_point[0])
        y_list.append(current_point[1])

    return (np.array(x_list[:-1:]), np.array(y_list[:-1:]))


def get_horizontal_trajectory(starting_point, width, x_shift, n_points=50):
    y_trajectory = np.zeros(n_points)
    y_trajectory = np.linspace(starting_point, starting_point, n_points)
    x_trajectory = np.linspace(x_shift - width/2, x_shift + width/2, n_points)
    return (x_trajectory, y_trajectory)


def add_auxilary_points_to_trajectory(trajectory: Tuple[np.array], initial_point=np.array([0, -0.4]), number_points=50):
    first_point = np.array([trajectory[0][0], trajectory[1][0]])
    vector = first_point-initial_point
    # length = np.linalg.norm(vector)
    multipliers = np.linspace(0, 1, number_points+1, endpoint=False)[1:]
    # multipliers = np.linspace(0,1,number_points,endpoint=False)
    new_x = np.array([initial_point[0]+vector[0]*m for m in multipliers])
    new_y = np.array([initial_point[1]+vector[1]*m for m in multipliers])
    result_x = np.concatenate((new_x, trajectory[0]))
    result_y = np.concatenate((new_y, trajectory[1]))

    return (result_x, result_y)
