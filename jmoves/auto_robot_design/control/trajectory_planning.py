import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.linalg as la


def calculate_polynom_coeffs_by_n_points(time_point_arr, v0, vf, alp_f):
    """
    Calculates the polynomial coefficients based on the given time points and boundary conditions.

    Parameters:
    - time_point_arr (numpy.ndarray): Array of time points and corresponding positions.
    - v0 (float): Initial velocity.
    - vf (float): Final velocity.
    - alp_f (float): Final acceleration.

    Returns:
    - numpy.ndarray: Array of polynomial coefficients.

    Raises:
    - Warning: If the system fails to solve for the polynomial coefficients.

    """
    t0 = time_point_arr[0, 0]
    p0 = time_point_arr[0, 1]
    tf = time_point_arr[-1, 0]
    pf = time_point_arr[-1, 1]
    N = time_point_arr.shape[0] + 3
    A = np.array(
        [
            [t0**i for i in range(N)],
            [i * t0 ** np.abs(i - 1) for i in range(N)],
            *[[t**i for i in range(N)] for t in time_point_arr[1:-1, 0]],
            [tf**i for i in range(N)],
            [i * tf ** np.abs(i - 1) for i in range(N)],
            [i * (i - 1) * tf ** np.abs(i - 2) for i in range(N)],
        ]
    )
    rhs = np.array([p0, v0, *time_point_arr[1:-1, 1], pf, vf, alp_f])
    x = la.solve(A, rhs)
    if not np.isclose(A @ x, rhs).all():
        warnings.warn("Failed to solve the system for polynom, try again")
        x = la.solve(A, rhs)
    return x


def trajectory_planning(
    q_via_points, v0, vf, alpf, time_end: float = 1.0, dt: float = 0.1, plot=False
):
    """
    Generate a trajectory plan based on via points using polynomial interpolation.

    Args:
        q_via_points (list): List of via points for each joint. Each element of the list
            is a 1D array representing the via points for a single joint.
        v0 (float): Initial velocity.
        vf (float): Final velocity.
        alpf (float): Acceleration limit per joint.
        time_end (float, optional): End time of the trajectory. Defaults to 1.0.
        dt (float, optional): Time step for evaluating the trajectory. Defaults to 0.1.
        plot (bool, optional): Whether to plot the trajectory. Defaults to False.

    Returns:
        tuple: A tuple containing the time array, joint position trajectory array,
            joint velocity trajectory array, and joint acceleration trajectory array.
    """

    N = len(q_via_points[0]) + 3
    q_traj = lambda t, b: np.sum(np.array([t**i for i in range(N)]).T * b, axis=1)
    dq_traj = lambda t, b: np.sum(
        np.array([i * t ** np.abs(i - 1) for i in range(N)]).T * b, axis=1
    )
    ddq_traj = lambda t, b: np.sum(
        np.array([i * (i - 1) * t ** np.abs(i - 1) for i in range(N)]).T * b, axis=1
    )

    time_array = np.linspace(0, time_end, len(q_via_points[0]))
    polynom_coefs = []
    # Sample data points
    for q in q_via_points:
        assert len(q) == N - 3
        time_point_arr = np.array([time_array, q]).T
        polynom_coefs.append(
            calculate_polynom_coeffs_by_n_points(time_point_arr, v0, vf, alpf)
        )

    # Create a dense set of points where we evaluate the spline
    time_arr = np.arange(0, time_end + dt, dt)
    q_traj_arr = np.zeros((time_arr.size, np.array(q_via_points).shape[0]))
    dq_traj_arr = np.zeros((time_arr.size, np.array(q_via_points).shape[0]))
    ddq_traj_arr = np.zeros((time_arr.size, np.array(q_via_points).shape[0]))

    for i, b in enumerate(polynom_coefs):
        q_traj_arr[:, i] = q_traj(time_arr, b)
        dq_traj_arr[:, i] = dq_traj(time_arr, b)
        ddq_traj_arr[:, i] = ddq_traj(time_arr, b)

    if plot:
        # Plot the spline interpolation
        plt.figure()
        for i in range(np.array(q_via_points).shape[0]):
            plt.subplot(
                np.array(q_via_points).shape[0],
                3,
                3 * i + 1,
            )
            plt.plot(time_arr, q_traj_arr[:, i], label="interpolated")
            plt.plot(time_array, q_via_points[i], "o", label="via points")
            plt.ylabel(f"q{i}")
            plt.legend()
            plt.grid()
            plt.xlim([0, time_end])
            plt.subplot(
                np.array(q_via_points).shape[0],
                3,
                3 * i + 2,
            )
            plt.plot(time_arr, dq_traj_arr[:, i])
            plt.ylabel(f"dq{i}")
            plt.grid()
            plt.xlim([0, time_end])
            plt.subplot(
                np.array(q_via_points).shape[0],
                3,
                3 * i + 3,
            )
            plt.plot(time_arr, ddq_traj_arr[:, i])
            plt.ylabel(f"ddq{i}")
            plt.grid()
            plt.xlim([0, time_end])
        plt.show()
    return (time_arr, q_traj_arr, dq_traj_arr, ddq_traj_arr)


if __name__ == "__main__":
    q_via_points = [
        [0, 0.1,   0.2, 0.3, 0.4],
        [0.5, 0.3, 1, 2, 3],
        [0.5, 0.7, 0.2, 0.1, 0.3],
    ]
    trajectory_planning(q_via_points, 0, 0, 0, 2, dt=0.01, plot=True)
