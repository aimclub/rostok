from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line

def power_quality(time: np.ndarray, power: np.ndarray, plot=False):
    """
    Evaluate the power quality of the robot
    Args:
        time (np.ndarray): time (s)
        power (np.ndarray): power consumption (W)
        plot (bool): power plot in power space and power over time
    Returns:
        float: power quality
    """
    
    PQ = np.zeros((power.shape[0], 1))
    
    for i in range(power.shape[0]):
        PQ[i] = np.sum(power[i])**2 - np.sum(power[i]**2)
        
    if plot:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(time, power[:, 0], label='P_1', linewidth=2)
        plt.plot(time, power[:, 1], label='P_2')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(time, PQ)
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Power Quality')
        plt.grid()
        plt.figure()

        plt.plot(power[:, 0], power[:, 1])
        plt.xlabel('P_1')
        plt.ylabel('P_2')
        plt.grid()
        plt.axis('equal')
        plt.show()
        
    return np.mean(PQ)

def movments_in_xz_plane(time: np.ndarray, x: np.ndarray, des_x: np.ndarray, plot=False):
    """
    Evaluate the movements in the xz plane
    Args:
        time (np.ndarray): time (s)
        x (np.ndarray): posiotion from simulation
        des_x (np.ndarray): desired position
        plot (bool): plot the movements
    Returns:
        float: error tracking trajectory in the xz plane
    """
    
    error = np.zeros((x.shape[0], 1))
    
    for i in range(x.shape[0]):
        error[i] = np.linalg.norm((x[i] - des_x[i]))
        
    if plot:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(time, des_x[:, 0], label='des', linestyle='--', linewidth=3)
        plt.plot(time, x[:, 0], label='real')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('X (m)')
        plt.grid()
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(time, x[:, 2], label='des', linestyle='--', linewidth=3)
        plt.plot(time, x[:, 2], label='real')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Z (m)')
        plt.grid()
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(time, error)
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.grid()
        plt.figure()
        plt.plot(des_x[:, 0], des_x[:, 2], label='des', linestyle='--', linewidth=3)
        plt.plot(x[:, 0], x[:, 2], label="real")
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        plt.show()
        
    return np.mean(error)

def torque_evaluation(time: np.ndarray, torque: np.ndarray, plot = False):
    """
    Evaluate the torque
    Args:
        time (np.ndarray): time (s)
        torque (np.ndarray): torque
        plot (bool): plot the torque
    Returns:
        float: torque evaluation
    """
    if plot:
        plt.figure()
        for i in range(torque.shape[1]):
            plt.plot(time, torque[:, i], label='tau_' + str(i))
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.grid()
        plt.legend()
        plt.show()
    
    return np.max(np.abs(torque), axis=0)