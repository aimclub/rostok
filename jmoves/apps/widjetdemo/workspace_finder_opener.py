import numpy as np
import matplotlib.pyplot as plt


from auto_robot_design.pinokla.default_traj import get_workspace_trajectory

from auto_robot_design.user_interface.check_in_ellips import Ellipse, check_points_in_ellips

def get_indices_by_point(mask: np.ndarray, reach_array: np.ndarray):
    mask_true_sum = np.sum(mask)
    reachability_sums = reach_array @ mask
    target_indices = np.where(reachability_sums == mask_true_sum)
    return target_indices[0]

data = np.load("test_workspace_BF_RES_0.npz")
reach_arrays = data["reach_array"]
q_arrays = data["q_array"]


mask[55] = False
mask[0] = False
mask[10] = False


# def plot_ellipse(ellipse):
TOPOLGY_NAME = 0
points_x, points_y = get_workspace_trajectory([-0.15, -0.35], 0.2, 0.3, 10, 10)
ellipse = Ellipse(np.array([0, -0.25]), np.deg2rad(30), np.array([0.05, 0.1]))
point_ellipse = ellipse.get_points()


points = np.vstack([points_x.flatten(), points_y.flatten()])
mask = check_points_in_ellips(points, ellipse)
rev_mask = np.array(1-mask, dtype="bool")

 
target_indices = get_indices_by_point(mask, reach_arrays)
 
 
print(len(target_indices))



plt.figure(figsize=(10,10))
plt.plot(point_ellipse[0,:], point_ellipse[1,:], "g", linewidth=3)
plt.scatter(points[:,rev_mask][0],points[:,rev_mask][1])
plt.scatter(points[:,mask][0],points[:,mask][1])
plt.show()
