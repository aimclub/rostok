from typing import Optional
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def rotation_matrix(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


class Ellipse:
    def __init__(self, p_center: np.ndarray, angle: float, axis: np.ndarray) -> None:
        self.p_center: np.ndarray = p_center
        self.angle: float = angle
        self.axis: np.ndarray = axis

    def get_points(self, step=0.1):
        E = np.linalg.inv(np.diag(self.axis) ** 2)
        R = rotation_matrix(-self.angle)
        En = R.T @ E @ R
        t = np.arange(0, 2 * np.pi, step)
        y = np.vstack([np.cos(t), np.sin(t)])
        x = sp.linalg.sqrtm(np.linalg.inv(En)) @ y
        x[0, :] = x[0, :] + self.p_center[0]
        x[1, :] = x[1, :] + self.p_center[1]
        return x

    def fill_area_with_points(self, points, turns=10):
        E = np.linalg.inv(np.diag(self.axis) ** 2)
        R = rotation_matrix(-self.angle)
        En = R.T @ E @ R
        t = np.linspace(0, 2 * turns * np.pi, points)
        y = np.vstack(
            [
                np.cos(t) * np.exp(-t / (turns * np.pi)),
                np.sin(t) * np.exp(-t / (turns * np.pi)),
            ]
        )
        x = sp.linalg.sqrtm(np.linalg.inv(En)) @ y
        x[0, :] = x[0, :] + self.p_center[0]
        x[1, :] = x[1, :] + self.p_center[1]
        return x


class SnakePathFinder:
    def __init__(
        self,
        start_point: np.ndarray,
        ellipse: Ellipse,
        max_len_btw_pts: Optional[float] = None,
        coef_reg: Optional[float] = None,
    ) -> None:
        self.start_point = start_point
        self.ellipse = ellipse
        if max_len_btw_pts is None:
            self.max_len = np.inf
        else:
            self.max_len = max_len_btw_pts
        if coef_reg is None:
            self.coef_ref = 1e-8
        else:
            self.coef_ref = coef_reg

    def _nearest_neighbor(self, points, current, visited):
        next_points = []
        filt_points = np.array(
            list(filter(lambda x: tuple(x.tolist()) not in visited, points))
        )

        if len(filt_points) > 0:
            dist2center = np.linalg.norm(filt_points - self.ellipse.p_center, axis=1)
            distance2current = (
                np.linalg.norm(current - filt_points, axis=1)
                + self.coef_ref * 1 / dist2center
            )
            pts_dist = np.hstack((filt_points, distance2current[:, np.newaxis]))
            next_points = list(sorted(pts_dist, key=lambda x: x[-1]))
            next_points = np.array(next_points)[:, :-1]
        return next_points

    def do_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[0]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _is_valid_move(self, current_point, next_point):
        # check = True
        # for i in range(len(trajectory) -1):
        #     if do_intersect(trajectory[i], trajectory[i+1], current_point, next_point):
        #         check = False
        check = False
        if np.linalg.norm(current_point - next_point) <= self.max_len:
            check = True
        return check

    def create_snake_traj(self, points):

        trajectory = [self.start_point]
        visited = set([tuple(self.start_point.tolist())])

        while len(visited) < len(points):
            curr_p = trajectory[-1]
            nx_pts = self._nearest_neighbor(points, curr_p, visited)

            # if len(nx_pts)>0 and is_valid_move(curr_p, nx_pts[0], trajectory):
            #     trajectory.append(nx_pts[0])
            #     visited.add(tuple(nx_pts[0].tolist()))
            #     print(len(visited),"/", len(points))

            if len(nx_pts) > 0:
                for nx_p in nx_pts:
                    if self._is_valid_move(curr_p, nx_p):
                        trajectory.append(nx_p)
                        visited.add(tuple(nx_p.tolist()))
                        break
            else:
                break
        return np.array(trajectory)


def check_points_in_ellips(points: np.ndarray, ellipse: Ellipse, tolerance=0.2):
    # https://en.wikipedia.org/wiki/Ellipse
    a = ellipse.axis[0] * (1 + tolerance)
    b = ellipse.axis[1] * (1 + tolerance)
    ang = ellipse.angle
    x0 = ellipse.p_center[0]
    y0 = ellipse.p_center[1]

    A = a**2 * np.sin(ang) ** 2 + b**2 * np.cos(ang) ** 2
    B = 2 * (b**2 - a**2) * np.sin(ang) * np.cos(ang)
    C = a**2 * np.cos(ang) ** 2 + b**2 * np.sin(ang) ** 2
    D = -2 * A * x0 - B * y0
    E = -B * x0 - 2 * C * y0
    F = A * x0**2 + B * x0 * y0 + C * y0**2 - a**2 * b**2

    ellps_impct_func = (
        lambda point: A * point[0] ** 2
        + C * point[1] ** 2
        + B * np.prod(point)
        + D * point[0]
        + E * point[1]
        + F
    )

    if points.size == 2:
        check = np.zeros(1, dtype="bool")
        check[0] = True if ellps_impct_func(points) < 0 else False
    else:
        check = np.zeros(points.shape[0], dtype="bool")
        for i in range(points.shape[0]):
            check[i] = True if ellps_impct_func(points[i, :]) < 0 else False
    return check


if __name__ == "__main__":
    # def plot_ellipse(ellipse):
    ellipse = Ellipse(np.array([-4, 2]), np.deg2rad(45), np.array([1, 2]))
    point_ellipse = ellipse.get_points()

    points_x = np.linspace(-5, 5, 50)
    points_y = np.linspace(-5, 5, 50)
    xv, yv = np.meshgrid(points_x, points_y)
    points = np.vstack([xv.flatten(), yv.flatten()]).T
    mask = check_points_in_ellips(points, ellipse, 0.2)
    rev_mask = np.array(1 - mask, dtype="bool")
    plt.figure(figsize=(10, 10))
    plt.plot(point_ellipse[0, :], point_ellipse[1, :], "g", linewidth=3)
    plt.scatter(points[rev_mask, :][:, 0], points[rev_mask, :][:, 1])
    plt.scatter(points[mask, :][:, 0], points[mask, :][:, 1])
    plt.show()
