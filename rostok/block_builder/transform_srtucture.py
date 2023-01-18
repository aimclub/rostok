from collections import namedtuple

import pychrono as chrono
import numpy as np

FrameTransform = namedtuple('FrameTransform',["position", "rotation"])

def rotation(alpha):
    quat_Y_ang_alpha = chrono.Q_from_AngY(np.deg2rad(alpha))
    return [quat_Y_ang_alpha.e0, quat_Y_ang_alpha.e1, quat_Y_ang_alpha.e2,quat_Y_ang_alpha.e3]
