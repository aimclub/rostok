from collections import namedtuple
from enum import Enum

FrameTransform = namedtuple('FrameTransform', ["position", "rotation"])

DefaultFrame = FrameTransform([0, 0, 0], [1, 0, 0, 0])


class JointInputType(Enum):
    TORQUE = "Torque"
    VELOCITY = "Speed"
    POSITION = "Angle"
    UNCONTROL = "Uncontrol"
