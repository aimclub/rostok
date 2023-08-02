from collections import namedtuple
from enum import Enum
from dataclasses import dataclass

FrameTransform = namedtuple('FrameTransform', ["position", "rotation"])

DefaultFrame = FrameTransform([0, 0, 0], [1, 0, 0, 0])


class JointInputType(Enum):
    TORQUE = "Torque"
    VELOCITY = "Speed"
    POSITION = "Angle"
    UNCONTROL = "Uncontrol"


# @dataclass
# class Material:
#     Friction = 0.5
#     DampingF = 0.5
#     Compliance = 0.0001
