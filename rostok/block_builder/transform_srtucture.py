from collections import namedtuple

FrameTransform = namedtuple('FrameTransform', ["position", "rotation"])

OriginWorldFrame = FrameTransform([0, 0, 0], [1, 0, 0, 0])