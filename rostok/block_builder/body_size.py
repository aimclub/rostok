from collections import namedtuple

BoxSize = namedtuple('BoxSize', ["width", "length", "height"])
CylinderSize = namedtuple('CylinderSize', ["radius", "height"])
SphereSize = namedtuple('SphereSize', ["radius"])
EllipsoidSize = namedtuple('EllipsoidSize', ["radius_a", "radius_b", "radius_c"])
