from dataclasses import dataclass
from typing import Union


@dataclass
class Box:
    width_x: float = 0.1
    length_y: float = 0.2
    height_z: float = 0.2


@dataclass
class Cylinder:
    """The cylinder is created along the Y axis and centered at the center of mass
    """
    radius: float = 0.1
    height_y: float = 0.5


@dataclass
class Sphere:
    radius: float = 0.15


@dataclass
class Ellipsoid:
    radius_x: float = 0.1
    radius_y: float = 0.2
    radius_z: float = 0.3

# All types of shape
ShapeTypes = Union[Box, Cylinder, Sphere, Ellipsoid]