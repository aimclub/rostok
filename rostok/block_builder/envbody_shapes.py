from dataclasses import dataclass
from typing import Union


@dataclass
class Box:
    width: float = 0.1
    length: float = 0.2
    height: float = 0.2


@dataclass
class Cylinder:
    radius: float = 0.1
    height: float = 0.5


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
