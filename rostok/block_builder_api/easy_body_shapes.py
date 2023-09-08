from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass
class Box:
    width_x: float = 0.1
    length_y: float = 0.5
    height_z: float = 0.1
    
    def __hash__(self) -> int:
        return hash(("Box", self.width_x, self.length_y, self.height_z))


@dataclass
class Cylinder:
    """The cylinder is created along the Y axis and centered at the center of mass
    """
    radius: float = 0.1
    height_y: float = 0.5
    
    def __hash__(self) -> int:
        return hash(("Cylinder", self.radius, self.height_y))


@dataclass
class Sphere:
    radius: float = 0.15
    
    def __hash__(self) -> int:
        return hash(("Sphere", self.radius))


@dataclass
class Ellipsoid:
    radius_x: float = 0.1
    radius_y: float = 0.2
    radius_z: float = 0.3
    
    def __hash__(self) -> int:
        return hash(("Ellipsoid", self.radius_x, self.radius_y, self.radius_z))


@dataclass
class FromMesh:
    path: Path
    
    def __hash__(self) -> int:
        return hash(("FromMesh", self.path))

@dataclass
class ConvexHull:
    points: list[tuple[float, float, float]] = field(default_factory=list)

    def __hash__(self) -> int:    
        return hash(("ConvexHull", self.points))


# All types of shape
ShapeTypes = Union[Box, Cylinder, Sphere, Ellipsoid, FromMesh, ConvexHull]
