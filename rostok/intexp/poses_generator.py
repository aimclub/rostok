from random import uniform
from math import pi, sin, cos
from scipy.spatial.transform import Rotation

import xml.etree.ElementTree
# import open3d as o3d

def genRandomPosesAroundLine(num_pose_on_layer: int, 
                             min_dist: float, 
                             max_dist: float, 
                             step: float, 
                             length: float) -> list[float]:
    """Generates poses with random angular displacement and random distances on predetermined range.
    Generation is performed along a straight line with a fixed step.

    Args:
        num_pose_on_layer (int): number of poses per layer
        min_dist (float): minimum distance from line
        max_dist (float): maximum distance from line
        step (float): step between layers
        length (float): line length

    Returns:
        list[(x,y,z), (x, y, z, w)]: poses as positions and rotations
    """       
    poses = []
    gape = max_dist - min_dist
    
    for i in range(round(length/step)):
        for j in range(num_pose_on_layer):
            phi = uniform(0, 2*pi)
            ro = gape * uniform(0, 1) + min_dist
            
            coord = [ro*sin(phi), ro*cos(phi), i*step]
            rot = Rotation.from_euler('xyz', [phi, 0, -pi/2])
            orient = rot.as_quat()
            
            poses.append([coord, [orient[0], orient[1], orient[2], orient[3]]])
    
    return poses

def genCylindricalSurfaceFromPoses(num_pose_on_layer: int, 
                                    dist: float, 
                                    step: float, 
                                    length: float) -> list[float]:
    """Generates poses with discrete angular displacement on predetermined distance.
    Generation is performed along a straight line with a fixed step.

    Args:
        num_pose_on_layer (int): number of poses per layer
        dist (float): predetermined distance from line
        step (float): step between layers
        length (float): line length

    Returns:
        list[(x,y,z), (x, y, z, w)]: poses as positions and rotations
    """
    poses = []
    dphi = 2*pi / num_pose_on_layer
    
    for i in range(round(length/step)):
        for j in range(num_pose_on_layer):
            phi = dphi*j
            coord = [dist*sin(phi), dist*cos(phi), i*step]
            rot = Rotation.from_euler('xyz', [phi, 0, -pi/2])
            orient = rot.as_quat()
            poses.append([coord, [orient[0], orient[1], orient[2], orient[3]]])
    
    return poses

# def genPosesFromConvexHull(mesh: o3d.geometry.TriangleMesh, offset: float, scale_factor: float = 3) -> list:
#     poses = []
#     mesh_convex_hull, _ = mesh.compute_convex_hull()
#     triangles_scaled = round(len(mesh_convex_hull.triangles)/scale_factor)
#     mesh_convex_hull_simplify = mesh_convex_hull.simplify_quadric_decimation(triangles_scaled)
#     return poses

def rewritePosesOnXMLFile(poses: list, xml_file_name: str) -> None:
    '''
    Deletes all poses in the file and overwrites the passed sheet
    '''
    doc = xml.etree.ElementTree.parse(xml_file_name)
    doc.find('grasping_poses').clear()
    new_poses = []
    
    index = 0
    for item in poses:
        new_pos = xml.etree.ElementTree.SubElement(doc.find('grasping_poses'), 
                                                    'pose', 
                                                    index = str(index), 
                                                    coordinates = str(item[0]), 
                                                    orientation = str(item[1])) 
        new_poses.append(new_pos)
        index += 1
        
    doc.write(xml_file_name, encoding="utf-8", xml_declaration=True) 

if __name__ == '__main__':
    pass