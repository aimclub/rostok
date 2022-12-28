from random import uniform
from math import pi, sin, cos
import xml.etree.ElementTree
from scipy.spatial.transform import Rotation
import numpy as np
from .entity import TesteeObject

def gen_random_poses_around_line(num_pose_on_layer: int,
                                 min_dist: float,
                                 max_dist: float,
                                 step: float,
                                 length: float):
    """Generates poses with random angular displacement and random distances
    on predetermined range. Generation is performed along a straight line with
    a fixed step.

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
    gap = max_dist - min_dist

    for i in range(round(length/step)):
        for _ in range(num_pose_on_layer):
            phi = uniform(0, 2*pi)
            des_dist = gap * uniform(0, 1) + min_dist

            coord = [des_dist*sin(phi), des_dist*cos(phi), i*step]
            rot = Rotation.from_euler('xyz', [phi, 0, -pi/2])
            orient = rot.as_quat()

            poses.append([coord, [orient[0], orient[1], orient[2], orient[3]]])

    return poses

def gen_random_poses_around_object_axis(testee_obj: TesteeObject,
                                        num_pose_on_layer: int,
                                        max_dist: float,
                                        step: float,
                                        axis = 'z'):
    """Generates poses with random angular displacement and random distances
    on predetermined range. Generation occurs around the selected axis of the object,
    taking into account its geometric dimensions.

    Args:
        testee_obj (TesteeObject): _description_
        num_pose_on_layer (int): number of poses per layer
        max_dist (float): maximum distance from object
        step (float): step between layers
        axis (str, optional): orientation around one of the axes 'x','y','z'. Defaults to 'z'.

    Returns:
        list[(x,y,z), (x, y, z, w)]: poses as positions and rotations
    """

    poses = []
    size = {'x': testee_obj.mesh.get_max_bound()[0] - testee_obj.mesh.get_min_bound()[0],
            'y': testee_obj.mesh.get_max_bound()[1] - testee_obj.mesh.get_min_bound()[1],
            'z': testee_obj.mesh.get_max_bound()[2] - testee_obj.mesh.get_min_bound()[2]}

    if axis == 'x':
        min_dist = (size['y']+size['z'])/4 * 1.2
        for item in np.arange(testee_obj.mesh.get_min_bound()[0],
                              testee_obj.mesh.get_max_bound()[0],
                              step):

            for _ in range(num_pose_on_layer):
                phi = uniform(0, 2*pi)
                des_dist = max_dist * uniform(0, 1) + min_dist
                coord = [item,
                         des_dist*sin(phi) + testee_obj.mesh.get_center()[1],
                         des_dist*cos(phi) + testee_obj.mesh.get_center()[2]]
                rot = Rotation.from_euler('xyz', [pi/2, -phi, pi])
                orient = rot.as_quat()
                poses.append([coord, [orient[0], orient[1], orient[2], orient[3]]])

    elif axis == 'y':
        min_dist = (size['x']+size['z'])/4 * 1.2
        for item in np.arange(testee_obj.mesh.get_min_bound()[1],
                              testee_obj.mesh.get_max_bound()[1],
                              step):
            for _ in range(num_pose_on_layer):
                phi = uniform(0, 2*pi)
                des_dist = max_dist * uniform(0, 1) + min_dist
                coord = [des_dist*sin(phi) + testee_obj.mesh.get_center()[0],
                         item,
                         des_dist*cos(phi) + testee_obj.mesh.get_center()[2]]
                rot = Rotation.from_euler('xyz', [0, phi, pi])
                orient = rot.as_quat()
                poses.append([coord, [orient[0], orient[1], orient[2], orient[3]]])

    elif axis == 'z':
        min_dist = (size['x']+size['y'])/4 * 1.2
        for item in np.arange(testee_obj.mesh.get_min_bound()[2],
                              testee_obj.mesh.get_max_bound()[2],
                              step):
            for _ in range(num_pose_on_layer):
                phi = uniform(0, 2*pi)
                des_dist = max_dist * uniform(0, 1) + min_dist
                coord = [des_dist*sin(phi) + testee_obj.mesh.get_center()[0],
                         des_dist*cos(phi) + testee_obj.mesh.get_center()[1],
                         item]
                rot = Rotation.from_euler('xyz', [phi, 0, -pi/2])
                orient = rot.as_quat()
                poses.append([coord, [orient[0], orient[1], orient[2], orient[3]]])
    else:
        raise Exception("The axis of the body is incorrect. \
                        You only need to use \'x\', \'y\', \'z\'")

    return poses

def gen_cylindrical_surface_from_poses(num_pose_on_layer: int,
                                       dist: float,
                                       step: float,
                                       length: float):
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

def gen_cylindrical_surface_around_object_axis(testee_obj: TesteeObject,
                                               num_pose_on_layer: int,
                                               dist: float,
                                               step: float,
                                               axis = 'z'):
    """Generates poses with discrete angular displacement on predetermined distance.
    Generation occurs around the selected axis of the object,
    taking into account its geometric dimensions.

    Args:
        testee_obj (TesteeObject): _description_
        num_pose_on_layer (int): number of poses per layer
        dist (float): distance from object
        step (float): step between layers
        axis (str, optional): orientation around one of the axes 'x','y','z'. Defaults to 'z'.

    Returns:
        list[(x,y,z), (x, y, z, w)]: poses as positions and rotations
    """

    poses = []
    dphi = 2*pi / num_pose_on_layer
    size = {'x': testee_obj.mesh.get_max_bound()[0] - testee_obj.mesh.get_min_bound()[0],
            'y': testee_obj.mesh.get_max_bound()[1] - testee_obj.mesh.get_min_bound()[1],
            'z': testee_obj.mesh.get_max_bound()[2] - testee_obj.mesh.get_min_bound()[2]}

    if axis == 'x':
        min_dist = (size['y']+size['z'])/4 * 1.1
        for item in np.arange(testee_obj.mesh.get_min_bound()[0],
                              testee_obj.mesh.get_max_bound()[0],
                              step):
            for j in range(num_pose_on_layer):
                phi = dphi*j
                des_dist = dist + min_dist
                coord = [item,
                         des_dist*sin(phi) + testee_obj.mesh.get_center()[1],
                         des_dist*cos(phi) + testee_obj.mesh.get_center()[2]]
                rot = Rotation.from_euler('xyz', [pi/2, -phi, pi])
                orient = rot.as_quat()
                poses.append([coord, [orient[0], orient[1], orient[2], orient[3]]])

    elif axis == 'y':
        min_dist = (size['x']+size['z'])/4 * 1.1
        for item in np.arange(testee_obj.mesh.get_min_bound()[1],
                              testee_obj.mesh.get_max_bound()[1],
                              step):
            for j in range(num_pose_on_layer):
                phi = dphi*j
                des_dist = dist + min_dist
                coord = [des_dist*sin(phi) + testee_obj.mesh.get_center()[0],
                         item,
                         des_dist*cos(phi) + testee_obj.mesh.get_center()[2]]
                rot = Rotation.from_euler('xyz', [0, phi, pi])
                orient = rot.as_quat()
                poses.append([coord, [orient[0], orient[1], orient[2], orient[3]]])

    elif axis == 'z':
        min_dist = (size['x']+size['y'])/4 * 1.1
        for item in np.arange(testee_obj.mesh.get_min_bound()[2],
                              testee_obj.mesh.get_max_bound()[2],
                              step):
            for j in range(num_pose_on_layer):
                phi = dphi*j
                des_dist = dist
                coord = [des_dist*sin(phi) + testee_obj.mesh.get_center()[0],
                         des_dist*cos(phi) + testee_obj.mesh.get_center()[1],
                         item]
                rot = Rotation.from_euler('xyz', [phi, 0, -pi/2])
                orient = rot.as_quat()
                poses.append([coord, [orient[0], orient[1], orient[2], orient[3]]])
    else:
        raise Exception("The axis of the body is incorrect. \
                        You only need to use \'x\', \'y\', \'z\'")

    return poses

def rewrite_poses_in_xmlconfig(poses: list, xml_file_name: str):
    '''
    Deletes all poses in the file and overwrites the passed sheet
    '''
    config_file = xml.etree.ElementTree.parse(xml_file_name)
    config_file.find('grasping_poses').clear()
    new_poses = []

    index = 0
    for item in poses:
        new_pos = xml.etree.ElementTree.SubElement(config_file.find('grasping_poses'),
                                                    'pose',
                                                    index = str(index),
                                                    coordinates = str(item[0]),
                                                    orientation = str(item[1]))
        new_poses.append(new_pos)
        index += 1

    config_file.write(xml_file_name, encoding="utf-8", xml_declaration=True)

if __name__ == '__main__':
    pass
