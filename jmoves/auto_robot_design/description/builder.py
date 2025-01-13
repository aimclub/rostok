from copy import deepcopy
from itertools import combinations
from typing import Union
from pyparsing import List, Tuple

import odio_urdf as urdf
import networkx as nx
import numpy.linalg as la
import numpy as np
from scipy.spatial.transform import Rotation as R
import modern_robotics as mr

from auto_robot_design.description.actuators import Actuator, RevoluteUnit, TMotor_AK80_9, MIT_Actuator
from auto_robot_design.description.kinematics import (
    Joint,
    JointPoint,
    Link,
    Mesh,
    Sphere,
    Box,
)
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph, KinematicGraph
from auto_robot_design.description.utils import tensor_inertia_sphere_by_mass
from auto_robot_design.pino_adapter.pino_adapter import get_pino_description, get_pino_description_3d_constraints
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions

RED_COLOR = np.array([[245/ 255, 84/ 255, 84/ 255, 1]])

BLUE_COLOR = np.array([[39, 105, 205, 1],
                    [40, 109, 204, 1],
                    [95, 128, 213, 1],
                    [132, 149, 222, 1],
                    [164, 170, 231, 1],
                    [193, 193, 240, 1]], dtype=np.float64)
BLUE_COLOR[:,:3] = BLUE_COLOR[:,:3] / 255

GREEN_COLOR = np.array([[17, 90, 57, 1],
                        [4, 129, 75, 1],
                        [0, 169, 92, 1],
                        [0, 211, 107, 1],
                        [0, 255, 119, 1]], dtype=np.float64)
GREEN_COLOR[:,:3] = GREEN_COLOR[:,:3] / 255



DEFAULT_DENSITY = 2700 / 2.8
DEFAULT_THICKNESS = 0.04
DEFAULT_JOINT_DAMPING = 0.05
DEFAULT_JOINT_FRICTION = 0
DEFAULT_ACTUATOR = TMotor_AK80_9()

DEFAULT_PARAMS_DICT = {
    "density": DEFAULT_DENSITY,
    "thickness": DEFAULT_THICKNESS,
    "joint_damping": DEFAULT_JOINT_DAMPING,
    "joint_friction": DEFAULT_JOINT_FRICTION,
    "actuator": DEFAULT_ACTUATOR,
}

MIT_DENSITY = 723.52
MIT_THICKNESS = 0.0335
MIT_JOINT_DAMPING = 0.05
MIT_JOINT_FRICTION = 0
MIT_ACTUATOR = MIT_Actuator()
MIT_BODY_SIZE = [0.28, 0.19, 0.098]
MIT_OFFSET_GROUND_FL = np.array([-MIT_BODY_SIZE[0]/2 - 0.096/2, MIT_BODY_SIZE[1]/2 + MIT_THICKNESS/2, 0])
MIT_OFFSET_GROUND_RL = np.array([MIT_BODY_SIZE[0]/2 + 0.096/2, MIT_BODY_SIZE[1]/2 + MIT_THICKNESS/2, 0])
MIT_BODY_DENSITY = (3.3 + (0.54-0.44))/ 4 / np.prod(MIT_BODY_SIZE)


MIT_CHEETAH_PARAMS_DICT = {
    "density": MIT_DENSITY,
    "thickness": MIT_THICKNESS,
    "joint_damping": MIT_JOINT_DAMPING,
    "joint_friction": MIT_JOINT_FRICTION,
    "actuator": MIT_ACTUATOR,
    "body_density": MIT_BODY_DENSITY,
    "size_ground": MIT_BODY_SIZE,
    "offset_ground_fl": MIT_OFFSET_GROUND_FL,
    "offset_ground_rl": MIT_OFFSET_GROUND_RL,
}

def add_branch(G: nx.Graph, branch: Union[List[JointPoint], List[List[JointPoint]]]):
    """
    Add a branch to the given graph.

    Parameters:
    - G (nx.Graph): The graph to which the branch will be added.
    - branch (Union[List[JointPoint], List[List[JointPoint]]]): The branch to be added. It can be a list of JointPoints or a list of lists of JointPoints.

    Returns:
    None
    """

    is_list = [isinstance(br, List) for br in branch]
    if all(is_list):
        for b in branch:
            add_branch(G, b)
    else:
        for i in range(len(branch) - 1):
            if isinstance(branch[i], List):
                for b in branch[i]:
                    G.add_edge(b, branch[i + 1])
            elif isinstance(branch[i + 1], List):
                for b in branch[i + 1]:
                    G.add_edge(branch[i], b)
            else:
                G.add_edge(branch[i], branch[i + 1])


def add_branch_with_attrib(
    G: nx.Graph,
    branch: Union[List[Tuple[JointPoint, dict]], List[List[Tuple[JointPoint, dict]]]],
):
    is_list = [isinstance(br, List) for br in branch]
    if all(is_list):
        for b in branch:
            add_branch_with_attrib(G, b)
    else:
        for ed in branch:
            G.add_edge(ed[0], ed[1], **ed[2])


def calculate_transform_with_2points(p1: np.ndarray,
                                     p2: np.ndarray,
                                     vec: np.ndarray = np.array([0, 0, 1])):
    """Calculate transformation from `vec` to vector build with points `p1` and `p2`

    Args:
        p1 (np.ndarray): point of vector's start
        p2 (np.ndarray): point of vector's end
        vec (np.ndarray, optional): Vector tansform from. Defaults to np.array([0, 0, 1]).

    Returns:
        tuple: position: np.ndarray, rotation: scipy.spatial.rotation, length: float
    """
    v_l = p2 - p1
    angle = np.arccos(np.inner(vec, v_l) / la.norm(v_l) / la.norm(vec))
    axis = mr.VecToso3(vec[:3]) @ v_l[:3]
    if not np.isclose(np.sum(axis), 0):
        axis /= la.norm(axis)

    rot = R.from_rotvec(axis * angle)
    pos = (p2 + p1) / 2
    length = la.norm(v_l)

    return pos, rot, length

class URDFLinkCreator:
    """
    Class responsible for creating URDF links and joints.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def create_link(cls, link: Link):
        """
        Create a URDF link based on the given Link object.

        Args:
            link (Link): The Link object containing the link information.

        Returns:
            urdf_link: The created URDF link.
        """
        if link.geometry.shape == "mesh":
            pos_joint_in_local = []
            H_l_w = mr.TransInv(link.frame)
            for j in link.joints:
                pos_joint_in_local.append(H_l_w @ np.r_[j.jp.r, 1])

            joint_pos_pairs = combinations(pos_joint_in_local, 2)
            body_origins = []
            for j_p in joint_pos_pairs:
                pos, rot, vec_len = calculate_transform_with_2points(j_p[0][:3], j_p[1][:3])
                if vec_len > link.geometry.get_thickness():
                    length = vec_len - link.geometry.get_thickness()
                else:
                    length = vec_len
                body_origins.append(
                    (pos.tolist(), rot.as_euler("xyz").tolist(), length)
                )
            inertia = (
                link.inertial_frame,
                link.geometry.size.moment_inertia_frame(link.inertial_frame),
            )
            urdf_link = cls._create_mesh(
                link.geometry, link.name, inertia, body_origins, cls.trans_matrix2xyz_rpy(link.inertial_frame)
            )
        elif link.geometry.shape == "box":
            origin = cls.trans_matrix2xyz_rpy(link.inertial_frame)
            # link_origin = cls.trans_matrix2xyz_rpy(link.frame)
            urdf_link = cls._create_box(link.geometry, link.name, origin, origin)
        elif link.geometry.shape == "sphere":
            origin = cls.trans_matrix2xyz_rpy(link.inertial_frame)
            # link_origin = cls.trans_matrix2xyz_rpy(link.frame)
            urdf_link = cls._create_sphere(link.geometry, link.name, origin, origin)
        else:
            pass
        return urdf_link

    @classmethod
    def create_joint(cls, joint: Joint):
        """
        Create a URDF joint based on the given Joint object.

        Args:
            joint (Joint): The Joint object containing the joint information.

        Returns:
            dict: A dictionary containing the created URDF joint and additional information.
        """
        if joint.link_in is None or joint.link_out is None:
            return {"joint": []}
        origin = cls.trans_matrix2xyz_rpy(joint.frame)
        if joint.is_constraint:
            color1 = joint.link_in.geometry.color
            color1[3] = 0.5
            color2 = joint.link_out.geometry.color
            color2[3] = 0.5

            name_link_in = joint.jp.name + "_" + joint.link_in.name + "Pseudo"
            rad_in = joint.link_in.geometry.get_thickness() / 1.4
            urdf_pseudo_link_in = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(urdf.Sphere(float(rad_in))),
                    urdf.Material(
                        urdf.Color(rgba=color1), name=name_link_in + "_Material"
                    ),
                    # name=name_link_in + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **cls.convert_inertia(
                            tensor_inertia_sphere_by_mass(
                                joint.actuator.mass / 2, rad_in
                            )
                        )
                    ),
                ),
                name=name_link_in,
            )
            urdf_joint_in = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=name_link_in),
                urdf.Origin(
                    xyz=origin[0],
                    rpy=origin[1],
                ),
                urdf.Axis(joint.jp.w.tolist()),
                urdf.Limit(
                    lower=joint.pos_limits[0],
                    upper=joint.pos_limits[1],
                    effort=joint.actuator.get_max_effort(),
                    velocity=joint.actuator.get_max_vel(),
                ),
                urdf.Dynamics(
                    damping=joint.damphing_friction[0],
                    friction=joint.damphing_friction[1],
                ),
                name=joint.jp.name + "_" + joint.link_in.name + "_revolute",
                type="revolute",
            )

            name_link_out = joint.jp.name + "_" + joint.link_out.name + "Pseudo"
            rad_out = joint.link_out.geometry.get_thickness() / 1.4
            urdf_pseudo_link_out = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(urdf.Sphere(float(rad_out))),
                    urdf.Material(
                        urdf.Color(rgba=color2), name=name_link_out + "_Material"
                    ),
                    # name=name_link_out + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **cls.convert_inertia(
                            tensor_inertia_sphere_by_mass(
                                joint.actuator.mass / 2, rad_out
                            )
                        )
                    ),
                ),
                name=name_link_out,
            )

            H_in_j = joint.frame
            H_w_in = joint.link_in.frame

            H_w_out = joint.link_out.frame

            H_out_j = mr.TransInv(H_w_out) @ H_w_in @ H_in_j

            out_origin = cls.trans_matrix2xyz_rpy(H_out_j)

            urdf_joint_out = urdf.Joint(
                urdf.Parent(link=joint.link_out.name),
                urdf.Child(link=name_link_out),
                urdf.Origin(
                    xyz=out_origin[0],
                    rpy=out_origin[1],
                ),
                name=joint.jp.name + "_" + joint.link_in.name + "_Weld",
                type="fixed",
            )

            out = {
                "joint": [
                    urdf_pseudo_link_in,
                    urdf_joint_in,
                    urdf_joint_out,
                    urdf_pseudo_link_out,
                ],
                "constraint": [name_link_in, name_link_out],
            }
        else:
            urdf_joint = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=joint.link_out.name),
                urdf.Origin(
                    xyz=origin[0],
                    rpy=origin[1],
                ),
                urdf.Axis(joint.jp.w.tolist()),
                urdf.Limit(
                    lower=joint.pos_limits[0],
                    upper=joint.pos_limits[1],
                    effort=joint.actuator.get_max_effort(),
                    velocity=joint.actuator.get_max_vel(),
                ),
                urdf.Dynamics(
                    damping=joint.damphing_friction[0],
                    friction=joint.damphing_friction[1],
                ),
                name=joint.jp.name,
                type="revolute",
            )
            out = {"joint": [urdf_joint]}
            if joint.jp.active:
                connected_unit = RevoluteUnit()
                connected_unit.size = [
                    joint.link_in.geometry.get_thickness() / 2,
                    joint.link_in.geometry.get_thickness(),
                ]
            elif not joint.actuator.size:
                unit_size = [
                    joint.link_in.geometry.get_thickness() / 2,
                    joint.link_in.geometry.get_thickness(),
                ]
                joint.actuator.size = unit_size
                connected_unit = joint.actuator
            else:
                connected_unit = joint.actuator

            name_joint_link = joint.jp.name + "_" + joint.link_in.name + "Unit"
            name_joint_weld = joint.jp.name + "_" + joint.link_in.name + "_WeldUnit"
            Rp_j = mr.TransToRp(joint.frame)
            color = joint.link_in.geometry.color
            color[3] = 0.9
            rot_a = R.from_matrix(
                Rp_j[0] @ R.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
            ).as_euler("xyz")
            urdf_joint_weld = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=name_joint_link),
                urdf.Origin(
                    xyz=Rp_j[1].tolist(),
                    rpy=rot_a.tolist(),
                ),
                name=name_joint_weld,
                type="fixed",
            )
            urdf_unit_link = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(
                        urdf.Cylinder(
                            length=connected_unit.size[1], radius=connected_unit.size[0]
                        )
                    ),
                    urdf.Material(
                        urdf.Color(rgba=color), name=name_joint_link + "_Material"
                    ),
                    # name=name_joint_link + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Inertia(
                        **cls.convert_inertia(connected_unit.calculate_inertia())
                    ),
                    urdf.Mass(float(connected_unit.mass)),
                ),
                name=name_joint_link,
            )

            if joint.jp.active:
                out["active"] = joint.jp.name
                name_actuator_link = (
                    joint.jp.name + "_" + joint.link_in.name + "Actuator"
                )
                name_actuator_weld = (
                    joint.jp.name + "_" + joint.link_in.name + "_WeldActuator"
                )
                pos = Rp_j[1] + joint.jp.w * (
                    joint.actuator.size[1] / 2 + connected_unit.size[1] / 2
                )
                urdf_actuator_weld = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=name_actuator_link),
                    urdf.Origin(
                        xyz=pos.tolist(),
                        rpy=rot_a.tolist(),
                    ),
                    name=name_actuator_weld,
                    type="fixed",
                )
                urdf_actuator_link = urdf.Link(
                    urdf.Visual(
                        urdf.Geometry(
                            urdf.Cylinder(
                                length=joint.actuator.size[1],
                                radius=joint.actuator.size[0],
                            )
                        ),
                        urdf.Material(
                            urdf.Color(rgba=color),
                            name=name_actuator_link + "_Material",
                        ),
                        # name=name_actuator_link + "_Visual",
                    ),
                    urdf.Inertial(
                        urdf.Inertia(
                            **cls.convert_inertia(joint.actuator.calculate_inertia())
                        ),
                        urdf.Mass(float(joint.actuator.mass)),
                    ),
                    name=name_actuator_link,
                )
                out["joint"].append(urdf_actuator_weld)
                out["joint"].append(urdf_actuator_link)

            out["joint"].append(urdf_unit_link)
            out["joint"].append(urdf_joint_weld)
        return out

    @classmethod
    def trans_matrix2xyz_rpy(cls, H):
        """
        Convert a transformation matrix to XYZ and RPY representation.

        Args:
            H: The transformation matrix.

        Returns:
            tuple: A tuple containing the XYZ position and RPY orientation.
        """
        Rp = mr.TransToRp(H)
        rpy = R.from_matrix(Rp[0]).as_euler("xyz").tolist()
        return (Rp[1].tolist(), rpy)

    @classmethod
    def convert_inertia(cls, tensor_inertia):
        """
        Convert the tensor inertia to a dictionary representation.

        Args:
            tensor_inertia: The tensor inertia.

        Returns:
            dict: A dictionary containing the converted inertia values.
        """
        x, y, z = tuple(range(3))
        Ixx = tensor_inertia[x][x]
        Iyy = tensor_inertia[y][y]
        Izz = tensor_inertia[z][z]
        Ixy = tensor_inertia[x][y]
        Ixz = tensor_inertia[x][z]
        Iyz = tensor_inertia[y][z]
        return {"ixx": Ixx, "ixy": Ixy, "ixz": Ixz, "iyy": Iyy, "iyz": Iyz, "izz": Izz}

    @classmethod
    def _create_box(cls, geometry: Box, name, origin, inertia_origin):
        """
        Create a URDF box based on the given Box geometry.

        Args:
            geometry (Box): The Box geometry object.
            name: The name of the box.
            origin: The origin of the box.
            inertia_origin: The origin of the inertia.

        Returns:
            urdf.Link: The created URDF link.
        """
        name_m = name + "_" + "Material"
        urdf_material = urdf.Material(urdf.Color(rgba=geometry.color), name=name_m)
        name_c = name + "_" + "Collision"
        name_v = name + "_" + "Visual"
        urdf_geometry = urdf.Geometry(urdf.Box(geometry.size))
        urdf_inertia_origin = urdf.Origin(
            xyz=inertia_origin[0],
            rpy=inertia_origin[1],
        )
        urdf_origin = urdf.Origin(
            xyz=origin[0],
            rpy=origin[1],
        )

        visual = urdf.Visual(
            urdf_origin,
            urdf_geometry,
            urdf_material,
            # name = name_v
        )
        collision = urdf.Collision(urdf_origin, urdf_geometry, name=name_c)
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(float(geometry.mass)),
            urdf.Inertia(**cls.convert_inertia(geometry.inertia)),
        )

        return urdf.Link(visual, collision, inertial, name=name)

    @classmethod
    def _create_sphere(cls, geometry: Sphere, name, origin, inertia_origin):
        """
        Create a URDF sphere based on the given Sphere geometry.

        Args:
            geometry (Sphere): The Sphere geometry object.
            name: The name of the sphere.
            origin: The origin of the sphere.
            inertia_origin: The origin of the inertia.

        Returns:
            urdf.Link: The created URDF link.
        """
        name_m = name + "_" + "Material"
        urdf_material = urdf.Material(urdf.Color(rgba=geometry.color), name=name_m)

        name_c = name + "_" + "Collision"
        name_v = name + "_" + "Visual"
        urdf_geometry = urdf.Geometry(urdf.Sphere(geometry.size[0]))
        urdf_inertia_origin = urdf.Origin(
            xyz=inertia_origin[0],
            rpy=inertia_origin[1],
        )
        urdf_origin = urdf.Origin(
            xyz=origin[0],
            rpy=origin[1],
        )

        visual = urdf.Visual(
            urdf_origin,
            urdf_geometry,
            urdf_material,
            # name = name_v
        )
        collision = urdf.Collision(urdf_origin, urdf_geometry, name=name_c)
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(geometry.mass),
            urdf.Inertia(**cls.convert_inertia(geometry.inertia)),
        )

        return urdf.Link(visual, collision, inertial, name=name)

    @classmethod
    def _create_mesh(cls, geometry: Mesh, name, inertia, body_origins, link_origin=None):
        """
        Create a URDF mesh based on the given Mesh geometry.

        Args:
            geometry (Mesh): The Mesh geometry object.
            name: The name of the mesh.
            inertia: The inertia of the mesh.
            body_origins: The origins of the mesh bodies.

        Returns:
            urdf.Link: The created URDF link.
        """
        name_m = name + "_" + "Material"
        urdf_material = urdf.Material(urdf.Color(rgba=geometry.color), name=name_m)
        origin_I = cls.trans_matrix2xyz_rpy(inertia[0])
        urdf_inertia_origin = urdf.Origin(xyz=origin_I[0], rpy=origin_I[1])
        visual_n_collision = []
        to_mesh = "D:\\Files\\Working\\auto-robotics-design\\testing_ground\\mesh\\" + name + ".obj"
        urdf_geometry = urdf.Geometry(urdf.Mesh(to_mesh, 1))
        urdf_origin = urdf.Origin(
            xyz=link_origin[0],
            rpy=link_origin[1],
        )
        for id, origin in enumerate(body_origins):
            name_c = name + "_" + str(id) + "_Collision"
            name_v = name + "_" + str(id) + "_Visual"
            thickness = geometry.get_thickness()
            urdf_geometry = urdf.Geometry(urdf.Box([thickness, thickness, origin[2]]))
            urdf_origin = urdf.Origin(
                xyz=origin[0],
                rpy=origin[1],
            )
            visual = urdf.Visual(
                urdf_origin,
                urdf_geometry,
                urdf_material,
                # name = name_v
            )

            collision = urdf.Collision(urdf_origin, urdf_geometry, name=name_c)
            visual_n_collision += [visual, collision]
            visual_n_collision += [collision]
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(float(geometry.size.mass)),
            urdf.Inertia(**cls.convert_inertia(inertia[1])),
        )
        return urdf.Link(*visual_n_collision, inertial, name=name)


class DetailedURDFCreatorFixedEE(URDFLinkCreator):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def create_joint(cls, joint: Joint):
        if joint.link_in is None or joint.link_out is None:
            return {"joint": []}
        origin = cls.trans_matrix2xyz_rpy(joint.frame)
        if joint.is_constraint:
            color1 = joint.link_in.geometry.color
            color1[3] = 0.5
            color2 = joint.link_out.geometry.color
            color2[3] = 0.5

            name_link_in = joint.jp.name + "_" + joint.link_in.name + "Pseudo"
            rad_in = joint.link_in.geometry.get_thickness() / 1.4
            urdf_pseudo_link_in = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(urdf.Sphere(float(rad_in))),
                    urdf.Material(
                        urdf.Color(rgba=color1), name=name_link_in + "_Material"
                    ),
                    # name=name_link_in + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **cls.convert_inertia(
                            tensor_inertia_sphere_by_mass(
                                joint.actuator.mass / 2, rad_in
                            )
                        )
                    ),
                ),
                name=name_link_in,
            )
            urdf_joint_in = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=name_link_in),
                urdf.Origin(
                    xyz=origin[0],
                    rpy=origin[1],
                ),
                urdf.Axis(joint.jp.w.tolist()),
                urdf.Limit(
                    lower=joint.pos_limits[0],
                    upper=joint.pos_limits[1],
                    effort=joint.actuator.get_max_effort(),
                    velocity=joint.actuator.get_max_vel(),
                ),
                urdf.Dynamics(
                    damping=joint.damphing_friction[0],
                    friction=joint.damphing_friction[1],
                ),
                name=joint.jp.name + "_" + joint.link_in.name + "_revolute",
                type="revolute",
            )

            name_link_out = joint.jp.name + "_" + joint.link_out.name + "Pseudo"
            rad_out = joint.link_out.geometry.get_thickness() / 1.4
            urdf_pseudo_link_out = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(urdf.Sphere(float(rad_out))),
                    urdf.Material(
                        urdf.Color(rgba=color2), name=name_link_out + "_Material"
                    ),
                    # name=name_link_out + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **cls.convert_inertia(
                            tensor_inertia_sphere_by_mass(
                                joint.actuator.mass / 2, rad_out
                            )
                        )
                    ),
                ),
                name=name_link_out,
            )

            H_in_j = joint.frame
            H_w_in = joint.link_in.frame
            H_w_out = joint.link_out.frame
            H_out_j = mr.TransInv(H_w_out) @ H_w_in @ H_in_j
            out_origin = cls.trans_matrix2xyz_rpy(H_out_j)
            urdf_joint_out = urdf.Joint(
                urdf.Parent(link=joint.link_out.name),
                urdf.Child(link=name_link_out),
                urdf.Origin(
                    xyz=out_origin[0],
                    rpy=out_origin[1],
                ),
                name=joint.jp.name + "_" + joint.link_in.name + "_Weld",
                type="fixed",
            )

            out = {
                "joint": [
                    urdf_pseudo_link_in,
                    urdf_joint_in,
                    urdf_joint_out,
                    urdf_pseudo_link_out,
                ],
                "constraint": [name_link_in, name_link_out],
            }
        else:
            if "EE" in [l.name for l in joint.links]:
                urdf_joint = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=joint.link_out.name),
                    urdf.Origin(
                        xyz=origin[0],
                        rpy=origin[1],
                    ),
                    name=joint.jp.name,
                    type="fixed",
                )
            else:
                urdf_joint = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=joint.link_out.name),
                    urdf.Origin(
                        xyz=origin[0],
                        rpy=origin[1],
                    ),
                    urdf.Axis(joint.jp.w.tolist()),
                    urdf.Limit(
                        lower=joint.pos_limits[0],
                        upper=joint.pos_limits[1],
                        effort=joint.actuator.get_max_effort(),
                        velocity=joint.actuator.get_max_vel(),
                    ),
                    urdf.Dynamics(
                        damping=joint.damphing_friction[0],
                        friction=joint.damphing_friction[1],
                    ),
                    name=joint.jp.name,
                    type="revolute",
                )
            out = {"joint": [urdf_joint]}
            if joint.jp.active:
                connected_unit = RevoluteUnit()
                if joint.link_in.name == "G":
                    connected_unit.size = [
                        joint.link_out.geometry.get_thickness() / 2,
                        joint.link_out.geometry.get_thickness(),
                    ]
                else:
                    connected_unit.size = [
                        joint.link_in.geometry.get_thickness() / 2,
                        joint.link_in.geometry.get_thickness(),
                    ]
            elif not joint.actuator.size:
                if joint.link_in.name == "G":
                    unit_size = [
                        joint.link_out.geometry.get_thickness() / 2,
                        joint.link_out.geometry.get_thickness(),
                    ]
                else:
                    unit_size = [
                        joint.link_in.geometry.get_thickness() / 2,
                        joint.link_in.geometry.get_thickness(),
                    ]
                joint.actuator.size = unit_size
                connected_unit = joint.actuator
            else:
                connected_unit = joint.actuator

            name_joint_link = joint.jp.name + "_" + joint.link_in.name + "Unit"
            name_joint_weld = joint.jp.name + "_" + joint.link_in.name + "_WeldUnit"
            Rp_j = mr.TransToRp(joint.frame)
            color = joint.link_in.geometry.color
            color[3] = 0.9
            rot_a = R.from_matrix(
                Rp_j[0] @ R.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
            ).as_euler("xyz")
            urdf_joint_weld = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=name_joint_link),
                urdf.Origin(
                    xyz=Rp_j[1].tolist(),
                    rpy=rot_a.tolist(),
                ),
                name=name_joint_weld,
                type="fixed",
            )
            urdf_unit_link = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(
                        urdf.Cylinder(
                            length=connected_unit.size[1], radius=connected_unit.size[0]
                        )
                    ),
                    urdf.Material(
                        urdf.Color(rgba=color), name=name_joint_link + "_Material"
                    ),
                    # name=name_joint_link + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Inertia(
                        **cls.convert_inertia(connected_unit.calculate_inertia())
                    ),
                    urdf.Mass(float(connected_unit.mass)),
                ),
                name=name_joint_link,
            )

            if joint.jp.active:
                out["active"] = joint.jp.name
                name_actuator_link = (
                    joint.jp.name + "_" + joint.link_in.name + "Actuator"
                )
                name_actuator_weld = (
                    joint.jp.name + "_" + joint.link_in.name + "_WeldActuator"
                )
                pos = Rp_j[1] + joint.jp.w * (
                    joint.actuator.size[1] / 2 + connected_unit.size[1] / 2
                )
                urdf_actuator_weld = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=name_actuator_link),
                    urdf.Origin(
                        xyz=pos.tolist(),
                        rpy=rot_a.tolist(),
                    ),
                    name=name_actuator_weld,
                    type="fixed",
                )
                urdf_actuator_link = urdf.Link(
                    urdf.Visual(
                        urdf.Geometry(
                            urdf.Cylinder(
                                length=joint.actuator.size[1],
                                radius=joint.actuator.size[0],
                            )
                        ),
                        urdf.Material(
                            urdf.Color(rgba=color),
                            name=name_actuator_link + "_Material",
                        )
                        # name=name_actuator_link + "_Visual",
                    ),
                    urdf.Inertial(
                        urdf.Inertia(
                            **cls.convert_inertia(joint.actuator.calculate_inertia())
                        ),
                        urdf.Mass(float(joint.actuator.mass)),
                    ),
                    name=name_actuator_link
                )
                out["joint"].append(urdf_actuator_weld)
                out["joint"].append(urdf_actuator_link)

            out["joint"].append(urdf_unit_link)
            out["joint"].append(urdf_joint_weld)
        return out


class URDFLinkCreater3DConstraints(URDFLinkCreator):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def create_joint(cls, joint: Joint):
        """
        Create a URDF joint based on the given Joint object.

        Args:
            joint (Joint): The Joint object containing the joint information.

        Returns:
            dict: A dictionary containing the created URDF joint and additional information.
        """
        if joint.link_in is None or joint.link_out is None:
            return {"joint": []}
        origin = cls.trans_matrix2xyz_rpy(joint.frame)
        if joint.is_constraint:
            color1 = joint.link_in.geometry.color
            color1[3] = 0.5
            color2 = joint.link_out.geometry.color
            color2[3] = 0.5

            name_link_in = joint.jp.name + "_" + joint.link_in.name + "Pseudo"
            rad_in = joint.link_in.geometry.get_thickness() / 1.4
            urdf_pseudo_link_in = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(urdf.Sphere(float(rad_in))),
                    urdf.Material(
                        urdf.Color(rgba=color1), name=name_link_in + "_Material"
                    ),
                    # name=name_link_in + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **cls.convert_inertia(
                            tensor_inertia_sphere_by_mass(
                                joint.actuator.mass / 2, rad_in
                            )
                        )
                    ),
                ),
                name=name_link_in,
            )
            urdf_joint_in = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=name_link_in),
                urdf.Origin(
                    xyz=origin[0],
                    rpy=origin[1],
                ),
                urdf.Axis(joint.jp.w.tolist()),
                urdf.Limit(
                    lower=joint.pos_limits[0],
                    upper=joint.pos_limits[1],
                    effort=joint.actuator.get_max_effort(),
                    velocity=joint.actuator.get_max_vel(),
                ),
                urdf.Dynamics(
                    damping=joint.damphing_friction[0],
                    friction=joint.damphing_friction[1],
                ),
                name=joint.jp.name + "_" + joint.link_in.name + "_Weld",
                type="fixed",
            )

            name_link_out = joint.jp.name + "_" + joint.link_out.name + "Pseudo"
            rad_out = joint.link_out.geometry.get_thickness() / 1.4
            urdf_pseudo_link_out = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(urdf.Sphere(float(rad_out))),
                    urdf.Material(
                        urdf.Color(rgba=color2), name=name_link_out + "_Material"
                    ),
                    # name=name_link_out + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **cls.convert_inertia(
                            tensor_inertia_sphere_by_mass(
                                joint.actuator.mass / 2, rad_out
                            )
                        )
                    ),
                ),
                name=name_link_out,
            )

            H_in_j = joint.frame
            H_w_in = joint.link_in.frame

            H_w_out = joint.link_out.frame

            H_out_j = mr.TransInv(H_w_out) @ H_w_in @ H_in_j

            out_origin = cls.trans_matrix2xyz_rpy(H_out_j)

            urdf_joint_out = urdf.Joint(
                urdf.Parent(link=joint.link_out.name),
                urdf.Child(link=name_link_out),
                urdf.Origin(
                    xyz=out_origin[0],
                    rpy=out_origin[1],
                ),
                name=joint.jp.name + "_" + joint.link_out.name + "_Weld",
                type="fixed",
            )

            out = {
                "joint": [
                    urdf_pseudo_link_in,
                    urdf_joint_in,
                    urdf_joint_out,
                    urdf_pseudo_link_out,
                ],
                "constraint": [name_link_in, name_link_out],
            }
        else:
            if "EE" in [l.name for l in joint.links]:
                urdf_joint = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=joint.link_out.name),
                    urdf.Origin(
                        xyz=origin[0],
                        rpy=origin[1],
                    ),
                    urdf.Axis(joint.jp.w.tolist()),
                    urdf.Limit(
                        lower=joint.pos_limits[0],
                        upper=joint.pos_limits[1],
                        effort=joint.actuator.get_max_effort(),
                        velocity=joint.actuator.get_max_vel(),
                    ),
                    urdf.Dynamics(
                        damping=joint.damphing_friction[0],
                        friction=joint.damphing_friction[1],
                    ),
                    name=joint.jp.name,
                    type="revolute",
                )
            else:
                urdf_joint = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=joint.link_out.name),
                    urdf.Origin(
                        xyz=origin[0],
                        rpy=origin[1],
                    ),
                    urdf.Axis(joint.jp.w.tolist()),
                    urdf.Limit(
                        lower=joint.pos_limits[0],
                        upper=joint.pos_limits[1],
                        effort=joint.actuator.get_max_effort(),
                        velocity=joint.actuator.get_max_vel(),
                    ),
                    urdf.Dynamics(
                        damping=joint.damphing_friction[0],
                        friction=joint.damphing_friction[1],
                    ),
                    name=joint.jp.name,
                    type="revolute",
                )
            # urdf_joint = urdf.Joint(
            #     urdf.Parent(link=joint.link_in.name),
            #     urdf.Child(link=joint.link_out.name),
            #     urdf.Origin(
            #         xyz=origin[0],
            #         rpy=origin[1],
            #     ),
            #     urdf.Axis(joint.jp.w.tolist()),
            #     urdf.Limit(
            #         lower=joint.pos_limits[0],
            #         upper=joint.pos_limits[1],
            #         effort=joint.actuator.get_max_effort(),
            #         velocity=joint.actuator.get_max_vel(),
            #     ),
            #     urdf.Dynamics(
            #         damping=joint.damphing_friction[0],
            #         friction=joint.damphing_friction[1],
            #     ),
            #     name=joint.jp.name,
            #     type="revolute",
            # )
            out = {"joint": [urdf_joint]}
            if joint.jp.active:
                connected_unit = RevoluteUnit()
                if joint.link_in.name == "G":
                    connected_unit.size = [
                        joint.link_out.geometry.get_thickness() / 2,
                        joint.link_out.geometry.get_thickness(),
                    ]
                else:
                    connected_unit.size = [
                        joint.link_in.geometry.get_thickness() / 2,
                        joint.link_in.geometry.get_thickness(),
                    ]
            elif not joint.actuator.size:
                if joint.link_in.name == "G":
                    unit_size = [
                        joint.link_out.geometry.get_thickness() / 2,
                        joint.link_out.geometry.get_thickness(),
                    ]
                else:
                    unit_size = [
                        joint.link_in.geometry.get_thickness() / 2,
                        joint.link_in.geometry.get_thickness(),
                    ]
                joint.actuator.size = unit_size
                connected_unit = joint.actuator
            else:
                connected_unit = joint.actuator

            name_joint_link = joint.jp.name + "_" + joint.link_in.name + "Unit"
            name_joint_weld = joint.jp.name + "_" + joint.link_in.name + "_WeldUnit"
            Rp_j = mr.TransToRp(joint.frame)
            color = joint.link_in.geometry.color
            color[3] = 0.9
            rot_a = R.from_matrix(
                Rp_j[0] @ R.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
            ).as_euler("xyz")
            urdf_joint_weld = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=name_joint_link),
                urdf.Origin(
                    xyz=Rp_j[1].tolist(),
                    rpy=rot_a.tolist(),
                ),
                name=name_joint_weld,
                type="fixed",
            )
            urdf_unit_link = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(
                        urdf.Cylinder(
                            length=connected_unit.size[1], radius=connected_unit.size[0]
                        )
                    ),
                    urdf.Material(
                        urdf.Color(rgba=color), name=name_joint_link + "_Material"
                    ),
                    # name=name_joint_link + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Inertia(
                        **cls.convert_inertia(connected_unit.calculate_inertia())
                    ),
                    urdf.Mass(float(connected_unit.mass)),
                ),
                name=name_joint_link,
            )

            if joint.jp.active:
                out["active"] = joint.jp.name
                name_actuator_link = (
                    joint.jp.name + "_" + joint.link_in.name + "Actuator"
                )
                name_actuator_weld = (
                    joint.jp.name + "_" + joint.link_in.name + "_WeldActuator"
                )
                pos = Rp_j[1] + joint.jp.w * (
                    joint.actuator.size[1] / 2 + connected_unit.size[1] / 2
                )
                urdf_actuator_weld = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=name_actuator_link),
                    urdf.Origin(
                        xyz=pos.tolist(),
                        rpy=rot_a.tolist(),
                    ),
                    name=name_actuator_weld,
                    type="fixed",
                )
                urdf_actuator_link = urdf.Link(
                    urdf.Visual(
                        urdf.Geometry(
                            urdf.Cylinder(
                                length=joint.actuator.size[1],
                                radius=joint.actuator.size[0],
                            )
                        ),
                        urdf.Material(
                            urdf.Color(rgba=color),
                            name=name_actuator_link + "_Material",
                        ),
                        # name=name_actuator_link + "_Visual",
                    ),
                    urdf.Inertial(
                        urdf.Inertia(
                            **cls.convert_inertia(joint.actuator.calculate_inertia())
                        ),
                        urdf.Mass(float(joint.actuator.mass)),
                    ),
                    name=name_actuator_link,
                )
                out["joint"].append(urdf_actuator_weld)
                out["joint"].append(urdf_actuator_link)

            out["joint"].append(urdf_unit_link)
            out["joint"].append(urdf_joint_weld)
        return out

class Builder:
    def __init__(self, creater) -> None:
        self.creater = creater

    def create_kinematic_graph(self, kinematic_graph, name="Robot"):

        links = kinematic_graph.nodes()
        joints = dict(
            filter(lambda kv: len(kv[1]) > 0, kinematic_graph.joint2edge.items())
        )

        urdf_links = []
        urdf_joints = []
        for link in links:
            urdf_links.append(self.creater.create_link(link))

        active_joints = []
        constraints = []
        for joint in joints:
            info_joint = self.creater.create_joint(joint)

            urdf_joints += info_joint["joint"]

            if "active" in info_joint.keys():
                active_joints.append(info_joint["active"])

            if "constraint" in info_joint.keys():
                constraints.append(info_joint["constraint"])

        urdf_objects = urdf_links + urdf_joints

        urdf_robot = urdf.Robot(*urdf_objects, name=name)

        return urdf_robot, active_joints, constraints


class ParametrizedBuilder(Builder):
    """
    A builder class that allows for parameterized construction of objects.

    Args:
        creater: The object that creates the instance of the builder.
        density (Union[float, dict]): The density of the object being built. Defaults to 2700 / 2.8.
        thickness (float): The thickness of the object being built. Defaults to 0.04.
        joint_damping (Union[float, dict]): The damping of the joints in the object being built. Defaults to 0.05.
        joint_friction (Union[float, dict]): The friction of the joints in the object being built. Defaults to 0.
        size_ground (np.ndarray): The size of the ground for the object being built. Defaults to np.zeros(3).
        actuator: The actuator used in the object being built. Defaults to TMotor_AK80_9().

    Attributes:
        density (Union[float, dict]): The density of the object being built.
        actuator: The actuator used in the object being built.
        thickness (float): The thickness of the object being built.
        size_ground (np.ndarray): The size of the ground for the object being built.
        joint_damping (Union[float, dict]): The damping of the joints in the object being built.
        joint_friction (Union[float, dict]): The friction of the joints in the object being built.
    """

    def __init__(
        self,
        creater,
        density: Union[float, dict] = 2700 / 2.8,
        thickness: Union[float, dict] = 0.01,
        joint_damping: Union[float, dict] = 0.05,
        joint_friction: Union[float, dict] = 0,
        joint_limits: Union[dict, tuple] = (-np.pi, np.pi),
        size_ground: np.ndarray = np.zeros(3),
        offset_ground: np.ndarray = np.zeros(3),
        actuator: Union[Actuator, dict]=TMotor_AK80_9(),
    ) -> None:
        super().__init__(creater)
        self.density = density
        self.actuator = actuator
        self.thickness = thickness
        self.size_ground = size_ground
        self.offset_ground = offset_ground
        self.joint_damping = joint_damping
        self.joint_friction = joint_friction
        self.joint_limits = joint_limits
        self.attributes = ["density", "joint_damping", "joint_friction", "joint_limits", "actuator", "thickness"]
        self.joint_attributes = ["joint_damping", "joint_friction", "actuator", "joint_limits"]
        self.link_attributes = ["density", "thickness"]

    def create_kinematic_graph(self, kinematic_graph: KinematicGraph, name="Robot"):
        # kinematic_graph = deepcopy(kinematic_graph)
        # kinematic_graph.G = list(filter(lambda n: n.name == "G", kinematic_graph.nodes()))[0]
        # kinematic_graph.EE = list(filter(lambda n: n.name == "EE", kinematic_graph.nodes()))[0]
        for attr in self.attributes:
            self.check_default(getattr(self, attr), attr)
        joints = kinematic_graph.joint_graph.nodes()
        for joint in joints:
            self._set_joint_attributes(joint)
        links = kinematic_graph.nodes()
        for link in links:
            self._set_link_attributes(link)

        return super().create_kinematic_graph(kinematic_graph, name)


    def _set_joint_attributes(self, joint):
        if joint.jp.active:
            joint.actuator = self.actuator[joint.jp.name] if joint.jp.name in self.actuator else self.actuator["default"]
        damping = self.joint_damping[joint.jp.name] if joint.jp.name in self.joint_damping else self.joint_damping["default"]
        friction = self.joint_friction[joint.jp.name] if joint.jp.name in self.joint_friction else self.joint_friction["default"]
        limits = self.joint_limits[joint.jp.name] if joint.jp.name in self.joint_limits else self.joint_limits["default"]
        joint.damphing_friction = (damping, friction)
        joint.pos_limits = limits

    def _set_link_attributes(self, link):
        if link.name == "G" and self.size_ground.any():
            link.geometry.size = list(self.size_ground)
            pos = self.offset_ground
            link.inertial_frame = mr.RpToTrans(np.eye(3), pos)
        else:
            link.thickness = self.thickness[link.name] if link.name in self.thickness else self.thickness["default"]
        link.geometry.density = self.density[link.name] if link.name in self.density else self.density["default"]

    def check_default(self, params, name):
        if not isinstance(params, dict):
            setattr(self, name, {"default": params})
        if "default" not in getattr(self, name):
            getattr(self, name)["default"] = DEFAULT_PARAMS_DICT[name]


def jps_graph2urdf_by_bulder(
    graph: nx.Graph,
    builder: ParametrizedBuilder
):
    """
    Converts a graph representation of a robot's kinematic structure to a URDF file using a builder.

    Args:
        graph (nx.Graph): The graph representation of the robot's kinematic structure.
        builder (ParametrizedBuilder): The builder object used to create the URDF.

    Returns:
        tuple: A tuple containing the URDF representation of the robot, the actuator description, and the constraints descriptions.
    """
    kinematic_graph = JointPoint2KinematicGraph(graph)
    kinematic_graph.define_main_branch()
    kinematic_graph.define_span_tree()
    # thickness_aux_branch = 0.025
    i = 1
    k = 1
    name_link_in_aux_branch = []
    for link in kinematic_graph.nodes():
        if link in kinematic_graph.main_branch.nodes():
            # print("yes")
            link.geometry.color = BLUE_COLOR[i,:].tolist()
            i = (i + 1) % 6
        else:
            link.geometry.color = GREEN_COLOR[k,:].tolist()
            name_link_in_aux_branch.append(link.name)
            k = (k + 1) % 5

    # builder.thickness = {link: thickness_aux_branch for link in name_link_in_aux_branch}
    
    kinematic_graph.define_link_frames()

    robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

    act_description, constraints_descriptions = get_pino_description(
        ative_joints, constraints
    )

    return robot.urdf(), act_description, constraints_descriptions

def jps_graph2pinocchio_robot(
    graph: nx.Graph,
    builder: ParametrizedBuilder
):
    """
    Converts a Joint Point Structure (JPS) graph to a Pinocchio robot model.

    Args:
        graph (nx.Graph): The Joint Point Structure (JPS) graph representing the robot's kinematic structure.
        builder (ParametrizedBuilder): The builder object used to create the kinematic graph.

    Returns:
        tuple: A tuple containing the robot model with fixed base and free base.
    """

    kinematic_graph = JointPoint2KinematicGraph(graph)
    kinematic_graph.define_main_branch()
    kinematic_graph.define_span_tree()
    
    # thickness_aux_branch = 0.025
    i = 1
    k = 1
    name_link_in_aux_branch = []
    for link in kinematic_graph.nodes():
        if link in kinematic_graph.main_branch.nodes():
            # print("yes")
            link.geometry.color = BLUE_COLOR[i,:].tolist()
            i = (i + 1) % 6
        else:
            link.geometry.color = GREEN_COLOR[k,:].tolist()
            name_link_in_aux_branch.append(link.name)
            k = (k + 1) % 5

    # builder.thickness = {link: thickness_aux_branch for link in name_link_in_aux_branch}
    kinematic_graph.define_link_frames()

    robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

    act_description, constraints_descriptions = get_pino_description(
        ative_joints, constraints
    )

    fixed_robot = build_model_with_extensions(robot.urdf(),
                                joint_description=act_description,
                                loop_description=constraints_descriptions,
                                actuator_context=kinematic_graph,
                                fixed=True)

    free_robot = build_model_with_extensions(robot.urdf(),
                                joint_description=act_description,
                                loop_description=constraints_descriptions,
                                actuator_context=kinematic_graph,
                                fixed=False)

    return fixed_robot, free_robot

def jps_graph2pinocchio_robot_3d_constraints(
    graph: nx.Graph,
    builder: ParametrizedBuilder,
    back_urdf_str = False
):
    """
    Converts a Joint Point Structure (JPS) graph to a Pinocchio robot model.

    Args:
        graph (nx.Graph): The Joint Point Structure (JPS) graph representing the robot's kinematic structure.
        builder (ParametrizedBuilder): The builder object used to create the kinematic graph.

    Returns:
        tuple: A tuple containing the robot model with fixed base and free base.
    """

    kinematic_graph = JointPoint2KinematicGraph(graph)
    kinematic_graph.define_main_branch()
    kinematic_graph.define_span_tree()
    
    # thickness_aux_branch = 0.025
    i = 1
    k = 1
    name_link_in_aux_branch = []
    for link in kinematic_graph.nodes():
        if link in kinematic_graph.main_branch.nodes():
            # print("yes")
            link.geometry.color = BLUE_COLOR[i,:].tolist()
            i = (i + 1) % 6
        else:
            link.geometry.color = GREEN_COLOR[k,:].tolist()
            name_link_in_aux_branch.append(link.name)
            k = (k + 1) % 5

    # builder.thickness = {link: thickness_aux_branch for link in name_link_in_aux_branch}

    kinematic_graph.define_link_frames()

    robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

    act_description, constraints_descriptions = get_pino_description_3d_constraints(
        ative_joints, constraints
    )

    if back_urdf_str:
        yaml_output = (f"closed_loop: {constraints} \n",
            f"type: {['3d' for __ in range(len(constraints))]} \n",
            f"name_mot: {ative_joints} \n",
            f"joint_name: {[]} \n",
            f"joint_type: {[]} \n"
        )
        return (robot.urdf(), "".join(yaml_output))
    fixed_robot = build_model_with_extensions(robot.urdf(),
                                joint_description=act_description,
                                loop_description=constraints_descriptions,
                                actuator_context=kinematic_graph,
                                fixed=True)

    free_robot = build_model_with_extensions(robot.urdf(),
                                joint_description=act_description,
                                loop_description=constraints_descriptions,
                                actuator_context=kinematic_graph,
                                fixed=False)


    return fixed_robot, free_robot

def create_dict_jp_limit(joints, limit):
    jp2limits = {}
    for jp, lim in zip(joints, limit):
        jp2limits[jp] = lim
    return jp2limits
