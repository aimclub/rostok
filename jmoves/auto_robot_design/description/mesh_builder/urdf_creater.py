
from itertools import combinations
from attr import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R

import manifold3d as m3d
import trimesh
import modern_robotics as mr
import odio_urdf as urdf

from auto_robot_design.description.actuators import RevoluteUnit
from auto_robot_design.description.builder import URDFLinkCreator
from auto_robot_design.description.kinematics import (
    Box,
    Joint,
    Link,
    Mesh,
    Sphere
)
from auto_robot_design.description.utils import tensor_inertia_sphere_by_mass
from auto_robot_design.utils.geom import calculate_rot_vec2_to_vec1, calculate_transform_with_2points

def manifold2trimesh(manifold):
    mesh = manifold.to_mesh()

    if mesh.vert_properties.shape[1] > 3:
        vertices = mesh.vert_properties[:, :3]
        colors = (mesh.vert_properties[:, 3:] * 255).astype(np.uint8)
    else:
        vertices = mesh.vert_properties
        colors = None

    return trimesh.Trimesh(
        vertices=vertices, faces=mesh.tri_verts, vertex_colors=colors
    )

def beautiful_cutout_holders(height, length, alp=1):

    x1 = np.linspace(-1,2,200)
    y1 = np.tanh(x1*alp)
    x2 = np.linspace(2,3,100)
    y2 = np.ones_like(x2) * y1[-1]
    x3 = np.linspace(-2,1,200)
    y3 = -np.tanh(x3*alp)

    x3_r = np.linspace(3,6,200)

    x4 = np.linspace(x3_r[-1], x1[0], 100)
    y4 = np.linspace(y3[-1], y1[0], 100)

    x = np.r_[x1, x2, x3_r, x4]
    y = np.r_[y1, y2, y3, y4]

    coeff_x = length / (np.max(x) - np.min(x))
    coeff_y = height / (np.max(y) - np.min(y))


    y = coeff_y*y
    x = coeff_x*x

    offset_x = -(np.max(x) + np.min(x)) / 2
    offset_y = -(np.max(y) + np.min(y)) / 2
    x += offset_x
    y += offset_y
    return x,y


def create_mesh_manipulator_base(link):
    height_base = 0.15
    radius_rack = 0.02
    thickness = 0.01
    pos_jps = [j.jp.r for j in link.joints]

    raduis_actuators = [j.actuator.size[0] for j in link.joints]
    height_actuators = [j.actuator.size[1] for j in link.joints]
    jps_pos_mean = np.mean(pos_jps, axis=0)+np.array([0, height_base/2+np.mean(height_actuators)/2, 0])

    pos_jps_db = np.repeat(pos_jps, 2, axis=0)

    k = 0
    for m in range(len(pos_jps_db)):
        pos_jps_db[m,[0,2]] = pos_jps_db[m,[0,2]] + (-1)**m * raduis_actuators[k]
        if m // 2 == 1:
            k +=1
    
    up_right_point_table = np.max(pos_jps_db, axis=0)
    down_left_point_table = np.min(pos_jps_db, axis=0)

    width = up_right_point_table[0] - down_left_point_table[0]
    length = up_right_point_table[2] - down_left_point_table[2]

    table_m3d = m3d.Manifold.cube([width, length, 0.03], True)

    diag_table_length = np.sqrt(width**2 + length**2)
    length_rack = diag_table_length * 1.3

    rot = np.rad2deg(np.arctan2(length, width))

    x,y = beautiful_cutout_holders(height_base, length_rack, 5)
    custom_md = m3d.CrossSection.hull_points(np.c_[x,y]).extrude(1).translate([0, 0, -0.5]).rotate([90, 0,0]).translate([0,0,-height_base*0.1])
    box = m3d.Manifold.cube([length_rack,thickness, height_base], True)
    cylinder_1 = m3d.Manifold.cylinder(height_base,radius_rack,circular_segments=40, center=True).translate([length_rack/2+radius_rack,0,0])
    cylinder_2 = m3d.Manifold.cylinder(height_base,radius_rack,circular_segments=40, center=True).translate([-(length_rack/2+radius_rack),0,0])
    holder = (box - custom_md + cylinder_1 + cylinder_2)
    first_holder = holder.rotate([0,0,rot])
    second_holder = holder.rotate([0,0,-rot])
    body = first_holder + second_holder+table_m3d.translate([0,0,height_base/2-0.03/2])

    return body.rotate([90, 0, 0]).translate(jps_pos_mean)

class MeshCreator:
    LOCOMOTION_RADIUS_SCALER = {"InActuator": 1, "InRevoluteUnit": 1, "OutActuator": 0.4, "OutRevoluteUnit": 1, "EE": 1.7}
    LOCOMOTION_HEIGHT_SCALER = {"InActuator": 1, "InRevoluteUnit": 0.6, "OutActuator": 1.5, "OutRevoluteUnit": 1.5, "EE": 1.4}
    MANIPULATOR_RADIUS_SCALER = {"InActuator": 1, "InRevoluteUnit": 1, "OutActuator": 0.4, "OutRevoluteUnit": 1, "EE": 0.5}
    MANIPULATOR_HEIGHT_SCALER = {"InActuator": 1, "InRevoluteUnit": 0.6, "OutActuator": 1.5, "OutRevoluteUnit": 1.5, "EE": 1.4}

    def __init__(self, predefind_mesh: dict[str, str] = {}):
        self.predefind_mesh = predefind_mesh
        self.radius_scaler = self.LOCOMOTION_RADIUS_SCALER
        self.height_scaler = self.LOCOMOTION_HEIGHT_SCALER

    
    def build_link_mesh(self, link: Link):
        if link.name in self.predefind_mesh:
            if isinstance(self.predefind_mesh[link.name],str):
                mesh = trimesh.load_mesh(self.predefind_mesh[link.name])
            else:
                body = self.predefind_mesh[link.name](link)
                mesh = manifold2trimesh(body)
        else:
            body = self.build_link_m3d(link)
            mesh = manifold2trimesh(body)
        
        return mesh
    
    def build_link_m3d(self, link: Link):
        in_joints = [j for j in link.joints if j.link_in == link]
        out_joints = [j for j in link.joints if j.link_out == link]
        joint_bodies = []
        substract_bodies = []
        frame = mr.TransInv(link.frame @ link.inertial_frame)
        for j in out_joints:
            pos = (frame @ np.hstack((j.jp.r, [1])))[:3]
            ort_move = frame[:3,:3] @ j.jp.w
            rot = calculate_rot_vec2_to_vec1(ort_move)
            tranform = mr.RpToTrans(rot.as_matrix(), pos)
            size = j.actuator.size
            if "G" in [l.name for l in j.links]:
                joint_bodies.append(
                    m3d.Manifold.
                    cylinder(size[1], size[0], circular_segments=32, center=True).
                    transform(tranform[:3,:])
                )
                substract_bodies.append(
                    m3d.Manifold()
                )
            else:
                if isinstance(j.actuator, RevoluteUnit):
                    r_scale = self.radius_scaler["OutRevoluteUnit"] # 1
                    height_scale = self.height_scaler["OutRevoluteUnit"] # 1.5
                    # joint_bodies.append(
                    #     m3d.Manifold.
                    #     cylinder(size[1]*height_scale, size[0]*r_scale, circular_segments=32, center=True).
                    #     transform(tranform[:3,:])
                    # )
                    subtract_size = [0.001, 0]
                else:
                    r_scale = self.radius_scaler["OutActuator"]#0.4
                    height_scale = self.height_scaler["OutActuator"]#1.5
                    subtract_size = [0.001, 0]
                joint_bodies.append(
                    m3d.Manifold.
                    cylinder(size[1]*height_scale, size[0]*r_scale, circular_segments=32, center=True).
                    transform(tranform[:3,:])
                )
                substract_bodies.append(
                    m3d.Manifold.
                    cylinder(size[1]+subtract_size[1], size[0]+subtract_size[0], circular_segments=32, center=True).
                    transform(tranform[:3,:])
                )
        for j in in_joints:
            pos = (frame @ np.hstack((j.jp.r, [1])))[:3]
            ort_move = frame[:3,:3] @ j.jp.w
            rot = calculate_rot_vec2_to_vec1(ort_move)
            tranform = mr.RpToTrans(rot.as_matrix(), pos)
            size = j.actuator.size
            
            if isinstance(j.actuator, RevoluteUnit):
                r_scale = self.radius_scaler["InRevoluteUnit"] #1
                height_scale = self.height_scaler["InRevoluteUnit"]#0.6
            else:
                r_scale = self.radius_scaler["InActuator"]#1
                height_scale = self.height_scaler["InActuator"]#1
            if "EE" in [l.name for l in j.links]:
                r_scale = self.radius_scaler["EE"]#1.7
                height_scale = self.height_scaler["EE"]#1.4
            joint_bodies.append(
                m3d.Manifold.
                cylinder(size[1]*height_scale, size[0]*r_scale, circular_segments=32, center=True).
                transform(tranform[:3,:])
            )

        js = in_joints + out_joints
        num_joint = len(in_joints + out_joints)
        
        j_points = []
        for j in js:
            j_points.append(
                (frame @ np.hstack((j.jp.r, [1])))[:3]
            ) 
        if num_joint == 1:
            body_link = joint_bodies[0]
        elif num_joint == 2:
            pos, rot, vec_len = calculate_transform_with_2points(j_points[1], j_points[0])
            if link.name == "G":
                thickness = 0.02
            else:
                thickness = link.geometry.get_thickness()/2
            if vec_len > thickness:
                length = vec_len - thickness
            else:
                length = vec_len
            Hb = mr.RpToTrans(rot.as_matrix(), pos)
            link_part1 = (m3d.Manifold.cylinder(length, thickness, circular_segments=32, center=True).
                        transform(Hb[:3,:]))
            link_part2 = (m3d.Manifold.cylinder(length, thickness, circular_segments=32, center=True).
                        transform(Hb[:3,:]))
            
            body_1 = m3d.Manifold.batch_hull([link_part1, joint_bodies[0]])
            body_2 = m3d.Manifold.batch_hull([link_part2, joint_bodies[1]])
            
            body_link = body_1 + body_2
        elif num_joint == 3:
            joint_pos_pairs = combinations(range(len(j_points)), 2)
            body_link = m3d.Manifold()
            for k, m in joint_pos_pairs:
                oth_point = (set(range(3)) - set([k,m])).pop()
                pos, rot, vec_len = calculate_transform_with_2points(j_points[k], j_points[m])
                thickness = link.geometry.get_thickness()/2
                if vec_len > thickness:
                    length = vec_len - thickness
                else:
                    length = vec_len
                Hb = mr.RpToTrans(rot.as_matrix(), pos)
                link_part = (m3d.Manifold.cylinder(length, thickness, circular_segments=32, center=True).
                        transform(Hb[:3,:]))
                body_part = m3d.Manifold.batch_hull([link_part, joint_bodies[oth_point]])
                # body_part = link_part
                body_link = body_link + body_part
        for sub_body in substract_bodies:
            body_link = body_link - sub_body
            
        return body_link

class URDFMeshCreator(URDFLinkCreator):
    def __init__(self) -> None:
        super().__init__()
        self.mesh_path = None
        self.prefix_name = None
        
    def set_path_to_mesh(self, path_to_mesh):
        self.mesh_path = path_to_mesh
        
    def set_prefix_name_mesh(self, prefix):
        self.prefix_name = prefix

    def create_link(self, link: Link):
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
            urdf_link = self._create_mesh(
                link.geometry, link.name, inertia, body_origins, self.trans_matrix2xyz_rpy(link.inertial_frame)
            )
        elif link.geometry.shape == "box":
            origin = self.trans_matrix2xyz_rpy(link.inertial_frame)
            # link_origin = self.trans_matrix2xyz_rpy(link.frame)
            urdf_link = self._create_box(link.geometry, link.name, origin, origin)
        elif link.geometry.shape == "sphere":
            origin = self.trans_matrix2xyz_rpy(link.inertial_frame)
            # link_origin = self.trans_matrix2xyz_rpy(link.frame)
            urdf_link = self._create_sphere(link.geometry, link.name, origin, origin)
        else:
            pass
        return urdf_link

    def create_joint(self, joint: Joint):
        """
        Create a URDF joint based on the given Joint object.

        Args:
            joint (Joint): The Joint object containing the joint information.

        Returns:
            dict: A dictionary containing the created URDF joint and additional information.
        """
        if joint.link_in is None or joint.link_out is None:
            return {"joint": []}
        origin = self.trans_matrix2xyz_rpy(joint.frame)
        if joint.is_constraint:
            color1 = joint.link_in.geometry.color
            color1[3] = 0.5
            color2 = joint.link_out.geometry.color
            color2[3] = 0.5

            name_link_in = joint.jp.name + "_" + joint.link_in.name + "Pseudo"
            rad_in = joint.link_in.geometry.get_thickness() / 1.4
            urdf_pseudo_link_in = urdf.Link(
                # urdf.Visual(
                #     urdf.Geometry(urdf.Sphere(float(rad_in))),
                #     urdf.Material(
                #         urdf.Color(rgba=color1), name=name_link_in + "_Material"
                #     ),
                #     # name=name_link_in + "_Visual",
                # ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **self.convert_inertia(
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
                # urdf.Visual(
                #     urdf.Geometry(urdf.Sphere(float(rad_out))),
                #     urdf.Material(
                #         urdf.Color(rgba=color2), name=name_link_out + "_Material"
                #     ),
                #     # name=name_link_out + "_Visual",
                # ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **self.convert_inertia(
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

            out_origin = self.trans_matrix2xyz_rpy(H_out_j)

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
                # urdf.Visual(
                #     urdf.Geometry(
                #         urdf.Cylinder(
                #             length=connected_unit.size[1], radius=connected_unit.size[0]
                #         )
                #     ),
                #     urdf.Material(
                #         urdf.Color(rgba=color), name=name_joint_link + "_Material"
                #     ),
                #     # name=name_joint_link + "_Visual",
                # ),
                urdf.Inertial(
                    urdf.Inertia(
                        **self.convert_inertia(connected_unit.calculate_inertia())
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
                    # urdf.Visual(
                    #     urdf.Geometry(
                    #         urdf.Cylinder(
                    #             length=joint.actuator.size[1],
                    #             radius=joint.actuator.size[0],
                    #         )
                    #     ),
                    #     urdf.Material(
                    #         urdf.Color(rgba=color),
                    #         name=name_actuator_link + "_Material",
                    #     ),
                    #     # name=name_actuator_link + "_Visual",
                    # ),
                    urdf.Inertial(
                        urdf.Inertia(
                            **self.convert_inertia(joint.actuator.calculate_inertia())
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

    def _create_box(self, geometry: Box, name, origin, inertia_origin):
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
        # urdf_geometry = urdf.Geometry(urdf.Box(geometry.size))
        to_mesh = str(self.mesh_path.joinpath(self.prefix_name + name + ".stl"))
        urdf_geometry = urdf.Geometry(urdf.Mesh(to_mesh, [1,1,1]))
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
            urdf.Inertia(**self.convert_inertia(geometry.inertia)),
        )

        return urdf.Link(visual, collision, inertial, name=name)

    def _create_sphere(self, geometry: Sphere, name, origin, inertia_origin):
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
        # urdf_geometry = urdf.Geometry(urdf.Sphere(geometry.size[0]))
        to_mesh = str(self.mesh_path.joinpath(self.prefix_name + name + ".stl"))
        urdf_geometry = urdf.Geometry(urdf.Mesh(to_mesh, [1,1,1]))
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
            urdf.Inertia(**self.convert_inertia(geometry.inertia)),
        )

        return urdf.Link(visual, collision, inertial, name=name)
    
    def _create_mesh(self, geometry: Mesh, name, inertia, body_origins, link_origin=None):
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
        origin_I = self.trans_matrix2xyz_rpy(inertia[0])
        urdf_inertia_origin = urdf.Origin(xyz=origin_I[0], rpy=origin_I[1])
        visual_n_collision = []
        to_mesh = str(self.mesh_path.joinpath(self.prefix_name + name + ".stl"))
        urdf_geometry = urdf.Geometry(urdf.Mesh(to_mesh, [1,1,1]))
        urdf_origin = urdf.Origin(
            xyz=link_origin[0],
            rpy=link_origin[1],
        )
        visual = urdf.Visual(
            urdf_origin,
            urdf_geometry,
            urdf_material,
            # name = name_v
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

            collision = urdf.Collision(urdf_origin, urdf_geometry, name=name_c)
            # visual_n_collision += [visual, collision]
            visual_n_collision += [collision]
        visual_n_collision += [visual]
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(float(geometry.size.mass)),
            urdf.Inertia(**self.convert_inertia(inertia[1])),
        )
        return urdf.Link(*visual_n_collision, inertial, name=name)