from dataclasses import dataclass
from lxml import etree
import open3d as o3d
import numpy as np

@dataclass
class PhysicsParameters:
    mass: float = 0
    density: float = 0
    mu_contact: float = 0.8
    kn_contact: float = 2e4
    gn_contact: float = 1e6

class TesteeObject():
    """Geometry and physical parameters of the testee object

    Init State:
        - empty mesh
        - empty physical parameters
        - empty poses
    """

    def __init__(self) -> None:
        self.__parameters = PhysicsParameters()
        self.__grasping_poses_list: list[list[float]] = []
        self.__linked_obj_file: str = ""
        self.mesh = o3d.geometry.TriangleMesh()

    def load_object_mesh(self, file_mesh: str):
        """Loading a volumetric mesh from a Wavefront OBJ file

        Args:
            file_mesh (str): mesh file like name 'body1.obj' or path '../folder1/body1.obj'

        TODO: Checking the integrity of the grid and the possibility of its use
        """

        self.mesh = o3d.io.read_triangle_mesh(filename = file_mesh,
                                              enable_post_processing = False,
                                              print_progress = True)

    def demonstrate_object(self):
        """Visualization of the object model through OpenGL
        """
        o3d.visualization.draw_geometries(geometry_list = [self.mesh],
                                          window_name = 'Testee Object',
                                          width = 1280,
                                          height = 720,
                                          mesh_show_wireframe = True)

    def demonstrate_object_and_grasping_poses(self):
        """Visualization of the object shape and gripping poses through OpenGL.
        The location and direction of capture is indicated by an arrow.

        TODO: Change arrow on frame
        """
        demo_list = []
        demo_list.append(self.mesh)
        for pose in self.__grasping_poses_list:
            demo_list.append(self.__create_arrow_sign(pose))

        demo_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1))
        o3d.visualization.draw_geometries(geometry_list = demo_list,
                                          window_name = "Testee Object and "+
                                                        f"{self.grasping_poses_number} poses",
                                          width = 1280,
                                          height = 720,
                                          mesh_show_wireframe = True)

        demo_list.clear()

    def __create_arrow_sign(self, pose) -> o3d.geometry.TriangleMesh:
        """Creates an o3d.geometry.TriangleMesh() object in the form of an
        arrow at the grip position. Arrow aligned with grip direction.

        Args:
            pose (list[(x,y,z), (x, y, z, w)]): position and orientation

        Returns:
            o3d.geometry.TriangleMesh: Arrow icon
        """
        coordinate, orientation = pose
        arrow_size = (self.mesh.get_max_bound()[2] - self.mesh.get_min_bound()[2]) / 10
        arrow_mesh = o3d.geometry.TriangleMesh().create_arrow(cylinder_radius=arrow_size*0.1,
                                                              cone_radius=arrow_size*0.15,
                                                              cylinder_height=arrow_size * 2/3,
                                                              cone_height=arrow_size * 1/3,
                                                              resolution=20,
                                                              cylinder_split=4,
                                                              cone_split=1)
        arrow_mesh.paint_uniform_color([220/255, 20/255, 60/255])
        arrow_mesh.translate(coordinate)
        arrow_mesh.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(orientation))
        return arrow_mesh

    def load_object_description(self, fname: str):
        """Loading physical parameters and required grip positions in xml format

        Args:
            fname (str): physical parameters like name 'body1_param.xml' or
            path '../folder1/body1_param.xml'

        Returns:
            error (ErrorReport): zero or error code
        """
        try:
            description_file = etree.parse(fname)
        except OSError:
            print("Wrong file name")
            raise

        xmlread_error = self.__check_xml_linked_obj_file( description_file.find('obj_file') )
        if xmlread_error:
            raise Exception("Not found paramater <obj_file>")

        xmlread_error = self.__check_xml_weight_param( description_file.find('weight') )
        if xmlread_error:
            raise Exception("Not found paramater <weight>")

        xmlread_error = self.__check_xml_mu_contact( description_file.find('mu_contact') )
        if xmlread_error:
            raise Exception("Not found paramater <mu_contact>")

        xmlread_error = self.__check_xml_kn_contact( description_file.find('kn_contact') )
        if xmlread_error:
            raise Exception("Not found paramater <kn_contact>")

        xmlread_error = self.__check_xml_gn_contact( description_file.find('gn_contact') )
        if xmlread_error:
            raise Exception("Not found paramater <gn_contact>")

        xmlread_error = self.__check_xml_grasping_poses( description_file.find('grasping_poses') )
        if xmlread_error:
            raise Exception("Not found paramater <grasping_poses>")

    def __check_xml_linked_obj_file(self, xmlsubelem_linked_obj_file) -> bool:
        """Checking the correctness of loaded parameters

        Args:
            xmlsubelem_linked_obj_file (lxmlsubelem): linked .obj file name

        Returns:
            error (bool): the presence of an error
        """
        if (xmlsubelem_linked_obj_file is not None) and \
            (xmlsubelem_linked_obj_file.text.endswith('.obj')):
            self.__linked_obj_file = xmlsubelem_linked_obj_file.text
            return False

        return True

    def __check_xml_weight_param(self, xmlsubelem_weight) -> bool:
        """Checking the correctness of loaded parameters

        Args:
            xmlsubelem_weight (lxmlsubelem): weight paramaeter (density or mass)

        Returns:
            error (bool): the presence of an error
        """
        if xmlsubelem_weight.get('parameter') == 'mass':
            weight_value = float(xmlsubelem_weight.text)
            if weight_value > 0:
                self.__parameters.mass = weight_value
                return False

            return True

        elif xmlsubelem_weight.get('parameter') == 'density':
            weight_value = float(xmlsubelem_weight.text)
            if weight_value > 0:
                self.__parameters.density = weight_value
                return False

            return True

        else:
            return True

    def __check_xml_mu_contact(self, xmlsubelem_friction) -> bool:
        """Checking the correctness of loaded parameters

        Args:
            xmlsubelem_friction (lxmlsubelem): friction coefficient for surface

        Returns:
            error (bool): the presence of an error
        """
        if xmlsubelem_friction is not None:
            value = float(xmlsubelem_friction.text)
            if (value > 0) or (value <= 1):
                self.__parameters.mu_contact = value
                return False

            return True

        return True

    def __check_xml_kn_contact(self, xmlsubelem_friction) -> bool:
        """Checking the correctness of loaded parameters

        Args:
            xmlsubelem_friction (lxmlsubelem): damping coefficient for surface

        Returns:
            error (bool): the presence of an error
        """
        if xmlsubelem_friction is not None:
            value = float(xmlsubelem_friction.text)
            if (value > 0) or (value <= 1):
                self.__parameters.kn_contact = value
                return False

            return True

        return True

    def __check_xml_gn_contact(self, xmlsubelem_friction) -> bool:
        """Checking the correctness of loaded parameters

        Args:
            xmlsubelem_friction (lxmlsubelem): damping coefficient for surface

        Returns:
            error (bool): the presence of an error
        """
        if xmlsubelem_friction is not None:
            value = float(xmlsubelem_friction.text)
            if (value > 0) or (value <= 1):
                self.__parameters.gn_contact = value
                return False

            return True

        return True

    def __check_xml_grasping_poses(self, xmlsubelem_grasping_poses) -> bool:
        """Checking the correctness of loaded parameters

        Args:
            xmlsubelem_grasping_poses (lxmlsubelem): list of user's grasping poses (desired)

        Returns:
            error (bool): the presence of an error
        """
        if xmlsubelem_grasping_poses is not None:
            for xmlsubelem_pose in xmlsubelem_grasping_poses.getchildren():
                coordinates = xmlsubelem_pose.get('coordinates')
                coordinates = coordinates.replace('[', '')
                coordinates = coordinates.replace(']', '')
                coordinates = [float(x) for x in coordinates.split(',')]
                if len(coordinates) != 3:
                    return True

                orientation = xmlsubelem_pose.get('orientation')
                orientation = orientation.replace('[', '')
                orientation = orientation.replace(']', '')
                orientation = [float(x) for x in orientation.split(',')]
                if len(orientation) != 4:
                    return True

                self.__grasping_poses_list.append([coordinates, orientation])
        else:
            return True

        return False

    @property
    def grasping_poses_number(self) -> int:
        """Property for getting number of poses

        Returns:
            int: number of poses
        """
        return len(self.__grasping_poses_list)

    def get_grasping_pose(self, index: int) -> list:
        """Property for getting pose with index

        Args:
            index (int): pose with index

        Returns:
            [(x,y,z), (x, y, z, w)]: pose as position and rotation
        """
        return self.__grasping_poses_list[index]

    def get_grasping_poses_list(self) -> list:
        """Property for getting all poses

        Returns:
            list[(x,y,z), (x, y, z, w)]: poses as position and rotation
        """
        return self.__grasping_poses_list

    def add_grasping_pose(self, coord, orient):
        """Add new pose in poses list

        Args:
            coord (list[(x,y,z)]): place of pose in local coordinate system
            orient (list[(x,y,z,w)]): orientation of gripper in local coordinate system
        """
        self.__grasping_poses_list.append([coord, orient])

    def add_grasping_poses_list(self, new_poses):
        """Add new poses in poses list

        Args:
            new_poses list[(x,y,z), (x, y, z, w)]: new poses of gripper
        """
        self.__grasping_poses_list.extend(new_poses)

    def clear_grasping_poses_list(self):
        """Clear list of poses.
        """
        self.__grasping_poses_list.clear()

    def rewrite_grasping_poses_list(self, new_poses):
        """Replaces an existing list of capture poses with a new one.

        Args:
            new_poses list[(x,y,z), (x, y, z, w)]: new poses of gripper
        """
        self.__grasping_poses_list.clear()
        self.__grasping_poses_list.extend(new_poses)

    @property
    def density(self) -> float:
        """Property of density

        Returns:
            float: kg/m3
        """
        if self.__parameters.density:
            return self.__parameters.density

        return self.__parameters.mass / self.mesh.get_volume()

    @density.setter
    def density(self, density_val: float):
        """Setter of density

        Args:
            density_val (float): new value in kg/m3
        """
        if density_val <= 0:
            raise Exception("Wrong density value")

        self.__parameters.density = density_val

    @property
    def mu_contact(self) -> float:
        """Property of surface friction coefficient

        Returns:
            float: from 0 to 1
        """
        if self.__parameters.mu_contact:
            return self.__parameters.mu_contact

        raise Exception("Mu friction not setup")

    @mu_contact.setter
    def mu_contact(self, mu_val: float):
        """Setter of surface friction coefficient

        Args:
            mu_val (float): from 0 to 1
        """
        if mu_val > 0 and mu_val <= 1.0:
            self.__parameters.mu_contact = mu_val
            return

        raise Exception("Wrong mu friction value")

    @property
    def kn_contact(self) -> float:
        """Property of surface damping coefficient

        Returns:
            float: from 0 to 1
        """
        if self.__parameters.kn_contact:
            return self.__parameters.kn_contact

        raise Exception("Damping contact not setup")

    @kn_contact.setter
    def kn_contact(self, d_val: float):
        """Setter of surface friction coefficient

        Args:
            d_val (float): from 0 to 1
        """
        if d_val > 0 and d_val <= 1.0:
            self.__parameters.kn_contact = d_val
            return

        raise Exception("Wrong damping contact value")

    @property
    def gn_contact(self) -> float:
        """Property of surface damping coefficient

        Returns:
            float: from 0 to 1
        """
        if self.__parameters.gn_contact:
            return self.__parameters.gn_contact

        raise Exception("Damping contact not setup")

    @gn_contact.setter
    def gn_contact(self, d_val: float):
        """Setter of surface friction coefficient

        Args:
            d_val (float): from 0 to 1
        """
        if d_val > 0 and d_val <= 1.0:
            self.__parameters.gn_contact = d_val
            return

        raise Exception("Wrong damping contact value")

    @property
    def obj_file_name(self) -> str:
        """Property of linked file name

        Returns:
            str: name of testee object
        """
        return self.__linked_obj_file

    @property
    def bound_box(self) -> list[float]:
        total_bound_box = np.absolute(self.mesh.get_max_bound()) + \
                            np.absolute(self.mesh.get_min_bound())
        return total_bound_box

class Crutch():
    def __init__(self, horn_width = 0.05, base_height = 0.05, gap = 0.01, depth_k: float = 0.05):
        if (horn_width > 0) and (base_height > 0) and (gap > 0) and (depth_k > 0):
            self.mesh = o3d.geometry.TriangleMesh()
            self.__horn_width = horn_width
            self.__base_height = base_height
            self.__gap = gap
            self.__depth_k = depth_k
        else:
            raise Exception("Invalid argument value. Arguments must be positive.")

    def build_for_testee_object(self, obj: o3d.geometry.TriangleMesh):
        if obj.is_empty():
            raise Exception("Unable not build Crutch for TesteeObject without mesh")

        bound_box = self.__calc_summary_bound_box(obj)
        crutch_depth = bound_box[2] * self.__depth_k
        left_horn = o3d.geometry.TriangleMesh().create_box(width = self.__horn_width,
                                                            height = bound_box[1]*0.5,
                                                            depth = crutch_depth)
        right_horn = o3d.geometry.TriangleMesh().create_box(width = self.__horn_width,
                                                            height = bound_box[1]*0.5,
                                                            depth = crutch_depth)
        base = o3d.geometry.TriangleMesh().create_box(width = bound_box[1] +
                                                            self.__gap*2 + self.__horn_width*2,
                                                        height = self.__base_height,
                                                        depth = crutch_depth)
        bottom = o3d.geometry.TriangleMesh().create_box(width = base.get_max_bound()[0],
                                                        height = 0.01,
                                                        depth = bound_box[2])

        left_horn.translate([0, self.__base_height, 0])
        right_horn.translate([base.get_max_bound()[0]-self.__horn_width,
                              self.__base_height,
                              0])
        self.mesh = base + left_horn + right_horn + bottom

        left_horn.translate([0, 0, bound_box[2]-crutch_depth])
        right_horn.translate([0, 0, bound_box[2]-crutch_depth])
        base.translate([0, 0, bound_box[2]-crutch_depth])
        self.mesh += base + left_horn + right_horn

        self.mesh.translate([-self.mesh.get_max_bound()[0]/2,
                             0,
                             -self.mesh.get_max_bound()[2]/2])

    def __calc_summary_bound_box(self, mesh) -> list[float]:
        return np.absolute(mesh.get_max_bound()) + \
                          np.absolute(mesh.get_min_bound())

if __name__ == '__main__':
    pass
