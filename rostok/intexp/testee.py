from lxml import etree
import open3d as o3d

from enum import Enum
from dataclasses import dataclass

@dataclass
class PhysicsParameters:
    mass: float = None
    density: float = None
    mu_friction: float = None

class ErrorReport(Enum):
    NONE = 0
    WRONG_FILE_NAME = 1
    WRONG_LINKED_FILE_DESCRIPTION = 2
    WRONG_WEIGHT_PARAMETER_DESCRIPTION = 3
    WRONG_FRICTION_DESCRIPTION = 4
    GRASPING_POSES_MISSING = 5
    WRONG_XML_STRUCTURE = 6
    WRONG_WEIGHT_VALUE = 7
    WRONG_FRICTION_VALUE = 8
    WRONG_GRASPING_POSES_DESCRIPTION = 9
    WRONG_POSE_COORDINATE_DESCRIPTION = 10
    WRONG_POSE_ORIENTATION_DESCRIPTION = 11
    REQUIRED_OBJ_MESH_FORMAT = 12
    BROKEN_3D_MESH = 13
    MESH_NOT_FOUND = 14
    WEIGHT_NOT_DEFINED = 15

class TesteeObject(object):
    """Geometry and physical parameters of the testee object
    
    Init State: 
        - empty mesh
        - empty physical parameters
        - empty poses
        
    TODO: Added @property and @setter 
    """
    
    def __init__(self) -> None:
        self.__parameters = PhysicsParameters()
        self.__grasping_poses_list = []
        self.__linked_obj_file = None
        self._mesh = o3d.geometry.TriangleMesh()

    def loadObject3DMesh(self, file_mesh: str) -> ErrorReport:
        """Loading a volumetric mesh from a Wavefront OBJ file

        Args:
            file_mesh (str): mesh file like name 'body1.obj' or path '../folder1/body1.obj'

        Returns:
            error (ErrorReport): zero or error code
            
        TODO: Checking the integrity of the grid and the possibility of its use
        """
        
        mesh = o3d.io.read_triangle_mesh(filename=file_mesh,
                                                enable_post_processing=False,
                                                print_progress=True)
        
        self._mesh = mesh
        return ErrorReport.NONE
    
    def getMeshVertices(self):
        """Vertex coordinates

        Returns:
            float64 array of shape (num_vertices, 3), use numpy.asarray() to access data
        """
        return self._mesh.vertices
    
    def showObject3DModel(self):
        """Visualization of the object model through OpenGL
        """
        mesh_list = [self._mesh]
        o3d.visualization.draw_geometries(geometry_list = mesh_list,
                                  window_name = 'Testee Object',
                                  width = 1280,
                                  height = 720,
                                  mesh_show_wireframe = True)

    def showObjectWithGraspingPoses(self):
        """Visualization of the object shape and gripping poses through OpenGL.
        The location and direction of capture is indicated by an arrow.
        
        TODO: Change arrow on frame
        """
        demo_list = []
        demo_list.append(self._mesh)
        
        for pose in self.__grasping_poses_list:
            demo_list.append(self.__createArrowSignGraspingPose(pose))
        
        demo_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1))
        o3d.visualization.draw_geometries(geometry_list = demo_list,
                                  window_name = 'Testee Object and Poses',
                                  width = 1280,
                                  height = 720,
                                  mesh_show_wireframe = True)
        
        demo_list.clear()
            
    def __createArrowSignGraspingPose(self, pose) -> o3d.geometry.TriangleMesh:
        """Creates an o3d.geometry.TriangleMesh() object in the form of an arrow at the grip position.
        Arrow aligned with grip direction.

        Args:
            pose (list[(x,y,z), (x, y, z, w)]): position and orientation

        Returns:
            o3d.geometry.TriangleMesh: Arrow icon
        """
        coordinate, orientation = pose
        arrow_size = (self._mesh.get_max_bound()[2] - self._mesh.get_min_bound()[2]) / 10
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

    def loadObjectDescription(self, fname: str) -> ErrorReport:
        """Loading physical parameters and required grip positions in xml format

        Args:
            fname (str): physical parameters like name 'body1_param.xml' or path '../folder1/body1_param.xml'

        Returns:
            error (ErrorReport): zero or error code
        """
        try:
            description_file = etree.parse(fname)
            
            xmlread_error = self.__checkXMLLinkedObjFile( description_file.find('obj_file') )
            if xmlread_error.value: return xmlread_error 

            xmlread_error = self.__checkXMLWeightParam( description_file.find('weight') )
            if xmlread_error.value: return xmlread_error

            xmlread_error = self.__checkXMLMuFriction( description_file.find('mu_friction') )
            if xmlread_error.value: return xmlread_error

            xmlread_error = self.__checkXMLGraspingPoses( description_file.find('grasping_poses') )
            if xmlread_error.value: return xmlread_error

            return ErrorReport.NONE

        except OSError:
            return ErrorReport.WRONG_FILE_NAME
        except:
            return ErrorReport.WRONG_XML_STRUCTURE

    def __checkXMLLinkedObjFile(self, xmlsubelem_linked_obj_file) -> ErrorReport:
        """Checking the correctness of loaded parameters

        Args:
            xmlsubelem_linked_obj_file (lxmlsubelem): linked .obj file name

        Returns:
            error (ErrorReport): zero or error code
        """
        if (xmlsubelem_linked_obj_file is not None) and (xmlsubelem_linked_obj_file.text.endswith('.obj')):
            self.__linked_obj_file = xmlsubelem_linked_obj_file.text
            return ErrorReport.NONE
        else:
            return ErrorReport.WRONG_LINKED_FILE_DESCRIPTION

    def __checkXMLWeightParam(self, xmlsubelem_weight) -> ErrorReport:
        """Checking the correctness of loaded parameters

        Args:
            xmlsubelem_weight (lxmlsubelem): weight paramaeter (density or mass)

        Returns:
            error (ErrorReport): zero or error code
        """
        if(xmlsubelem_weight.get('parameter') == 'mass'):
            weight_value = float(xmlsubelem_weight.text)
            if weight_value > 0:
                self.__parameters.mass = weight_value
                return ErrorReport.NONE
            else:
                return ErrorReport.WRONG_WEIGHT_VALUE

        elif(xmlsubelem_weight.get('parameter') == 'density'):
            weight_value = float(xmlsubelem_weight.text)
            if weight_value > 0:
                self.__parameters.density = weight_value
                return ErrorReport.NONE
            else:
                return ErrorReport.WRONG_WEIGHT_VALUE

        else:
            return ErrorReport.WRONG_WEIGHT_PARAMETER_DESCRIPTION

    def __checkXMLMuFriction(self, xmlsubelem_friction) -> ErrorReport:
        """Checking the correctness of loaded parameters

        Args:
            xmlsubelem_friction (lxmlsubelem): friction coefficient for surface

        Returns:
            error (ErrorReport): zero or error code
        """
        if (xmlsubelem_friction is not None):
            value = float(xmlsubelem_friction.text)
            if (value > 0) or (value <= 1):
                self.__parameters.mu_friction = value
                return ErrorReport.NONE
            else:
                return ErrorReport.WRONG_FRICTION_VALUE
        else:
            return ErrorReport.WRONG_FRICTION_DESCRIPTION

    def __checkXMLGraspingPoses(self, xmlsubelem_grasping_poses) -> ErrorReport:
        """Checking the correctness of loaded parameters

        Args:
            xmlsubelem_grasping_poses (lxmlsubelem): list of user's grasping poses (desired)

        Returns:
            error (ErrorReport): zero or error code
        """
        if xmlsubelem_grasping_poses is not None:
            for xmlsubelem_pose in xmlsubelem_grasping_poses.getchildren():
                coordinates = xmlsubelem_pose.get('coordinates')
                coordinates = coordinates.replace('[', '')
                coordinates = coordinates.replace(']', '')
                coordinates = [float(x) for x in coordinates.split(',')]
                if len(coordinates) != 3: return ErrorReport.WRONG_POSE_COORDINATE_DESCRIPTION

                orientation = xmlsubelem_pose.get('orientation')
                orientation = orientation.replace('[', '')
                orientation = orientation.replace(']', '')
                orientation = [float(x) for x in orientation.split(',')]
                if len(orientation) != 4: return ErrorReport.WRONG_POSE_ORIENTATION_DESCRIPTION

                self.__grasping_poses_list.append([coordinates, orientation])
        else:
            return ErrorReport.GRASPING_POSES_MISSING
        
        return ErrorReport.NONE
    
    def __checkIntersectionGraspingPosesAndMesh(self) -> ErrorReport:
        """ !!!_summary_!!!

        Returns:
            error (ErrorReport): zero or error code
            
        TODO: Create check method for detection instersection btw grasping points
            adnd mesh. If grasping pose was inside -> remove pos from poses_list
        """

        pass

    def getGraspingPosesNumber(self) -> int:
        """Property for getting number of poses

        Returns:
            int: number of poses
        """
        return len(self.__grasping_poses_list)

    def getGraspingPose(self, index: int) -> list:
        """Property for getting pose with index

        Args:
            index (int): pose with index

        Returns:
            [(x,y,z), (x, y, z, w)]: pose as position and rotation
        """
        return self.__grasping_poses_list[index]

    def getGraspingPosesList(self) -> list:
        """Property for getting all poses

        Returns:
            list[(x,y,z), (x, y, z, w)]: poses as position and rotation
        """
        return self.__grasping_poses_list
    
    def addGraspingPose(self, coord, orient):
        """Add new pose in poses list

        Args:
            coord (list[(x,y,z)]): place of pose in local coordinate system
            orient (list[(x,y,z,w)]): orientation of gripper in local coordinate system
        """
        self.__grasping_poses_list.append([coord, orient])
    
    def addGraspingPosesList(self, new_poses):
        """Add new poses in poses list

        Args:
            new_poses list[(x,y,z), (x, y, z, w)]: new poses of gripper
        """
        self.__grasping_poses_list.extend(new_poses)
        
    def clearGraspingPosesList(self):
        """Clear list of poses.
        """
        self.__grasping_poses_list.clear()
        
    def setGraspingPosesList(self, new_poses):
        """Replaces an existing list of capture poses with a new one.

        Args:
            new_poses list[(x,y,z), (x, y, z, w)]: new poses of gripper
        """
        self.__grasping_poses_list.clear()
        self.__grasping_poses_list.extend(new_poses)
    
    def setDensity(self, density_val: float):
        """Setter of density

        Args:
            density_val (float): new value in kg/m3
        """
        self.__parameters.density = density_val
    
    def getDensity(self) -> float:
        """Property of density

        Returns:
            float: kg/m3
        """
        if self.__parameters.density:
            return self.__parameters.density
        elif self.__parameters.mass and self._mesh:
            return self.__parameters.mass / self._mesh.get_volume()
        else:
            return None
    
    def setMuFriction(self, mu_val: float):
        """Setter of surface friction coefficient

        Args:
            mu_val (float): from 0 to 1
        """
        if mu_val > 0 and mu_val <= 1: self.__parameters.mu_friction = mu_val
    
    def getMuFriction(self) -> float:
        """Property of surface friction coefficient

        Returns:
            float: from 0 to 1 or None if not defined
        """
        if self.__parameters.mu_friction:
            return self.__parameters.mu_friction
        else:
            return None      
        
    def getName(self):
        """Property of linked file name

        Returns:
            str: name of testee object
        """
        return self.__linked_obj_file

if __name__ == '__main__':
    pass