import pychrono as chrono
from numpy import asarray

from .testee import TesteeObject, ErrorReport
from dataclasses import dataclass

class ChTesteeObject(TesteeObject):  
    """Class for creation of rigid bodies from a mesh shape for PyChrono Engine.
    Supported files for loading mesh - .obj, for loading physicals parameters - .xml

    Init State: 
        - empty mesh
        - empty physical parameters
        - empty poses
    """
    chrono_body_mesh = None
    chrono_material = None
            
    def __init__(self) -> None:
        super().__init__()
        self.chrono_body_mesh = None
        self.chrono_material = None
        pass
    
    def createChronoBodyMeshFromFile(self, obj_fname: str, xml_fname: str):
        """ Method created ChBodyEasyMesh object from .obj file with physicals
        parameters from .xml file

        Args:
            obj_fname (str): mesh file like name 'body1.obj' or path '../folder1/body1.obj'
            xml_fname (str): physical parameters like name 'body1_param.xml' or path '../folder1/body1_param.xml'
            
        TODO: Error cheking when file imported
        """
        error1 = super().loadObject3DMesh(obj_fname)
        error2 = super().loadObjectDescription(xml_fname)
        if (error1 is ErrorReport.NONE) and (error2 is ErrorReport.NONE):
            self.chrono_material = chrono.ChMaterialSurfaceNSC()
            self.chrono_material.SetFriction(super().getMuFriction())
            self.chrono_material.SetDampingF(0.001) # GAG, need import from xml config
            
            self.chrono_body_mesh = chrono.ChBodyEasyMesh(self.__convert03DMeshToChTriangleMeshConnected(),
                                                        self.getDensity(),
                                                        True,
                                                        True,
                                                        True,
                                                        self.chrono_material)
            self.chrono_body_mesh.SetName(super().getName())
            self.chrono_body_mesh.GetVisualShape(0).SetColor(chrono.ChColor(240/255, 100/255, 55/255))
            print('OK')
        
        return (error1, error1)
        
    def __convert03DMeshToChTriangleMeshConnected(self) -> chrono.ChTriangleMeshConnected:
        """Converting the spatial mesh format from O3D to ChTriangleMeshConnected to create
        ChBodyEasyMesh simulation object

        Returns:
            chrono.ChTriangleMeshConnected: The spatial mesh for describing ChBodyEasyMesh
        """
        triangles = asarray(self._mesh.triangles)
        vertices = asarray(self._mesh.vertices)
        ch_mesh = chrono.ChTriangleMeshConnected()
        
        for item in triangles:
            x, y, z = 0, 1, 2
            vert1_index, vert2_index, vert3_index = item[0], item[1], item[2]
            ch_mesh.addTriangle(chrono.ChVectorD(vertices[vert1_index][x], vertices[vert1_index][y], vertices[vert1_index][z]),
                                chrono.ChVectorD(vertices[vert2_index][x], vertices[vert2_index][y], vertices[vert2_index][z]),
                                chrono.ChVectorD(vertices[vert3_index][x], vertices[vert3_index][y], vertices[vert3_index][z]))
       
        return ch_mesh
        
    def getChronoGraspingPosesList(self) -> list[chrono.ChFrameD]:
        """Returned list of all poses for interaction with testee object.
        All poses described like position + rotation in local coordinates of object.

        Returns:
            list[chrono.ChFrameD]: Vecotor and Quaternion of poses
        """
        poses = self.getGraspingPosesList()
        print(len(poses))
        chrono_poses = []
        for i in poses:
            chrono_poses.append(chrono.ChFrameD(chrono.ChVectorD(i[0][0], i[0][1], i[0][2]),
                                                chrono.ChQuaternionD(i[1][0], i[1][1], i[1][2], i[1][3])))         
        return chrono_poses
    
    def setChronoBodyMeshOnPose(self, pose_to_ref: chrono.ChFrameD, pose_to_abs: chrono.ChFrameD):
        """Positions the object in the capture pose relative to the global coordinate system

        Args:
            pose_to_ref (chrono.ChFrameD): gripping pose in local coordinate system
            pose_to_abs (chrono.ChFrameD): place in global coordinate system
        """
        cog_to_ref = self.chrono_body_mesh.GetFrame_COG_to_REF()
        myX = pose_to_ref.GetInverse() * cog_to_ref
        desired_cog_to_abs = pose_to_abs * myX
        self.chrono_body_mesh.SetCoord(desired_cog_to_abs.GetPos(), desired_cog_to_abs.GetRot())
        self.chrono_body_mesh.SetNoSpeedNoAcceleration()
        
    def setChronoBodyMeshRefFrameInPoint(self, desired_ref_to_abs_point: chrono.ChFrameD):
        """Positions the object relative local frame to the global coordinate system

        Args:
            desired_ref_to_abs_point (chrono.ChFrameD): place in global coordinate system
        """
        cog_to_ref = self.chrono_body_mesh.GetFrame_COG_to_REF()
        desired_cog_to_abs = desired_ref_to_abs_point * cog_to_ref
        self.chrono_body_mesh.SetCoord(desired_cog_to_abs.GetPos(), desired_cog_to_abs.GetRot())
        self.chrono_body_mesh.SetNoSpeedNoAcceleration()
        
    def addStaticImpactToChronoBodyMesh(self):
        pass
    
    def addSineImpactToChronoBodyMesh(self):
        pass
    
    def applyKickImpactToChronoBodyMesh(self):
        pass
    
    def clearAllExternalForces(self):
        self.chrono_body_mesh.RemoveAllForces()
        pass
    
    def updateExternalForcesTimer(self):
        pass
        
@dataclass
class ImpactTimerParameters:
    test_num: int = 0   # number of actual impact
    test_bound: int = 0 # number of all impacts
    clock: float = 0 # actual time
    clock_bound: float = 0 # time to change of impact
    step: float = 0       

def updateImpact(chtestee: ChTesteeObject, 
                 force: chrono.ChForce,
                 solver: ImpactTimerParameters, 
                 impacts_point:list, 
                 impacts_dir: list, 
                 impacts_magnitude: list):
    """The function of passing through all the impacts at regular intervals

    Args:
        chtestee (ChTesteeObject): Chrono entity of Testee Object
        force (ChForce): Constant force
        solver (ImpactTimerParameters): timer entity for tracking
        impacts_point (list): sets the application point, in rigid body coordinates
        impacts_dir (list): gets the force direction, in rigid body coordinates
        impacts_magnitude (list): force modulus
    """
    if solver.clock > solver.clock_bound:
        if solver.test_num < solver.test_bound:
            chtestee.chrono_body_mesh.RemoveAllForces()
            chtestee.setChronoBodyMeshRefFrameInPoint(chrono.ChFrameD(chrono.ChVectorD(0,0.2,0), chrono.ChQuaternionD(1,0,0,0)))
            chtestee.chrono_body_mesh.SetNoSpeedNoAcceleration()
            
            chtestee.chrono_body_mesh.AddForce(force)
            force.SetVrelpoint(impacts_point[solver.test_num])
            force.SetRelDir(impacts_dir[solver.test_num])
            force.SetMforce(impacts_magnitude[solver.test_num])
            solver.test_num += 1
        else:
            solver.test_num = 0 
        solver.clock = 0
        
    else:    
        solver.clock += solver.step

if __name__ == '__main__':
    pass