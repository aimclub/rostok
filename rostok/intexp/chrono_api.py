from dataclasses import dataclass
import pychrono as chrono
from .entity import TesteeObject, Crutch
from .utils import o3d_to_chrono_trianglemesh

class ChTesteeObject(TesteeObject):
    """Class for creation of rigid bodies from a mesh shape for PyChrono Engine.
    Supported files for loading mesh - .obj, for loading physicals parameters - .xml

    Init State:
        - empty mesh: o3d.geometry.TriangleMesh()
        - empty physical parameters
        - empty poses
    """

    def __init__(self, obj_fname: str = "", xml_fname: str = "") -> None:
        super().__init__()

        if (not obj_fname) and (not xml_fname):
            self.chrono_body: chrono.ChBodyEasyMesh = None
            self.chrono_material: chrono.ChMaterialSurfaceSMC = None
            return

        self.create_chrono_body_from_file(obj_fname, xml_fname)

    def create_chrono_body_from_file(self, obj_fname: str, xml_fname: str):
        """ Method created ChBodyEasyMesh object from .obj file with physicals
        parameters from .xml file

        Args:
            obj_fname (str): mesh file like name 'body1.obj' or path '../folder1/body1.obj'
            xml_fname (str): physical parameters like name 'body1_param.xml'
                             or path '../folder1/body1_param.xml'
        """
        super().load_object_mesh(obj_fname)
        super().load_object_description(xml_fname)

        self.chrono_material = chrono.ChMaterialSurfaceSMC()
        self.chrono_material.SetFriction(self.mu_contact)
        self.chrono_material.SetKn(self.kn_contact)
        self.chrono_material.SetGn(self.gn_contact)

        self.chrono_body = chrono.ChBodyEasyMesh(o3d_to_chrono_trianglemesh(self.mesh),
                                                 self.density,
                                                 True,
                                                 True,
                                                 True,
                                                 self.chrono_material)
        self.chrono_body.SetName(super().obj_file_name)
        self.chrono_body.GetVisualShape(0).SetColor(chrono.ChColor(240/255, 100/255, 55/255))

    def get_chrono_grasping_poses_list(self) -> list[chrono.ChFrameD]:
        """Returned list of all poses for interaction with testee object.
        All poses described like position + rotation in local coordinates of object.

        Returns:
            list[chrono.ChFrameD]: Vecotor and Quaternion of poses
        """
        poses = self.get_grasping_poses_list()
        print(len(poses))
        chrono_poses = []
        for i in poses:
            chrono_poses.append(chrono.ChFrameD(chrono.ChVectorD(i[0][0], i[0][1], i[0][2]),
                                                chrono.ChQuaternionD(i[1][0], i[1][1], i[1][2], i[1][3])))
        return chrono_poses

    def set_chrono_body_on_pose(self, pose_to_ref: chrono.ChFrameD, pose_to_abs: chrono.ChFrameD):
        """Positions the object in the capture pose relative to the global coordinate system

        Args:
            pose_to_ref (chrono.ChFrameD): gripping pose in local coordinate system
            pose_to_abs (chrono.ChFrameD): place in global coordinate system
        """
        cog_to_ref = self.chrono_body.GetFrame_COG_to_REF()
        temp = pose_to_ref.GetInverse() * cog_to_ref
        desired_cog_to_abs = pose_to_abs * temp
        self.chrono_body.SetCoord(desired_cog_to_abs.GetPos(), desired_cog_to_abs.GetRot())
        self.chrono_body.SetNoSpeedNoAcceleration()

    def set_chrono_body_ref_frame_in_point(self, desired_ref_to_abs_point: chrono.ChFrameD):
        """Positions the object relative local frame to the global coordinate system

        Args:
            desired_ref_to_abs_point (chrono.ChFrameD): place in global coordinate system
        """
        cog_to_ref = self.chrono_body.GetFrame_COG_to_REF()
        desired_cog_to_abs = desired_ref_to_abs_point * cog_to_ref
        self.chrono_body.SetCoord(desired_cog_to_abs.GetPos(), desired_cog_to_abs.GetRot())
        self.chrono_body.SetNoSpeedNoAcceleration()

@dataclass
class ImpactTimerParameters:
    test_num: int = 0   # number of actual impact
    test_bound: int = 0 # number of all impacts
    clock: float = 0 # actual time
    clock_bound: float = 0 # time to change of impact
    step: float = 0

def update_impact(chtestee: ChTesteeObject,
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
            chtestee.chrono_body.RemoveAllForces()
            chtestee.set_chrono_body_ref_frame_in_point(chrono.ChFrameD(chrono.ChVectorD(0,0.2,0),
                                                                        chrono.ChQuaternionD(1,0,0,0)))
            chtestee.chrono_body.SetNoSpeedNoAcceleration()

            chtestee.chrono_body.AddForce(force)
            force.SetVrelpoint(impacts_point[solver.test_num])
            force.SetRelDir(impacts_dir[solver.test_num])
            force.SetMforce(impacts_magnitude[solver.test_num])
            solver.test_num += 1
        else:
            solver.test_num = 0
        solver.clock = 0

    solver.clock += solver.step

class ChCrutch(Crutch):
    def __init__(self, horn_width=0.05, base_height=0.05, gap=0.01, depth_k=0.05) -> None:
        super().__init__(horn_width, base_height, gap, depth_k)
        self.chrono_body: chrono.ChBodyEasyMesh = None
        self.chrono_material = chrono.ChMaterialSurfaceNSC()
        self.chrono_frame_for_object_placing: chrono.ChFrameD = chrono.ChFrameD()
        self.__holded_object_place: chrono.ChFrameD = None

    def build_chrono_body(self, chrono_obj: ChTesteeObject,
                          mu_contact: float = 0.8,
                          kn_contact: float = 2e4,
                          gn_contact: float = 1e6,
                          density = 2700,
                          start_pos: list[float] = [0.0, 0.0, 0.0]):

        super().build_for_testee_object(chrono_obj.mesh)

        self.chrono_material = chrono.ChMaterialSurfaceSMC()
        self.chrono_material.SetFriction(mu_contact)
        self.chrono_material.SetKn(kn_contact)
        self.chrono_material.SetGn(gn_contact)

        self.chrono_body = chrono.ChBodyEasyMesh(o3d_to_chrono_trianglemesh(self.mesh),
                                                 density,
                                                 True,
                                                 True,
                                                 True,
                                                 self.chrono_material)
        self.chrono_body.SetPos(chrono.ChVectorD(start_pos[0],
                                                 start_pos[1],
                                                 start_pos[2]))
        self.chrono_body.SetBodyFixed(True)
        self.chrono_body.SetName("Crutch")
        self.chrono_body.GetVisualShape(0).SetColor(chrono.ChColor(165/255, 165/255, 165/255))

        self.__holded_object_place = chrono.ChFrameD(chrono.ChVectorD(0,
                                                                      self.mesh.get_max_bound()[1] * 0.8,
                                                                      -self.mesh.get_max_bound()[2]),
                                                     chrono.ChQuaternionD(1, 0, 0, 0))

    def set_position(self, desired_ref_to_abs_point: chrono.ChFrameD):
        cog_to_ref = self.chrono_body.GetFrame_COG_to_REF()
        desired_cog_to_abs = desired_ref_to_abs_point * cog_to_ref
        self.chrono_body.SetCoord(desired_cog_to_abs.GetPos(), desired_cog_to_abs.GetRot())
        self.chrono_body.SetNoSpeedNoAcceleration()

    @property
    def place_for_object(self) -> chrono.ChFrameD:
        return self.__holded_object_place

    @place_for_object.setter
    def place_for_object(self, new_place: chrono.ChFrameD):
        self.__holded_object_place = new_place

if __name__ == '__main__':
    pass
