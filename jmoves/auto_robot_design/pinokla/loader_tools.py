from copy import deepcopy
from dataclasses import dataclass, field
import unittest
from typing import Optional, Tuple, Union
import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import networkx as nx
import re
import yaml
from yaml.loader import SafeLoader
from warnings import warn


from auto_robot_design.pinokla.actuation_model import ActuationModel
import pinocchio as pin


# Robot = namedtuple(
#     "Robot",
#     [
#         "model",
#         "constraint_models",
#         "actuation_model",
#         "visual_model",
#         "constraint_data",
#         "data",
#     ],
# )

class MotionSpace:
    wrap = {"x": 0, "y": 1, "z": 2, "ang_x": 3, "ang_y": 4, "ang_z": 5}
    
    def __init__(self, *cartesian_terms):
        self.terms = cartesian_terms
    
    @property
    def mask(self):
        idx = self.indexes
        out = np.zeros(len(self.wrap))
        out[idx,] = np.ones_like(idx)
        
        return out
    
    @property
    def indexes(self):
        return tuple(self.wrap[t] for t in self.terms)
    
    def get_6d_traj(self, traj: np.ndarray):
        
        traj_shape = np.shape(traj)
        
        if traj_shape[1] != len(self.terms):
            raise Exception("Wrong size of trajactory")
        
        out = np.zeros((traj_shape[0], len(self.wrap)))
        
        out[:, self.indexes] = traj
        
        return out
    
    def get_6d_point(self, point: np.ndarray):
        
        out = np.zeros(len(self.wrap))
        out[self.indexes,] = point
        
        return out
    
    def rewind_6d_point(self, point_6d: np.ndarray):
        return point_6d[self.indexes,]
    
    def rewind_6d_traj(self, traj_6d: np.ndarray):
        return traj_6d[:, self.indexes]

@dataclass
class Robot:
    model:  pin.Model
    constraint_models: list = field(default_factory=list)
    actuation_model: ActuationModel = field(default_factory=ActuationModel)
    visual_model: pin.GeometryModel = field(default_factory=pin.GeometryModel)
    constraint_data: list = field(default_factory=list)
    data: pin.Data = field(default_factory=pin.Data)
    ee_name: str = "EE"
    motion_space: MotionSpace = MotionSpace("x", "z")
        
        

def make_Robot_copy(robo: Robot):
    # Make real copy
    copied_constrains = []
    for con in robo.constraint_models:
        copied_con = pin.RigidConstraintModel(con)
        copied_constrains.append(copied_con)
        pass

    copied_con_dates = []
    for con in copied_constrains:
        copied_con_data =  con.createData()
        copied_con_dates.append(copied_con_data)
        pass

    copied_model = pin.Model(robo.model)
    copied_data = copied_model.createData()
    copied_actuator_model = deepcopy(robo.actuation_model)
    copied_visual_model = pin.GeometryModel(robo.visual_model)
    return copied_model, copied_constrains, copied_actuator_model, copied_visual_model, copied_con_dates, copied_data

def nameFrameConstraint(model, nomferme="fermeture", Lid=[]):
    """
    nameFrameConstraint(model, nomferme="fermeture", Lid=[])

    Takes a robot model and returns a list of frame names that are constrained to be in contact: Ln=[['name_frame1_A','name_frame1_B'],['name_frame2_A','name_frame2_B'].....]
    where names_frameX_A and names_frameX_B are the frames in forced contact by the kinematic loop.
    The frames must be named: "...nomfermeX_..." where X is the number of the corresponding kinematic loop.
    The kinematics loop can be selectionned with Lid=[id_kinematcsloop1, id_kinematicsloop2 .....] = [1,2,...]
    if Lid = [] all the kinematics loop will be treated.

    Argument:
        model - Pinocchio robot model
        nom_ferme - nom de la fermeture
        Lid - List of kinematic loop indexes to select
    Return:
        Lnames - List of frame names that should be in contact
    """
    warn(
        "Function nameFrameConstraint depreceated - prefer using a YAML file as complement to the URDF. Should only be used to generate a YAML file"
    )
    if Lid == []:
        Lid = range(len(model.frames) // 2)
    Lnames = []
    for id in Lid:
        pair_names = []
        for f in model.frames:
            name = f.name
            match = re.search(nomferme + str(id), name)
            match2 = re.search("frame", f.name)
            if match and not (match2):
                pair_names.append(name)
        if len(pair_names) == 2:
            Lnames.append(pair_names)
    return Lnames


def generateYAML(
    path, name_mot="mot", name_spherical="to_rotule", nomferme="fermeture", file=None
):
    """
    if robot.urdf inside the path, write a yaml file associate to the the robot.
    Write the name of the frame constrained, the type of the constraint, the presence of rotule articulation,
    the name of the motor, idq and idv (with the sphrical joint).
    """

    rob = RobotWrapper.BuildFromURDF(path + "/robot.urdf", path)
    Ljoint = []
    Ltype = []
    Lmot = []
    for name in rob.model.names:
        match = re.search(name_spherical, name)
        match_mot = re.search(name_mot, name)
        if match:
            Ljoint.append(name)
            Ltype.append("SPHERICAL")
        if match_mot:
            Lmot.append(name)

    name_frame_constraint = nameFrameConstraint(rob.model, nomferme)
    # Constraint is default to 6D... that is not very general...
    constraint_type = ["6d"] * len(name_frame_constraint)

    if file is None:
        with open(path + "/robot.yaml", "w") as f:
            f.write("closed_loop: " + str(name_frame_constraint) + "\n")
            f.write("type: " + str(constraint_type) + "\n")
            f.write("name_mot: " + str(Lmot) + "\n")
            f.write("joint_name: " + str(Ljoint) + "\n")
            f.write("joint_type: " + str(Ltype) + "\n")
    else:
        file.write("closed_loop: " + str(name_frame_constraint) + "\n")
        file.write("type: " + str(constraint_type) + "\n")
        file.write("name_mot: " + str(Lmot) + "\n")
        file.write("joint_name: " + str(Ljoint) + "\n")
        file.write("joint_type: " + str(Ltype) + "\n")


def getYAMLcontents(path, name_yaml="robot.yaml"):
    with open(path + "/" + name_yaml, "r") as yaml_file:
        contents = yaml.load(yaml_file, Loader=SafeLoader)
    return contents


def completeRobotLoader(
    path, name_urdf="robot.urdf", name_yaml="robot.yaml", fixed=True
):
    """
    Return  model and constraint model associated to a directory, where the name od the urdf is robot.urdf and the name of the yam is robot.yaml
    if no type assiciated, 6D type is applied
    """
    # Load the robot model using the pinocchio URDF parser
    if fixed:
        robot = RobotWrapper.BuildFromURDF(path + "/" + name_urdf, path)
    else:
        robot = RobotWrapper.BuildFromURDF(
            path + "/robot.urdf", path, root_joint=pin.JointModelFreeFlyer()
        )
        robot.model.names[1] = "root_joint"

    model = robot.model

    yaml_content = getYAMLcontents(path, name_yaml)

    # try to update model
    update_joint = yaml_content["joint_name"]
    joints_types = yaml_content["joint_type"]
    LjointFixed = []
    new_model = pin.Model()
    visual_model = robot.visual_model
    for place, iner, name, parent, joint in list(
        zip(
            model.jointPlacements,
            model.inertias,
            model.names,
            model.parents,
            model.joints,
        )
    )[1:]:
        if name in update_joint:
            joint_type = joints_types[update_joint.index(name)]
            if joint_type == "SPHERICAL":
                jm = pin.JointModelSpherical()
            if joint_type == "FIXED":
                jm = joint
                LjointFixed.append(joint.id)
        else:
            jm = joint
        jid = new_model.addJoint(parent, jm, place, name)
        new_model.appendBodyToJoint(jid, iner, pin.SE3.Identity())

    for f in model.frames:
        n, parent, placement = f.name, f.parentJoint, f.placement
        frame = pin.Frame(n, parent, placement, f.type)
        new_model.addFrame(frame, False)

    new_model.frames.__delitem__(0)
    new_model, visual_model = pin.buildReducedModel(
        new_model, visual_model, LjointFixed, pin.neutral(new_model)
    )

    model = new_model

    # check if type is associated,else 6D is used
    try:
        name_frame_constraint = yaml_content["closed_loop"]
        constraint_type = yaml_content["type"]

        # construction of constraint model
        Lconstraintmodel = []
        for L, ctype in zip(name_frame_constraint, constraint_type):
            name1, name2 = L
            id1 = model.getFrameId(name1)
            id2 = model.getFrameId(name2)
            Se3joint1 = model.frames[id1].placement
            Se3joint2 = model.frames[id2].placement
            parentjoint1 = model.frames[id1].parentJoint
            parentjoint2 = model.frames[id2].parentJoint
            if ctype == "3D" or ctype == "3d":
                constraint = pin.RigidConstraintModel(
                    pin.ContactType.CONTACT_3D,
                    model,
                    parentjoint1,
                    Se3joint1,
                    parentjoint2,
                    Se3joint2,
                    pin.ReferenceFrame.LOCAL,
                )
                constraint.name = name1 + "C" + name2
            else:
                constraint = pin.RigidConstraintModel(
                    pin.ContactType.CONTACT_6D,
                    model,
                    parentjoint1,
                    Se3joint1,
                    parentjoint2,
                    Se3joint2,
                    pin.ReferenceFrame.LOCAL,
                )
                constraint.name = name1 + "C" + name2
            Lconstraintmodel.append(constraint)

        constraint_models = Lconstraintmodel
    except:
        print("no constraint")
    if fixed:
        actuation_model = ActuationModel(model, yaml_content["name_mot"])
    else:
        Lmot = yaml_content["name_mot"]
        Lmot.append("root_joint")
        actuation_model = ActuationModel(model, Lmot)

    return (model, constraint_models, actuation_model, visual_model)


def completeRobotLoaderFromStr(
    udf_str: str, joint_description: dict, loop_description: dict, fixed=True, root_joint_type=pin.JointModelFreeFlyer(), is_act_root_joint=True
):
    """Build pinocchio model from urdf string, actuator(joint) descriptor and loop description.
    You have 2 options:
    1) You can create a model whose base will be rigidly attached to the world. 
    For this set fixed = True. Args root_joint_type and is_act_root_joint not working for this option.
    2) You can create a model whose base will be attached to the world by different type of joint. 
    If you set is_act_root_joint = True, root_joint is actuated. Generalized coordinates associated with 
    root_joint locate first in q vector.
    Args:
        udf_str (str): _description_
        joint_description (dict): _description_
        loop_description (dict): _description_
        fixed (bool, optional): _description_. Defaults to True.
        root_joint_type (_type_, optional): _description_. Defaults to pin.JointModelFreeFlyer().
        is_act_root_joint (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if fixed:
        model = pin.buildModelFromXML(udf_str)
    else:
        model = pin.buildModelFromXML(udf_str, root_joint=root_joint_type)
        model.names[1] = "root_joint"

    visual_model = pin.buildGeomFromUrdfString(
        model, udf_str, pin.pinocchio_pywrap_default.GeometryType.VISUAL
    )
    robot = RobotWrapper(model, visual_model=visual_model)

    model = robot.model

    # try to update model
    update_joint = joint_description["joint_name"]
    joints_types = joint_description["joint_type"]
    LjointFixed = []
    new_model = pin.Model()
    visual_model = robot.visual_model
    for place, iner, name, parent, joint in list(
        zip(
            model.jointPlacements,
            model.inertias,
            model.names,
            model.parents,
            model.joints,
        )
    )[1:]:
        if name in update_joint:
            joint_type = joints_types[update_joint.index(name)]
            if joint_type == "SPHERICAL":
                jm = pin.JointModelSpherical()
            if joint_type == "FIXED":
                jm = joint
                LjointFixed.append(joint.id)
        else:
            jm = joint
        jid = new_model.addJoint(parent, jm, place, name)
        new_model.appendBodyToJoint(jid, iner, pin.SE3.Identity())

    for f in model.frames:
        n, parent, placement = f.name, f.parentJoint, f.placement
        frame = pin.Frame(n, parent, placement, f.type)
        new_model.addFrame(frame, False)

    new_model.frames.__delitem__(0)
    new_model, visual_model = pin.buildReducedModel(
        new_model, visual_model, LjointFixed, pin.neutral(new_model)
    )

    model = new_model

    # check if type is associated,else 6D is used
    try:
        name_frame_constraint = loop_description["closed_loop"]
        constraint_type = loop_description["type"]

        # construction of constraint model
        Lconstraintmodel = []
        for L, ctype in zip(name_frame_constraint, constraint_type):
            name1, name2 = L
            id1 = model.getFrameId(name1)
            id2 = model.getFrameId(name2)
            Se3joint1 = model.frames[id1].placement
            Se3joint2 = model.frames[id2].placement
            parentjoint1 = model.frames[id1].parentJoint
            parentjoint2 = model.frames[id2].parentJoint
            if ctype == "3D" or ctype == "3d":
                constraint = pin.RigidConstraintModel(
                    pin.ContactType.CONTACT_3D,
                    model,
                    parentjoint1,
                    Se3joint1,
                    parentjoint2,
                    Se3joint2,
                    pin.ReferenceFrame.LOCAL,
                )
                constraint.name = name1 + "-" + name2
            else:
                constraint = pin.RigidConstraintModel(
                    pin.ContactType.CONTACT_6D,
                    model,
                    parentjoint1,
                    Se3joint1,
                    parentjoint2,
                    Se3joint2,
                    pin.ReferenceFrame.LOCAL,
                )
                constraint.name = name1 + "-" + name2
            Lconstraintmodel.append(constraint)

        constraint_models = Lconstraintmodel
    except:
        print("no constraint")
    if fixed:
        actuation_model = ActuationModel(model, joint_description["name_mot"])
    else:
        Lmot = joint_description["name_mot"]
        if is_act_root_joint:
            Lmot.append("root_joint")
        actuation_model = ActuationModel(model, Lmot)

    return (model, constraint_models, actuation_model, visual_model)


def build_model_with_extensions(
    urdf_str: str,
    joint_description: dict,
    loop_description: dict,
    actuator_context: Union[None, tuple, dict, nx.Graph] = None,
    fixed=True,
    root_joint_type = pin.JointModelFreeFlyer(),
    is_act_root_joint = True,
    
):
    """
    Builds a robot model with extensions based on the provided URDF string, joint 
    description, loop description,actuator context, and fixed flag. If the actuator_context 
    is not None, an armature is set for each active connection based on the actuator rotor 
    inertia and reduction ratio.

    Args:
        urdf_str (str): The URDF string representing the robot model.
        joint_description (dict): A dictionary describing the active joints of the robot.
        loop_description (dict): A dictionary describing the kinematics loops of the robot.
        actuator_context (Union[None, tuple, dict, nx.Graph], optional): Field, which have information about what actuator is used for each joint. Defaults to None.
        fixed (bool, optional): A flag indicating whether the base robot is fixed. Defaults to True.
        is_act_root_joint (bool): See docs for completeRobotLoaderFromStr 

    Returns:
        Robot: The built robot model with extensions.
    """

    model, constraint_models, actuation_model, visual_model = (
        completeRobotLoaderFromStr(urdf_str, joint_description, loop_description, fixed, root_joint_type, is_act_root_joint)
    )
    constraint_data = [c.createData() for c in constraint_models]
    data = model.createData()
    if actuator_context is not None:
        # Perform additional operations based on the actuator context
        if isinstance(actuator_context, dict):

            actuator_context = tuple(filter(lambda x: x[0] != "default", actuator_context.items()))
        elif isinstance(actuator_context, nx.Graph):
            active_joints = actuator_context.active_joints
            actuator_context = []
            for act_j in active_joints:
                actuator_context.append((act_j.jp.name, act_j.actuator))

        for joint, actuator in actuator_context:
            # It works if motname and idvmot in actuation_model are in the same order
            place_mot = actuation_model.motname2id_v[joint]
            model.armature[place_mot] = (
                actuator.inertia * actuator.reduction_ratio**-2
            )

    return Robot(
        model, constraint_models, actuation_model, visual_model, constraint_data, data
    )


nle = pin.nonLinearEffects


def buildModelsFromUrdf(
    filename,
    package_dirs=None,
    root_joint=None,
    verbose=False,
    meshLoader=None,
    geometry_types=[pin.GeometryType.COLLISION, pin.GeometryType.VISUAL],
) -> Tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
    """Parse the URDF file given in input and return a Pinocchio Model followed by corresponding GeometryModels of types specified by geometry_types, in the same order as listed.
    Examples of usage:
        # load model, collision model, and visual model, in this order (default)
        model, collision_model, visual_model = buildModelsFromUrdf(filename[, ...], geometry_types=[pin.GeometryType.COLLISION,pin.GeometryType.VISUAL])
        model, collision_model, visual_model = buildModelsFromUrdf(filename[, ...]) # same as above

        model, collision_model = buildModelsFromUrdf(filename[, ...], geometry_types=[pin.GeometryType.COLLISION]) # only load the model and the collision model
        model, collision_model = buildModelsFromUrdf(filename[, ...], geometry_types=pin.GeometryType.COLLISION)   # same as above
        model, visual_model    = buildModelsFromUrdf(filename[, ...], geometry_types=pin.GeometryType.VISUAL)      # only load the model and the visual model

        model = buildModelsFromUrdf(filename[, ...], geometry_types=[])  # equivalent to buildModelFromUrdf(filename[, root_joint])
    """
    if geometry_types is None:
        geometry_types = [pin.GeometryType.COLLISION, pin.GeometryType.VISUAL]
    if root_joint is None:
        model = pin.buildModelFromUrdf(filename)
    else:
        model = pin.buildModelFromUrdf(filename, root_joint)

    if verbose and not WITH_HPP_FCL and meshLoader is not None:
        print(
            "Info: MeshLoader is ignored. Pinocchio has not been compiled with HPP-FCL."
        )
    if verbose and not WITH_HPP_FCL_BINDINGS and meshLoader is not None:
        print(
            "Info: MeshLoader is ignored. The HPP-FCL Python bindings have not been installed."
        )
    if package_dirs is None:
        package_dirs = []

    lst = [model]

    if not hasattr(geometry_types, "__iter__"):
        geometry_types = [geometry_types]

    for geometry_type in geometry_types:
        if meshLoader is None or (not WITH_HPP_FCL and not WITH_HPP_FCL_BINDINGS):
            geom_model = pin.buildGeomFromUrdf(
                model, filename, geometry_type, package_dirs=package_dirs
            )
        else:
            geom_model = pin.buildGeomFromUrdf(
                model,
                filename,
                geometry_type,
                package_dirs=package_dirs,
                mesh_loader=meshLoader,
            )
        lst.append(geom_model)

    return tuple(lst)


########## TEST ZONE ##########################


class TestRobotLoader(unittest.TestCase):
    def test_complete_loader(self):
        import io

        robots_paths = [
            ["robot_simple_iso3D", "unittest_iso3D.txt"],
            ["robot_simple_iso6D", "unittest_iso6D.txt"],
        ]

        for rp in robots_paths:
            path = "robots/" + rp[0]
            m, cm, am, vm, collm = completeRobotLoader(path)
            joints_info = [
                (j.id, j.shortname(), j.idx_q, j.idx_v) for j in m.joints[1:]
            ]
            frames_info = [
                (f.name, f.inertia, f.parentJoint, f.parentFrame, f.type)
                for f in m.frames
            ]
            constraint_info = [
                (
                    cmi.name,
                    cmi.joint1_id,
                    cmi.joint2_id,
                    cmi.joint1_placement,
                    cmi.joint2_placement,
                    cmi.type,
                )
                for cmi in cm
            ]
            mot_info = [(am.idqfree, am.idqmot, am.idvfree, am.idvmot)]

            results = io.StringIO()
            results.write(
                "\n".join(f"{x[0]} {x[1]} {x[2]} {x[3]}" for x in joints_info)
            )
            results.write(
                "\n".join(f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]}" for x in frames_info)
            )
            results.write(
                "\n".join(
                    f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]} {x[5]}"
                    for x in constraint_info
                )
            )
            results.write("\n".join(f"{x[0]} {x[1]} {x[2]} {x[3]}" for x in mot_info))
            results.seek(0)

            # Ground truth is defined from a known good result
            with open("unittest/" + rp[1], "r") as truth:
                assert truth.read() == results.read()

    def test_generate_yaml(self):
        import io

        robots_paths = [
            ["robot_simple_iso3D", "unittest_iso3D_yaml.txt"],
            ["robot_simple_iso6D", "unittest_iso6D_yaml.txt"],
            ["robot_delta", "unittest_delta_yaml.txt"],
        ]

        for rp in robots_paths:
            path = "robots/" + rp[0]
            results = io.StringIO()
            generateYAML(path, file=results)
            results.seek(0)

            # Ground truth is defined from a known good result
            with open("unittest/" + rp[1], "r") as truth:
                assert truth.read() == results.read()


if __name__ == "__main__":
    unittest.main()
