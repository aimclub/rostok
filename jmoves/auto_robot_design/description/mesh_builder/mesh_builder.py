
from pathlib import Path
import os

from typing import Union
import networkx as nx
import numpy as np
import modern_robotics as mr
import odio_urdf as urdf
from auto_robot_design.description.actuators import Actuator, TMotor_AK80_9
from auto_robot_design.description.builder import BLUE_COLOR, DEFAULT_PARAMS_DICT, GREEN_COLOR, RED_COLOR, Builder, ParametrizedBuilder
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph, KinematicGraph
from auto_robot_design.description.mesh_builder.urdf_creater import MeshCreator, URDFMeshCreator
from auto_robot_design.pino_adapter.pino_adapter import get_pino_description_3d_constraints
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions


class MeshBuilder(ParametrizedBuilder):
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
        creator: URDFMeshCreator,
        mesh_creator: MeshCreator,
        mesh_path = None,
        density: Union[float, dict] = 2700 / 2.8,
        thickness: Union[float, dict] = 0.01,
        joint_damping: Union[float, dict] = 0.05,
        joint_friction: Union[float, dict] = 0,
        joint_limits: Union[dict, tuple] = (-np.pi, np.pi),
        size_ground: np.ndarray = np.zeros(3),
        offset_ground: np.ndarray = np.zeros(3),
        actuator: Union[Actuator, dict]=TMotor_AK80_9(),
    ) -> None:
        super().__init__(creator, density, thickness, joint_damping, joint_friction, joint_limits, size_ground, offset_ground, actuator)
        self.creater = creator
        self.mesh_creator: MeshCreator = mesh_creator
        self.mesh_path = mesh_path

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
        self.create_meshes(kinematic_graph)
        
        links = kinematic_graph.nodes()
        joints = dict(
            filter(lambda kv: len(kv[1]) > 0, kinematic_graph.joint2edge.items())
        )

        urdf_links = []
        urdf_joints = []
        for link in links:
            urdf_links.append(self.creater.create_link(link))
            # print(link.name, link.geometry.mass)

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

    def create_meshes(self, kinematic_graph, prefix=""):
        if self.mesh_path is None:
            dirpath = Path().parent.absolute()
            path_to_mesh = dirpath.joinpath("mesh")
            if not path_to_mesh.exists():
                os.mkdir(path_to_mesh)
            self.mesh_path = path_to_mesh
        
        self.creater.set_path_to_mesh(self.mesh_path)
        self.creater.set_prefix_name_mesh(prefix)
        
        links = kinematic_graph.nodes()
        for link in links:
            link_mesh = self.mesh_creator.build_link_mesh(link)
            link_mesh.apply_scale([1,1,1])
            name = prefix + link.name + ".stl"
            link_mesh.export(Path(self.mesh_path).joinpath(name))


def jps_graph2pinocchio_meshes_robot(
    graph: nx.Graph,
    builder: MeshBuilder
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
        if link.name == "G":
            link.geometry.color = RED_COLOR[0,:].tolist()
        elif link in kinematic_graph.main_branch.nodes():
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

    # with open("robot.urdf", "w") as f:
    #     f.write(robot.urdf())

    act_description, constraints_descriptions = get_pino_description_3d_constraints(
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


if __name__ == "__main__":
    import pinocchio as pin
    import meshcat
    import time
    from pinocchio.visualize import MeshcatVisualizer
    from auto_robot_design.description.mesh_builder.urdf_creater import URDFMeshCreator, MeshCreator, create_mesh_manipulator_base
    from auto_robot_design.description.builder import MIT_CHEETAH_PARAMS_DICT
    from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
    from auto_robot_design.description.actuators import TMotor_AK60_6
    
    thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]
    actuator = TMotor_AK60_6()#MIT_CHEETAH_PARAMS_DICT["actuator"]
    density = MIT_CHEETAH_PARAMS_DICT["density"]
    body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]
    
    # predined_mesh = {"G":"mesh/body.stl",
    #                  "EE":"mesh/wheel_small.stl"}

    predined_mesh = {"G":create_mesh_manipulator_base,
                     "EE":"mesh/uhvat.stl"}
    
    mesh_creator = MeshCreator(predined_mesh)
    urdf_creator = URDFMeshCreator()
    
    # builder = MeshBuilder(urdf_creator,
    #                     mesh_creator,
    #                     density={"default": density, "G": body_density},
    #                     thickness={"default": thickness},
    #                     actuator={"default": actuator},
    #                     size_ground=np.array(
    #                         MIT_CHEETAH_PARAMS_DICT["size_ground"]),
    #                     #offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"]
    #                     )
    
    builder = MeshBuilder(urdf_creator,
                        mesh_creator,
                        density={"default": density, "G": body_density},
                        thickness={"default": 0.01},
                        actuator={"default": actuator},
                        size_ground=np.array(
                            MIT_CHEETAH_PARAMS_DICT["size_ground"]),
                        )

    gm = get_preset_by_index_with_bounds(0)
    x_centre = gm.generate_random_from_mutation_range()
    graph_jp = gm.get_graph(x_centre)
    
    robot, __ = jps_graph2pinocchio_meshes_robot(graph_jp, builder)
    
    viz = MeshcatVisualizer(
    robot.model, robot.visual_model, robot.visual_model)
    viz.viewer = meshcat.Visualizer().open()
    time.sleep(2)
    viz.viewer["/Background"].set_property("visible", False)
    viz.viewer["/Grid"].set_property("visible", False)
    viz.viewer["/Axes"].set_property("visible", False)
    viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [
                                                                    0, -0.1, 0.5])
    viz.clean()
    viz.loadViewerModel()
    q = pin.neutral(robot.model)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    viz.display(q)
    time.sleep(10)