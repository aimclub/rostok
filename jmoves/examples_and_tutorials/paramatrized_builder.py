import time
import numpy as np

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.mesh_builder.mesh_builder import jps_graph2pinocchio_meshes_robot
from auto_robot_design.description.utils import all_combinations_active_joints_n_actuator
from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder, MIT_CHEETAH_PARAMS_DICT, jps_graph2pinocchio_robot_3d_constraints
from auto_robot_design.generator.topologies.graph_manager_2l import get_preset_by_index
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
from auto_robot_design.vizualization.meshcat_utils import create_meshcat_vizualizer
from auto_robot_design.utils.configs import get_standard_builder, get_mesh_builder, get_standard_crag, get_standard_rewards

gm = get_preset_by_index(0)

graph = gm.get_graph(gm.generate_central_from_mutation_range())


thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]

density = MIT_CHEETAH_PARAMS_DICT["density"]
body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]


builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE,
                              density={"default": density, "G":body_density},
                              thickness={"default": thickness, "EE":0.033},
                              actuator={"default": MIT_CHEETAH_PARAMS_DICT["actuator"]},
                            #   size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
                              offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"]
)

# builder = get_mesh_builder(True)

# robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(graph, builder)
robot,__ = jps_graph2pinocchio_meshes_robot(graph, builder)

viz = create_meshcat_vizualizer(robot)
time.sleep(1)
viz.display(np.zeros(robot.model.nq))


# with open("parametrized_builder_test.urdf", "w") as f:
#     f.write(robo_urdf)

