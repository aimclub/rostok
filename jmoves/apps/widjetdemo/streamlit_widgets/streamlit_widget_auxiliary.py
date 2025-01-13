import streamlit as st
import streamlit.components.v1 as components
import time
from auto_robot_design.motion_planning.trajectory_ik_manager import TrajectoryIKManager
import meshcat
import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
from auto_robot_design.description.mesh_builder.mesh_builder import (
    MeshBuilder,
    jps_graph2pinocchio_meshes_robot,
)
import dill
from auto_robot_design.description.utils import draw_joint_point, draw_joint_point_widjet
from auto_robot_design.utils.configs import get_standard_builder, get_mesh_builder, get_standard_crag, get_standard_rewards
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds


@st.cache_resource
def get_visualizer(user_key=0, **camera_params):
    _gm = get_preset_by_index_with_bounds(8)
    base_graph = _gm.get_graph(_gm.generate_central_from_mutation_range())
    builder = get_mesh_builder(manipulation=False)
    fixed_robot, _ = jps_graph2pinocchio_meshes_robot(base_graph, builder)
    # create a pinocchio visualizer object with current value of a robot
    visualizer = MeshcatVisualizer(
        fixed_robot.model, fixed_robot.visual_model, fixed_robot.visual_model
    )
    # create and setup a meshcat visualizer
    visualizer.viewer = meshcat.Visualizer()
    # visualizer.viewer["/Background"].set_property("visible", False)
    visualizer.viewer["/Grid"].set_property("visible", False)
    visualizer.viewer["/Axes"].set_property("visible", False)
    visualizer.viewer["/Cameras/default/rotated/<object>"].set_property(
        "position", camera_params.get("cam_pos", [0, 0.0, 1])
    )
    # load a model to the visualizer and set it into the neutral position
    visualizer.clean()
    visualizer.loadViewerModel()
    visualizer.display(pin.neutral(fixed_robot.model))

    return visualizer, visualizer.viewer.url()

def send_graph_to_visualizer(graph, visualizer, visualization_builder):
    fixed_robot, _ = jps_graph2pinocchio_meshes_robot(
        graph, visualization_builder)
    visualizer.model = fixed_robot.model
    visualizer.collision = fixed_robot.visual_model
    visualizer.visual_model = fixed_robot.visual_model
    visualizer.rebuildData()
    visualizer.clean()
    visualizer.loadViewerModel()
    visualizer.display(pin.neutral(fixed_robot.model))

def send_robot_to_visualizer(robot, visualizer):
    visualizer.model = robot.model
    visualizer.collision = robot.visual_model
    visualizer.visual_model = robot.visual_model
    visualizer.rebuildData()
    visualizer.clean()
    visualizer.loadViewerModel()
    visualizer.display(pin.neutral(robot.model))

def graph_mesh_visualization(graph, visualizer, url,**kwargs):
    send_graph_to_visualizer(graph, visualizer, st.session_state.visualization_builder)
    col_1, col_2 = st.columns([0.7, 0.3], gap="medium")
    with col_1:
        st.markdown("Граф выбранной структуры:")
        draw_joint_point_widjet(graph,**kwargs)
        plt.gcf().set_size_inches(4, 4)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.markdown("Визуализация механизма:")
        components.iframe(url, width=400,
                          height=400, scrolling=True)
    st.markdown("Используйте мышь для вращения, сдвига и масштабирования модели.")

def robot_move_visualization(fixed_robot, trajectory, visualizer):
    ik_manager = TrajectoryIKManager()
    ik_manager.register_model(
        fixed_robot.model, fixed_robot.constraint_models, fixed_robot.visual_model
    )
    ik_manager.set_solver("Closed_Loop_PI")
    _ = ik_manager.follow_trajectory(
        trajectory, viz=visualizer
    )
    time.sleep(1)
    visualizer.display(pin.neutral(fixed_robot.model))
    st.session_state.run_simulation_flag = False