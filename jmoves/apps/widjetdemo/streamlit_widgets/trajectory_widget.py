import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from matplotlib.patches import Circle
import streamlit as st
from auto_robot_design.pinokla.default_traj import (
    add_auxilary_points_to_trajectory, convert_x_y_to_6d_traj_xz,
    create_simple_step_trajectory, get_vertical_trajectory)

def set_vertical_trajectory():
    height = st.slider(
        label="длина вдоль Z", min_value=0.02*st.session_state.scale, max_value=0.3*st.session_state.scale, value=0.1*st.session_state.scale)
    x = st.slider(label="X координата начала", min_value=-0.3*st.session_state.scale,
                    max_value=0.3*st.session_state.scale, value=0.0*st.session_state.scale)
    z = st.slider(label="Z координата начала", min_value=-0.4*st.session_state.scale,
                    max_value=-0.2*st.session_state.scale, value=-0.3*st.session_state.scale)
    trajectory = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(get_vertical_trajectory(z, height, x, 100),initial_point=np.array([0,-0.4])*st.session_state.scale))
    return trajectory
    
def set_step_trajectory():
    start_x = st.slider(
        label="X координата начала", min_value=-0.3*st.session_state.scale, max_value=0.3*st.session_state.scale, value=-0.14*st.session_state.scale)
    start_z = st.slider(
        label="Z координата начала", min_value=-0.4*st.session_state.scale, max_value=-0.2*st.session_state.scale, value=-0.34*st.session_state.scale)
    height = st.slider(
        label="Высота", min_value=0.02*st.session_state.scale, max_value=0.3*st.session_state.scale, value=0.1*st.session_state.scale)
    width = st.slider(label="Ширина", min_value=0.1*st.session_state.scale,
                        max_value=0.6*st.session_state.scale, value=0.28*st.session_state.scale)
    trajectory = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(
            create_simple_step_trajectory(
                starting_point=[start_x, start_z],
                step_height=height,
                step_width=width,
                n_points=100,
            ),initial_point=np.array([0,-0.4])*st.session_state.scale
        )
    )
    return trajectory

def add_point_toTrajectory(X,Y):
    # in our current framework a trajectory is represented as two lists, one for x coordinates and one for z coordinates
    prev_trajectory = st.session_state.trajectory
    st.session_state.trajectory_history.append([X,Y])
    if prev_trajectory is None:
        st.session_state.trajectory = [np.array([X]),np.array([Y])]
    else:
        prev_x = prev_trajectory[0][-1]
        prev_y = prev_trajectory[1][-1]
        step_length = np.linalg.norm(np.array([X,Y])-np.array([prev_x, prev_y]))
        new_x = np.linspace(prev_x, X, int(step_length/0.005))[1:]
        new_y = np.linspace(prev_y, Y, int(step_length/0.005))[1:]
        prev_trajectory[0] = np.concatenate((prev_trajectory[0], new_x))
        prev_trajectory[1] = np.concatenate((prev_trajectory[1], new_y))
        st.session_state.trajectory = prev_trajectory

def clear_trajectory():
    st.session_state.trajectory = None
    st.session_state.trajectory_history = []

def user_trajectory(x_range,z_range,initial_point = np.array([0,-0.4]),):
    st.markdown('Добавляйте точки для построения траектории. Слайдеры задают положение следующей точки. траектория должна содержать минимум 2 точки')
    X = st.slider(label="X координата", min_value=x_range[0], max_value=x_range[1], value=initial_point[0], step=0.01, key="x")
    Y = st.slider(label="Z координата", min_value=z_range[0], max_value=z_range[1], value=initial_point[1], step=0.01, key="z")
    Drawing_colored_circle = Circle((X, Y), radius=0.005, color="g",zorder=4)
    plt.gca().add_artist(Drawing_colored_circle)
    st.button(label = "Добавить точку", key="add_point", help="Add point to trajectory", on_click=add_point_toTrajectory, args=(X,Y))
    st.button(label = "Очистить траекторию", key="clear_trajectory",  on_click=clear_trajectory)
    if len(st.session_state.trajectory_history) > 0:
        st.write("Опорные точки траектории:")
    for point in st.session_state.trajectory_history:
        st.write(f"X: {point[0]}, Z: {point[1]}")

    return st.session_state.trajectory
