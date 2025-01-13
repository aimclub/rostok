import time
from copy import deepcopy
from pathlib import Path
import zipfile
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import streamlit as st
import streamlit.components.v1 as components
from apps.widjetdemo.streamlit_widgets.reward_descriptions.md_rawards import MD_REWARD_DESCRIPTION
from forward_init import (
    add_trajectory_to_vis,
    build_constant_objects,
    get_russian_reward_description,
)
from matplotlib.patches import Circle
from streamlit_widget_auxiliary import get_visualizer, send_graph_to_visualizer

from auto_robot_design.description.builder import (
    jps_graph2pinocchio_robot_3d_constraints,
)
from auto_robot_design.description.mesh_builder.mesh_builder import (
    jps_graph2pinocchio_meshes_robot,
)
from auto_robot_design.description.utils import draw_joint_point, draw_joint_point_widjet
from auto_robot_design.generator.topologies.bounds_preset import (
    get_preset_by_index_with_bounds,
)
from auto_robot_design.generator.topologies.graph_manager_2l import (
    plot_2d_bounds,
    plot_one_jp_bounds,
)
from auto_robot_design.motion_planning.bfs_ws import Workspace
from auto_robot_design.motion_planning.dataset_generator import set_up_reward_manager
from auto_robot_design.motion_planning.many_dataset_api import ManyDatasetAPI
from auto_robot_design.motion_planning.trajectory_ik_manager import TrajectoryIKManager
from auto_robot_design.pinokla.default_traj import (
    add_auxilary_points_to_trajectory,
    convert_x_y_to_6d_traj_xz,
)
from auto_robot_design.user_interface.check_in_ellips import (
    Ellipse,
    SnakePathFinder,
    check_points_in_ellips,
)

# constant objects
(
    graph_managers,
    optimization_builder,
    manipulation_builder,
    suspension_builder,
    crag,
    reward_dict,
) = build_constant_objects()
reward_description = get_russian_reward_description()
general_reward_keys = [
    "actuated_inertia_matrix",
    "z_imf",
    "manipulability",
    "min_manipulability",
    "min_force",
    "dexterity",
    "min_acceleration",
    "mean_heavy_lifting",
    "min_heavy_lifting",
]
suspension_reward_keys = [
    "z_imf",
    "min_acceleration",
    "mean_heavy_lifting",
    "min_heavy_lifting",
]
manipulator_reward_keys = [
    "manipulability",
    "min_manipulability",
    "min_force",
    "dexterity",
    "min_acceleration",
]
USER_KEY = 0 
WORKSPACE_COLORS_VIZUALIZATION_RED = "#dd2e44"
WORKSPACE_COLORS_VIZUALIZATION_YELLOW = "#fdcb58"
dataset_paths = [Path("./datasets/top_0"), Path("./datasets/top_1"),Path("./datasets/top_2"), Path("./datasets/top_3"),Path("./datasets/top_4"),Path("./datasets/top_5"),Path("./datasets/top_6"), Path("./datasets/top_7"), Path("./datasets/top_8")]
user_visualizer, user_vis_url = get_visualizer(USER_KEY)

st.title("Генерация механизмов по заданной рабочей области")
# starting stage
if not hasattr(st.session_state, "stage"):
    st.session_state.stage = "class_choice"
    st.session_state.gm = get_preset_by_index_with_bounds(-1)
    st.session_state.run_simulation_flag = False
    st.session_state["slider_version"] = 1
    path_to_robots = Path().parent.absolute().joinpath(f"robots/user_{USER_KEY}")
    if os.path.exists(path_to_robots):
        shutil.rmtree(path_to_robots)

def type_choice(t):
    if t == "free":
        st.session_state.type = "free"
        st.session_state.visualization_builder = optimization_builder
        st.session_state.reward_keys = general_reward_keys
    elif t == "suspension":
        st.session_state.type = "suspension"
        st.session_state.visualization_builder = suspension_builder
        st.session_state.reward_keys = suspension_reward_keys
    elif t == "manipulator":
        st.session_state.type = "manipulator"
        st.session_state.visualization_builder = manipulation_builder
        st.session_state.reward_keys = manipulator_reward_keys
    st.session_state.stage = "topology_choice"


# chose the class of optimization
if st.session_state.stage == "class_choice":
    some_text = r"""В данном сценарии происходит генерация механизмов по заданной рабочей области. Предлагается выбрать один из трёх типов задач для синтеза механизма:

- Абстрактный механизм;
- Подвеска колёсного робота;
- Робот-манипулятор.

Для каждого типа подготовлен свой набор критериев, используемых при генерации механизма и модель визуализации."""
    st.markdown(some_text)
    col_1, col_2, col_3 = st.columns(3, gap="medium",  vertical_alignment= 'bottom')
    with col_1:
        st.button(
            label="Абстрактный механизм",
            key="free",
            on_click=type_choice,
            args=["free"],
        )
        st.image("./apps/kin_struct.png")
    with col_2:
        st.button(
            label="Подвеска",
            key="suspension",
            on_click=type_choice,
            args=["suspension"],
        )
        st.image("./apps/hybrid_loco.png")
    with col_3:
        st.button(
            label="Манипулятор",
            key="manipulator",
            on_click=type_choice,
            args=["manipulator"],
        )
        st.image("./apps/manipulator.png")


def confirm_topology(topology_list, topology_mask):
    """Confirm the selected topology and move to the next stage."""
    # if only one topology is chosen, there is an option to choose the optimization ranges
    if len(topology_list) == 1:
        st.session_state.stage = "jp_ranges"
        st.session_state.gm = topology_list[0][1]
        graph = st.session_state.gm.get_graph(st.session_state.gm.generate_central_from_mutation_range())
        _,_=jps_graph2pinocchio_meshes_robot(graph, st.session_state.visualization_builder)
        st.session_state.gm_clone = deepcopy(st.session_state.gm)
        st.session_state.current_generator_dict = deepcopy(
            st.session_state.gm.generator_dict
        )
        # st.session_state.gm_clone = deepcopy(st.session_state.gm)
        st.session_state.datasets = [
            x for i,x in enumerate(dataset_paths) if topology_mask[i] is True
        ]
    else:
        for _, gm in topology_list:
            graph = gm.get_graph(gm.generate_central_from_mutation_range())
            _,_=jps_graph2pinocchio_meshes_robot(graph, st.session_state.visualization_builder)
        st.session_state.gm_clone = deepcopy(st.session_state.gm)
        st.session_state.stage = "ellipsoid"
        st.session_state.datasets = [
            x for i, x in enumerate(dataset_paths) if topology_mask[i] is True
        ]
    # create a deep copy of the graph manager for further updates
    st.session_state.topology_list = topology_list
    st.session_state.topology_mask = topology_mask


if st.session_state.stage == "topology_choice":
    some_text = """Предлагается выбор из девяти топологических структур механизмов.
В процессе генерации будут учитываться только выбранные топологические структуры.
Для визуализации выбора предлагаются примеры механизмов каждой структуры."""
    st.text(some_text)
    topology_name = lambda x:  f"Топология {x}"
    with st.sidebar:
        st.header("Выбор структуры")
        st.write(
            "При выборе только одной структуры доступна опция выбора границ для параметров генерации"
        )
        topology_mask = [0]*9
        for i, gm in enumerate(graph_managers.items()):
            if i == 0:
                topology_mask[i] = st.checkbox(label=topology_name(i), value=True) 
            else:
                topology_mask[i]=st.checkbox(label=topology_name(i), value=False)
        chosen_topology_list = [
            x for i, x in enumerate(graph_managers.items()) if topology_mask[i] is True
        ]

        if sum(topology_mask) > 0:
            st.button(
                label="Подтвердить выбор",
                key="confirm_topology",
                on_click=confirm_topology,
                args=[chosen_topology_list, topology_mask],
                type="primary"
            )

    plt.figure(figsize=(10, 10))
    for i in range(9):
        if i < len(chosen_topology_list):
            gm = chosen_topology_list[i][1]
            plt.subplot(3, 3, i + 1)
            gm.get_graph(gm.generate_central_from_mutation_range())
            draw_joint_point_widjet(gm.graph, labels=2, draw_legend=False)
            plt.title(topology_name(chosen_topology_list[i][0][-1]))
        else:
            plt.subplot(3, 3, i + 1)
            plt.axis("off")

    st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)


def confirm_ranges():
    """Confirm the selected ranges and move to the next stage."""
    st.session_state.stage = "ellipsoid"
    gm_clone = st.session_state.gm_clone
    for key, value in gm_clone.generator_dict.items():
        for i, values in enumerate(value.mutation_range):
            if values is None:
                continue
            if values[0] == values[1]:
                current_fp = gm.generator_dict[key].freeze_pos
                current_fp[i] = values[0]
                gm_clone.freeze_joint(key, current_fp)

    gm_clone.set_mutation_ranges()


def return_to_topology():
    """Return to the topology choice stage."""
    st.session_state.stage = "topology_choice"


def joint_choice():
    st.session_state.current_generator_dict = deepcopy(
        st.session_state.gm_clone.generator_dict
    )


# second stage
if st.session_state.stage == "jp_ranges":
    axis = ["x", "y", "z"]
    # form for optimization ranges. All changes affects the gm_clone and it should be used for optimization
    # initial nodes
    initial_generator_info = st.session_state.gm.generator_dict
    initial_mutation_ranges = st.session_state.gm.mutation_ranges
    gm = st.session_state.gm_clone
    generator_info = gm.generator_dict
    graph = gm.graph
    labels = {n: i for i, n in enumerate(graph.nodes())}
    with st.sidebar:
        # return button
        st.button(
            label="Назад к выбору топологии",
            key="return_to_topology",
            on_click=return_to_topology,
        )

        # set of joints that have mutation range in initial generator and get current jp and its index on the graph picture

        mutable_jps = [key[0] for key in initial_mutation_ranges.keys()]
        options = [(jp, idx) for jp, idx in labels.items() if jp in mutable_jps]
        current_jp = st.radio(
            label="Выбор сочленения для установки границ",
            options=options,
            index=0,
            format_func=lambda x: x[1],
            key="joint_choice",
            on_change=joint_choice,
        )
        # we can get current jp generator info in the cloned gm which contains all the changes
        current_generator_info = generator_info[current_jp[0]]
        for i, mut_range in enumerate(current_generator_info.mutation_range):
            if mut_range is None:
                continue
            # we can get mutation range from previous activation of the corresponding radio button
            left_value, right_value = st.session_state.current_generator_dict[
                current_jp[0]
            ].mutation_range[i]
            name = f"{labels[current_jp[0]]}_{axis[i]}"
            toggle_value = not left_value == right_value
            current_on = st.toggle(f"Отключить оптимизацию " + name, value=toggle_value)
            init_values = initial_generator_info[current_jp[0]].mutation_range[i]
            if current_on:
                mut_range = st.slider(
                    label=name,
                    min_value=init_values[0],
                    max_value=init_values[1],
                    value=(left_value, right_value),
                )
                generator_info[current_jp[0]].mutation_range[i] = mut_range
            else:
                current_value = st.number_input(
                    label="Insert a value",
                    value=(left_value + right_value) / 2,
                    key=name,
                    min_value=init_values[0],
                    max_value=init_values[1],
                )
                # if current_value < init_values[0]:
                #     current_value = init_values[0]
                # if current_value > init_values[1]:
                #     current_value = init_values[1]
                mut_range = (current_value, current_value)
                generator_info[current_jp[0]].mutation_range[i] = mut_range

        st.button(
            label="подтвердить диапазоны оптимизации",
            key="ranges_confirm",
            on_click=confirm_ranges,
        )
    # here should be some kind of visualization for ranges
    gm.set_mutation_ranges()
    plot_one_jp_bounds(gm, current_jp[0].name)
    center = gm.generate_central_from_mutation_range()
    graph = gm.get_graph(center)
    # here I can insert the visualization for jp bounds

    draw_joint_point_widjet(graph, labels=1, draw_legend=True, draw_lines=True)
    # here gm is a clone

    # plot_2d_bounds(gm)
    st.pyplot(plt.gcf(), clear_figure=True)
    # this way we set ranges after each step, but without freezing joints
    some_text = """Диапазоны оптимизации определяют границы пространства поиска механизмов в процессе 
оптимизации. x - горизонтальные координаты, z - вертикальные координаты.
Отключенные координаты не будут участвовать в оптимизации и будут иметь постоянные 
значения во всех механизмах."""
    st.text(some_text)
    # st.text("x - горизонтальные координаты, z - вертикальные координаты")


def reward_choice():
    st.session_state.stage = "rewards"

def reset_sliders():
    st.session_state["slider_version"] = st.session_state["slider_version"] + 1

if st.session_state.stage == "ellipsoid":
    st.markdown("""Задайте необходимую рабочую область для генерации механизмов.
    Рабочее пространство всех сгенерированных решений будет включать заданную область.
    Область задаётся в виде эллипса, определяемого своим центром, радиусами и углом.
                
:large_yellow_square: Желтая область - допустимая область для задния рабочего пространства.  
:large_red_square: Красная область - желаемая область рабочего пространства.""")
#     some_text = """Задайте необходимую рабочую область для генерации механизмов.
# Рабочее пространство всех сгенерированных решений будет включать заданную область.
# Область задаётся в виде эллипса, определяемого своим центром, радиусами и углом."""
    warning_text = """<!DOCTYPE html>
                    <html lang="ru-RU">
                    <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Warning Alert</title>
                    <style>
                        /* The warning alert message box */
                        .warning-alert {
                        padding: 25px;
                        background-color: #f0b33a; /* Yellow */
                        color: #000; /* Black text */
                        margin-bottom: 20px;
                        border-radius: 5px;
                        border: 1px solid #ccc;
                        }

                        /* Optional: Add a transition effect to fade out the alert */
                        .warning-alert {
                        opacity: 1;
                        transition: opacity 0.6s; /* 600ms to fade out */
                        }
                    </style>
                    </head>
                    <body>

                    <div class="alert warning-alert">
                    <strong>Внимаение! Граница эллипса выходит за область определения:</strong> \n
                    Пожалуйста, выберете параметры эллипса, входящий в область определения
                    </div>
                    </body>
                    </html>
                    """
    with st.sidebar:
        st.header("Выбор рабочего пространства")
        with st.form(key="ellipse"):
            x = st.slider(
                label="х координата центра", min_value=-0.3, max_value=0.3, value=0.0, key=f"ellips_params_x_{st.session_state['slider_version']}"
            )
            y = st.slider(
                label="z координата центра", min_value=-0.4, max_value=-0.2, value=-0.33, key=f"ellips_params_y_{st.session_state['slider_version']}"
            )
            x_rad = st.slider(
                label="х радиус", min_value=0.02, max_value=0.3, value=0.06, key=f"ellips_params_x_rad_{st.session_state['slider_version']}"
            )
            y_rad = st.slider(
                label="z радиус", min_value=0.02, max_value=0.3, value=0.05, key=f"ellips_params_y_rad_{st.session_state['slider_version']}"
            )
            angle = st.slider(label="наклон", min_value=0, max_value=180, value=33, key=f"ellips_angle_{st.session_state['slider_version']}")
            st.form_submit_button(label="Задать рабочее пространство")
        st.button(
        label="Вернуть эллипс к начальным параметрам", key="back_ellipse", on_click=reset_sliders,
    )
    st.session_state.ellipsoid_params = [x, y, x_rad, y_rad, angle]
    ellipse = Ellipse(np.array([x, y]), np.deg2rad(angle), np.array([x_rad, y_rad]))
    point_ellipse = ellipse.get_points()

    size_box_bound = np.array([0.5, 0.42])
    center_bound = np.array([0, -0.21])
    bounds = np.array(
        [
            [-size_box_bound[0] / 2 - 0.001, size_box_bound[0] / 2],
            [-size_box_bound[1] / 2, size_box_bound[1] / 2],
        ]
    )
    bounds[0, :] += center_bound[0]
    bounds[1, :] += center_bound[1]
    start_pos = np.array([0, -0.4])
    workspace_obj = Workspace(None, bounds, np.array([0.01, 0.01]))
    points = workspace_obj.points
    mask = check_points_in_ellips(points, ellipse, 0.02)
    rev_mask = np.array(1 - mask, dtype="bool")
    plt.figure(figsize=(10, 10))
    plt.scatter(points[rev_mask, :][:, 0], points[rev_mask, :][:, 1], s=2, marker="s", c=WORKSPACE_COLORS_VIZUALIZATION_RED)
    plt.scatter(points[mask, :][:, 0], points[mask, :][:, 1], s=2, marker="s", c=WORKSPACE_COLORS_VIZUALIZATION_YELLOW)

    # plt.plot(point_ellipse[0, :], point_ellipse[1, :], "g", linewidth=1)
    graph = st.session_state.gm.get_graph(
        st.session_state.gm.generate_central_from_mutation_range()
    )
    draw_joint_point(graph, labels=2, draw_legend=False, draw_lines=True, offset_lim=0.05)
    plt.gcf().set_size_inches(4, 4)
    st.pyplot(plt.gcf(), clear_figure=True)

    if not workspace_obj.point_in_bound(point_ellipse.T):
        st.html(warning_text)
    else:
        with st.sidebar:
            st.button(
            label="Перейти к целевой функции", key="rewards", on_click=reward_choice, type="primary"
        )


def generate():
    st.session_state.stage = "generate"


if st.session_state.stage == "rewards":
    some_text = """Укажите критерий оценки для отбора лучших механизмов.
Необходимо задать точку рассчёта критерия в рабочей области механизма.
Используйте боковую панель для установки точки расчёта."""
    st.text(some_text)
    x, y, x_rad, y_rad, angle = st.session_state.ellipsoid_params
    ellipse = Ellipse(np.array([x, y]), np.deg2rad(angle), np.array([x_rad, y_rad]))
    point_ellipse = ellipse.get_points()
    size_box_bound = np.array([0.5, 0.42])
    center_bound = np.array([0, -0.21])
    bounds = np.array(
        [
            [-size_box_bound[0] / 2 - 0.001, size_box_bound[0] / 2],
            [-size_box_bound[1] / 2, size_box_bound[1] / 2],
        ]
    )
    bounds[0, :] += center_bound[0]
    bounds[1, :] += center_bound[1]
    start_pos = np.array([0, -0.4])
    workspace_obj = Workspace(None, bounds, np.array([0.01, 0.01]))
    st.session_state.ws = workspace_obj
    points = workspace_obj.points
    mask = check_points_in_ellips(points, ellipse, 0.02)
    rev_mask = np.array(1 - mask, dtype="bool")
    plt.figure(figsize=(10, 10))
    plt.scatter(points[rev_mask, :][:, 0], points[rev_mask, :][:, 1], s=2, marker="s", c=WORKSPACE_COLORS_VIZUALIZATION_RED)
    plt.scatter(points[mask, :][:, 0], points[mask, :][:, 1], s=2, marker="s", c=WORKSPACE_COLORS_VIZUALIZATION_YELLOW)
    # plt.plot(point_ellipse[0, :], point_ellipse[1, :], "g", linewidth=1)
    with st.sidebar:
        st.header("Выбор точки вычисления")
        x_p = st.slider(
            label="х координата", min_value=-0.25, max_value=0.25, value=0.0
        )
        y_p = st.slider(
            label="z координата", min_value=-0.42, max_value=0.0, value=-0.3
        )
        if st.session_state.type == "free":
            reward_keys = general_reward_keys
            chosen_reward_idx = st.radio(
                label="Выбор целевой функции",
                options=range(len(reward_keys)),
                index=0,
                format_func=lambda x: reward_description[reward_keys[x]][0],
            )
            st.session_state.chosen_reward = reward_dict[reward_keys[chosen_reward_idx]]
            st.session_state.reward_name = reward_description[reward_keys[chosen_reward_idx]][0]
        if st.session_state.type == 'suspension':
            reward_keys = suspension_reward_keys
            chosen_reward_idx = st.radio(label='Выбор целевой функции', options=range(len(reward_keys)), index=0, format_func=lambda x: reward_description[reward_keys[x]][0])
            st.session_state.chosen_reward = reward_dict[reward_keys[chosen_reward_idx]]
            st.session_state.reward_name = reward_description[reward_keys[chosen_reward_idx]][0]
        if st.session_state.type == "manipulator":
            reward_keys = manipulator_reward_keys
            chosen_reward_idx = st.radio(label='Выбор целевой функции', options=range(len(reward_keys)), index=0, format_func=lambda x: reward_description[reward_keys[x]][0])
            st.session_state.chosen_reward = reward_dict[reward_keys[chosen_reward_idx]]
            st.session_state.reward_name = reward_description[reward_keys[chosen_reward_idx]][0]
        st.button(label="Сгенерировать механизмы", key="generate", on_click=generate, type="primary")
    st.session_state.point = [x_p, y_p]

    Drawing_colored_circle = Circle((x_p, y_p), radius=0.01, color="g")
    plt.gca().add_artist(Drawing_colored_circle)
    plt.gcf().set_size_inches(4, 4)
    plt.gca().axes.set_aspect(1)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.sidebar.button(label="Посмотреть подробное описание критериев", key="show_reward_description",on_click=lambda: st.session_state.__setitem__('stage', 'reward_description'))

def show_results():
    st.session_state.stage = "results"


def reset():
    delattr(st.session_state, "stage")


if st.session_state.stage == "generate":
    empt = st.empty()
    st.text("Начался процесс генерации. Подождите пару минут...")
    with empt:
        st.image(str(Path("./apps/widjetdemo/mechanical-wolf-running.gif").absolute()))
    dataset_api = ManyDatasetAPI(st.session_state.datasets)

    x, y, x_rad, y_rad, angle = st.session_state.ellipsoid_params
    ellipse = Ellipse(np.array([x, y]), np.deg2rad(angle), np.array([x_rad, y_rad]))
    index_list = dataset_api.get_indexes_cover_ellipse(ellipse)
    print(len(index_list))
    des_point = np.array(st.session_state.point)
    traj = np.array(
        add_auxilary_points_to_trajectory(([des_point[0]], [des_point[1]]))
    ).T
    dataset = dataset_api.datasets[0]
    graph = dataset.graph_manager.get_graph(
        dataset.graph_manager.generate_random_from_mutation_range()
    )
    robot, __ = jps_graph2pinocchio_robot_3d_constraints(graph, dataset.builder)
    traj_6d = robot.motion_space.get_6d_traj(traj)
    reward_manager = set_up_reward_manager(traj_6d, st.session_state.chosen_reward)
    sorted_indexes = dataset_api.sorted_indexes_by_reward(
        index_list, 10, reward_manager
    )
    if len(sorted_indexes) == 0:
        st.markdown(
            """Для заданного рабочего пространства и топологий не удалось найти решений, рекомендуется изменить требуемую рабочую область и/или топологии"""
        )
        st.button(label="Перезапуск сценария", on_click=reset)
    else:
        n = min(len(sorted_indexes), 10)
        graphs = []
        for topology_idx, index, value in sorted_indexes[:n]:
            gm = dataset_api.datasets[topology_idx].graph_manager
            x = dataset_api.datasets[topology_idx].get_design_parameters_by_indexes(
                [index]
            )
            graph = gm.get_graph(x[0])
            graphs.append((deepcopy(graph),value))
        st.session_state.graphs = graphs
        with empt:
            st.button(
                label="Результаты генерации", key="show_results", on_click=show_results
            )


def run_simulation(**kwargs):
    st.session_state.run_simulation_flag = True

def create_file(graph, user_key=0, id_robot=0):
    path_to_robots = Path().parent.absolute().joinpath(f"robots/user_{user_key}")
    if not os.path.exists(path_to_robots):
        os.makedirs(path_to_robots)
    zip_file_name = path_to_robots / f"robot_{id_robot}.zip"
    if os.path.exists(zip_file_name):
        return zip_file_name
    robot_urdf_str, yaml_out = jps_graph2pinocchio_robot_3d_constraints(graph, optimization_builder, True)
    path_to_urdf = path_to_robots / f"robot_{id_robot}.urdf"
    path_to_yaml = path_to_robots / f"robot_{id_robot}.yaml"
    with open(path_to_urdf, "w") as f:
        f.write(robot_urdf_str)
    with open(path_to_yaml, "w") as f:
        f.write(yaml_out)
    file_names = [f"robot_{id_robot}.urdf", f"robot_{id_robot}.yaml"]
    with zipfile.ZipFile(zip_file_name, 'w') as zip_object:
        # Add multiple files to the zip file
        for file_name in file_names:
            zip_object.write(path_to_robots / file_name, file_name)
    return zip_file_name


if st.session_state.stage == "results":
    description_text = r"""По заданной рабочей области и кртерию сгенерировано 10 механизмов. Механизмы ранжированные по убыванию значения критерия в указанной точке.
    Для верификации механизма можно провизуализировать движение по рабочему пространству. На схеме механизма показан траектория движения по рабочей области. 

Вы можете скачать URDF модель полученного механизма для дальнейшего использования. Данный виджет служит для первичной генерации кинематических структур,
вы можете использовать редакторы URDF для детализации робота и физические симуляторы для имитационного модеирования.
    """
    st.markdown(description_text)
    vis_builder = st.session_state.visualization_builder
    idx = st.select_slider(
        label="Лучшие по заданному критерию механизмы:",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        value=1,
        help="Перемещайте ползунок для выбора одного из 10 лучших дизайнов",
    )
    graph = st.session_state.graphs[idx - 1][0]
    reward = st.session_state.graphs[idx - 1][1]
    st.text(f"Значение критерия '{st.session_state.reward_name}' для дизайна {reward:.2f}")
    send_graph_to_visualizer(graph, user_visualizer, vis_builder)
    create_file(graph, USER_KEY, id_robot=idx )
    col_1, col_2 = st.columns(2, gap="medium", vertical_alignment= 'center')
    x, y, x_rad, y_rad, angle = st.session_state.ellipsoid_params
    ellipse = Ellipse(np.array([x, y]), np.deg2rad(angle), np.array([x_rad, y_rad]))
    points_on_ellps = ellipse.get_points(0.1).T
    ws = st.session_state.ws
    reach_ws_points = ws.points
    mask_ws_n_ellps = check_points_in_ellips(reach_ws_points, ellipse, 0.1)
    # plt.plot(points_on_ellps[:,0], points_on_ellps[:,1], "r", linewidth=3)
    # plt.scatter(pts[:,0],pts[:,1])
    snake_finder = SnakePathFinder(
        points_on_ellps[0], ellipse, coef_reg=np.prod(ws.resolution)
    )  # max_len_btw_pts= np.linalg.norm(dataset.workspace.resolution),
    traj = snake_finder.create_snake_traj(reach_ws_points[mask_ws_n_ellps, :])

    final_trajectory = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory((traj[:, 0], traj[:, 1]))
    )

    with col_1:
        st.header("Cхема механизма")
        draw_joint_point(graph, labels=2, draw_legend=False, draw_lines=True, offset_lim=0.05)
        rev_mask = np.array(1 - mask_ws_n_ellps, dtype="bool")
        # plt.plot(points_on_ellps[:, 0], points_on_ellps[:, 1], "g")
        plt.scatter(
            reach_ws_points[rev_mask, :][:, 0], reach_ws_points[rev_mask, :][:, 1], s=2, marker="s",
            c=WORKSPACE_COLORS_VIZUALIZATION_RED
        )
        plt.scatter(
            reach_ws_points[mask_ws_n_ellps, :][:, 0],
            reach_ws_points[mask_ws_n_ellps, :][:, 1],
            marker="s",
            c=WORKSPACE_COLORS_VIZUALIZATION_YELLOW
        )
        plt.plot(traj[:, 0], traj[:, 1], "r")
        Drawing_colored_circle = Circle(st.session_state.point, radius=0.01, color="r")
        plt.gca().add_artist(Drawing_colored_circle)
        plt.gcf().set_size_inches(7, 7)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Робот")
        add_trajectory_to_vis(user_visualizer, final_trajectory, step_balls=1)
        # add_trajectory_to_vis(get_visualizer(vis_builder, cam_pos=[0.09, 0.09, 0.09]), final_trajectory, step_balls=1, y_offset_balls=0.04)
        components.iframe(
            user_vis_url,
            width=310,
            height=310,
            scrolling=True,
        )
    st.button(
        label="Визуализация движения", key="run_simulation", on_click=run_simulation
    )
    # with open(path_to_urdf, "r") as f:
    #     st.download_button(
    #         "Скачать URDF описание робота",
    #         data=f,
    #         file_name="robot.urdf",
    #         mime="robot/urdf",

    #     )
    with open(create_file(graph, USER_KEY, id_robot=idx), "rb") as file:
        st.download_button(
            "Скачать URDF описание робота",
            data=file,
            file_name="robot_inv_description.zip",
            mime="robot/urdf",
        )
    if st.session_state.type == "free":
        if st.session_state.run_simulation_flag:
            ik_manager = TrajectoryIKManager()
            # fixed_robot, free_robot = jps_graph2pinocchio_robot(gm.graph, builder)
            fixed_robot, _ = jps_graph2pinocchio_robot_3d_constraints(
                graph, vis_builder
            )
            ik_manager.register_model(
                fixed_robot.model,
                fixed_robot.constraint_models,
                fixed_robot.visual_model,
            )
            ik_manager.set_solver("Closed_Loop_PI")
            # with st.status("simulation..."):
            _ = ik_manager.follow_trajectory(
                final_trajectory, viz=user_visualizer
            )
            time.sleep(1)
            user_visualizer.display(pin.neutral(fixed_robot.model))
            st.session_state.run_simulation_flag = False
    else:
        if st.session_state.run_simulation_flag:
            ik_manager = TrajectoryIKManager()
            # fixed_robot, free_robot = jps_graph2pinocchio_robot(gm.graph, builder)
            fixed_robot, _ = jps_graph2pinocchio_meshes_robot(graph, vis_builder)
            ik_manager.register_model(
                fixed_robot.model,
                fixed_robot.constraint_models,
                fixed_robot.visual_model,
            )
            ik_manager.set_solver("Closed_Loop_PI")
            # with st.status("simulation..."):
            _ = ik_manager.follow_trajectory(
                final_trajectory, viz=user_visualizer
            )
            time.sleep(1)
            user_visualizer.display(pin.neutral(fixed_robot.model))
            st.session_state.run_simulation_flag = False

if st.session_state.stage == 'reward_description':
    st.button(label="Вернуться к выбору критериев", key="return_to_criteria_calculation",on_click=lambda: st.session_state.__setitem__('stage', 'rewards'))
    st.markdown(MD_REWARD_DESCRIPTION)