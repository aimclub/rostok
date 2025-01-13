import time
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import streamlit as st
import streamlit.components.v1 as components


from forward_init import add_trajectory_to_vis, build_constant_objects, get_russian_reward_description
from streamlit_widget_auxiliary import get_visualizer, send_graph_to_visualizer

from auto_robot_design.description.builder import jps_graph2pinocchio_robot_3d_constraints
from auto_robot_design.description.mesh_builder.mesh_builder import (
    jps_graph2pinocchio_meshes_robot)
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.generator.topologies.bounds_preset import \
    get_preset_by_index_with_bounds
from auto_robot_design.motion_planning.bfs_ws import (
    BreadthFirstSearchPlanner, Workspace)
from auto_robot_design.motion_planning.trajectory_ik_manager import \
    TrajectoryIKManager
from auto_robot_design.pinokla.default_traj import (
    add_auxilary_points_to_trajectory, convert_x_y_to_6d_traj_xz,
    create_simple_step_trajectory, get_vertical_trajectory)
from auto_robot_design.utils.configs import get_standard_builder, get_mesh_builder, get_standard_crag, get_standard_rewards
# build and cache constant objects
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, MIT_CHEETAH_PARAMS_DICT
from auto_robot_design.optimization.rewards.reward_base import NotReacablePoints
from auto_robot_design.generator.topologies.graph_manager_2l import scale_graph_manager, scale_jp_graph, plot_one_jp_bounds


graph_managers, optimization_builder, _, _, crag, reward_dict = build_constant_objects()
reward_description = get_russian_reward_description()


st.title("Расчёт характеристик рычажных механизмов")
# create gm variable that will be used to store the current graph manager and set it to be update for a session
if 'gm' not in st.session_state:
    # the session variable for chosen topology, it gets a value after topology confirmation button is clicked
    st.session_state.stage = 'topology_choice'
    st.session_state.run_simulation_flag = False


def confirm_topology():
    st.session_state.stage = 'joint_point_choice'
    st.session_state.gm = deepcopy(graph_managers[st.session_state.topology_choice])
    st.session_state.gm.set_mutation_ranges()
    st.session_state.current_gm = st.session_state.gm
    st.session_state.jp_positions = st.session_state.gm.generate_central_from_mutation_range()
    st.session_state.slider_constants = deepcopy(st.session_state.jp_positions)
    st.session_state.scale = 1


# the radio button and confirm button are only visible until the topology is selected
if st.session_state.stage == 'topology_choice':
    st.markdown("""В данном сценарии предлагается выбрать одну из девяти структур рычажных механизмов и задать положение сочленений кинематической схемы. 
После этого будет рассчитано рабочее пространство кинематической схемы и предложены на выбор критерии, которые можно для неё рассчитать.

Первый шаг - выбор структуры механизма, выберите структуру при помощи кнопок на боковой панели. Для каждой структуры визуализируется пример графа и механизма.""")
    with st.sidebar:
        topology_choice = st.radio(label="Выбор структуры рычажного механизма:",
                 options=graph_managers.keys(), index=0, key='topology_choice')
        st.button(label='Подтвердить выбор структуры', key='confirm_topology',
                  on_click=confirm_topology)

    st.markdown(
    """Для управления инерциальными характеристиками механизма можно задать плотность и сечение элементов конструкции.""")
    density = st.number_input(label="Плотность [кг/м^3]", min_value=0.01, max_value=None,
                        value=MIT_CHEETAH_PARAMS_DICT["density"], step=10.0, key='density')
    thickness = st.number_input(label="Толщина [м]", min_value=0.01, max_value=None,
                          value=MIT_CHEETAH_PARAMS_DICT["thickness"], step=0.01, key='thickness')
    st.session_state.visualization_builder = get_mesh_builder(manipulation=True, thickness=thickness, density=density)
    st.session_state.gm = graph_managers[st.session_state.topology_choice]
    gm = st.session_state.gm
    values = gm.generate_central_from_mutation_range()
    graph = st.session_state.gm.get_graph(values)
    send_graph_to_visualizer(graph, st.session_state.visualization_builder)
    col_1, col_2 = st.columns([0.7, 0.3], gap="medium")
    with col_1:
        st.markdown("Граф выбранной структуры:")
        draw_joint_point(graph, labels=2, draw_lines=True)
        plt.gcf().set_size_inches(4, 4)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.markdown("Визуализация мехнизма:")
        components.iframe(get_visualizer(st.session_state.visualization_builder).viewer.url(), width=400,
                          height=400, scrolling=True)
    st.session_state.optimization_builder = get_standard_builder(thickness, density)

from auto_robot_design.optimization.rewards.inertia_rewards import MassReward
from auto_robot_design.pinokla.criterion_math import calculate_mass
def evaluate_construction(tolerance):
    """Calculate the workspace of the robot and display it"""
    st.session_state.stage = 'workspace_visualization'
    st.session_state.slider_constants = deepcopy(st.session_state.jp_positions)
    gm = st.session_state.current_gm
    graph = gm.graph
    fixed_robot, free_robot= jps_graph2pinocchio_robot_3d_constraints(
        graph, builder=st.session_state.optimization_builder)
    size_box_bound = np.array([0.5, 0.42])*st.session_state.scale
    center_bound = np.array([0, -0.21])*st.session_state.scale
    bounds = np.array(
        [
            [-size_box_bound[0] / 2 - 0.001, size_box_bound[0] / 2],
            [-size_box_bound[1] / 2, size_box_bound[1] / 2],
        ]
    )
    bounds[0, :] += center_bound[0]
    bounds[1, :] += center_bound[1]
    start_pos = np.array([0, -0.4])*st.session_state.scale
    q = np.zeros(fixed_robot.model.nq)
    workspace_obj = Workspace(fixed_robot, bounds, np.array([0.01*st.session_state.scale, 0.01*st.session_state.scale]))

    # tolerance = [0.004, 400]
    ws_bfs = BreadthFirstSearchPlanner(
        workspace_obj, 0, dexterous_tolerance=tolerance)
    workspace = ws_bfs.find_workspace(start_pos, q)
    st.session_state.workspace = workspace
    st.session_state.mass = calculate_mass(fixed_robot)


def slider_change():
    st.session_state.slider_constants = deepcopy(st.session_state.jp_positions)


def scale_change():
    graph_scale = st.session_state.scaler/st.session_state.scale
    st.session_state.scale = st.session_state.scaler
    tmp = deepcopy(st.session_state.jp_positions.copy())
    st.session_state.jp_positions = [i*graph_scale for i in tmp]
    st.session_state.slider_constants = deepcopy(st.session_state.jp_positions)
    current_gm = deepcopy(st.session_state.gm)
    current_gm = scale_graph_manager(current_gm, st.session_state.scale)
    current_gm.set_mutation_ranges()
    st.session_state.current_gm = current_gm


# choose the mechanism for optimization
if st.session_state.stage == 'joint_point_choice':
    st.write("""Установите необходимые положения для координат центров сочленений.
Каждое сочленение выбирается отдельно при помощи кнопок и слайдеров на боковой панели.
Если для сочленения нет слайдеров, то данное сочленение в соответствующей структуре является неизменяемым.""")
    st.markdown("""Каждое сочленение относится к одному из четырёх типов:  
                1. Неподвижное сочленение - неизменяемое положение.  
                2. Cочленение в абсолютных координатах - положение задаётся в абсолютной системе координат в метрах.  
                3. Сочленение в относительных координатах - положение задаётся относительно другого сочленения в метрах.  
                4. Сочленени задаваемое относительно звена - положение задаётся относительно центра звена в процентах от длины звена.  
                Для каждого сочленения на боковой панели указан его тип.""")
    gm = st.session_state.current_gm
    mut_ranges = gm.mutation_ranges
    graph = gm.graph
    labels = {n: i for i, n in enumerate(graph.nodes())}
    with st.sidebar:
        st.button(label='Вернуться к выбору топологии', key='return_to_topology_choice',
                  on_click=lambda: st.session_state.__setitem__('stage', 'topology_choice'))
    with st.sidebar:
        st.header('Выбор положений сочленений')
        jp_label = st.radio(label='Сочленение', options=labels.values(
        ), index=0, key='joint_point_choice', horizontal=True, on_change=slider_change)
        jp = list(labels.keys())[jp_label]
        if st.session_state.gm.generator_dict[list(labels.keys())[jp_label]].mutation_type.value == 1:
            if None in st.session_state.gm.generator_dict[list(labels.keys())[jp_label]].freeze_pos:
                st.write("Тип сочленения: Сочленение в абсолютных координатах")
            else:
                st.write("Тип сочленения: Неподвижное сочленение")
        if st.session_state.gm.generator_dict[list(labels.keys())[jp_label]].mutation_type.value == 2:
            st.write("Тип сочленения: Сочленение в относительных координатах")
            st.write("координаты относительно сочленения: "+str(
                labels[st.session_state.gm.generator_dict[list(labels.keys())[jp_label]].relative_to]))
        if st.session_state.gm.generator_dict[list(labels.keys())[jp_label]].mutation_type.value == 3:
            st.write("Тип сочленения: Сочленение задаваемое относительно звена")
            st.write("координаты относительно звена: "+str(labels[st.session_state.gm.generator_dict[list(labels.keys())[
                     jp_label]].relative_to[0]])+'->'+str(labels[st.session_state.gm.generator_dict[list(labels.keys())[jp_label]].relative_to[1]]))

        chosen_range = gm.generator_dict[list(labels.keys())[jp_label]].mutation_range
        for i, tpl in enumerate(mut_ranges.items()):
            key, value = tpl
            if key[0] == list(labels.keys())[jp_label]:
                slider = st.slider(
                    label=str(labels[key[0]])+'_'+key[1], min_value=value[0], max_value=value[1], value=st.session_state.slider_constants[i],
                    key="slider_"+str(labels[key[0]])+'_'+key[1])
                st.session_state.jp_positions[i] = slider
        lower = 0
        upper = np.inf
        # lower_toggle = st.toggle(label='Задать нижний предел манипулируемости', value = False,key='lower_toggle')
        # if lower_toggle:
        #     lower = st.slider(label="нижний предел манипулируемости",min_value=0.0001, max_value=0.001,value=0.0001,step=0.0001,key='lower', format="%f")
        # upper_toggle = st.toggle(label = 'Задать верхний предел манипулируемости', value = False, key='upper_toggle')
        # if upper_toggle:
        #     upper = st.slider(label="верхний предел манипулируемости",min_value=10, max_value=100,value=100,step=10, key='upper')
    st.markdown("""Высоту механизма можно настроить при помощи изменения общего масштаба механизма.""")
    st.number_input(label="Масштаб", min_value=0.1, max_value=None,value=1.0, step=0.1, key='scaler', on_change=scale_change)
    with st.sidebar:
        
        st.button(label="Рассчитать рабочее пространство",
                  on_click=evaluate_construction, key="get_workspace", args=[[lower, upper]])
    # draw the graph
    graph = gm.get_graph(st.session_state.jp_positions)
    send_graph_to_visualizer(graph, st.session_state.visualization_builder)
    draw_joint_point(graph, labels=1, draw_lines=True)
    plot_one_jp_bounds(gm, jp.name)

    plt.gcf().set_size_inches(4, 4)
    st.pyplot(plt.gcf(), clear_figure=True)
    # link lengths calculations
    with st.sidebar:
        for edge in graph.edges():
            vector = edge[0].r - edge[1].r
            st.write(
                f"Ребро {labels[edge[0]]}->{labels[edge[1]]} имеет длину {np.linalg.norm(vector):.3f} метров")


def to_trajectory_choice():
    st.session_state.stage = 'trajectory_choice'


def run_simulation():
    st.session_state.run_simulation_flag = True


def calculate_and_display_rewards(trajectory, reward_mask):
    gm = st.session_state.current_gm
    fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(
        gm.graph, st.session_state.optimization_builder)
    point_criteria_vector, trajectory_criteria, res_dict_fixed = crag.get_criteria_data(
        fixed_robot, free_robot, trajectory, viz=None)
    some_text = """Критерии представлены в виде поточечных значений вдоль траектории."""
    st.text(some_text)
    col_1, col_2 = st.columns([0.5, 0.5], gap="small")
    counter = 0
    try:
        for i, reward in enumerate(reward_dict.items()):
            if counter % 2 == 0:
                col = col_1
            else:
                col = col_2
            with col:
                if reward_mask[i]:
                    
                        calculate_result = reward[1].calculate(
                            point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=st.session_state.optimization_builder.actuator['default'])
                        # st.text(reward_description[reward[0]][0]+":\n   " )
                        reward_vector = np.array(calculate_result[1])
                        # plt.gcf().set_figheight(3)
                        # plt.gcf().set_figwidth(3)
                        plt.plot(reward_vector)
                        plt.xticks(fontsize=10)
                        plt.yticks(fontsize=10)
                        plt.xlabel('шаг траектории', fontsize=12)
                        plt.ylabel('значение критерия на шаге', fontsize=12)
                        plt.title(reward_description[reward[0]][0], fontsize=12)
                        plt.legend(
                            [f'Итоговое значение критерия: {calculate_result[0]:.2f}'], fontsize=12)
                        st.pyplot(plt.gcf(), clear_figure=True,
                                use_container_width=True)
                        counter += 1
    except NotReacablePoints:
        st.text_area(
            label="", value="Траектория содержит точки за пределами рабочего пространства. Для рассчёта критериев укажите траекторию внутри рабочей области.")



def create_file(graph):
    robot_urdf_str = jps_graph2pinocchio_robot_3d_constraints(
        graph, st.session_state.optimization_builder, True)
    path_to_robots = Path().parent.absolute().joinpath("robots")
    path_to_urdf = path_to_robots / "robot_forward.urdf"
    return robot_urdf_str


if st.session_state.stage == 'workspace_visualization':
    st.markdown("""Рабочее пространство изображено совместно с графовым представлением механизма.   
:large_yellow_square: Жёлтая область - рабочее пространство механизма.  
:large_red_square: Красные область - недостижимые точки.  
Для выбранной кинематической схемы можно рассчитать набор критериев. Для успешного вычисления критериев необходимо задать желаемую траекторию лежащую внутри рабочего пространства механизма.""")
    st.text("Выберите траекторию и критерии при помощи конопок на боковой панели:")
    gm = st.session_state.current_gm
    graph = gm.graph
    # points = st.session_state.points
    points = st.session_state.workspace.points
    workspace = st.session_state.workspace
    x = points[:, 0]
    y = points[:, 1]
    values = workspace.reachabilty_mask.flatten()
    x_0 = x[values == 0]
    y_0 = y[values == 0]
    x_1 = x[values == 1]
    y_1 = y[values == 1]
    # # Plot the points
    plt.plot(x_0, y_0, "sr", markersize=3, label = "недостижимые точки",zorder=1)
    plt.legend()
    plt.plot(x_1, y_1, "sy", markersize=3, label = "достижимые точки",zorder=1)
    plt.legend()
    # trajectory setting script
    trajectory = None
    with st.sidebar:
        st.button(label="Вернуться к выбору механизма", key="return_to_joint_point_choice",
                  on_click=lambda: st.session_state.__setitem__('stage', 'joint_point_choice'))
        trajectory_type = st.radio(label='Выберите тип траектории', options=[
            "вертикальная", "шаг"], index=None, key="trajectory_type")
        if trajectory_type == "вертикальная":
            height = st.slider(
                label="высота", min_value=0.02*st.session_state.scale, max_value=0.3*st.session_state.scale, value=0.1*st.session_state.scale)
            x = st.slider(label="x", min_value=-0.3*st.session_state.scale,
                          max_value=0.3*st.session_state.scale, value=0.0*st.session_state.scale)
            z = st.slider(label="z", min_value=-0.4*st.session_state.scale,
                          max_value=-0.2*st.session_state.scale, value=-0.3*st.session_state.scale)
            trajectory = convert_x_y_to_6d_traj_xz(
                *add_auxilary_points_to_trajectory(get_vertical_trajectory(z, height, x, 100),initial_point=np.array([0,-0.4])*st.session_state.scale))
        if trajectory_type == "шаг":
            start_x = st.slider(
                label="начало_x", min_value=-0.3*st.session_state.scale, max_value=0.3*st.session_state.scale, value=-0.14*st.session_state.scale)
            start_z = st.slider(
                label="начало_z", min_value=-0.4*st.session_state.scale, max_value=-0.2*st.session_state.scale, value=-0.34*st.session_state.scale)
            height = st.slider(
                label="высота", min_value=0.02*st.session_state.scale, max_value=0.3*st.session_state.scale, value=0.1*st.session_state.scale)
            width = st.slider(label="width", min_value=0.1*st.session_state.scale,
                              max_value=0.6*st.session_state.scale, value=0.2*st.session_state.scale)
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
        if trajectory_type is not None:
            st.button(label="Симуляция движения по траектории", key="run_simulation",
                      on_click=run_simulation)
            with st.form(key="rewards"):
                st.header("Критерии")
                reward_mask = []
                for key, reward in reward_dict.items():
                    reward_mask.append(st.checkbox(
                        label=reward_description[key][0], value=False, help=reward_description[key][1]))
                cr = st.form_submit_button(
                    "Рассчитать значения выбранных критериев")

    col_1, col_2 = st.columns([0.7, 0.3], gap="medium")
    with col_1:
        draw_joint_point(graph, labels=2, draw_legend=True, draw_lines=True)
        plt.gcf().set_size_inches(6, 6)
        if trajectory_type is not None:
            # plt.plot(trajectory[50:, 0], trajectory[50:, 2], 'green', markersize=2)
            plt.plot(trajectory[:, 0], trajectory[:, 2], 'green', markersize=2)
        plt.legend(loc="lower left", bbox_to_anchor=(0, 1.02))
        plt.text(0,0.95,'Масса механизма: '+f"{st.session_state.mass:.3f}", transform=plt.gca().transAxes, fontsize=12)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.text("\n ")
        st.text("\n ")
        st.text("\n ")
        st.text("\n ")
        st.text("\n ")
        st.text("\n ")
        st.text("\n ")
        if trajectory_type is not None:
            add_trajectory_to_vis(get_visualizer(
                st.session_state.visualization_builder), trajectory[50:])
        components.iframe(get_visualizer(st.session_state.visualization_builder).viewer.url(), width=400,
                          height=400, scrolling=True)

    if trajectory_type is not None:
        if st.session_state.run_simulation_flag or cr:
            calculate_and_display_rewards(trajectory, reward_mask)

    st.download_button(
        "Скачать URDF описание робота",
        data=create_file(graph),
        file_name="robot_forward.urdf",
        mime="robot/urdf",
    )
    st.markdown(r"""||.j4||Вы можете скачать URDF модель полученного механизма для дальнейшего использования. Данный виджет служит для первичной оценки кинематических структур, вы можете использовать редакторы URDF для более точной настройки параметров и физические симуляторы для имитационного модеирования.""")
    # st.button(label="Посмотреть подробное описание критериев", key="show_reward_description",on_click=lambda: st.session_state.__setitem__('stage', 'reward_description'))
    if st.session_state.run_simulation_flag:
        ik_manager = TrajectoryIKManager()
        fixed_robot, _ = jps_graph2pinocchio_meshes_robot(
            graph, st.session_state.visualization_builder)
        ik_manager.register_model(
            fixed_robot.model, fixed_robot.constraint_models, fixed_robot.visual_model
        )
        ik_manager.set_solver("Closed_Loop_PI")
        _ = ik_manager.follow_trajectory(
            trajectory, viz=get_visualizer(st.session_state.visualization_builder)
        )
        time.sleep(1)
        get_visualizer(st.session_state.visualization_builder).display(
            pin.neutral(fixed_robot.model))
        st.session_state.run_simulation_flag = False

# if st.session_state.stage == 'reward_description':
#     st.button(label="Вернуться к расчёту критериев", key="return_to_criteria_calculation",on_click=lambda: st.session_state.__setitem__('stage', 'workspace_visualization'))
#     # st.markdown(MD_REWARD_DESCRIPTION)