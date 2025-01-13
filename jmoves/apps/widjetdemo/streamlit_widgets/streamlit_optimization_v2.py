import subprocess
import time
from copy import deepcopy
from pathlib import Path
import os 
import shutil
import zipfile
import dill
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import streamlit as st
import streamlit.components.v1 as components
from forward_init import (add_trajectory_to_vis, build_constant_objects,
                          get_russian_reward_description)
from pymoo.decomposition.asf import ASF
from streamlit_widget_auxiliary import get_visualizer, send_graph_to_visualizer, graph_mesh_visualization, robot_move_visualization
from pathlib import Path
from auto_robot_design.description.builder import (jps_graph2pinocchio_robot_3d_constraints)
from auto_robot_design.description.mesh_builder.mesh_builder import jps_graph2pinocchio_meshes_robot
from auto_robot_design.description.utils import draw_joint_point, draw_joint_point_widjet
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
from auto_robot_design.generator.topologies.graph_manager_2l import plot_one_jp_bounds, scale_graph_manager
from auto_robot_design.motion_planning.trajectory_ik_manager import \
    TrajectoryIKManager
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.problems import (MultiCriteriaProblem,
                                                     SingleCriterionProblem)
from auto_robot_design.optimization.rewards.reward_base import (
    PositioningConstrain, PositioningErrorCalculator, RewardManager)
from auto_robot_design.optimization.saver import load_checkpoint
from auto_robot_design.pinokla.default_traj import (
    add_auxilary_points_to_trajectory, convert_x_y_to_6d_traj_xz,
    create_simple_step_trajectory, get_vertical_trajectory)
from auto_robot_design.utils.configs import get_standard_builder, get_mesh_builder, get_standard_crag, get_standard_rewards
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, MIT_CHEETAH_PARAMS_DICT
from auto_robot_design.optimization.rewards.reward_base import NotReacablePoints
from apps.widjetdemo.streamlit_widgets.reward_descriptions.md_rawards import MD_REWARD_DESCRIPTION
from widget_html_tricks import ChangeWidgetFontSize, font_size
from apps.widjetdemo.streamlit_widgets.trajectory_widget import set_step_trajectory, set_vertical_trajectory, user_trajectory


# st.set_page_config(layout = "wide", initial_sidebar_state = "expanded")
graph_managers, default_optimization_builder, default_mesh_builder,_, crag, reward_dict = build_constant_objects()
reward_description = get_russian_reward_description()
default_x_range = np.array([-0.3,0.3])
default_z_range = np.array([-0.42,0.0])
user_key=1
axis = ['x', 'y', 'z']
font_size(20)
user_visualizer, user_vis_url = get_visualizer(user_key)

def show_loaded_results(dir="./results/optimization_widget/current_results (copy_2)"):
    st.session_state.results_saved = True
    st.session_state.stage = "results"
    st.session_state.results_exist = True
    st.session_state.run_simulation_flag = False
    selected_directory = dir
    with open(Path(dir).absolute().joinpath("out.txt"),'r')as file:
        n_obj = int(file.readline())
    st.session_state.n_obj = n_obj
    if n_obj == 1:
        problem = SingleCriterionProblem.load(selected_directory)
        checkpoint = load_checkpoint(selected_directory)
        optimizer = PymooOptimizer(problem, checkpoint)
        optimizer.load_history(selected_directory)
        res = optimizer.run()
        st.session_state.optimizer = optimizer
        st.session_state.problem = problem
        st.session_state.res = res
        st.session_state.reward_manager=problem.rewards_and_trajectories
        st.session_state.optimization_builder = problem.builder
        st.session_state.visualization_builder = default_mesh_builder
        st.session_state.gm = problem.graph_manager
    if n_obj >= 2:
        problem = MultiCriteriaProblem.load(selected_directory)
        checkpoint = load_checkpoint(selected_directory)
        optimizer = PymooOptimizer(problem, checkpoint)
        optimizer.load_history(selected_directory)
        res = optimizer.run()
        st.session_state.optimizer = optimizer
        st.session_state.problem = problem
        st.session_state.res = res
        st.session_state.reward_manager=problem.rewards_and_trajectories
        st.session_state.optimization_builder = problem.builder
        st.session_state.visualization_builder = default_mesh_builder
        st.session_state.gm = problem.graph_manager
        


@st.dialog("Выберите папку с результатами оптимизации")
def load_results():
    st.session_state.load_results = True
    options = [f for f in Path(f"./results/optimization_widget/user_{user_key}").iterdir() if (f.is_dir()and("current_results" not in f.name and "buffer" not in f.name))]
    path = st.radio(label="Выберите папку с результатами оптимизации", options=options, index=0, key='results_dir',format_func=lambda x:x.name)
    if st.button("Загрузить результаты"):
        show_loaded_results(path)
        st.rerun()


st.title("Оптимизация рычажных механизмов")

# gm is the first value that gets set. List of all values that should be update for each session
if 'stage' not in st.session_state:
    st.session_state.user_key = user_key
    st.session_state.reward_manager = RewardManager(crag=crag)
    error_calculator = PositioningErrorCalculator(jacobian_key="Manip_Jacobian")
    st.session_state.soft_constraint = PositioningConstrain(
        error_calculator=error_calculator, points=[])
    st.session_state.stage = "topology_choice"
    st.session_state.gm_clone = None
    
    # states that I need to create trajectory groups and associated rewards
    st.session_state.trajectory_idx = 0
    st.session_state.trajectory_groups = []
    st.session_state.trajectory_buffer = {}
    st.session_state.opt_rewards_dict = {}

    st.session_state.run_simulation_flag = False
    st.session_state.results_exist = False

    path_to_robots = Path().parent.absolute().joinpath(f"robots/user_{user_key}")
    if os.path.exists(path_to_robots):
        shutil.rmtree(path_to_robots)


def confirm_topology():
    """Confirm the selected topology and move to the next stage."""
    #next stage
    st.session_state.stage = "ranges_choice"
    # create a deep copy of the graph manager for further updates
    # We need three instances of the gm. Initial gm has initial parameters, gm_clone has scaled parameters, 
    # and current_gm has current parameters that will be used in optimization
    st.session_state.gm.set_mutation_ranges()
    st.session_state.gm_clone = deepcopy(st.session_state.gm)
    st.session_state.current_gm = deepcopy(st.session_state.gm)
    st.session_state.current_generator_dict = deepcopy(st.session_state.gm.generator_dict)
    st.session_state.scale = 1

# def topology_choice():
#     """Update the graph manager based on the selected topology."""
#     st.session_state.gm = graph_managers[st.session_state.topology_choice]

# the radio button and confirm button are only visible until the topology is selected
if st.session_state.stage == "topology_choice":
    some_text = """<p class="big-font"> Данный сценарий предназначен для оптимизации рычажных механизмов.
Первый шаг - выбор структуры механизма для оптимизации. Структура определяет звенья 
и сочленения механизма. Рёбра графа соответствуют твердотельным звеньям, 
а вершины - сочленениям и концевому эффектору.
Предлагается выбор из девяти структур, основанных на двухзвенной главной цепи.</p>"""
    st.markdown(some_text, unsafe_allow_html=True)
    with st.sidebar:
        st.radio(label="Выбор структруры для оптимизации:", options=graph_managers.keys(),
                 index=0, key='topology_choice')
        st.button(label='Подтвердить выбор структуры', key='confirm_topology',
                  on_click=confirm_topology,type='primary')
        ChangeWidgetFontSize("Выбор структруры для оптимизации:", "16px")
    st.markdown(
    """<p class="big-font">Для управления инерциальными характеристиками механизма можно задать плотность и сечение элементов конструкции.</p>""", unsafe_allow_html=True)
    density = st.slider(label="Плотность [кг/м^3]", min_value=250, max_value=8000,
                        value=int(MIT_CHEETAH_PARAMS_DICT["density"]), step=50, key='density')
    thickness = st.slider(label="Толщина [м]", min_value=0.01, max_value=0.1,
                          value=MIT_CHEETAH_PARAMS_DICT["thickness"], step=0.01, key='thickness')

    st.session_state.visualization_builder = get_mesh_builder(thickness=thickness, density=density)
    st.session_state.optimization_builder = get_standard_builder(thickness, density)
    st.session_state.gm = graph_managers[st.session_state.topology_choice]
    gm = st.session_state.gm
    values = gm.generate_central_from_mutation_range()
    graph = st.session_state.gm.get_graph(values)
    graph_mesh_visualization(graph,user_visualizer,user_vis_url, labels=2, draw_lines=True, draw_legend=False)
    if Path(f"./results/optimization_widget/user_{user_key}").exists():
        options = [f for f in Path(f"./results/optimization_widget/user_{user_key}").iterdir() if (f.is_dir()and("current_results" not in f.name and "buffer" not in f.name))]
        if len(options) > 0:
            if st.button("Загрузить одну из прошлых оптимизаций"):
                load_results()
    ChangeWidgetFontSize('Подтвердить выбор структуры', "16px")
    ChangeWidgetFontSize("Плотность [кг/м^3]", "16px")
    ChangeWidgetFontSize("Толщина [м]", "16px")

def confirm_ranges():
    """Confirm the selected ranges and move to the next stage."""
    st.session_state.stage = "trajectory_choice"
    current_gm = st.session_state.current_gm
    for key, value in current_gm.generator_dict.items():
        for i, values in enumerate(value.mutation_range):
            if values is None:
                continue
            if values[0] == values[1]:
                current_fp = gm.generator_dict[key].freeze_pos
                current_fp[i] = values[0]
                current_gm.freeze_joint(key, current_fp)

    current_gm.set_mutation_ranges()
    # this object is used only for user trajectory
    st.session_state.trajectory = None
    st.session_state.trajectory_history = []


def return_to_topology():
    """Return to the topology choice stage."""
    st.session_state.__delattr__("stage")

def joint_choice():
    st.session_state.current_generator_dict = deepcopy(st.session_state.current_gm.generator_dict)

def scale_change():
    graph_scale = st.session_state.scaler/st.session_state.scale
    st.session_state.scale = st.session_state.scaler
    tmp = deepcopy(st.session_state.gm)
    st.session_state.gm_clone = scale_graph_manager(tmp, st.session_state.scale)
    st.session_state.gm_clone.set_mutation_ranges()
    current_gm = deepcopy(st.session_state.current_gm)
    st.session_state.current_gm  = scale_graph_manager(current_gm, graph_scale)
    st.session_state.current_gm.set_mutation_ranges()
    st.session_state.current_generator_dict = deepcopy(st.session_state.current_gm.generator_dict)


# second stage
if st.session_state.stage == "ranges_choice":
    st.markdown("""Для выбранной топологии необходимо задать диапазоны оптимизации. В нашей системе есть 4 типа сочленений:  
                1. Неподвижное сочленение - неизменяемое положение. Нельзя выбрать для изменения.  
                2. Cочленение в абсолютных координатах - положение задаётся в абсолютной системе координат в метрах.  
                3. Сочленение в относительных координатах - положение задаётся относительно другого сочленения в метрах.   
                4. Сочленени задаваемое относительно звена - положение задаётся относительно центра звена в процентах от длины звена.  
                Для каждого сочленения на боковой панели указан его тип.  
                x - горизонтальные координаты, z - вертикальные координаты. Размеры указаны в метрах. Для изменения высоты конструкции необходимо изменять общий масштаб.  
                По умолчанию активированы все возможные оптимизируемые величины для каждого сочленения и заданы максимальные диапазоны оптимизации. Используйте  переключатель на боковой панели, чтобы визуализировать и изменять диапазоны каждого оптимизируемого сочленения.
                Если отключить оптимизацию величины, то её значение будет постоянным и его можно задать в соответствующем окне на боковой панели. Значение должно быть в максимальном диапазоне оптимизации""", unsafe_allow_html=True)
    
    # form for optimization ranges. All changes affects the gm_clone and it should be used for optimization
    # initial nodes
    st.markdown("""Высоту механизма можно настроить при помощи изменения общего масштаба механизма.""")
    st.slider(label="Масштаб", min_value=0.5, max_value=2.0,value=1.0, step=0.1, key='scaler', on_change=scale_change)
    # gm is initial graph, gm_clone is scaled gm so it has scaled initial ranges, current_gm is scaled current gm with current ranges
    initial_generator_info = st.session_state.gm_clone.generator_dict
    initial_mutation_ranges = st.session_state.gm_clone.mutation_ranges
    gm = st.session_state.current_gm
    generator_info = gm.generator_dict
    graph = gm.get_graph(gm.generate_central_from_mutation_range())
    labels = {n:i for i,n in enumerate(graph.nodes())}
    with st.sidebar:
        # return button
        st.button(label="Назад к выбору топологии",
                  key="return_to_topology", on_click=return_to_topology)
        
        # set of joints that have mutation range in initial generator and get current jp and its index on the graph picture
        
        mutable_jps = [key[0] for  key in initial_mutation_ranges.keys()]
        options = [(jp, idx) for jp, idx in labels.items() if jp in mutable_jps]
        current_jp = st.radio(label="Выбор сочленения для установки диапазона оптимизации", options=options, index=0, format_func=lambda x:x[1],key='joint_choice', on_change=joint_choice,horizontal=True)
        st.markdown("""Переключатель позволяет выбрать диапазоны оптимизации для каждого сочленения. Значения по умолчанию соответствуют максимыльным диапазонам оптимизации.""")
        jp_label = current_jp[1]
        jp = list(labels.keys())[jp_label]
        if st.session_state.gm.generator_dict[jp].mutation_type.value == 1:
            if None in st.session_state.gm.generator_dict[jp].freeze_pos:
                st.write("Тип сочленения: Сочленение в абсолютных координатах")
            else:
                st.write("Тип сочленения: Неподвижное сочленение")
        if st.session_state.gm.generator_dict[jp].mutation_type.value == 2:
            st.write("Тип сочленения: Сочленение в относительных координатах")
            st.write("координаты относительно сочленения: "+str(
                labels[st.session_state.gm.generator_dict[jp].relative_to]))
        if st.session_state.gm.generator_dict[jp].mutation_type.value == 3:
            st.write("Тип сочленения: Сочленение задаваемое относительно звена")
            st.write("координаты относительно звена: "+str(labels[st.session_state.gm.generator_dict[jp].relative_to[0]])+':arrow_right:'+str(labels[st.session_state.gm.generator_dict[jp].relative_to[1]]))


        # we can get current jp generator info in the cloned gm which contains all the changes
        current_generator_info = generator_info[current_jp[0]]
        for i, mut_range in enumerate(current_generator_info.mutation_range):
            if mut_range is None:
                continue
            # we can get mutation range from previous activation of the corresponding radio button
            left_value, right_value = st.session_state.current_generator_dict[current_jp[0]].mutation_range[i]
            # name = f"{labels[current_jp[0]]}_{axis[i]}"
            name = f"{axis[i]}".upper()
            toggle_value = not left_value == right_value
            current_on = st.toggle(f"Отключить оптимизацию "+name+" координаты", value=toggle_value)
            init_values = initial_generator_info[current_jp[0]].mutation_range[i]
            if current_on:
                mut_range = st.slider(
                    label=name+' координата сочленения '+str(labels[current_jp[0]]), min_value=init_values[0], max_value=init_values[1], value=(left_value, right_value))
                generator_info[current_jp[0]].mutation_range[i] = mut_range
            else:
                current_value = st.number_input(label="Insert a value", value=(
                    left_value + right_value)/2, key=name, min_value=init_values[0], max_value=init_values[1])
                # if current_value < init_values[0]:
                #     current_value = init_values[0]
                # if current_value > init_values[1]:
                #     current_value = init_values[1]
                mut_range = (current_value, current_value)
                generator_info[current_jp[0]].mutation_range[i] = mut_range

        st.button(label="подтвердить диапазоны оптимизации",
                  key='ranges_confirm', on_click=confirm_ranges, type='primary')
    # here should be some kind of visualization for ranges
    gm.set_mutation_ranges()
    plot_one_jp_bounds(gm, current_jp[0].name)
    center = gm.generate_central_from_mutation_range()
    graph = gm.get_graph(center)
    # here I can insert the visualization for jp bounds
    draw_joint_point_widjet(graph, labels=1, draw_lines=True)
    # draw_joint_point(graph, labels=1, draw_legend=True,draw_lines=True)
    st.pyplot(plt.gcf(), clear_figure=True)


def add_trajectory(trajectory, idx, name='unnamed'):
    """Create a new trajectory group with a single trajectory."""
    # trajectory buffer is necessary to store all trajectories until the confirmation and adding to reward manager
    st.session_state.trajectory_buffer[idx] = (trajectory,name)
    st.session_state.trajectory_groups.append([idx])
    st.session_state.trajectory_idx += 1
    # this object is used only for user trajectory
    st.session_state.trajectory = None
    st.session_state.trajectory_history = []

def remove_trajectory_group():
    """Remove the last added trajectory group."""
    # we only allow to remove the last added group and that should be enough
    for idx in st.session_state.trajectory_groups[-1]:
        del st.session_state.trajectory_buffer[idx]
    st.session_state.trajectory_groups.pop()


def add_to_group(trajectory, idx, name='unnamed'):
    """Add a trajectory to the last added group."""
    st.session_state.trajectory_buffer[idx] = (trajectory, name)
    st.session_state.trajectory_groups[-1].append(idx)
    st.session_state.trajectory_idx += 1
    # this object is used only for user trajectory
    st.session_state.trajectory = None
    st.session_state.trajectory_history = []

def start_optimization(rewards_tf):
    """Start the optimization process."""
    # print(st.session_state.trajectory_groups)
    st.session_state.stage = "optimization"
    #auxilary parameter just to rerun once in before optimization
    st.session_state.rerun = True
    # rewards_tf = trajectories
    # add all trajectories to the reward manager and soft constraint
    for idx_trj, trj in st.session_state.trajectory_buffer.items():
        st.session_state.reward_manager.add_trajectory(trj[0], idx_trj, trj[1])
        st.session_state.soft_constraint.add_points_set(trj[0])
    # add all rewards to the reward manager according to trajectory groups
    rewards = list(reward_dict.values())
    for trj_list_idx, trajectory_list in enumerate(st.session_state.trajectory_groups):
        for trj in trajectory_list:
            for r_idx, r in enumerate(rewards_tf[trj_list_idx]):
                if r:
                    st.session_state.reward_manager.add_reward(
                        rewards[r_idx], trj, 1)
        # we only allow mean aggregation for now
        st.session_state.reward_manager.add_trajectory_aggregator(
            trajectory_list, 'mean')
    # add all necessary objects to a buffer folder for the optimization script
    graph_manager = deepcopy(st.session_state.current_gm)
    reward_manager = deepcopy(st.session_state.reward_manager)
    sf = deepcopy(st.session_state.soft_constraint)
    builder = deepcopy(st.session_state.optimization_builder)
    data = (graph_manager, builder, crag, reward_manager, sf)
    if not Path(f"./results/optimization_widget/user_{user_key}/buffer").exists():
        Path(f"./results/optimization_widget/user_{user_key}/buffer").mkdir(parents=True)
    with open(Path(f"./results/optimization_widget/user_{user_key}/buffer/data.pkl"), "wb+") as f:
        dill.dump(data, f)


def return_to_ranges():
    """Return to the ranges choice stage."""
    st.session_state.stage = "ranges_choice"
    st.session_state.trajectory_groups = []
    st.session_state.trajectory_buffer = {}
    st.session_state.trajectory_idx = 0
    st.session_state.reward_manager = RewardManager(crag=crag)
    st.session_state.gm.set_mutation_ranges()
    st.session_state.gm_clone = deepcopy(st.session_state.gm)
    st.session_state.current_gm = deepcopy(st.session_state.gm)
    st.session_state.current_generator_dict = deepcopy(st.session_state.gm.generator_dict)
    st.session_state.scale = 1


# when ranges are set we start to choose the reward+trajectory
# each trajectory should be added to the manager
if st.session_state.stage == "trajectory_choice":
    # graph is only for visualization so it still gm
    graph = st.session_state.current_gm.graph
    trajectory = None
    with st.sidebar:
        st.button(label="Назад к выбору диапазонов оптимизации",
                  key="return_to_ranges", on_click=return_to_ranges)
        # currently only choice between predefined parametrized trajectories
        trajectory_type = st.radio(label='Выберите тип траектории:', options=[
            "Тип 1 (линия)", "Тип 2 (дуга)", "Тип 3 (ломаная)"], index=0, key="trajectory_type")
        ChangeWidgetFontSize("Выберите тип траектории:", "16px")
        if trajectory_type == "Тип 1 (линия)":
            trajectory = set_vertical_trajectory()
        if trajectory_type == "Тип 2 (дуга)":
            trajectory = set_step_trajectory()
        if trajectory_type == "Тип 3 (ломаная)":
            trajectory = user_trajectory(default_x_range*st.session_state.scale,default_z_range*st.session_state.scale,initial_point=np.array([0,-0.4])*st.session_state.scale)
            if trajectory is not None:
                trajectory = convert_x_y_to_6d_traj_xz(*add_auxilary_points_to_trajectory(trajectory,initial_point=np.array([0,-0.4])*st.session_state.scale))

        if trajectory_type != "Тип 3 (ломаная)" or (trajectory_type == "Тип 3 (ломаная)" and len(st.session_state.trajectory_history)>1):
            # no more than 2 groups for now
            if len(st.session_state.trajectory_groups) < 2:
                st.button(label="Добавить траекторию к новой группе", key="add_trajectory", args=(
                    trajectory, st.session_state.trajectory_idx,f"Траектория {st.session_state.trajectory_idx} {trajectory_type}"), on_click=add_trajectory)
            # if there is at leas one group we can add to group or remove group
            if st.session_state.trajectory_groups:
                st.button(label="Добавить траекторию к текущей группе", key="add_to_group", args=[
                    trajectory, st.session_state.trajectory_idx,f"Траектория {st.session_state.trajectory_idx} {trajectory_type}"], on_click=add_to_group)
                st.button(label="Удалить текущую группу", key="remove_group",
                        on_click=remove_trajectory_group)
        # for each reward trajectories should be assigned
    # top visualization of current trajectory

    st.markdown("""<p class="big-font">Для оптимизации используются кинематические критерии, рассчитываемые вдоль траекторий. Траектория определяет множество точек в котором будут рассчитаны выбранные критерии.
Если критерий нужно рассчитать вдоль более чем одной траектории необходимо создать группу траекторий. При помощи кнопок на боковой панели выберите траектории и соответствующие им критерии.</p>
""", unsafe_allow_html=True)
    st.button(label="Посмотреть подробное описание критериев", key="show_reward_description",on_click=lambda: st.session_state.__setitem__('stage', 'reward_description'))

    draw_joint_point_widjet(graph,labels=2, draw_legend=False, draw_lines=True)
    plt.gcf().set_size_inches(4, 4)
    if trajectory is not None:
        plt.plot(trajectory[:, 0], trajectory[:, 2])
    st.pyplot(plt.gcf(), clear_figure=True)

    trajectories = [[0]*len(list(reward_dict.keys()))]*len(st.session_state.trajectory_groups)
    if st.session_state.trajectory_groups:
        st.write("Выберите критерии для каждой группы траекторий:")
    rewards_counter = []
    for i, t_g in enumerate(st.session_state.trajectory_groups):
        st.write(f"Группа {i} траектории и критерии:")
        cols = st.columns(2)
        with cols[0]:
            st.text("Граф и выбранные траектории:")
            draw_joint_point(graph, labels=2, draw_legend=False)
            for idx in st.session_state.trajectory_groups[i]:
                current_trajectory = st.session_state.trajectory_buffer[idx][0]
                plt.plot(current_trajectory[:, 0], current_trajectory[:, 2])
            st.pyplot(plt.gcf(), clear_figure=True)
        with cols[1]:
            st.header("Критерии:")
            reward_idxs = [0]*len(list(reward_dict.keys()))
            for reward_idx, reward in enumerate(reward_dict.items()):
                current_checkbox = st.checkbox(
                    label=reward_description[reward[0]][0], value=False, key=reward[1].reward_name+str(i), help=reward_description[reward[0]][1])
                reward_idxs[reward_idx] = current_checkbox
            trajectories[i] = reward_idxs
        rewards_counter.append(sum(reward_idxs))
    # we only allow to start optimization if there is at least one group and all groups have at least one reward
    if st.session_state.trajectory_groups and all([r > 0 for r in rewards_counter]):
        st.button(label="Старт оптимизации",
                  key="start_optimization", on_click=start_optimization, args=[trajectories], type='primary')


def show_results():
    st.session_state.stage = "results"
    st.session_state.results_exist = True
    n_obj = st.session_state.reward_manager.close_trajectories()
    selected_directory = Path(f"./results/optimization_widget/user_{user_key}/current_results")
    st.session_state.n_obj = n_obj
    if n_obj == 1:
        problem = SingleCriterionProblem.load(selected_directory)
        checkpoint = load_checkpoint(selected_directory)
        optimizer = PymooOptimizer(problem, checkpoint)
        optimizer.load_history(selected_directory)
        res = optimizer.run()
        st.session_state.optimizer = optimizer
        st.session_state.problem = problem
        st.session_state.res = res
    if n_obj >= 2:
        problem = MultiCriteriaProblem.load(selected_directory)
        checkpoint = load_checkpoint(selected_directory)
        optimizer = PymooOptimizer(problem, checkpoint)
        optimizer.load_history(selected_directory)
        res = optimizer.run()
        st.session_state.optimizer = optimizer
        st.session_state.problem = problem
        st.session_state.res = res


if st.session_state.stage == "optimization":
    # I have to rerun to clear the screen
    if st.session_state.rerun:
        st.session_state.rerun = False
        st.rerun()
    
    graph = st.session_state.current_gm.graph
    graph_mesh_visualization(graph,user_visualizer,user_vis_url, labels=2, draw_lines=True, draw_legend=False)
    # col_1, col_2 = st.columns([0.7, 0.3], gap="medium")
    # with col_1:
    #     # st.header("Графовое представление:")
    #     draw_joint_point(graph, labels=2, draw_legend=False, draw_lines=True)
    #     plt.gcf().set_size_inches(4, 4)
    #     st.pyplot(plt.gcf(), clear_figure=True)
    # with col_2:
    #     send_graph_to_visualizer(graph, user_visualizer,st.session_state.visualization_builder)
    #     components.iframe(user_vis_url, width=400,
    #                       height=400, scrolling=True)
    st.text("Идёт процесс оптимизации, пожалуйста подождите...")
    empt = st.empty()
    with empt:
        st.image(str(Path('./apps/widjetdemo/mechanical-wolf-running.gif').absolute()))
    if not Path(f"./results/optimization_widget/user_{user_key}/current_results").exists():
        Path(f"./results/optimization_widget/user_{user_key}/current_results").mkdir(parents=True)
    file = open(
        Path(f"./results/optimization_widget/user_{user_key}/current_results/out.txt"), 'w')
    subprocess.run(
        ['python', "apps/widjetdemo/streamlit_widgets/run.py", str(user_key)], stdout=file)
    file.close()

    # the button should appear after the optimization is done
    with empt:
        st.button(label="Show results", key="show_results", on_click=show_results)
    # st.button(label="Show results", key="show_results", on_click=show_results)


def run_simulation(**kwargs):
    st.session_state.run_simulation_flag = True

def translate_labels(labels, reward_dict, reward_description):
    for i, label in enumerate(labels):
        for key, value in reward_dict.items():
            if value.reward_name == label:
                labels[i] = reward_description[key][0]
def translate_reward_name(name, reward_dict, reward_description):
        for key, value in reward_dict.items():
            if value.reward_name == name:
                return  reward_description[key][0]


def calculate_and_display_rewards(graph,trajectory, reward_mask):
    
    fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(
        graph, st.session_state.optimization_builder)
    point_criteria_vector, trajectory_criteria, res_dict_fixed = crag.get_criteria_data(
        fixed_robot, free_robot, trajectory, viz=None)
    if sum(reward_mask)>0:
        some_text = """<p class="big-font"> Критерии представлены в виде поточечных значений вдоль траектории.</p>"""
        st.markdown(some_text,unsafe_allow_html=True)
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

def create_file(graph, user_key=0, id_robot=0):
    path_to_robots = Path().parent.absolute().joinpath(f"robots/user_{user_key}")
    if not os.path.exists(path_to_robots):
        os.makedirs(path_to_robots)
    zip_file_name = path_to_robots / f"robot_{id_robot}.zip"
    if os.path.exists(zip_file_name):
        return zip_file_name
    robot_urdf_str, yaml_out = jps_graph2pinocchio_robot_3d_constraints(graph, st.session_state.optimization_builder, True)
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

def save_results():
    initial_path = Path(f"./results/optimization_widget/user_{user_key}/current_results")
    new_path = Path(f"./results/optimization_widget/user_{user_key}/results_"+time.strftime("%Y-%m-%d_%H-%M-%S"))
    shutil.copytree(initial_path, new_path)
    st.session_state.results_saved = True

if st.session_state.stage == "results":
    n_obj = st.session_state.n_obj
    if n_obj == 1:
        optimizer = st.session_state.optimizer
        problem = st.session_state.problem
        ten_best = np.argsort(np.array(optimizer.history["F"]).flatten())[:10]
        st.markdown("""<p class="big-font"> Результатом оптимизации является набор механизмов с наилучшими значениями заданного критерия, найденными в процессе оптимизации. 
Для каждого полученного механизма можно рассчитать критерии вдоль траекторий использованных в процессе оптмизации</p>""",unsafe_allow_html=True)
        idx = st.select_slider(label="Лучшие по заданному критерию механизмы:", options=[
                               1, 2, 3, 4, 5, 6, 7, 8, 9, 10], value=1, help='10 механизмов с наибольшими значением выбранного критерия, 1 соответствует максимальному значению критерия')
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = []
        for i in ten_best:
            y.append(optimizer.history["F"][i][0]*-1)
        best_id = ten_best[idx-1]
        best_x = optimizer.history["X"][best_id]
        graph = problem.graph_manager.get_graph(best_x)
        send_graph_to_visualizer(graph, user_visualizer, st.session_state.visualization_builder)
        with st.sidebar:
            trajectories = problem.rewards_and_trajectories.trajectories
            trj_idx = st.radio(label="Выбор траектории:", options=trajectories.keys(
            ), index=0, key='opt_trajectory_choice', format_func=lambda x: problem.rewards_and_trajectories.trajectory_names[x])
            trajectory = trajectories[trj_idx]

            st.button(label='Визуализация движения', key='run_simulation', on_click=run_simulation, kwargs={
                      "graph": graph, "trajectory": trajectory})
            st.header("Характеристики:")
            reward_idxs = [0]*len(list(reward_dict.values()))
            for reward_idx, reward in enumerate(reward_dict.items()):
                current_checkbox = st.checkbox(
                    label=reward_description[reward[0]][0], value=False, key=reward[1].reward_name+str(reward_idx), help=reward_description[reward[0]][1])
                reward_idxs[reward_idx] = current_checkbox
        graph_mesh_visualization(graph, user_visualizer,user_vis_url, labels=2, draw_lines=True, draw_legend=False)
        with st.sidebar:
            bc = st.button(label="Рассчитать значения выбранных критериев", key="calculate_rewards", type='primary')
        
        plt.figure(figsize=(3,3))
        plt.scatter(x,np.array(y))
        st.markdown("""Значения критерия оптимизации для лучших механизмов. График показывыает величину разброса результатов. Для каждого механизма можно рассчитать критерии вдоль указанных для оптимизации траекторий.""")
        st.pyplot(plt.gcf(), clear_figure=True,use_container_width=False)
        if bc:
            calculate_and_display_rewards(graph, trajectory, reward_idxs)

    if n_obj >= 2:
        if n_obj>2:
            import itertools
            st.markdown("Для отображения результатов выберите пару критериев, для построения проекции Парето фронта")
            reward_manager:RewardManager = st.session_state.problem.rewards_and_trajectories
            supp = [[x[0], True] for x in reward_manager.agg_list]
            choice_list = []
            for key, value in reward_manager.rewards.items():
                for i, tp in enumerate(supp):
                    if key in tp[0] and tp[1]:
                        tp[1]=False
                        for reward in value:
                            choice_list.append((f'группы {i}',reward[0].reward_name))
            pairs = list(itertools.combinations(choice_list, 2))
            pairs_of_idx = list(itertools.combinations(list(range(len(choice_list))), 2))
            choice = st.radio(label="Выберите пару критериев для построения графика Парето фронта", options=list(range(len(pairs))), index=0, key='pair_choice',format_func = lambda x:f'Траектории {pairs[x][0][0]}, критерий {translate_reward_name(pairs[x][0][1], reward_dict, reward_description)} и траектории {pairs[x][1][0]}, критерий {translate_reward_name(pairs[x][1][1], reward_dict, reward_description)}')
            idx_pair = pairs_of_idx[choice]
            labels = [choice_list[idx_pair[0]][1], choice_list[idx_pair[1]][1]]
            translate_labels(labels, reward_dict, reward_description)
        else:
            idx_pair = [0,1]
            labels = []
            for trajectory_idx, rewards in st.session_state.problem.rewards_and_trajectories.rewards.items():
                for reward in rewards:
                    if reward[0].reward_name not in labels:
                        labels.append(reward[0].reward_name)
            translate_labels(labels, reward_dict, reward_description)

        st.markdown("""Результатом оптимизации является набор механизмов, которые образуют Парето фронт по заданным группам критериев.""")
        res = st.session_state.res
        optimizer = st.session_state.optimizer
        problem = st.session_state.problem
        F = res.F[:, idx_pair]
        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)
        nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
        w1 = st.slider(label="Выбор решения из Парето фронта при помощи указания относительного веса:", min_value=0.01,
                       max_value=0.99, value=0.5)
        weights = np.array([w1, 1-w1])
        decomp = ASF()
        b = decomp.do(nF, 1/weights).argmin()
        best_x = res.X[b]
        graph = problem.graph_manager.get_graph(best_x)
        with st.sidebar:
            trajectories = st.session_state.reward_manager.trajectories
            trj_idx = st.radio(label="Выберите траекторию из заданных перед оптимизацией:", options=trajectories.keys(
            ), index=0, key='opt_trajectory_choice', format_func=lambda x: problem.rewards_and_trajectories.trajectory_names[x])
            trajectory = trajectories[trj_idx]
            st.button(label='Визуализация движения', key='run_simulation', on_click=run_simulation, kwargs={
                      "graph": graph, "trajectory": trajectory})
            with st.form("reward_form_mlt"):
                st.header("Характеристики:")
                reward_idxs = [0]*len(list(reward_dict.values()))
                for reward_idx, reward in enumerate(reward_dict.items()):
                    current_checkbox = st.checkbox(
                        label=reward_description[reward[0]][0], value=False, key=reward[1].reward_name+str(reward_idx), help=reward_description[reward[0]][1])
                    reward_idxs[reward_idx] = current_checkbox
                bc = st.form_submit_button(label="Рассчитать значения выбранных критериев",  type='primary')
        plt.plot(trajectory[:, 0], trajectory[:, 2])
        graph_mesh_visualization(graph, user_visualizer, user_vis_url, labels=2, draw_lines=True, draw_legend=False)
        add_trajectory_to_vis(user_visualizer, trajectory)
        # send_graph_to_visualizer(graph, st.session_state.visualization_builder)
        # col_1, col_2 = st.columns([0.7,0.3], gap="medium")
        # with col_1:
        #     # st.header("Графовое представление")
        #     draw_joint_point(graph, labels=2, draw_legend=False, draw_lines=True)
        #     
        #     plt.gcf().set_size_inches(4, 4)
        #     st.pyplot(plt.gcf(), clear_figure=True)
        # with col_2:
        #     # st.header("Робот")

        #     components.iframe(get_visualizer(st.session_state.visualization_builder).viewer.url(), width=400,
        #                       height=400, scrolling=True)
        st.text('Красный маркер указывает точку соответствующую заданному весу')

        plt.figure(figsize=(7, 5))
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.scatter(F[:, 0]*-1, F[:, 1]*-1, s=30,
                    facecolors='none', edgecolors='blue')
        # plt.scatter(approx_ideal[0], approx_ideal[1], facecolors='none',
        #             edgecolors='red', marker="*", s=100, label="Ideal Point (Approx)")
        # plt.scatter(approx_nadir[0], approx_nadir[1], facecolors='none',
        #             edgecolors='black', marker="p", s=100, label="Nadir Point (Approx)")
        plt.scatter(F[b, 0]*-1, F[b, 1]*-1, marker="x", color="red", s=200)
        if n_obj==2:
            plt.title("Парето фронт")
        else:
            plt.title('Проекция Парето фронта на плоскость выбранных критериев')
        st.pyplot(plt.gcf(),clear_figure=True)
        
        if bc:
            calculate_and_display_rewards(graph, trajectory, reward_idxs)

    with open(create_file(graph, user_key), "rb") as file:
        st.download_button(
            "Скачать URDF описание робота",
            data=file,
            file_name="robot_opt_description.zip",
            mime="robot/urdf",
        )
    st.markdown("""Вы можете скачать URDF модель полученного механизма для дальнейшего использования. Данный виджет служит для оптимизации кинематических структур в рамках заданных ограничений, вы можете использовать редакторы URDF для более точной настройки параметров и физические симуляторы для имитационного модеирования.""")
    if  "results_saved" in st.session_state:
        st.markdown("""<p class="big-font">Результаты оптимизации сохранены.</p>""",unsafe_allow_html=True)
    else:
        st.button(label="Сохранить результаты оптимизации", key="save_results", on_click=save_results)
    with st.sidebar:
        st.button(label="Посмотреть подробное описание критериев", key="show_reward_description",on_click=lambda: st.session_state.__setitem__('stage', 'reward_description'))
    
    # We need a flag to run the simulation in the frame that was just created
    if st.session_state.run_simulation_flag:
        fixed_robot_vis, _ = jps_graph2pinocchio_meshes_robot(graph, st.session_state.visualization_builder)
        robot_move_visualization(fixed_robot_vis, trajectory, user_visualizer)


if st.session_state.stage == 'reward_description':
    if st.session_state.results_exist:
        st.button(label="Вернуться", key="return_to_criteria_calculation",on_click=lambda: st.session_state.__setitem__('stage', 'results'),type='primary')
    else:
        st.button(label="Вернуться", key="return_to_criteria_calculation",on_click=lambda: st.session_state.__setitem__('stage', 'trajectory_choice'),type='primary')
    st.markdown(MD_REWARD_DESCRIPTION)