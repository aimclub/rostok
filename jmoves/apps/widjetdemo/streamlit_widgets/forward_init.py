from dataclasses import dataclass

import meshcat
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import streamlit as st
from matplotlib.patches import Circle
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
from auto_robot_design.generator.topologies.graph_manager_2l import GraphManager2L
from auto_robot_design.optimization.rewards.reward_base import PositioningConstrain, RewardManager
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.utils.configs import get_standard_builder, get_mesh_builder, get_standard_crag, get_standard_rewards

pin.seed(1)

@st.cache_resource
def build_constant_objects():
    optimization_builder = get_standard_builder()
    manipulation_builder = get_mesh_builder(manipulation=True)
    suspension_builder = get_mesh_builder(manipulation=False)
    crag = get_standard_crag()
    graph_managers = {f"Структура_{i}": get_preset_by_index_with_bounds(i) for i in range(9)}
    reward_dict = get_standard_rewards()
    return graph_managers, optimization_builder, manipulation_builder, suspension_builder, crag, reward_dict

def add_trajectory_to_vis(pin_vis, trajectory, **balls_parameters):
    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(0xFF00FF)
    material.color = int(0x00FFFF)
    material.color = int(0xFFFF00)
    material.color = int(0x00FF00)
    material.opacity = 0.3
    for idx, point in enumerate(trajectory):
        if idx%balls_parameters.get("step_balls", 2)==0:
            ballID = "world/ball" + str(idx)
            pin_vis.viewer[ballID].set_object(meshcat.geometry.Sphere(balls_parameters.get("radius_balls",0.0075)), material)
            T = np.r_[np.c_[np.eye(3), point[:3]+np.array([0,balls_parameters.get("y_offset_balls", -0.04),0])], np.array([[0, 0, 0, 1]])]
            pin_vis.viewer[ballID].set_transform(T)

@st.cache_resource
def get_russian_reward_description():
    reward_description={}
    reward_description['mass'] = ("Масса механизма", "Общая масса всех звеньев и моторов")
    reward_description['actuated_inertia_matrix'] = ("Обратная инерция", "Величина обратная определителю матрицы инерции в обобщённых координатах актуированных сочленений. Характеризует полную инерцию механизма")
    reward_description['z_imf'] = ("Фактор распределения вертикального удара", "Характеризует насколько конструкция ноги ослабляет влияние вертикальных внешних сил на корпус робота")
    reward_description['trajectory_manipulability'] = ("Манипулируемость вдоль траектории", "Среднее значение проекции якобиана скоростей концевого эффектора на направление траектории. Показывает соотношение скоростей движения моторов")
    reward_description['manipulability'] = ("Манипулируемость", "Среднее значение определителя якобиана скоростей концевого эффектора. В каждой точке характеризует преобразование скоростей моторов в скорость концевого эффектора")
    reward_description['min_manipulability'] = ("Минимальная манипулируемость", "Минимальное значение манипулируемости концевого эффектора. Зависит от минимального значения преобразования скоростей моторов в скорость концевого эффектора")
    reward_description['min_force'] = ("Минимальное усилие", "Минимальное значение внешней силы необходимое для преодоления единичного момента актуаторов")
    reward_description['trajectory_zrr'] = ("Вертикальное передаточное отношение", "Среднее значение вертикального передаточного отношения вдоль траектории. Характеризует способность конструкции ноги выдерживать вертикальное усилие приложенное к концевому эффектору.")
    reward_description['dexterity'] = ("Индекс подвижности", "Среднее значение индекса подвижности вдоль траектории. Характеризует отношение силовой и скоростной характеристик механизма, большие значения соотвествуют лучшему балансу.")
    reward_description['trajectory_acceleration'] = ("Потенциальное ускорение вдоль траектории", "Среднее значение потенциального ускорения вдоль траектории. Характеризует способность двигателей разгонять концевой эффектор в заданноам направлении.")
    reward_description['min_acceleration'] = ("Минимальное потенциальное ускорение", "Минимальное значение потенциального ускорения в точке. Характеризует способность двигателей разгонять концевой эффектор.")
    reward_description['mean_heavy_lifting'] = ("Средняя грузоподъемность", "Среднее значение грузоподъемности. Характеризует способность конструкции ноги поднимать груз не превышая пороговых значений моментов актуаторов.")
    reward_description['min_heavy_lifting'] = ("Минимальная грузоподъемность", "Минимальное значение грузоподъемности на заданной траектории. Характеризует конструкции ноги непрерывно переносить груз без отклонений от заданной траектории.")
    return reward_description





