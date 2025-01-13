import numpy as np
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, MIT_CHEETAH_PARAMS_DICT
from auto_robot_design.pinokla.default_traj import add_auxilary_points_to_trajectory, convert_x_y_to_6d_traj_xz, get_vertical_trajectory, create_simple_step_trajectory, get_workspace_trajectory, get_horizontal_trajectory
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.optimization.rewards.reward_base import PositioningReward, PositioningConstrain, PositioningErrorCalculator, RewardManager
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability, MeanHeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import ManipulabilityReward, DexterityIndexReward, VelocityReward, ZRRReward, MinForceReward, MinManipulabilityReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward, ActuatedMassReward, TrajectoryIMFReward
from auto_robot_design.description.mesh_builder.urdf_creater import (
    URDFMeshCreator,
    MeshCreator,
)
from auto_robot_design.description.mesh_builder.mesh_builder import (
    MeshBuilder,
    jps_graph2pinocchio_meshes_robot,
)
from auto_robot_design.description.mesh_builder.urdf_creater import(
    create_mesh_manipulator_base
)
from auto_robot_design.description.actuators import TMotor_AK60_6
import os
from pathlib import Path


def is_in_subdirectory(file_path, current_directory):
    # Resolve both paths to ensure they are absolute and account for symlinks
    file_path = Path(file_path).resolve()
    current_directory = Path(current_directory).resolve()
    
    # Check if file_path is in a subdirectory of current_directory
    return file_path.is_relative_to(current_directory)

def get_mesh_builder(manipulation=False, thickness=None, density=None):
    if thickness is None:
        thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]
    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    if density is None:
        density = MIT_CHEETAH_PARAMS_DICT["density"]
    body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]

    if manipulation:
        cwd = Path.cwd()
        path = 'mesh/uhvat.stl'
        if is_in_subdirectory(path, cwd):
            uhvat_path = str(Path.joinpath(Path.cwd(), Path(path)))
        else:
            p = 0
            while not is_in_subdirectory(path, Path.cwd().parents[p]):
                p+=1
            uhvat_path = str(Path.joinpath(Path.cwd().parents[p], Path(path)))
        predefined_mesh = {"G": create_mesh_manipulator_base, "EE": uhvat_path}

        actuator = TMotor_AK60_6()
        mesh_creator = MeshCreator(predefined_mesh)
        urdf_creator = URDFMeshCreator()
        mesh_creator.height_scaler = mesh_creator.MANIPULATOR_HEIGHT_SCALER
        mesh_creator.radius_scaler = mesh_creator.MANIPULATOR_RADIUS_SCALER
        builder = MeshBuilder(urdf_creator,
                            mesh_creator,
                            density={"default": density, "G": density},
                            thickness={"default": thickness},
                            actuator={"default": actuator},
                            size_ground=np.array(
                                [0.1,0.1,0.1]),
                            )
    else:
        cwd = Path.cwd()
        path = Path('mesh/body.stl')
        if is_in_subdirectory(path, cwd):
            body_path = str(Path.joinpath(Path.cwd(), Path('mesh/body.stl')))
            whell_path = str(Path.joinpath(Path.cwd(), Path('mesh/wheel_small.stl')))
        else:
            p = 0 
            while not is_in_subdirectory(path, Path.cwd().parents[p]):
                p+=1
            body_path = str(Path.joinpath(Path.cwd().parents[p], Path('mesh/body.stl')))
            whell_path = str(Path.joinpath(Path.cwd().parents[p], Path('mesh/wheel_small.stl')))

        predefined_mesh = {"G": body_path, "EE": whell_path}
        mesh_creator = MeshCreator(predefined_mesh)
        urdf_creator = URDFMeshCreator()
        builder = MeshBuilder(
            urdf_creator,
            mesh_creator,
            density={"default": density, "G": body_density},
            thickness={"default": thickness, "EE": 0.003},
            actuator={"default": actuator},
            size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
            offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"],
        )
    return builder


def get_standard_builder(thickness=None, density=None):
    if thickness is None:
        thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]
    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    if density is None:
        density = MIT_CHEETAH_PARAMS_DICT["density"]
    body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]
    builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE,
                                  density={"default": density,
                                           "G": body_density},
                                  thickness={
                                      "default": thickness, "EE": 0.003},
                                  actuator={"default": actuator}
                                  )
    return builder


def get_standard_trajectories():
    trajectory_dict = {}
    workspace_trajectory = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(get_workspace_trajectory([-0.15, -0.35], 0.14, 0.3, 30, 60)))
    trajectory_dict['workspace'] = workspace_trajectory
    ground_symmetric_step1 = convert_x_y_to_6d_traj_xz(*add_auxilary_points_to_trajectory(create_simple_step_trajectory(
        starting_point=[-0.14, -0.34], step_height=0.12, step_width=0.28, n_points=200)))
    trajectory_dict['step1'] = ground_symmetric_step1
    ground_symmetric_step2 = convert_x_y_to_6d_traj_xz(*add_auxilary_points_to_trajectory(create_simple_step_trajectory(
        starting_point=[-0.14 + 0.015, -0.34], step_height=0.10, step_width=-2*(-0.14 + 0.015), n_points=200)))
    trajectory_dict['step2'] = ground_symmetric_step2
    ground_symmetric_step3 = convert_x_y_to_6d_traj_xz(*add_auxilary_points_to_trajectory(create_simple_step_trajectory(
        starting_point=[-0.14 + 0.025, -0.34], step_height=0.08, step_width=-2*(-0.14 + 0.025), n_points=200)))
    trajectory_dict['step3'] = ground_symmetric_step3
    central_vertical = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(get_vertical_trajectory(-0.34, 0.12, 0, 200)))
    trajectory_dict['central_vertical'] = central_vertical
    left_vertical = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(get_vertical_trajectory(-0.34, 0.12, -0.12, 200)))
    trajectory_dict['left_vertical'] = left_vertical
    right_vertical = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(get_vertical_trajectory(-0.34, 0.12, 0.12, 200)))
    trajectory_dict['right_vertical'] = right_vertical
    return trajectory_dict


def get_standard_crag(open_loop=False):
    # criteria that either calculated without any reference to points, or calculated through the aggregation of values from all points on trajectory
    dict_trajectory_criteria = {
        "MASS": NeutralPoseMass(),
        "POS_ERR": TranslationErrorMSE()  # MSE of deviation from the trajectory
    }
    # criteria calculated for each point on the trajectory
    dict_point_criteria = {
        # Impact mitigation factor along the axis
        "IMF": ImfCompute(ImfProjections.Z),
        "MANIP": ManipCompute(MovmentSurface.XZ),
        "Effective_Inertia": EffectiveInertiaCompute(),
        "Actuated_Mass": ActuatedMass(),
        "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ)
    }
    # special object that calculates the criteria for a robot and a trajectory
    if open_loop:
        crag = CriteriaAggregator(
            dict_point_criteria, dict_trajectory_criteria, alg_name="Open_Loop")
    else:
        crag = CriteriaAggregator(
            dict_point_criteria, dict_trajectory_criteria)
    return crag


def get_standard_rewards():
    reward_dict = {}
    # reward_dict['mass'] = MassReward(mass_key='MASS')
    reward_dict['actuated_inertia_matrix'] = ActuatedMassReward(
        mass_key='Actuated_Mass', reachability_key="is_reach")
    reward_dict['z_imf'] = TrajectoryIMFReward(
        imf_key='IMF', trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['trajectory_manipulability'] = VelocityReward(
        manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['manipulability'] = ManipulabilityReward(manipulability_key='MANIP',
                                                         trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['min_manipulability'] = MinManipulabilityReward(manipulability_key='Manip_Jacobian',
                                                                trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['min_force'] = MinForceReward(manipulability_key='Manip_Jacobian',
                                              trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['trajectory_zrr'] = ZRRReward(manipulability_key='Manip_Jacobian',
                                              trajectory_key="traj_6d",  reachability_key="is_reach")
    reward_dict['dexterity'] = DexterityIndexReward(manipulability_key='Manip_Jacobian',
                                                    trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['trajectory_acceleration'] = AccelerationCapability(manipulability_key='Manip_Jacobian',
                                                                    trajectory_key="traj_6d", reachability_key="is_reach", actuated_mass_key="Actuated_Mass")
    reward_dict['min_acceleration'] = MinAccelerationCapability(
        manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", reachability_key="is_reach", actuated_mass_key="Actuated_Mass")
    reward_dict['mean_heavy_lifting'] = MeanHeavyLiftingReward(
        manipulability_key='Manip_Jacobian', reachability_key="is_reach", mass_key="MASS")
    reward_dict['min_heavy_lifting'] = HeavyLiftingReward(
        manipulability_key='Manip_Jacobian', mass_key='MASS', reachability_key="is_reach")
    return reward_dict


def inertial_config_two_link_six_trajectories(workspace_based=False, open_loop=False):
    """Create objects for optimization of two link based robots

    Args:
        workspace_based (bool, optional): If true use the workspace trajectory for the soft constraint. Defaults to False.

    Returns:
        list: builder, crag, soft_constrain, reward_manager
    """
    builder = get_standard_builder()
    trajectories = get_standard_trajectories()
    crag = get_standard_crag(open_loop)
    workspace_trajectory = trajectories['workspace']
    ground_symmetric_step1 = trajectories['step1']
    ground_symmetric_step2 = trajectories['step2']
    ground_symmetric_step3 = trajectories['step3']
    central_vertical = trajectories['central_vertical']
    left_vertical = trajectories['left_vertical']
    right_vertical = trajectories['right_vertical']
    # set the rewards and weights for the optimization task
    rewards = get_standard_rewards()
    acceleration_capability = rewards['trajectory_acceleration']
    heavy_lifting = rewards['min_heavy_lifting']

    # set up special classes for reward calculations
    error_calculator = PositioningErrorCalculator(
        jacobian_key="Manip_Jacobian")
    if workspace_based:
        soft_constrain = PositioningConstrain(
            error_calculator=error_calculator, points=[workspace_trajectory])
    else:
        soft_constrain = PositioningConstrain(
            error_calculator=error_calculator, points=[ground_symmetric_step1, ground_symmetric_step2, ground_symmetric_step3, central_vertical, left_vertical, right_vertical])

    # manager should be filled with trajectories and rewards using the manager API
    reward_manager = RewardManager(crag=crag)

    reward_manager.add_trajectory(ground_symmetric_step1, 0)
    reward_manager.add_trajectory(ground_symmetric_step2, 1)
    reward_manager.add_trajectory(ground_symmetric_step3, 2)

    reward_manager.add_trajectory(central_vertical, 3)
    reward_manager.add_trajectory(left_vertical, 4)
    reward_manager.add_trajectory(right_vertical, 5)

    reward_manager.add_reward(acceleration_capability, 0, 1)
    reward_manager.add_reward(acceleration_capability, 1, 1)
    reward_manager.add_reward(acceleration_capability, 2, 1)

    reward_manager.add_reward(heavy_lifting, 3, 1)
    reward_manager.add_reward(heavy_lifting, 4, 1)
    reward_manager.add_reward(heavy_lifting, 5, 1)

    reward_manager.add_trajectory_aggregator([0, 1, 2], 'mean')
    reward_manager.add_trajectory_aggregator([3, 4, 5], 'mean')

    return builder, crag, soft_constrain, reward_manager

def inertial_config_two_link_six_trajectories_v2(workspace_based=False, open_loop=False):
    """Create objects for optimization of two link based robots

    Args:
        workspace_based (bool, optional): If true use the workspace trajectory for the soft constraint. Defaults to False.

    Returns:
        list: builder, crag, soft_constrain, reward_manager
    """
    builder = get_standard_builder()
    trajectories = get_standard_trajectories()
    crag = get_standard_crag(open_loop)
    workspace_trajectory = trajectories['workspace']
    ground_symmetric_step1 = trajectories['step1']
    ground_symmetric_step2 = trajectories['step2']
    ground_symmetric_step3 = trajectories['step3']
    central_vertical = trajectories['central_vertical']
    left_vertical = trajectories['left_vertical']
    right_vertical = trajectories['right_vertical']
    # set the rewards and weights for the optimization task
    rewards = get_standard_rewards()
    acceleration_capability = rewards['min_acceleration']
    heavy_lifting = rewards['mean_heavy_lifting']

    # set up special classes for reward calculations
    error_calculator = PositioningErrorCalculator(
        jacobian_key="Manip_Jacobian")
    if workspace_based:
        soft_constrain = PositioningConstrain(
            error_calculator=error_calculator, points=[workspace_trajectory])
    else:
        soft_constrain = PositioningConstrain(
            error_calculator=error_calculator, points=[ground_symmetric_step1, ground_symmetric_step2, ground_symmetric_step3, central_vertical, left_vertical, right_vertical])

    # manager should be filled with trajectories and rewards using the manager API
    reward_manager = RewardManager(crag=crag)

    reward_manager.add_trajectory(ground_symmetric_step1, 0)
    reward_manager.add_trajectory(ground_symmetric_step2, 1)
    reward_manager.add_trajectory(ground_symmetric_step3, 2)

    reward_manager.add_trajectory(central_vertical, 3)
    reward_manager.add_trajectory(left_vertical, 4)
    reward_manager.add_trajectory(right_vertical, 5)

    reward_manager.add_reward(acceleration_capability, 0, 1)
    reward_manager.add_reward(acceleration_capability, 1, 1)
    reward_manager.add_reward(acceleration_capability, 2, 1)

    reward_manager.add_reward(heavy_lifting, 3, 1)
    reward_manager.add_reward(heavy_lifting, 4, 1)
    reward_manager.add_reward(heavy_lifting, 5, 1)

    reward_manager.add_trajectory_aggregator([0, 1, 2], 'mean')
    reward_manager.add_trajectory_aggregator([3, 4, 5], 'mean')

    return builder, crag, soft_constrain, reward_manager
def inertial_config_two_link_workspace(open_loop=False):
    """Create objects for optimization of two link based robots

        Inertial rewards for optimization.

    Args:
        workspace_based (bool, optional): If true use the workspace trajectory for the soft constraint. Defaults to False.

    Returns:
        list: builder, crag, soft_constrain, reward_manager
    """
    builder = get_standard_builder()
    trajectories = get_standard_trajectories()
    crag = get_standard_crag(open_loop)
    workspace_trajectory = trajectories['workspace']
    # set the rewards and weights for the optimization task
    rewards = get_standard_rewards()
    acceleration_capability = rewards['min_acceleration']
    heavy_lifting = rewards['mean_heavy_lifting']

    # set up special classes for reward calculations
    error_calculator = PositioningErrorCalculator(
        jacobian_key="Manip_Jacobian")
    soft_constrain = PositioningConstrain(
        error_calculator=error_calculator, points=[workspace_trajectory])

    # manager should be filled with trajectories and rewards using the manager API
    reward_manager = RewardManager(crag=crag)

    reward_manager.add_trajectory(workspace_trajectory, 0)

    reward_manager.add_reward(acceleration_capability, 0, 1)
    reward_manager.add_reward(heavy_lifting, 0, 1)

    return builder, crag, soft_constrain, reward_manager


def jacobian_config_two_link_workspace(open_loop=False):
    """Create objects for optimization of two link based robots

        Inertial rewards for optimization.

    Args:
        workspace_based (bool, optional): If true use the workspace trajectory for the soft constraint. Defaults to False.

    Returns:
        list: builder, crag, soft_constrain, reward_manager
    """
    builder = get_standard_builder()
    trajectories = get_standard_trajectories()
    crag = get_standard_crag(open_loop)
    workspace_trajectory = trajectories['workspace']
    # set the rewards and weights for the optimization task
    rewards = get_standard_rewards()
    manipulability = rewards['manipulability']
    zrr = rewards['trajectory_zrr']

    # set up special classes for reward calculations
    error_calculator = PositioningErrorCalculator(
        jacobian_key="Manip_Jacobian")
    soft_constrain = PositioningConstrain(
        error_calculator=error_calculator, points=[workspace_trajectory])

    # manager should be filled with trajectories and rewards using the manager API
    reward_manager = RewardManager(crag=crag)

    reward_manager.add_trajectory(workspace_trajectory, 0)

    reward_manager.add_reward(manipulability, 0, 1)
    reward_manager.add_reward(zrr, 0, 1)

    return builder, crag, soft_constrain, reward_manager
