from operator import le
from exceptiongroup import catch
from matplotlib import pyplot as plt
import numpy as np
import pinocchio as pin
from scipy import optimize

import meshcat
from pinocchio.visualize import MeshcatVisualizer

from auto_robot_design.control.model_based import OperationSpacePDControl
from auto_robot_design.control.trajectory_planning import trajectory_planning
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
from auto_robot_design.motion_planning.trajectory_ik_manager import IK_METHODS, TrajectoryIKManager
from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopInverseKinematicsProximal, closedLoopProximalMount
from auto_robot_design.pinokla.default_traj import add_auxilary_points_to_trajectory, convert_x_y_to_6d_traj_xz, create_simple_step_trajectory

class TrajectoryMovements:
    def __init__(self, trajectory, final_time, time_step, name_ee_frame) -> None:
        """
        Initialization of the class for modeling the mechanism by the trajectory of end effector movement.
        For trajectory movement class use PD control in operational space with coefficients Kp = (7000, 4200) and Kd = (74, 90).
        
        Method `simulate` simulates the movement of the mechanism along the trajectory and returns the 
        position, velocity, acceleration in configuration space,
        torque, position of the end effector frame and power.
        
        For tune the control coefficients use `optimize_control` method. `optimize_control` method uses the scipy.optimize.shgo
        method for optimization of the control coefficients. The optimization function is the sum of the squared error of the position of the end effector frame and the sum of the squared torque.

        Args:
            trajectory (numpy.ndarray): The desired trajectory via points in x-z plane.
            final_time (numpy.ndarray): The time of the final point in trajectory.
            time_step (float): The time step for simulation.
            name_ee_frame (str): The name of the end-effector frame.

        """

        Kp = np.zeros((6,6))
        Kp[0,0] = 3000
        Kp[2,2] = 3000
        
        Kd = np.zeros((6,6))
        Kd[0,0] = 90
        Kd[2,2] = 90
        
        self.Kp = Kp
        self.Kd = Kd

        self.name_ee_frame = name_ee_frame
        self.traj = trajectory
        self.time = final_time
        self.time_step = time_step
        
        self.num_sim_steps = int(self.time / self.time_step)
        

    def setup_dynamic(self):
        """Initializes the dynamics calculator, also set time_step. 
        """
        accuracy = 1e-8
        mu_sim = 1e-8
        max_it = 10000
        self.dynamic_settings = pin.ProximalSettings(accuracy, mu_sim, max_it)

    def prepare_trajectory(self, robot):
        """
        Prepare the trajectory for simulation.

        Args:
            robot: The robot object.

        Returns:
            time_arr: Array of time values.
            des_traj_6d: Desired 6D trajectory.
            des_d_traj_6d: Desired 6D trajectory derivative.
        """
        des_trajectories = np.zeros((self.num_sim_steps, 2))
        des_trajectories[:,0] = np.linspace(self.traj[0,0], self.traj[-1,0], self.num_sim_steps)
        # cs_z_by_x = np.polyfit(self.traj[:,0], self.traj[:,1], 3)
        # des_trajectories[:,1] = np.polyval(cs_z_by_x, des_trajectories[:,0])

        if self.traj[:,0].max() - self.traj[:,0].min() < 1e-3:
            des_trajectories[:,1] = np.linspace(self.traj[0,1], self.traj[-1,1], self.num_sim_steps)
        else:
            cs_z_by_x = np.polyfit(self.traj[:,0], self.traj[:,1], 3)
            des_trajectories[:,1] = np.polyval(cs_z_by_x, des_trajectories[:,0])
        time_arr = np.linspace(0, self.time, self.num_sim_steps)
        
        des_traj_6d = convert_x_y_to_6d_traj_xz(des_trajectories[:,0], des_trajectories[:,1])
        
        des_d_traj_6d = np.diff(des_traj_6d, axis=0) / self.time_step
        des_d_traj_6d = np.vstack((des_d_traj_6d, des_d_traj_6d[-1]))

        # q = np.zeros(robot.model.nq)
        # Trajectory by points in joint space
        
        
        # traj_points = convert_x_y_to_6d_traj_xz(self.traj[:,0], self.traj[:,1])
        # q_des_points = np.zeros((len(traj_points), robot.model.nq))
        
        # frame_id = robot.model.getFrameId(self.name_ee_frame)
        # for num, i_pos in enumerate(traj_points):
        #     q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
        #         robot.model,
        #         robot.data,
        #         robot.constraint_models,
        #         robot.constraint_data,
        #         i_pos,
        #         frame_id,
        #         onlytranslation=True,
        #         q_start=q,
        #     )
        #     if not is_reach:
        #         q = closedLoopProximalMount(
        #             robot.model, robot.data, robot.constraint_models, robot.constraint_data, q
        #         )
        #     q_des_points[num] = q.copy()

        # q = q_des_points[0]

        # __, q_des_traj, dq_des_traj, ddq_des_traj = trajectory_planning(
        # q_des_points.T, 0, 0, 0, self.times[-1], self.time_step, False
        # )
        
        # self.des_trajectories = {
        #     "time": time_arr,
        #     # "q_ref": q_des_traj,
        #     # "dq_ref": dq_des_traj,
        #     # "ddq_ref": ddq_des_traj,
        #     "traj_6d_ref": des_traj_6d,
        #     "d_traj_6d_ref": des_d_traj_6d,
        # }
        return time_arr, des_traj_6d, des_d_traj_6d

    def simulate(self, robot, q_start, is_vis=False):
        """
        Simulates the trajectory movements of a robot.

        Args:
            robot (RobotModel): The robot model.
            is_vis (bool, optional): Whether to visualize the simulation. Defaults to False.

        Returns:
            tuple: A tuple containing the following simulation data:
                - q (numpy.ndarray): The joint positions at each simulation step.
                - vq (numpy.ndarray): The joint velocities at each simulation step.
                - acc (numpy.ndarray): The joint accelerations at each simulation step.
                - tau_act (numpy.ndarray): The actuator torques at each simulation step.
                - pos_ee_frame (numpy.ndarray): The end-effector frame positions at each simulation step.
                - power (numpy.ndarray): The mechanical power actuators exerted at each simulation step.
        """
        self.setup_dynamic()
        frame_id = robot.model.getFrameId(self.name_ee_frame)

        __, des_traj_6d, des_d_traj_6d = self.prepare_trajectory(robot)
        q = q_start

        control = OperationSpacePDControl(robot, self.Kp, self.Kd, frame_id)

        pin.initConstraintDynamics(robot.model, robot.data, robot.constraint_models)

        vq = np.zeros(robot.model.nv)
        tau_q = np.zeros(robot.model.nv)

        q_act = np.zeros((self.num_sim_steps, robot.model.nq))
        vq_act = np.zeros((self.num_sim_steps, robot.model.nv))
        acc_act = np.zeros((self.num_sim_steps, robot.model.nv))
        tau_act = np.zeros((self.num_sim_steps, 2))

        power = np.zeros((self.num_sim_steps, len(robot.actuation_model.idvmot)))

        pos_ee_frame = np.zeros((self.num_sim_steps, 3))

        if is_vis:
            viz = MeshcatVisualizer(robot.model, robot.visual_model, robot.visual_model)
            viz.viewer = meshcat.Visualizer().open()
            viz.viewer["/Background"].set_property("visible", False)
            viz.viewer["/Grid"].set_property("visible", False)
            viz.viewer["/Axes"].set_property("visible", False)
            viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [0,0,0.5])
            viz.clean()
            viz.loadViewerModel()
            viz.display(q)

        for i in range(self.num_sim_steps):
            a = pin.constraintDynamics(
                robot.model,
                robot.data,
                q,
                vq,
                tau_q,
                robot.constraint_models,
                robot.constraint_data,
                self.dynamic_settings,
            )

            vq += a * self.time_step

            q = pin.integrate(robot.model, q, vq * self.time_step)
            try:
                tau_q = control.compute(q, vq, des_traj_6d[i], des_d_traj_6d[i])
            except np.linalg.LinAlgError:
                return q_act, vq_act, acc_act, tau_act, pos_ee_frame, power
            # First coordinate is root_joint

            tau_a = tau_q[control.ids_vmot]
            vq_a = vq[control.ids_vmot]
            q_act[i] = q
            vq_act[i] = vq
            acc_act[i] = a
            tau_act[i] = tau_a
            pos_ee_frame[i] = robot.data.oMf[frame_id].translation
            power[i] = tau_a * vq_a

            if is_vis:
                viz.display(q)
        return q_act, vq_act, acc_act, tau_act, pos_ee_frame, power
    
    def optimize_control(self, robot):
        """
        Optimize the control coefficients for the robot. The optimization function is the sum of the squared error of the position of the end effector frame and the sum of the squared torque.
        The `scipy.optimize.shgo` method is used for optimization.

        Args:
            robot: The robot object.

        Returns:
            Kp: The optimized proportional gain matrix.
            Kd: The optimized derivative gain matrix.
        """
        def cost(x, robot):
            old_Kp = self.Kp
            old_Kd = self.Kd

            self.Kp = np.zeros((6,6))
            self.Kd = np.zeros((6,6))

            self.Kp[0,0] = x[0]
            self.Kp[2,2] = x[1]
            self.Kd[0,0] = x[2]
            self.Kd[2,2] = x[3]

            __, __, __, tau_act, pos_ee_frame, __ = self.simulate(robot,False)

            des_pos_ee_frame = self.des_trajectories["traj_6d_ref"][:,:3]

            pos_error = np.sum(np.linalg.norm(pos_ee_frame - des_pos_ee_frame, axis=1)**2)

            norm_tau = np.sum(np.linalg.norm(tau_act, axis=1)**2)/6e4

            self.Kp = old_Kp
            self.Kd = old_Kd

            return pos_error + norm_tau

        bounds = [[0, 1e4] for __ in range(2)]
        bounds = np.vstack((bounds, [[0, 5e3] for __ in range(2)]))

        results = optimize.shgo(cost, bounds, args=(robot,), n=10, iters=1)

        Kp = np.zeros((6,6))
        Kd = np.zeros((6,6))

        Kp[0,0] = results.x[0]
        Kp[2,2] = results.x[1]
        Kd[0,0] = results.x[2]
        Kd[2,2] = results.x[3]
        return Kp, Kd
    

def go_to_point(robot, point):

    to_start_from_init = add_auxilary_points_to_trajectory(np.array([point]).T)
    traj_6d = convert_x_y_to_6d_traj_xz(to_start_from_init[0], to_start_from_init[1])
    
    traj_manager = TrajectoryIKManager()
    traj_manager.register_model(robot.model, robot.constraint_models)
    traj_manager.set_solver(traj_manager.default_name)
    pos, q_arrs, __, reach_array = traj_manager.follow_trajectory(traj_6d, np.zeros(robot.model.nq))
    
    result_q = np.zeros(robot.model.nq)
    if reach_array[-1]:
        result_q = q_arrs[-1]
    else:
        raise Exception("Point is not reachable")
    
    return result_q


if __name__ == "__main__":
    from auto_robot_design.generator.restricted_generator.two_link_generator import (
    TwoLinkGenerator,
    )
    from auto_robot_design.description.builder import (
    ParametrizedBuilder,
    URDFLinkCreator,
    jps_graph2pinocchio_robot,
)   
    builder = ParametrizedBuilder(URDFLinkCreator)

    gm = get_preset_by_index_with_bounds(5)
    x_centre = gm.generate_central_from_mutation_range()
    graph_jp = gm.get_graph(x_centre)

    robo, __ = jps_graph2pinocchio_robot(graph_jp, builder)
    
    name_ee = "EE"
    
    
    ground_symmetric_step1 = create_simple_step_trajectory(
            starting_point=[-0.11, -0.32], step_height=0.07, step_width=0.22, n_points=4)
    
    start_q = go_to_point(robo, np.array(ground_symmetric_step1)[:,0])
    
    
    test = TrajectoryMovements(np.array(ground_symmetric_step1).T, 1, 0.01, name_ee)
    pin.framesForwardKinematics(robo.model, robo.data, start_q)
    
    Kp = np.zeros((6,6))
    Kd = np.zeros((6,6))
    
    Kp[0,0] = 3000
    Kd[0,0] = 100
    
    Kp[2,2] = 3000
    Kd[2,2] = 100
    
    test.Kp = Kp
    test.Kd = Kd

        
    # # q, vq, acc, tau, pos_ee, power
    __, __, __, tau_arr, pos_ee, __ = test.simulate(robo, start_q, True)
    
    
    des_traj = np.array(ground_symmetric_step1).T
    plt.plot(pos_ee[:,0], pos_ee[:,2])
    plt.plot(des_traj[:,0], des_traj[:,1], ".")
    
    plt.show()