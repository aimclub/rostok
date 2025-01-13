import random
from copy import deepcopy
from numpy.linalg import solve, norm, pinv
import pinocchio as pin
import numpy as np
from auto_robot_design.pinokla.robot_utils import freezeJoints, freezeJointsWithoutVis
from pinocchio.visualize import GepettoVisualizer
from pinocchio.visualize import MeshcatVisualizer
import meshcat


def openLoopInverseKinematicsProximal(
    rmodel,
    rdata,
    rconstraint_model,
    rconstraint_data,
    target_pos,
    ideff,
    q_start=None,
    onlytranslation=False,
    max_it=300,
    eps=1e-4,
    rho=1e-12,
    mu=1e-1,
    vis=None
):
    model = pin.Model(rmodel)
    data = model.createData()

    target_SE3 = pin.SE3.Identity()
    target_SE3.translation = np.array(target_pos[0:3])

    if vis is not None:
        ballID = "world/ball"
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = int(0xFF0000)
        vis.viewer[ballID].set_object(meshcat.geometry.Sphere(0.01), material)
        T = np.r_[np.c_[np.eye(3), target_SE3.translation],
                  np.array([[0, 0, 0, 1]])]
        vis.viewer[ballID].set_transform(T)

    success = False
    if q_start is None:
        q = pin.neutral(model)
    else:
        q = q_start

    err_arrs = []
    for k in range(max_it):
        pin.framesForwardKinematics(model, data, q)
        pin.forwardKinematics(model, data, q)
        err = data.oMf[ideff].translation-target_SE3.translation
        if norm(err) < eps:
            success = True
            break
        err_arrs.append(norm(err))
        J = pin.computeFrameJacobian(
            model, data, q, ideff, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        v = - pinv(J) @ err
        q = pin.integrate(model, q, v * mu)
        if vis is not None:
            vis.display(q)
    return q, norm(err), success


def closedLoopInverseKinematicsProximal(
    rmodel,
    rdata,
    rconstraint_model,
    rconstraint_data,
    target_pos,
    ideff,
    q_start=None,
    onlytranslation=False,

    max_it=300,
    eps=1e-4,
    rho=1e-10,
    mu=1e-3,
):
    """
    q=inverseGeomProximalSolver(rmodel,rdata,rconstraint_model,rconstraint_data,idframe,pos,only_translation=False,max_it=100,eps=1e-12,rho=1e-10,mu=1e-4)

    Perform inverse kinematics with a proximal solver.

    Args:
        rmodel (pinocchio.Model): Pinocchio model.
        rdata (pinocchio.Data): Pinocchio data.
        rconstraint_model (list): List of constraint models.
        rconstraint_data (list): List of constraint data.
        target_pos (np.array): Target position.
        name_eff (str, optional): Name of the frame. Defaults to "effecteur".
        onlytranslation (bool, optional): Only consider translation. Defaults to False.
        max_it (int, optional): Maximum number of iterations. Defaults to 100.
        eps (float, optional): Convergence threshold for primal and dual feasibility. Defaults to 1e-12.
        rho (float, optional): Scaling factor for the identity matrix. Defaults to 1e-10.
        mu (float, optional): Penalty parameter. Defaults to 1e-4.

    Returns:
        np.array: Joint positions that achieve the desired target position.

    raw here (L84-126):https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """
    TRAJ_CONS_DEVIDER = 1
    model = pin.Model(rmodel)
    constraint_model = [pin.RigidConstraintModel(x) for x in rconstraint_model]
    open_loop = False
    if not constraint_model:
        open_loop = True
    # add a contact constraint
    target_SE3 = pin.SE3.Identity()
    target_SE3.translation = np.array(target_pos[0:3])
    frame_constraint = model.frames[ideff]
    parent_joint = frame_constraint.parentJoint
    placement = frame_constraint.placement
    if onlytranslation:
        final_constraint = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,
                                                    model, parent_joint,
                                                    placement, 0, target_SE3,
                                                    pin.ReferenceFrame.LOCAL)
    else:
        final_constraint = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D, model, parent_joint, placement,
            model.getJointId("universel"), target_pos,
            pin.ReferenceFrame.LOCAL)
        raise Exception("Not implemented")

    final_constraint.name = "TrajCons"
    constraint_model.append(final_constraint)

    data = model.createData()
    constraint_data = [cm.createData() for cm in constraint_model]

    # proximal solver (black magic)
    if q_start is None:
        q = pin.neutral(model)
    else:
        q = q_start
    constraint_dim = 0
    for cm in constraint_model:
        constraint_dim += cm.size()
    is_reach = False
    # Solve the inverse kinematics for open loop kinematics
    # Only translation is considered
    # ref: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b_examples_d_inverse_kinematics.html#autotoc_md44
    if open_loop:
        DT = 1e-1  # Optimization step
        for k in range(max_it):
            pin.framesForwardKinematics(model, data, q)
            pin.forwardKinematics(model, data, q)
            err = data.oMf[ideff].translation-target_SE3.translation
            if norm(err) < eps:
                is_reach = True
                break
            J = pin.computeFrameJacobian(
                model, data, q, ideff, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            v = - pinv(J) @ err
            q = pin.integrate(model, q, v * DT)
        return q, norm(err), is_reach

    y = np.ones((constraint_dim))
    data.M = np.eye(model.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(model, constraint_model)
    primal_feas_array = np.zeros(max_it)
    real_feas_array = np.zeros(max_it)
    q_array = np.zeros((max_it, len(q)))
    for k in range(max_it):
        pin.computeJointJacobians(model, data, q)
        kkt_constraint.compute(model, data, constraint_model, constraint_data,
                               mu)

        constraint_value = np.concatenate([
            (pin.log(cd.c1Mc2).np[:cm.size()])
            for (cd, cm) in zip(constraint_data, constraint_model)
        ])

        LJ = []
        for cm, cd in zip(constraint_model, constraint_data):
            Jc = pin.getConstraintJacobian(model, data, cm, cd)
            LJ.append(Jc)
        J = np.concatenate(LJ)
        traj_cons_value = constraint_value[-3:]
        # if np.linalg.norm(traj_cons_value) < 0.01:
        #     traj_cons_value = np.zeros(3)
        constraint_value[-3:] = traj_cons_value / TRAJ_CONS_DEVIDER
        primal_feas = np.linalg.norm(constraint_value, np.inf)
        real_constrain_feas = np.linalg.norm(constraint_value[:-3])
        real_feas_array[k] = real_constrain_feas
        primal_feas_array[k] = primal_feas
        q_array[k] = q
        # dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        if primal_feas < eps:
            is_reach = True
            break

        rhs = np.concatenate([-constraint_value - y * mu, np.zeros(model.nv)])

        dz = kkt_constraint.solve(rhs)
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]

        alpha = 0.5
        q = pin.integrate(model, q, -alpha * dq)
        if np.linalg.norm(dq, np.inf) > 0.5:
            break
        y -= alpha * (-dy + y)

    pin.framesForwardKinematics(model, data, q)

    # pos_e = np.linalg.norm(data.oMf[id_frame].translation -
    #                     np.array(target_pos[0:3]))
    min_feas = primal_feas
    min_real_feas = real_constrain_feas
    if not is_reach:
        for_sort = np.column_stack(
            (primal_feas_array, real_feas_array, q_array))

        def key_sort(x): return x[0]
        for_sort = sorted(for_sort, key=key_sort)
        finish_q = for_sort[0][2:]
        q = finish_q
        min_feas = for_sort[0][0]
        min_real_feas = for_sort[0][1]
        pin.framesForwardKinematics(model, data, q)
    # print(min_real_feas," ", is_reach)

    return q, min_feas, is_reach


def closedLoopProximalMount(
    model,
    data,
    constraint_model,
    constraint_data,
    # actuation_model,
    q_prec=None,
    max_it=100,
    eps=1e-6,
    rho=1e-10,
    mu=1e-4,
):
    """
    q=proximalSolver(model,data,constraint_model,constraint_data,max_it=100,eps=1e-12,rho=1e-10,mu=1e-4)

    Build the robot in respect to the constraints using a proximal solver.

    Args:
        model (pinocchio.Model): Pinocchio model.
        data (pinocchio.Data): Pinocchio data.
        constraint_model (list): List of constraint models.
        constraint_data (list): List of constraint data.
        actuation_model (ActuationModelFreeFlyer): Actuation model.
        q_prec (list or np.array, optional): Initial guess for joint positions. Defaults to [].
        max_it (int, optional): Maximum number of iterations. Defaults to 100.
        eps (float, optional): Convergence threshold for primal and dual feasibility. Defaults to 1e-12.
        rho (float, optional): Scaling factor for the identity matrix. Defaults to 1e-10.
        mu (float, optional): Penalty parameter. Defaults to 1e-4.

    Returns:
        np.array: Joint positions of the robot respecting the constraints.

    raw here (L84-126):https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """

    # Lid = actuation_model.idqmot
    if q_prec is None:
        q_prec = pin.neutral(model)
    q = q_prec

    constraint_dim = 0
    for cm in constraint_model:
        constraint_dim += cm.size()
    # If constraint_dim is 0, then the robot have open loop kinematics
    if constraint_dim == 0:
        return q
    y = np.ones((constraint_dim))
    data.M = np.eye(model.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(model, constraint_model)

    for k in range(max_it):
        pin.computeJointJacobians(model, data, q)
        kkt_constraint.compute(
            model, data, constraint_model, constraint_data, mu)

        constraint_value = np.concatenate(
            [
                (pin.log(cd.c1Mc2).np[: cm.size()])
                for (cd, cm) in zip(constraint_data, constraint_model)
            ]
        )

        LJ = []
        for cm, cd in zip(constraint_model, constraint_data):
            Jc = pin.getConstraintJacobian(model, data, cm, cd)
            LJ.append(Jc)
        J = np.concatenate(LJ)

        primal_feas = np.linalg.norm(constraint_value, np.inf)
        dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        if primal_feas < eps and dual_feas < eps:
            # print("Convergence achieved")
            break
        # print("constraint_value:", np.linalg.norm(constraint_value))
        rhs = np.concatenate([-constraint_value - y * mu, np.zeros(model.nv)])

        dz = kkt_constraint.solve(rhs)
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]

        alpha = 1.0
        q = pin.integrate(model, q, -alpha * dq)
        y -= alpha * (-dy + y)
    return q


def ForwardK(
    model,
    constraint_model,
    actuation_model,
    q_prec=None,
    max_it=100,
    alpha=0.7,
    eps=1e-12,
    rho=1e-10,
    mu=1e-4,

):
    """
    q=proximalSolver(model,data,constraint_model,constraint_data,max_it=100,eps=1e-12,rho=1e-10,mu=1e-4)

    Build the robot in respect to the constraints using a proximal solver.

    Args:
        model (pinocchio.Model): Pinocchio model.
        data (pinocchio.Data): Pinocchio data.
        constraint_model (list): List of constraint models.
        constraint_data (list): List of constraint data.
        actuation_model (ActuationModelFreeFlyer): Actuation model.
        q_prec (list or np.array, optional): Initial guess for joint positions. Defaults to [].
        max_it (int, optional): Maximum number of iterations. Defaults to 100.
        eps (float, optional): Convergence threshold for primal and dual feasibility. Defaults to 1e-12.
        rho (float, optional): Scaling factor for the identity matrix. Defaults to 1e-10.
        mu (float, optional): Penalty parameter. Defaults to 1e-4.

    Returns:
        np.array: Joint positions of the robot respecting the constraints.

    raw here (L84-126):https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """

    Lid = actuation_model.idMotJoints
    Lid_q = actuation_model.idqmot

    (reduced_model, reduced_constraint_models, reduced_actuation_model) = freezeJointsWithoutVis(
        model, constraint_model, None, Lid, q_prec
    )

    reduced_data = reduced_model.createData()
    reduced_constraint_data = [c.createData()
                               for c in reduced_constraint_models]

    q = np.delete(q_prec, Lid_q, axis=0)
    constraint_dim = 0
    for cm in reduced_constraint_models:
        constraint_dim += cm.size()

    y = np.ones((constraint_dim))
    reduced_data.M = np.eye(reduced_model.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(
        reduced_model, reduced_constraint_models
    )

    for k in range(max_it):
        pin.computeJointJacobians(reduced_model, reduced_data, q)
        kkt_constraint.compute(
            reduced_model,
            reduced_data,
            reduced_constraint_models,
            reduced_constraint_data,
            mu,
        )

        constraint_value = np.concatenate(
            [
                (pin.log(cd.c1Mc2).np[: cm.size()])
                for (cd, cm) in zip(reduced_constraint_data, reduced_constraint_models)
            ]
        )

        # LJ = []
        # for cm, cd in zip(reduced_constraint_models, reduced_constraint_data):
        #     Jc = pin.getConstraintJacobian(reduced_model, reduced_data, cm, cd)
        #     LJ.append(Jc)
        # J = np.concatenate(LJ)

        primal_feas = np.linalg.norm(constraint_value, np.inf)
        # dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        if primal_feas < eps:
            # print("Convergence achieved")
            break
        # print("constraint_value:", np.linalg.norm(constraint_value))
        rhs = np.concatenate(
            [-constraint_value - y * mu, np.zeros(reduced_model.nv)])

        dz = kkt_constraint.solve(rhs)
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]

        q = pin.integrate(reduced_model, q, -alpha * dq)
        y -= alpha * (-dy + y)

    q_final = q_prec
    free_q_dict = zip(actuation_model.idqfree, q)
    for index, value in free_q_dict:
        q_final[index] = value
    return q_final, primal_feas


def ForwardK1(
    model,
    visual_model,
    constraint_model,
    collision_model,
    actuation_model,
    q_prec=None,
    max_it=100,
    eps=1e-12,
    rho=1e-10,
    mu=1e-4,
):
    Lid = actuation_model.idMotJoints

    Lid_q = actuation_model.idqmot
    q_prec2 = np.delete(q_prec, Lid, axis=0)
    model2 = model.copy()
    constraint_model2 = constraint_model.copy()
    reduced_model, reduced_constraint_models, reduced_actuation_model, reduced_visual_model, reduced_collision_model = freezeJoints(model2,
                                                                                                                                    constraint_model2,
                                                                                                                                    actuation_model,
                                                                                                                                    visual_model,
                                                                                                                                    collision_model,
                                                                                                                                    Lid,
                                                                                                                                    q_prec,
                                                                                                                                    )

    reduced_data = reduced_model.createData()
    data = model.createData()
    reduced_constraint_data = [c.createData()
                               for c in reduced_constraint_models]
    constraint_data = [c.createData() for c in constraint_model2]

    pin.framesForwardKinematics(reduced_model, reduced_data, q_prec2)
    pin.computeAllTerms(
        reduced_model,
        reduced_data,
        q_prec2,
        q_prec2)

    q_ooo = closedLoopProximalMount(
        reduced_model,
        reduced_data,
        reduced_constraint_models,
        reduced_constraint_data,
        reduced_actuation_model,
        q_prec2,
        max_it=4,
        rho=1e-8,
        mu=1e-3
    )

    return q_ooo




