import pinocchio as pin
import numpy as np
from numpy.linalg import norm
# from loader_tools import *
# from apps.results_humanoid_2023.criterion_test import Li
from auto_robot_design.pinokla.closed_loop_kinematics import *
from pinocchio.robot_wrapper import RobotWrapper
import os
from auto_robot_design.pinokla.loader_tools import completeRobotLoader

import meshcat
from pinocchio.visualize import MeshcatVisualizer

def jacobianFinitDiffClosedLoop(model,actuation_model,constraint_model, idframe: int, idref: int, qmot: np.array,q_prec, dq=1e-6,name_mot='mot',fermeture='fermeture'):
    """
    J=Jacobian_diff_finis(robot ,idframe: int,idref :int,qo :np.array,dq: float)
    return the jacobian of the frame id idframe in the reference frame number idref, with the configuration of the robot rob qo
    """
    LJ = []  # the transpose of the Jacobian ( list of list)

    data = model.createData()
    q,b=closedLoopForwardKinematics(model, data, qmot, q_prec, name_mot, fermeture)
    pin.framesForwardKinematics(model, data, q)
    oMf1 = data.oMf[idframe].copy()  # require to avoid bad pointing
    oMrep = data.oMf[idref].copy()

    RrefXframe = (oMrep.inverse() * oMf1).action
    Lidmot=getMotId_q(model,name_mot)
    for i in range(len(Lidmot)):  # finit difference algorithm
        qmot[i] = qmot[i] + dq
        nq,b=closedLoopForwardKinematics(model, data, qmot, q_prec, name_mot, fermeture)
        pin.framesForwardKinematics(model, data, nq)
        oMf1p = data.oMf[idframe]
        V = pin.log(oMf1.inverse() * oMf1p).vector / dq

        LJ.append(V.tolist())
        qmot[i] = qmot[i] - dq
    
    J = np.transpose(np.array(LJ))
    J = RrefXframe @ J
    return J


def sepJc(model,actuation_model,Jn):
    """
    Jmot,Jfree=sepJc(model,actuation_model,Jn)
    
    Separate a constraint Jacobian `Jn` into Jcmot and Jcfree, the constraint Jacobians associated with the motor joints and free joints.

    Args:
        model (pinocchio.Model): Pinocchio model.
        actuation_model (ActuationModelFreeFlyer): Actuation model.
        Jn (np.array): Constraint Jacobian.

    Returns:
        tuple: A tuple containing:
            - Jmot (np.array): Constraint Jacobian associated with the motor joints.
            - Jfree (np.array): Constraint Jacobian associated with the free joints.
    """
    Lidmot=actuation_model.idvmot
    Lidfree=actuation_model.idvfree


    Smot=np.zeros((model.nv,len(Lidmot)))
    Smot[Lidmot,range(len(Lidmot))]=1

    Sfree=np.zeros((model.nv,model.nv-len(Lidmot)))
    Sfree[Lidfree,range(len(Lidfree))]=1


    Jmot=Jn@Smot
    Jfree=Jn@Sfree
    return(Jmot,Jfree)

def dqRowReorder(model,actuation_model,dq):
    """
    q=dqRowReorder(model,actuation_model,dq)
    
    Reorganize the vector/matrix `dq` in accordance with the model.

    Args:
        model (pinocchio.Model): Pinocchio model.
        actuation_model (ActuationModelFreeFlyer): Actuation model.
        dq (np.array): Vector/matrix organized as [dqmot dqfree].

    Returns:
        np.array: Reorganized `dq` vector/matrix.
    """
    Lidmot=actuation_model.idvmot
    Lidfree=actuation_model.idvfree
    imot=0
    ifree=0
    nJ=dq.copy()
    for i,dq in enumerate(dq.tolist()):
        if i<len(Lidmot):
            nJ[Lidmot[imot]]=dq
            imot+=1
        else:
            nJ[Lidfree[ifree]]=dq
            ifree+=1
    return(nJ)


def dq_dqmot(model,actuation_model,LJ):
    """
    dq=dq_dq_mot(model,actuation_model,LJ)

    Compute the derivative `dq/dqmot` of the joint to the motor joint.

    Args:
        model (pinocchio.Model): Pinocchio model.
        actuation_model (ActuationModelFreeFlyer): Actuation model.
        LJ (list): List of constraint Jacobians.

    Returns:
        np.array: Derivative `dq/dqmot`.
    """
    Lidmot=actuation_model.idvmot
    Jmot=np.zeros((0,len(Lidmot)))
    Jfree=np.zeros((0,model.nv-len(Lidmot)))
    for J in LJ:
        [mot,free]=sepJc(model,actuation_model,J)
        Jmot=np.concatenate((mot,Jmot))
        Jfree=np.concatenate((free,Jfree))
    
    I=np.identity(len(Lidmot))
    pinvJfree=np.linalg.pinv(Jfree)
    dq=np.concatenate((I,-pinvJfree@Jmot))
    dq=dqRowReorder(model,actuation_model,dq)
    return(dq)


def constraint_jacobian_active_to_passive(model,data,constraint_model,constraint_data,actuation_model,q0, actuation_model_aligned = True):
    #update of the jacobian an constraint model
    pin.computeJointJacobians(model,data,q0)
    LJ=[np.array(())]*len(constraint_model)
    for (cm,cd,i) in zip(constraint_model,constraint_data,range(len(LJ))):
        LJ[i]=pin.getConstraintJacobian(model,data,cm,cd)
        

    #init of constant
    Lidmot=actuation_model.idvmot
    Lidfree=actuation_model.idvfree
    nv=model.nv
    nv_mot=len(Lidmot)
    nv_free=len(Lidfree)
    Lnc=[J.shape[0] for J in LJ]
    nc=int(np.sum(Lnc))
    
    
    Jmot=np.zeros((nc,nv_mot))
    Jfree=np.zeros((nc,nv_free))
    


    #separation between Jmot and Jfree
    
    nprec=0
    for J,n in zip(LJ,Lnc):
        Smot=np.zeros((nv,nv_mot))
        Smot[Lidmot,range(nv_mot)]=1
        Sfree=np.zeros((nv,nv_free))
        Sfree[Lidfree,range(nv_free)]=1


        Jmot[nprec:nprec+n,:]=J@Smot
        Jfree[nprec:nprec+n,:]=J@Sfree

        nprec=nprec+n

    E_tau = np.zeros((nv, nv))
    E_tau[range(nv_mot), Lidmot] = 1
    E_tau[range(nv_mot,nv), Lidfree] = 1
    # computation of dq/dqmot

    pinvJfree=np.linalg.pinv(Jfree)
    dq_dmot_no=np.concatenate((np.identity(nv_mot),-pinvJfree@Jmot))
    
    if actuation_model_aligned:
        #re order dq/dqmot
        dq_dmot=dq_dmot_no.copy()
        dq_dmot[Lidmot]=dq_dmot_no[:nv_mot,:]
        dq_dmot[Lidfree]=dq_dmot_no[nv_mot:,:]
    else:
        dq_dmot = dq_dmot_no
    
    return dq_dmot, E_tau


def jacobian_constraint(model,data,constraint_model,constraint_data,actuation_model,q0):
    #update of the jacobian an constraint model
    pin.computeJointJacobians(model,data,q0)
    LJ=[np.array(())]*len(constraint_model)
    for (cm,cd,i) in zip(constraint_model,constraint_data,range(len(LJ))):
        LJ[i]=pin.getConstraintJacobian(model,data,cm,cd)
        

    #init of constant
    Lidmot=actuation_model.idvmot
    Lidfree=actuation_model.idvfree
    nv=model.nv
    nv_mot=len(Lidmot)
    nv_free=len(Lidfree)
    Lnc=[J.shape[0] for J in LJ]
    nc=int(np.sum(Lnc))
    
    
    Jmot=np.zeros((nc,nv_mot))
    Jfree=np.zeros((nc,nv_free))
    


    #separation between Jmot and Jfree
    
    nprec=0
    for J,n in zip(LJ,Lnc):
        Smot=np.zeros((nv,nv_mot))
        Smot[Lidmot,range(nv_mot)]=1
        Sfree=np.zeros((nv,nv_free))
        Sfree[Lidfree,range(nv_free)]=1


        Jmot[nprec:nprec+n,:]=J@Smot
        Jfree[nprec:nprec+n,:]=J@Sfree

        nprec=nprec+n
    
    return Jmot, Jfree

# inverseConstraintKinematicsSpeed
def ConstraintFrameJacobian(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff):
    """
    vq,Jf_cloesd=ConstraintFrameJacobian(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff)
    
    Compute the joint velocity `vq` that generates the speed `veff` on frame `ideff`.
    Return also `Jf_closed`, the closed loop Jacobian on the frame `ideff`.

    Args:
        model (pinocchio.Model): Pinocchio model.
        data (pinocchio.Data): Pinocchio data associated with the model.
        constraint_model (list): List of constraint models.
        constraint_data (list): List of constraint data associated with the constraint models.
        actuation_model (ActuationModelFreeFlyer): Actuation model.
        q0 (np.array): Initial configuration.
        ideff (int): Frame index for which the joint velocity is computed.
        veff (np.array): Desired speed on frame `ideff`.

    Returns:
        tuple: A tuple containing:
            - vq (np.array): Joint velocity that generates the desired speed on frame `ideff`.
            - Jf_closed (np.array): Closed loop Jacobian on frame `ideff`.
    """
    #update of the jacobian an constraint model
    pin.computeJointJacobians(model,data,q0)
    LJ=[np.array(())]*len(constraint_model)
    arrs_oMc1c2 = []
    arrs_c1Mc2 = []
    for (cm,cd,i) in zip(constraint_model,constraint_data,range(len(LJ))):
        LJ[i]=pin.getConstraintJacobian(model,data,cm,cd)
        arrs_oMc1c2.append([cd.oMc1, cd.oMc2])
        arrs_c1Mc2.append(cd.oMc2.translation - cd.oMc1.translation)

    #init of constant
    Lidmot=actuation_model.idvmot
    Lidfree=actuation_model.idvfree
    nv=model.nv
    nv_mot=len(Lidmot)
    nv_free=len(Lidfree)
    Lnc=[J.shape[0] for J in LJ]
    nc=int(np.sum(Lnc))
    
    
    Jmot=np.zeros((nc,nv_mot))
    Jfree=np.zeros((nc,nv_free))
    


    #separation between Jmot and Jfree
    
    nprec=0
    for J,n in zip(LJ,Lnc):
        Smot=np.zeros((nv,nv_mot))
        Smot[Lidmot,range(nv_mot)]=1
        Sfree=np.zeros((nv,nv_free))
        Sfree[Lidfree,range(nv_free)]=1


        Jmot[nprec:nprec+n,:]=J@Smot
        Jfree[nprec:nprec+n,:]=J@Sfree

        nprec=nprec+n

    # computation of dq/dqmot
    

    pinvJfree=np.linalg.pinv(Jfree)
    dq_dmot_no=np.concatenate((np.identity(nv_mot),-pinvJfree@Jmot))
    
    
    #re order dq/dqmot
    dq_dmot=dq_dmot_no.copy()
    dq_dmot[Lidmot]=dq_dmot_no[:nv_mot,:]
    dq_dmot[Lidfree]=dq_dmot_no[nv_mot:,:]

    #computation of the closed-loop jacobian
    Jf_closed = pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED)@dq_dmot
    

    # if viz:
    #     for id, oMc1c2 in enumerate(arrs_oMc1c2):
        
    #         ballIDc1 = "world/ball_c1_" + str(id)
    #         material = meshcat.geometry.MeshPhongMaterial()
    #         material.color = int(0xFF0000)
            
    #         ballIDc2 = "world/ball_c2_" + str(id)
    #         material2 = meshcat.geometry.MeshPhongMaterial()
    #         material2.color = int(0x00FF00)
            
    #         material.opacity = 0.5
    #         viz.viewer[ballIDc1].set_object(meshcat.geometry.Sphere(0.002), material)
    #         viz.viewer[ballIDc1].set_transform(oMc1c2[0].np)
            
    #         viz.viewer[ballIDc2].set_object(meshcat.geometry.Sphere(0.002), material2)
    #         viz.viewer[ballIDc2].set_transform(oMc1c2[1].np)
        
    #     print(f"constrs: 1. {arrs_c1Mc2[0]}") #2. {arrs_c1Mc2[1]}")
    
    return Jf_closed

##########TEST ZONE ##########################
# import unittest

# class TestRobotInfo(unittest.TestCase):
#     #only test inverse constraint kineatics because it runs all precedent code
#     def test_inverseConstraintKinematics(self):
#         vapply=np.array([0,0,1,0,0,0])
#         vq=inverseConstraintKinematicsSpeed(model,data,constraint_models,constraint_datas,actuation_model,q0,34,vapply)[0]
#         pin.computeAllTerms(model,data,q0,vq)
#         vcheck=data.v[13].np #frame 34 is center on joint 13
#         #check that the computing vq give the good speed 
#         self.assertTrue(norm(vcheck-vapply)<1e-6)


# if __name__ == "__main__":
#     #load robot
#     path = os.getcwd()+"//robots//robot_marcheur_1"
#     model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(path)
#     data=model.createData()
#     constraint_datas=[cm.createData() for cm in constraint_models]
#     q0=proximalSolver(model,data,constraint_models,constraint_datas)
    
    
#     #test
#     unittest.main()