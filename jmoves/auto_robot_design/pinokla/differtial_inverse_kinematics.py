import pinocchio as pin
import numpy as np

from auto_robot_design.pinokla.closed_loop_kinematics import *


def dampfing_least_square(J, damp_coeff = 1e-8):
    U, S, Vh = np.linalg.svd(J)


    e_diag = S / (S**2 + damp_coeff**2)
    # E = S / (S**2 + l**2)
    
    # E = np.zeros((Vh.shape[0], U.shape[0]))
    
    # E[:Vh.shape[0],:Vh.shape[0]] = np.diag(e_diag)
    
    # pinvJg =  np.dot(Vh.T[:, :E.size], E * U.T)
    pinvJ = np.zeros((Vh.shape[0], U.shape[0]))
    for i in range(e_diag.shape[0]):
        pinvJ += e_diag[i] * np.dot(Vh[:,i][:,np.newaxis],  U[:,i][np.newaxis,:])
    
    return pinvJ


def DiffertialInverseKinematics(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff, viz=None):
    """
    vq,Jf_cloesd=inverseConstraintKinematicsSpeedOptimized(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff)
    
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
    Jf_closed = pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL)@dq_dmot
    # Jf_closed = pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED)@dq_dmot
    
    # pin.forwardKinematics(model,data, q0)
    # data.oMi[model.getFrameId(ideff)]
    
    #computation of the kinematics
    vqmot=np.linalg.pinv(Jf_closed)@veff 
    vqfree=-pinvJfree@Jmot@vqmot
    vqmotfree=np.concatenate((vqmot,vqfree))  # qmotfree=[qmot qfree]
    
    #reorder of vq
    vq=np.zeros(nv)
    vq[Lidmot]=vqmotfree[:nv_mot]
    vq[Lidfree]=vqmotfree[nv_mot:]

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
    
    return(vq,Jf_closed)

def PlaneDiffertialInverseKinematics(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff):
    """
    vq,Jf_cloesd=PlaneDiffertialInverseKinematics(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff)
    
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
    

    # pinvJfree=np.linalg.pinv(Jfree)
    pinvJfree=dampfing_least_square(Jfree)
    dq_dmot_no=np.concatenate((np.identity(nv_mot),-pinvJfree@Jmot))
    
    
    #re order dq/dqmot
    dq_dmot=dq_dmot_no.copy()
    dq_dmot[Lidmot]=dq_dmot_no[:nv_mot,:]
    dq_dmot[Lidfree]=dq_dmot_no[nv_mot:,:]

    #computation of the closed-loop jacobian
    # Jf_closed = pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL)@dq_dmot
    # Jf_closed = pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED)@dq_dmot
    Jf_closed = (pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED)@dq_dmot)[[0,2]]
    
    # pin.forwardKinematics(model,data, q0)
    # data.oMi[model.getFrameId(ideff)]
    
    #computation of the kinematics
    # vqmot=np.linalg.pinv(Jf_closed)@veff 
    # vqmot=dampfing_least_square(Jf_closed)@veff
    vqmot=dampfing_least_square(Jf_closed)@veff[[0,2]]
    vqfree=-pinvJfree@Jmot@vqmot
    vqmotfree=np.concatenate((vqmot,vqfree))  # qmotfree=[qmot qfree]
    
    #reorder of vq
    vq=np.zeros(nv)
    vq[Lidmot]=vqmotfree[:nv_mot]
    vq[Lidfree]=vqmotfree[nv_mot:]

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
    
    return(vq,Jf_closed)


def PlaneDiffertialInverseKinematicsMultitasking(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff):
    """
    vq,Jf_cloesd=PlaneDiffertialInverseKinematicsMultitasking(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff)
    
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
    # pin.computeJointJacobians(model,data,q0)
    constraint_frame_name = []
    constraint_frame_id = []
    arrs_c1Mc2 = []
    arrs_oMc1c2 = []
    LJ=[np.array(()) for __ in range(len(constraint_model))]
    for (cm,cd,i) in zip(constraint_model,constraint_data,range(len(LJ))):
        LJ[i]=pin.getConstraintJacobian(model,data,cm,cd)
        constraint_frame_name.append(cm.name.split("-"))
        constraint_frame_id.append([model.getFrameId(n) for n in constraint_frame_name[-1]])
        # arrs_c1Mc2.append(cd.c1Mc2)
        arrs_oMc1c2.append([cd.oMc1, cd.oMc2])
        
        # err_cnstr = (pin.log6(cd.oMc2) - pin.log6(cd.oMc1)).np * 10
        # err_cnstr[4] = 0
        
        # arrs_c1Mc2.append(cd.oMc1.action @ err_cnstr)
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
        Sfree[Lidfree,range(nv_free)] = 1


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
    # Jf_closed = pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL)@dq_dmot
    Jf_closed = (pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED)@dq_dmot)[[0,2],:]
    Jee = pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED)[[0,2],:]
    
    
    
    Jc1 = (pin.computeFrameJacobian(model,data,q0,constraint_frame_id[0][0],pin.LOCAL))[[0,2],:]
    Jc2 = (pin.computeFrameJacobian(model,data,q0,constraint_frame_id[1][0],pin.LOCAL))[[0,2],:]
    # Jc1_closed = (pin.computeFrameJacobian(model,data,q0,constraint_frame_id[0][0],pin.LOCAL_WORLD_ALIGNED))[[0,2],:]
    # Jc2_closed = (pin.computeFrameJacobian(model,data,q0,constraint_frame_id[1][0],pin.LOCAL_WORLD_ALIGNED))[[0,2],:]
    # Jf_closed = (pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL)@dq_dmot)[[0,2],:]
    # Jf_closed = (pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED)@dq_dmot)[[0,2],:]
    
    # pin.forwardKinematics(model,data, q0)
    # data.oMi[model.getFrameId(ideff)]

    vqmot=np.linalg.pinv(Jf_closed)@veff[[0,2]]
    Pee = np.round(np.eye(model.nv) - np.linalg.pinv(Jee) @ Jee, 6)
    # if not np.all(np.isclose(Pee, 0)):
    #computation of the kinematics
    vqfree=-pinvJfree@Jmot@vqmot
    vqmotfree=np.concatenate((vqmot,vqfree))  # qmotfree=[qmot qfree]

    #reorder of vq
    vq=np.zeros(nv)
    vq[Lidmot]=vqmotfree[:nv_mot]
    vq[Lidfree]=vqmotfree[nv_mot:]

    # vq += np.linalg.pinv(Jc1 @ Pee) @ (- pin.log6(arrs_c1Mc2[0]).np[[0,2]]/1e-5 - Jc1 @ vq)
    # vq += np.linalg.pinv(Jc1 @ Pee) @ (- arrs_c1Mc2[0][[0,2]]/1e-5 - Jc1 @ vq)
    vq += np.linalg.pinv(Jc1 @ Pee) @ (- arrs_c1Mc2[0][[0,2]]*70 - Jc1 @ vq)
    # vq += np.linalg.pinv(Jc1 @ Pee) @ (- pin.log6(arrs_c1Mc2[0].inverse()).np[[0,2]]/1e-5 - Jc1 @ vq)
    
    Pc2 = np.round(Pee - np.linalg.pinv(Jc1 @ Pee) @ Jc1 @ Pee, 6)
    
    # vq += np.linalg.pinv(Jc2 @ Pc2) @ (- arrs_c1Mc2[1][[0,2]]/1e-5 - Jc2 @ vq)
    vq += np.linalg.pinv(Jc2 @ Pc2) @ (- arrs_c1Mc2[1][[0,2]]*70 - Jc2 @ vq)
    # vq += np.linalg.pinv(Jc2 @ Pc2) @ (- pin.log6(arrs_c1Mc2[1]).np[[0,2]]/1e-5 - Jc2 @ vq)
    
    
    joint_off_ids = list(map(lambda x: model.getJointId(x), filter(lambda x: not x.find("Main_connection"), model.names)))
    
    # vq[joint_off_ids] = np.zeros_like(joint_off_ids)
    
    # print(f"c1c2 1 {np.linalg.norm(arrs_c1Mc2[0].translation):.4f}; 2 {np.linalg.norm(arrs_c1Mc2[1].translation):.4f}")
    # for id, oMc1c2 in enumerate(arrs_oMc1c2):
    
    #     ballIDc1 = "world/ball_c1_" + str(id)
    #     material = meshcat.geometry.MeshPhongMaterial()
    #     material.color = int(0xFF0000)
        
    #     ballIDc2 = "world/ball_c2_" + str(id)
    #     material2 = meshcat.geometry.MeshPhongMaterial()
    #     material2.color = int(0x00FF00)
        
    #     material.opacity = 0.5
    #     viz.viewer[ballIDc1].set_object(meshcat.geometry.Sphere(0.002), material)
    #     viz.viewer[ballIDc1].set_transform(oMc1c2[0].np)
        
    #     viz.viewer[ballIDc2].set_object(meshcat.geometry.Sphere(0.002), material2)
    #     viz.viewer[ballIDc2].set_transform(oMc1c2[1].np)
    
    # print(f"constrs: 1. {arrs_c1Mc2[0]} 2. {arrs_c1Mc2[1]}")
    return(vq,Jf_closed)


def PlaneDiffertialInverseKinematicsMultitaskingConstrained(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff, viz):
    """
    vq,Jf_cloesd=PlaneDiffertialInverseKinematicsMultitaskingConstrained(model,data,constraint_model,constraint_data,actuation_model,q0,ideff,veff)
    
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
    # pin.computeJointJacobians(model,data,q0)
    constraint_frame_name = []
    constraint_frame_id = []
    arrs_oMc1c2 = []
    arrs_c1Mc2 = []
    LJ=[np.array(()) for __ in range(len(constraint_model))]
    for (cm,cd,i) in zip(constraint_model,constraint_data,range(len(LJ))):
        LJ[i]=pin.getConstraintJacobian(model,data,cm,cd)
        constraint_frame_name.append(cm.name.split("-"))
        constraint_frame_id.append([model.getFrameId(n) for n in constraint_frame_name[-1]])
        arrs_oMc1c2.append([cd.oMc1, cd.oMc2])
        arrs_c1Mc2.append(cd.oMc1.translation - cd.oMc2.translation)

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
        Sfree[Lidfree,range(nv_free)] = 1


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
    Jf_closed = (pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED)@dq_dmot)[[0,2],:]
    # Jee = (pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED)@dq_dmot)[[0,2],:]
    # Jc1 = (pin.computeFrameJacobian(model,data,q0,constraint_frame_id[0][0],pin.LOCAL_WORLD_ALIGNED)@dq_dmot)[[0,2],:]
    # Jc2 = (pin.computeFrameJacobian(model,data,q0,constraint_frame_id[1][0],pin.LOCAL_WORLD_ALIGNED)@dq_dmot)[[0,2],:]
    Jee = (pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED))[[0,2],:]
    Jc1 = (pin.computeFrameJacobian(model,data,q0,constraint_frame_id[0][1],pin.LOCAL_WORLD_ALIGNED))[[0,2],:]
    # Jc2 = (pin.computeFrameJacobian(model,data,q0,constraint_frame_id[1][0],pin.LOCAL_WORLD_ALIGNED))[[0,2],:]
    
    Jg = np.vstack([Jee, Jc1])#, Jc2])

    # pin.forwardKinematics(model,data, q0)
    # data.oMi[model.getFrameId(ideff)]
    
    U, S, Vh = np.linalg.svd(Jg)

    l = 1e-8
    
    e_diag = S / (S**2 + l**2)
    # E = S / (S**2 + l**2)
    
    # E = np.zeros((Vh.shape[0], U.shape[0]))
    
    # E[:Vh.shape[0],:Vh.shape[0]] = np.diag(e_diag)
    
    # pinvJg =  np.dot(Vh.T[:, :E.size], E * U.T)
    pinvJg = np.zeros((Vh.shape[0], U.shape[0]))
    for i in range(e_diag.shape[0]):
        pinvJg += e_diag[i] * np.dot(Vh[:,i][:,np.newaxis],  U[:,i][np.newaxis,:])

    # vq = pinvJg @ np.hstack([veff[[0,2]], 10*pin.log6(arrs_c1Mc2[0].inverse()).np[[0,2]], 10*pin.log6(arrs_c1Mc2[1].inverse()).np[[0,2]]])
    # vqmot = pinvJg @ np.hstack([veff[[0,2]], arrs_c1Mc2[0][[0,2]], arrs_c1Mc2[1][[0,2]]])
    vq = pinvJg @ np.hstack([veff[[0,2]], arrs_c1Mc2[0][[0,2]]])#, arrs_c1Mc2[1][[0,2]]])
    
    joint_off_ids = list(map(lambda x: model.getJointId(x), filter(lambda x: not x.find("Main_connection"), model.names)))
    
    # vq[joint_off_ids] = np.zeros_like(joint_off_ids)
    
    # vqfree=-pinvJfree@Jmot@vqmot
    # vqmotfree=np.concatenate((vqmot,vqfree))  # qmotfree=[qmot qfree]

    # #reorder of vq
    # vq=np.zeros(nv)
    # vq[Lidmot]=vqmotfree[:nv_mot]
    # vq[Lidfree]=vqmotfree[nv_mot:]
    
    # print(f"c1c2 1 {np.linalg.norm(arrs_c1Mc2[0].translation):.4f}; 2 {np.linalg.norm(arrs_c1Mc2[1].translation):.4f}")
    # for id, oMc1c2 in enumerate(arrs_oMc1c2):
    
    #     ballIDc1 = "world/ball_c1_" + str(id)
    #     material = meshcat.geometry.MeshPhongMaterial()
    #     material.color = int(0xFF0000)
        
    #     ballIDc2 = "world/ball_c2_" + str(id)
    #     material2 = meshcat.geometry.MeshPhongMaterial()
    #     material2.color = int(0x00FF00)
        
    #     material.opacity = 0.5
    #     viz.viewer[ballIDc1].set_object(meshcat.geometry.Sphere(0.002), material)
    #     viz.viewer[ballIDc1].set_transform(oMc1c2[0].np)
        
    #     viz.viewer[ballIDc2].set_object(meshcat.geometry.Sphere(0.002), material2)
    #     viz.viewer[ballIDc2].set_transform(oMc1c2[1].np)
    
    # # print(f"constrs: 1. {arrs_c1Mc2[0]} 2. {arrs_c1Mc2[1]}")
    # print(f"constrs: 1. {arrs_c1Mc2[0]}")
    return(vq,Jf_closed)