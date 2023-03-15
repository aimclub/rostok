import numpy as np
import pychrono.core as chrono
from rostok.block_builder.transform_srtucture import FrameTransform
def frame_transform_to_chcoordsys(transform: FrameTransform):
    return chrono.ChCoordsysD(
                chrono.ChVectorD(transform.position[0], transform.position[1],
                                 transform.position[2]),
                chrono.ChQuaternionD(transform.rotation[0], transform.rotation[1],
                                     transform.rotation[2], transform.rotation[3]))
def rotation_z(alpha):
    quat_Z_ang_alpha = chrono.Q_from_AngZ(np.deg2rad(alpha))
    return [quat_Z_ang_alpha.e0, quat_Z_ang_alpha.e1, quat_Z_ang_alpha.e2,quat_Z_ang_alpha.e3]

def rotation_z_q(alpha):
    quat_Z_ang_alpha = chrono.Q_from_AngZ(np.deg2rad(alpha))
    return chrono.ChQuaternionD(quat_Z_ang_alpha.e0, quat_Z_ang_alpha.e1, quat_Z_ang_alpha.e2,quat_Z_ang_alpha.e3)

class ContactReporter(chrono.ReportContactCallback):

    def __init__(self, chrono_body):
        """Create a sensor of contact normal forces for the body.

        Args:
            chrono_body (ChBody): The body on which the sensor is installed
        """
        self._body = chrono_body
        self.__current_normal_forces = None
        self.__current_contact_coord = None
        self.__list_normal_forces = []
        self.__list_contact_coord = []
        super().__init__()

    def OnReportContact(self, pA: chrono.ChVectorD, pB: chrono.ChVectorD,
                        plane_coord: chrono.ChMatrix33D, distance: float, eff_radius: float,
                        react_forces: chrono.ChVectorD, react_torques: chrono.ChVectorD,
                        contactobjA: chrono.ChContactable, contactobjB: chrono.ChContactable):
        """Callback used to report contact points already added to the container

        Args:
            pA (ChVector): coordinates of contact point(s) in body A
            pB (ChVector): coordinates of contact point(s) in body B
            plane_coord (ChMatrix33): contact plane coordsystem
            distance (float): contact distance
            eff_radius (float)): effective radius of curvature at contact
            react_forces (ChVector): reaction forces in coordsystem 'plane_coord'
            react_torques (ChVector): reaction torques, if rolling friction
            contactobjA (ChContactable): model A
            contactobjB (ChContactable): model B
        Returns:
            bool: If returns false, the contact scanning will be stopped
        """

        body_a = chrono.CastToChBody(contactobjA)
        body_b = chrono.CastToChBody(contactobjB)
        if (body_a == self._body) or (body_b == self._body):
            self.__current_normal_forces = react_forces.x
            self.__list_normal_forces.append(react_forces.x)

            if (body_a == self._body):
                self.__current_contact_coord = [pA.x, pA.y, pA.z]
                self.__list_contact_coord.append(self.__current_contact_coord)
            elif(body_b == self._body):
                self.__current_contact_coord = [pB.x, pB.y, pB.z]
                self.__list_contact_coord.append(self.__current_contact_coord)

        return True

    def is_empty(self):
        return len(self.__list_normal_forces) == 0

    def list_clear(self):
        self.__list_normal_forces.clear()
    
    def list_cont_clear(self):
        self.__list_contact_coord.clear()

    def get_normal_forces(self):
        return self.__current_normal_forces

    def get_list_n_forces(self):
        return self.__list_normal_forces

    def get_list_c_coord(self):
        return self.__list_contact_coord 
    

class SpringTorque(chrono.TorqueFunctor):

    def __init__(self, spring_coef, damping_coef, rest_angle):
        super(SpringTorque, self).__init__()
        self.spring_coef = spring_coef
        self.damping_coef = damping_coef
        self.rest_angle = rest_angle

    def evaluate(self, time, angle, vel, link):
        """Calculation of torque, that is created by spring
        

        Args:
            time  :  current time
            angle :  relative angle of rotation
            vel   :  relative angular speed
            link  :  back-pointer to associated link


        Returns:
            torque: torque, that is created by spring
        """
        torque = 0
        if self.spring_coef > 10**-3:
            torque = -self.spring_coef * \
                (angle - self.rest_angle) - self.damping_coef * vel
        else:
            torque = -self.damping_coef * vel
        return torque



