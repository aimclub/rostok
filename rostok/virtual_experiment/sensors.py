from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import numpy as np
import pychrono.core as chrono

CoordinatesContact: TypeAlias = chrono.ChVectorD
ForceVector: TypeAlias = chrono.ChVectorD


class ContactReporter(chrono.ReportContactCallback):

    def __init__(self) -> None:
        """Create a sensor of contact normal forces for the body.

        Args:
            _body_list (List[Tuple[int, chrono.ChBody]]): list of tuples of type (index, body), 
                the index will be converted into the key in the contact dictionary
            __contact_dict_this_step (Dict[int, List[Tuple[chrono.ChVectorD, chrono.ChVectorD]]]): 
                dict of contacts obtained for the bodies from the body list in the current step. 
                Each value is a list of contacts of form (position, force)
        """
        super().__init__()
        self._body_map: Optional[Dict[int, chrono.ChBody]] = {}
        self.__contact_dict_this_step: Dict[int, List[Tuple[CoordinatesContact, ForceVector]]] = {}
        self.__outer_contact_dict_this_step: Dict[int, List[Tuple[CoordinatesContact,
                                                                  ForceVector]]] = {}

    def set_body_map(self, body_map_ordered: Dict[int, Any]):
        self._body_map = body_map_ordered

    def reset_contact_dict(self):
        for idx in self._body_map:
            self.__contact_dict_this_step[idx] = []
            self.__outer_contact_dict_this_step[idx] = []

    def OnReportContact(self, pA: CoordinatesContact, pB: CoordinatesContact,
                        plane_coord: chrono.ChMatrix33D, distance: float, eff_radius: float,
                        react_forces: ForceVector, react_torques: chrono.ChVectorD,
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
        # The threshold for the force sensitivity
        if react_forces.Length() < 0.001:
            return True
        body_a = chrono.CastToChBody(contactobjA)
        body_b = chrono.CastToChBody(contactobjB)
        idx_a = None
        idx_b = None
        for idx, body in self._body_map.items():
            if body_a == body.body:
                idx_a = idx
            elif body_b == body.body:
                idx_b = idx
        if not idx_a is None:
            temp_vec = -(plane_coord * react_forces)
            self.__contact_dict_this_step[idx_a].append(
                ([pA.x, pA.y, pA.z], [temp_vec.x, temp_vec.y, temp_vec.z]))
            if idx_b is None:
                self.__outer_contact_dict_this_step[idx_a].append(
                    ([pA.x, pA.y, pA.z], [temp_vec.x, temp_vec.y, temp_vec.z]))
        if not idx_b is None:
            temp_vec = -(plane_coord * react_forces)
            self.__contact_dict_this_step[idx_b].append(
                ([pB.x, pB.y, pB.z], [temp_vec.x, temp_vec.y, temp_vec.z]))
            if idx_a is None:
                self.__outer_contact_dict_this_step[idx_b].append(
                    ([pB.x, pB.y, pB.z], [temp_vec.x, temp_vec.y, temp_vec.z]))

        return True

    def get_contacts(self):
        return self.__contact_dict_this_step

    def get_outer_contacts(self):
        return self.__outer_contact_dict_this_step



class Sensor:
    """Control data obtained in the current step of the simulation"""

    def __init__(self, body_map_ordered, joint_map_ordered) -> None:
        self.contact_reporter: ContactReporter = ContactReporter()
        self.contact_reporter.set_body_map(body_map_ordered)
        self.body_map_ordered: Dict[int, Any] = body_map_ordered
        self.joint_map_ordered: Dict[int, Any] = joint_map_ordered

    def update_current_contact_info(self, system: chrono.ChSystem):
        system.GetContactContainer().ReportAllContacts(self.contact_reporter)

    def get_body_trajectory_point(self):
        output = {}
        for idx, body in self.body_map_ordered.items():
            output[idx] = [
                round(body.body.GetPos().x, 4),
                round(body.body.GetPos().y, 4),
                round(body.body.GetPos().z, 4)
            ]
        return output

    def get_velocity(self):
        output = {}
        for idx, body in self.body_map_ordered.items():
            output[idx] = [
                round(body.body.GetPos_dt().x, 4),
                round(body.body.GetPos_dt().y, 4),
                round(body.body.GetPos_dt().z, 4)
            ]
        return output
    
    def get_rotation_velocity(self):
        output = {}
        for idx, body in self.body_map_ordered.items():
            mat = body.body.GetA_dt()
            output[idx] = [[mat.Get_A_Xaxis.x, mat.Get_A_Yaxis.x, mat.Get_A_Zaxis.x], [mat.Get_A_Xaxis.y, mat.Get_A_Yaxis.y, mat.Get_A_Zaxis.y], [mat.Get_A_Xaxis.z, mat.Get_A_Yaxis.z, mat.Get_A_Zaxis.z]]
        return output

    def get_joint_z_trajectory_point(self):
        output = {}
        for idx, joint in self.joint_map_ordered.items():
            master_body: chrono.ChBodyFrame = joint.joint.GetBody2()
            slave_body: chrono.ChBodyFrame = joint.joint.GetBody1()
            relative_rot = (master_body.GetInverse() * slave_body)
            angle = chrono.Q_to_Euler123(chrono.ChQuaternionD(relative_rot.GetRot()))
            output[idx] = round(angle.z, 5)
        
        return output

    def get_forces(self):
        output = {}
        contacts = self.contact_reporter.get_contacts()
        for idx in self.body_map_ordered:
            contacts_idx = contacts[idx]
            if len(contacts_idx) > 0:
                output[idx] = contacts_idx
            else:
                output[idx] = []

        return output

    def get_amount_contacts(self):
        output = {}
        contacts = self.contact_reporter.get_outer_contacts()
        for idx in self.body_map_ordered:
            contacts_idx = contacts[idx]
            output[idx] = len(contacts_idx)

        return output

    def get_outer_force_center(self):
        output = {}
        contacts = self.contact_reporter.get_outer_contacts()
        for idx in self.body_map_ordered:
            contacts_idx = contacts[idx]
            if len(contacts_idx) > 0:
                body_contact_coordinates = [x[0] for x in contacts_idx]
                body_contact_coordinates_sum = np.zeros(3)
                for contact in body_contact_coordinates:
                    body_contact_coordinates_sum += np.array(contact)

                body_contact_coordinates_sum = body_contact_coordinates_sum * (1 /
                                                                               len(contacts_idx))
                output[idx] = list(body_contact_coordinates_sum)
            else:
                output[idx] = None

        return output

    # def get_COG(self):
    #     output = {}
    #     for idx, body in self.body_map_ordered.items():
    #         body = body.body
    #         output[idx] = [body.GetPos().x, body.GetPos().y, body.GetPos().z]

    #     return output
    def get_body_map(self):
        return self.body_map_ordered
    def get_joint_map(self):
        return self.joint_map_ordered
class SensorCalls(str, Enum):
    """
        BODY_TRAJECTORY: trajectories of all bodies from the body map,
        JOINT_TRAJECTORY : trajectories of all joints, 
        FORCE: all forces acting on the bodies from the body map,
        AMOUNT_FORCE : amount of forces acting on the bodies from the body_map,
        FORCE_CENTER: position of the center of the forces acting on a body from the body map
    """
    BODY_TRAJECTORY = Sensor.get_body_trajectory_point
    JOINT_TRAJECTORY = Sensor.get_joint_z_trajectory_point
    FORCE = Sensor.get_forces
    AMOUNT_FORCE = Sensor.get_amount_contacts
    FORCE_CENTER = Sensor.get_outer_force_center
    BODY_VELOCITY = Sensor.get_velocity


class SensorObjectClassification(str, Enum):
    BODY = Sensor.get_body_map
    JOINT = Sensor.get_joint_map

class DataStorage():
    """Class aggregates data from all steps of the simulation."""

    def __init__(self, sensor: Sensor):
        self.sensor = sensor
        self.callback_dict = {}
        self.main_storage = {}

    def add_data_type(self,
                      key: str,
                      sensor_callback: SensorCalls,
                      object_map: SensorObjectClassification,
                      step_number):
        empty_dict: Dict[int, np.NDArray] = {}
        self.callback_dict[key] = sensor_callback
        starting_values = sensor_callback(self.sensor)
        for idx in object_map(self.sensor):
            empty_dict[idx] = [np.nan] * (step_number + 1)
            if starting_values[idx] is None:
                empty_dict[idx][0] = np.nan
            else:
                empty_dict[idx][0] = np.array(starting_values[idx])
        self.main_storage[key] = empty_dict

    def add_data(self, key, data_list, step_n):
        if data_list:
            for idx, data in data_list.items():
                if not data is None:
                    self.main_storage[key][idx][step_n + 1] = np.array(data)
                else:
                    self.main_storage[key][idx][step_n + 1] = np.nan

    def update_storage(self, step_n):
        for key, sensor_callback in self.callback_dict.items():
            self.add_data(key, sensor_callback(self.sensor), step_n)

    def get_data(self, key):
        return self.main_storage[key]
