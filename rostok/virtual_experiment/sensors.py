from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pychrono.core as chrono
from typing_extensions import TypeAlias

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
            self.__contact_dict_this_step[idx_a].append((pA, -(plane_coord * react_forces)))
            if idx_b is None:
                self.__outer_contact_dict_this_step[idx_a].append((pA, -(plane_coord * react_forces)))
        if not idx_b is None:
            self.__contact_dict_this_step[idx_b].append((pB, -(plane_coord * react_forces)))
            if idx_a is None:
                self.__outer_contact_dict_this_step[idx_b].append((pB, -(plane_coord * react_forces)))

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
        self.body_map_ordered:Dict[int, Any] = body_map_ordered
        self.joint_map_ordered:Dict[int, Any] = joint_map_ordered

    def update_current_contact_info(self, system: chrono.ChSystem):
        system.GetContactContainer().ReportAllContacts(self.contact_reporter)

    def get_body_trajectory_point(self):
        output = []
        for idx, body in self.body_map_ordered.items():
            output.append((idx, [
                round(body.body.GetPos().x, 3),
                round(body.body.GetPos().y, 3),
                round(body.body.GetPos().z, 3)
            ]))
        return output

    def get_joint_trajectory_point(self)->List[Any]:
        output = []
        for idx, joint in self.joint_map_ordered.items():
            master_body:chrono.ChBodyFrame = joint.joint.GetBody2()
            slave_body:chrono.ChBodyFrame = joint.joint.GetBody1()
            angle = (master_body.GetInverse()*slave_body).GetRotAngle()
            output.append((idx, round(angle, 3)))
        return output

    def get_forces(self):
        output = []
        contacts = self.contact_reporter.get_contacts()
        for idx in self.body_map_ordered:
            output.append((idx, contacts[idx]))
        return output

    def get_amount_contacts(self):
        output = []
        contacts = self.contact_reporter.get_contacts()
        for idx in self.body_map_ordered:
            output.append((idx, len(contacts[idx])))

    def get_outer_force_center(self):
        output = []
        contacts = self.contact_reporter.get_outer_contacts()
        for idx in self.body_map_ordered:
            if len(contacts[idx])>0:
                body_contact_coordinates_sum = sum([x[0] for x in contacts[idx]])*(1/len(contacts[idx]))
                output.append((idx, [body_contact_coordinates_sum.GetPos().x, body_contact_coordinates_sum.GetPos().y, body_contact_coordinates_sum.GetPos().z] ))
            else:
                output.append((idx, None))

        return output
    
    def get_COG(self):
        output = []
        for idx, body in self.body_map_ordered.items():
            body = body.body
            output.append((idx, [body.GetPos().x, body.GetPos().y, body.GetPos().z]))

        return output

    def std_contact_forces(self, index: int = -1):
        """Sensor of standard deviation of contact forces that affect on object

        Args:
            contact_reporter(ContactReporter): the reset ContactReporter object 
        Returns:
            dict[int, float]: Dictionary which keys are id object and values of standard deviation
              of contact forces
        """
        contacts = self.contact_reporter.get_contacts()
        list_n_forces = []
        forces = contacts[index]
        for force in forces:
            list_n_forces.append(force[1].x)
        if len(list_n_forces) > 0:
            contact_force_obj = np.std(list_n_forces)
            return dict([(index, contact_force_obj)])
        else:
            return None

    def amount_contact_forces(self, index: int = -1):
        """The total amount of contact forces

        Args:s
            in_robot (Robot): Robot to measure sum of contact forces

        Returns:
            dict[int, float]: Dictionary which keys are id object and values of standard
            deviation of contact forces
        """
        contacts = self.contact_reporter.get_contacts()
        forces = contacts[index]
        if np.size(forces) > 0:
            amount_contact_force_obj = np.size(forces)
            return dict([(index, amount_contact_force_obj)])
        else:
            return None

    def amount_outer_contact_forces(self, index: int = -1):
        """The total amount of contact forces

        Args:s
            in_robot (Robot): Robot to measure sum of contact forces

        Returns:
            dict[int, float]: Dictionary which keys are id object and values of standard
            deviation of contact forces
        """
        contacts = self.contact_reporter.get_outer_contacts()
        forces = contacts[index]
        if np.size(forces) > 0:
            amount_contact_force_obj = np.size(forces)
            return dict([(index, amount_contact_force_obj)])
        else:
            return None

    def abs_coord_COG(self, index: int = -1):
        """Sensor of absolute coordinates of grasp object
        Args:
            obj (ChronoBodyEnv): Grasp object
        Returns:
            dict[int, chrono.ChVectorD]: Dictionary which keys are id of object 
            and value of object COG in XYZ format
        """
        for idx, body in self.body_map_ordered.items():
            if idx == index:
                body = body.body
                return dict([(index, [body.GetPos().x, body.GetPos().y, body.GetPos().z])])
        return None

    def contact_coord(self, index: int = -1):
        """Sensor of COG of contact points
        Args:
            obj (ChronoBodyEnv): Grasp object
        Returns:
            dict[int, float]: Dictionary which keys are id of object and values of COG of contact point volume in XYZ format
        """

        contacts = self.contact_reporter.get_contacts()
        forces = contacts[index]
        list_c_coord = []
        for force in forces:
            list_c_coord.append(force[0])
        if np.size(list_c_coord) > 0:
            coordinates = []
            coord_x = 0
            coord_y = 0
            coord_z = 0
            for coord in list_c_coord:
                coord_x += coord.x
                coord_y += coord.y
                coord_z += coord.z
            coordinates.append([
                coord_x / len(list_c_coord), coord_y / len(list_c_coord),
                coord_z / len(list_c_coord)
            ])
            return dict([(index, [
                coord_x / len(list_c_coord), coord_y / len(list_c_coord),
                coord_z / len(list_c_coord)
            ])])
        else:
            return None


class DataStorage():
    """Class aggregates data from all steps of the simulation."""

    def __init__(self):
        self.main_storage = {}

    def add_data_type(self, key: str, object_map):
        empty_dict:Dict[str, List[Any]] = {}
        for idx in object_map:
            empty_dict[idx] = []
        self.main_storage[key] = empty_dict

    def add_data(self, key, data_list, step_n):
        if data_list:
            for data in data_list:
                self.main_storage[key][data[0]].append((step_n, data[1]))

    def get_data(self, key):
        return self.main_storage[key]
