from typing import Dict, List, Optional, Tuple

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

        self._body_list: Optional[List[Tuple[int, chrono.ChBody]]] = None
        self.__contact_dict_this_step: Dict[int, List[Tuple[CoordinatesContact, ForceVector]]] = {}
        self.__outer_contact_dict_this_step: Dict[int, List[Tuple[CoordinatesContact, ForceVector]]] = {}

        super().__init__()

    def set_body_list(self, body_list: List[Tuple[int, chrono.ChBody]]):
        self._body_list = body_list

    def reset_contact_dict(self):
        for bt in self._body_list:
            self.__contact_dict_this_step[bt[0]] = []
            self.__outer_contact_dict_this_step[bt[0]] = []

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
        if react_forces.Length()<0.001:
            return True
        body_a = chrono.CastToChBody(contactobjA)
        body_b = chrono.CastToChBody(contactobjB)
        idx_a = None
        idx_b = None
        for body_tuple in self._body_list:
            if body_a == body_tuple[1].body:
                idx_a = body_tuple[0]
            elif body_b == body_tuple[1].body:
                idx_b = body_tuple[0]
        if idx_a:
            self.__contact_dict_this_step[idx_a].append((pA, - plane_coord*react_forces))
            if idx_b is None:
                self.__outer_contact_dict_this_step[idx_a].append((pA, - plane_coord*react_forces))
        if idx_b:
            self.__contact_dict_this_step[idx_b].append((pB, plane_coord*react_forces))
            if idx_a is None:
                self.__outer_contact_dict_this_step[idx_b].append((pB, plane_coord*react_forces ))

        return True

    def get_contacts(self):
        return self.__contact_dict_this_step
    def get_outer_contacts(self):
        return self.__outer_contact_dict_this_step


class Sensor:

    def __init__(self, body_list, joint_list) -> None:
        self.contact_reporter: ContactReporter = ContactReporter()
        self.contact_reporter.set_body_list(body_list)
        self.body_list = body_list
        self.joint_list = joint_list
        #self.joint_body_map:Dict[int, Tuple[int, int]] = joint_body_map
        self.body_trajectories={}
        self.joint_trajectories = {}
        for x in  body_list:
            self.body_trajectories[x[0]] = [[round(x[1].body.GetPos().x,3),round(x[1].body.GetPos().y,3), round(x[1].body.GetPos().z,3)]]
        for x in  joint_list:
            self.joint_trajectories[x[0]] = [round(x[1].joint.GetMotorRot(), 3)]

    def update_current_contact_info(self, system:chrono.ChSystem):
        system.GetContactContainer().ReportAllContacts(self.contact_reporter)

    def update_trajectories(self):
        for x in  self.body_list:
            self.body_trajectories[x[0]].append([round(x[1].body.GetPos().x,3),round(x[1].body.GetPos().y,3), round(x[1].body.GetPos().z,3)])
        for x in self.joint_list:
            self.joint_trajectories[x[0]].append(round(x[1].joint.GetMotorRot(), 3))

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
        for tp in self.body_list:
            if tp[0]==index:
                body = tp[1]
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
