import numpy as np
import pychrono.core as chrono
from typing import Optional, List, Tuple, Dict


class ContactReporter(chrono.ReportContactCallback):

    def __init__(self) -> None:
        """Create a sensor of contact normal forces for the body.

        Args:
            chrono_body (ChBody): The body on which the sensor is installed
        """
        self._body_list: Optional[List[Tuple[int, chrono.ChBody]]] = None
        self.__contact_dict: Dict[int, List[Tuple[chrono.ChVectorD, chrono.ChVectorD]]] = {}

        super().__init__()

    def set_body_list(self, body_list: List[chrono.ChBody]):
        self._body_list = body_list

    def reset_contact_dict(self):
        for bt in self._body_list:
            self.__contact_dict[bt[0]] = []

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
        idx_a = None
        idx_b = None
        
        for body_tuple in self._body_list:
            if body_a == body_tuple[1]:
                idx_a = body_tuple[0]
            elif body_b == body_tuple[1]:
                idx_b = body_tuple[0]
            
            if not(idx_a is None or  idx_b is None):
                break
        
        if idx_a is None and idx_b is None: return True

        if not(idx_a is None):
            self.__contact_dict[idx_a].append((pA, react_forces.x))

        if not(idx_b is None):
            self.__contact_dict[idx_b].append((pB, react_forces.x))

        # if (body_a in self._current_body_list) or (body_b == self._current_body_list):
        #     self.__current_normal_forces = react_forces.x  # is it the force that works on body a or b?
        #     self.__list_normal_forces.append(react_forces.x)

        #     if (body_a in self._current_body_list):
        #         self.__current_contact_coord = [pA.x, pA.y, pA.z]
        #         self.__list_contact_coord.append(self.__current_contact_coord)
        #     elif (body_b == self._current_body_list):
        #         self.__current_contact_coord = [pB.x, pB.y, pB.z]
        #         self.__list_contact_coord.append(self.__current_contact_coord)

        return True


    def get_contacts(self):
        return self.__contact_dict
    # def is_empty(self):
    #     return len(self.__list_normal_forces) == 0

    # def list_clear(self):
    #     self.__list_normal_forces.clear()

    # def list_cont_clear(self):
    #     self.__list_contact_coord.clear()

    # def get_normal_forces(self):
    #     return self.__current_normal_forces

    # def get_list_n_forces(self):
    #     return self.__list_normal_forces

    # def get_list_c_coord(self):
    #     return self.__list_contact_coord


class Sensor:
    def __init__(self, contact_reporter:ContactReporter):
        self.reporter:ContactReporter = contact_reporter
    #@staticmethod
    # the function fills the lists in the reporter object
    # def reset_reporter_for_objects(system: chrono.ChSystem, body_list: (List[chrono.ChBody]),
    #                                contact_reporter: ContactReporter):
    #     contact_reporter.reset()
    #     contact_reporter.set_body_list(body_list)
    #     system.GetContactContainer().ReportAllContacts(contact_reporter)

    # one should use all other functions after resetting the reporter for new objects

    def std_contact_forces(self, index:int=-1):
        """Sensor of standard deviation of contact forces that affect on object

        Args:
            contact_reporter(ContactReporter): the reset ContactReporter object 
        Returns:
            dict[int, float]: Dictionary which keys are id object and values of standard deviation
              of contact forces
        """
        contacts = self.reporter.get_contacts()
        list_n_forces = []
        forces = contacts[index]
        for force in forces:
            list_n_forces.append(force[1])
        if len(list_n_forces)>0:
            contact_force_obj = np.std(list_n_forces)
            return dict([(index, contact_force_obj)])
        else: return None
        # list_n_forces = contact_reporter.get_contacts()
        # if np.size(list_n_forces) > 0:
        #     contact_force_obj = np.std(list_n_forces)
        #     return dict([(index, contact_force_obj)])

    def amount_contact_forces(self, index:int=-1):
        """The total amount of contact forces

        Args:s
            in_robot (Robot): Robot to measure sum of contact forces

        Returns:
            dict[int, float]: Dictionary which keys are id object and values of standard
            deviation of contact forces
        """
        contacts= self.reporter.get_contacts()
        forces = contacts[index]
        #list_n_forces
        if np.size(forces) > 0:
            amount_contact_force_obj = np.size(forces)
            return dict([(index, amount_contact_force_obj)])
        else:
            return None

    def abs_coord_COG(self ,body: chrono.ChBody, index:int=-1):
        """Sensor of absolute coordinates of grasp object
        Args:
            obj (ChronoBodyEnv): Grasp object
        Returns:
            dict[int, chrono.ChVectorD]: Dictionary which keys are id of object 
            and value of object COG in XYZ format
        """
        return dict([(index, [body.GetPos().x, body.GetPos().y, body.GetPos().z])])

    def contact_coord(self, index:int=-1):
        """Sensor of COG of contact points
        Args:
            obj (ChronoBodyEnv): Grasp object
        Returns:
            dict[int, float]: Dictionary which keys are id of object and values of COG of contact point volume in XYZ format
        """
        contacts = self.reporter.get_contacts()
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
                coord_x += coord[0]
                coord_y += coord[1]
                coord_z += coord[2]
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