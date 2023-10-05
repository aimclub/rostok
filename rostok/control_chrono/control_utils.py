from dataclasses import dataclass, field

from rostok.control_chrono.external_force import ForceControllerTemplate


@dataclass
class ForceTorqueContainer:
    controller_list: list[ForceControllerTemplate] = field(default_factory=list)

    def update_all(self, time: float, data=None):
        for i in self.controller_list:
            i.update(time, data)

    def add(self, controller: ForceControllerTemplate):
        if controller.is_bound:
            self.controller_list.append(controller)
        else:
            raise Exception("Force controller should be bound to body, before use")
