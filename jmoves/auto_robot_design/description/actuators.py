from dataclasses import dataclass, field

import numpy as np


@dataclass
class Actuator:
    mass: float = 0.0
    inertia: float = 0.0
    peak_effort: float = 0.0
    peak_velocity: float = 0.0
    size: list[float] = field(default_factory=list)
    reduction_ratio: float = 0.0
    nominal_effort: float = 0.0
    nominal_speed: float = 0.0

    def get_max_effort(self):
        return self.peak_effort * 0.6

    def get_max_vel(self):
        max_vel_rads = self.peak_velocity * 2 * np.pi / 60
        return max_vel_rads * 0.6

    def torque_weight_ratio(self):
        return self.get_max_effort() / self.mass


@dataclass
class RevoluteActuator:
    mass: float = 0.0
    inertia: float = 0.0
    peak_effort: float = 0.0
    peak_velocity: float = 0.0
    size: list[float] = field(default_factory=list)
    reduction_ratio: float = 0.0

    def get_max_effort(self):
        return self.peak_effort * 0.6

    def get_max_vel(self):
        max_vel_rads = self.peak_velocity * 2 * np.pi / 60
        return max_vel_rads * 0.6

    def torque_weight_ratio(self):
        return self.get_max_effort() / self.mass

    def calculate_inertia(self):
        Izz = 1 / 2 * self.mass * self.size[0]
        Iyy = 1 / 12 * self.mass * self.size[1] + 1 / 4 * self.mass * self.size[0]
        Ixx = Iyy
        return np.diag([Ixx, Iyy, Izz])


@dataclass
class RevoluteUnit(RevoluteActuator):
    def __init__(self) -> None:
        self.mass = 0.1
        self.peak_effort = 1000
        self.peak_velocity = 100
        self.size = [0.016, 0.03]


@dataclass
class CustomActuators_KG3(RevoluteActuator):
    def __init__(self):
        self.mass = 3
        self.peak_effort = 420
        self.peak_velocity = 320
        self.size = [0.06, 0.08]
        self.reduction_ratio = 1 / 9


@dataclass
class CustomActuators_KG2(RevoluteActuator):
    def __init__(self):
        self.mass = 2
        self.peak_effort = 180
        self.peak_velocity = 300
        self.size = [0.045, 0.06]
        self.reduction_ratio = 1 / 6


@dataclass
class CustomActuators_KG1(RevoluteActuator):
    def __init__(self):
        self.mass = 1
        self.peak_effort = 80
        self.peak_velocity = 220
        self.size = [0.048, 0.06]
        self.reduction_ratio = 1 / 10


@dataclass
class TMotor_AK10_9(RevoluteActuator):
    def __init__(self):
        self.mass = 0.960
        self.inertia: float = 1002 * 1e-07
        self.peak_effort = 48
        self.peak_velocity = 297.5
        self.size = [0.045, 0.062]
        self.reduction_ratio = 1 / 9
        self.nominal_effort = 18
        self.nominal_speed = 220 * 2 * np.pi / 60


@dataclass
class TMotor_AK70_10(RevoluteActuator):
    def __init__(self):
        self.mass = 0.521
        self.inertia: float = 414 * 1e-07
        self.peak_effort = 24.8
        self.peak_velocity = 382.5
        self.size = [0.0415, 0.05]
        self.reduction_ratio = 1 / 10
        self.nominal_effort = 10
        self.nominal_speed = 310 * 2 * np.pi / 60


@dataclass
class TMotor_AK60_6(RevoluteActuator):
    def __init__(self):
        self.mass = 0.368
        self.inertia: float = 243.5 * 1e-07
        self.peak_effort = 9
        self.peak_velocity = 285
        self.size = [0.034, 0.0395]
        self.reduction_ratio = 1 / 6
        self.nominal_effort = 3
        self.nominal_speed = 420 * 2 * np.pi / 60


@dataclass
class TMotor_AK80_64(RevoluteActuator):
    def __init__(self):
        self.mass = 0.850
        self.inertia: float = 564.5 * 1e-07
        self.peak_effort = 120
        self.peak_velocity = 54.6
        self.size = [0.0445, 0.062]
        self.reduction_ratio = 1 / 64
        self.nominal_effort = 6
        self.nominal_speed = 600 * 2 * np.pi / 60


@dataclass
class TMotor_AK80_9(RevoluteActuator):
    def __init__(self):
        self.mass = 0.485
        self.inertia: float = 607 * 1e-07
        self.peak_effort = 18
        self.peak_velocity = 475
        self.size = [0.0425, 0.0385]
        self.reduction_ratio = 1 / 9
        self.nominal_effort = 9
        self.nominal_speed = 390 * 2 * np.pi / 60

@dataclass
class MIT_Actuator(RevoluteActuator):
    def __init__(self):
        self.mass = 0.440
        self.inertia: float = 0.0023
        self.peak_effort = 17
        self.peak_velocity = 381.97
        self.size = [0.048, 0.04]
        self.reduction_ratio = 1 / 6
        self.continous_effort = 6.9
        self.nominal_effort = 9
        self.nominal_speed = 390 * 2 * np.pi / 60


@dataclass
class Unitree_H1_Motor(RevoluteActuator):
    def __init__(self):
        self.reduction_ratio = 1 / 24
        self.mass = 2.3
        self.inertia: float = 260 * 1e-6
        self.peak_effort = 15 / self.reduction_ratio
        self.peak_velocity = 3000 / 2 / np.pi * 60 * self.reduction_ratio
        self.size = [0.108/2, 0.074]
        self.nominal_effort = 4.5 / self.reduction_ratio
        self.nominal_speed = 2640 * 2 * np.pi / 60 * self.reduction_ratio


@dataclass
class Unitree_GO_Motor(RevoluteActuator):
    def __init__(self):
        self.mass = 0.530
        self.peak_effort = 23.7
        self.peak_velocity = 30 / 2 / np.pi * 60
        self.size = [0.0478, 0.041]
        self.reduction_ratio = 1 / 6.33
        self.nominal_effort = 0.7
        self.nominal_speed = 1600 * 2 * np.pi / 60


@dataclass
class Unitree_B1_Motor(RevoluteActuator):
    def __init__(self):
        self.mass = 1.740
        self.peak_effort = 140
        self.peak_velocity = 297.5
        self.size = [0.0535, 0.074]
        self.reduction_ratio = 1 / 10
        
@dataclass
class Unitree_B2_Motor(RevoluteActuator):
    def __init__(self):
        self.reduction_ratio = 1 / 15
        
        self.mass = 2.2
        self.inertia = 2630 * 1e-07
        self.peak_effort = 13.4 / self.reduction_ratio
        self.peak_velocity = 297.5
        self.size = [0.120/2, 0.075]
        self.nominal_effort = 4.5 / self.reduction_ratio
        self.nominal_speed = 380 * self.reduction_ratio



@dataclass
class Unitree_A1_Motor(RevoluteActuator):
    def __init__(self):
        self.mass = 0.605
        self.peak_effort = 33.5
        self.peak_velocity = 21 / 2 / np.pi * 60
        self.size = [0.0459, 0.044]
        self.reduction_ratio = 1 / 6.33


@dataclass
class MyActuator_RMD_MT_RH_17_100_N(RevoluteActuator):
    def __init__(self):
        self.mass = 0.590
        self.peak_effort = 40
        self.peak_velocity = 100
        self.size = [0.076 / 2, 0.0605]
        self.reduction_ratio = 1 / 36
        self.nominal_effort = 18
        self.nominal_speed = 90 * 2 * np.pi / 60


t_motor_actuators = [
    TMotor_AK10_9(),
    TMotor_AK60_6(),
    TMotor_AK70_10(),
    TMotor_AK80_64(),
    TMotor_AK80_9(),
]

unitree_actuators = [Unitree_A1_Motor(), Unitree_B1_Motor(), Unitree_GO_Motor()]

all_actuators = (
    t_motor_actuators + unitree_actuators + [MyActuator_RMD_MT_RH_17_100_N()]
)

main_actuators = [*t_motor_actuators, Unitree_B2_Motor(), Unitree_H1_Motor()]