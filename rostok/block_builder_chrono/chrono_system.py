import pychrono.core as chrono
from typing import Optional


_chrono_system:Optional[chrono.ChSystem] = None

def reset_chrono_system():
    global _chrono_system
    _chrono_system = chrono.ChSystemNSC()
    _chrono_system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    _chrono_system.SetSolverMaxIterations(100)
    _chrono_system.SetSolverForceTolerance(1e-6)
    _chrono_system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    _chrono_system.Set_G_acc(chrono.ChVectorD(0, 0, 0))

def register_chrono_system(system: chrono.ChSystem):
    global _chrono_system
    _chrono_system = system

def get_chrono_system() ->chrono.ChSystem:
    return _chrono_system
