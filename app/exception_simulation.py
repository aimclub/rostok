from abc import ABC
import pychrono as chrono

class ChronoError(Exception,ABC):
    pass

class ChronoSimulationError(Exception):
    def __init__(self, chrono_system: chrono.ChSystem) -> None:
        self.system = chrono_system
        
        super().__init__(f"")