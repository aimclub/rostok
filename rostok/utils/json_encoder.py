import json
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable
import numpy as np


class RostokJSONEncoder(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value
        elif (isinstance(o, MappingProxyType)):
            return [type(o).__name__]
        elif (isinstance(o, np.ndarray)):
            return [type(o).__name__, str(o)]

        return [type(o).__name__, o.__dict__]