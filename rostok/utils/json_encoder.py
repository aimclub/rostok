import json
from enum import Enum
from typing import Any, Callable


class RostokJSONEncoder(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value
        return [type(o).__name__, o.__dict__]