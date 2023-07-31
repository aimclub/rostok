import json
from typing import Any


class RostokJSONEncoder(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        return [type(o).__name__, o.__dict__]