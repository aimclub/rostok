import pickle
from datetime import datetime
from pathlib import Path


def load_saveable(path):
    with open(path, "rb") as file:
        return  pickle.load(file)

class Saveable():

    def __init__(self, path = None, filename = 'default'):
        self.path = path
        self.file_name = filename 

    def set_path(self, path):
        self.path = path

    def make_time_dependent_path(self):
        time = datetime.now()
        time = str(time.date()) + "_" + str(time.hour) + "-" + str(time.minute) + "-" + str(
            time.second)
        self.path = Path(self.path, "Reports_" + datetime.now().strftime("%yy_%mm_%dd_%HH_%MM"))
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def save(self):
        if self.path is None:
            raise Exception("Set the path to save for", type(self))

        with open(Path(self.path, self.file_name + '.pickle'), "wb+") as file:
            pickle.dump(self, file)
