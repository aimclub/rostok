import pickle
from datetime import datetime
from pathlib import Path


def load_saveable(path):
    """The function to load a saveable object.
    
    Args:
        path (Path): path to the corresponding pickle file
    """
    with open(path, "rb+") as file:
        return  pickle.load(file)

class Saveable():
    """Class that represents the objects that can be saved using pickle module.

    CLasses inherited from that class would obtain functionality to save objects
    and set the path to save.

    Attributes:
        path (Path): a path to directory for saving the object
        file_name (str): name of the file to save the object
    """
    def __init__(self, path = None, file_name = 'default'):
        """Set initial path and file_name.

        Args:
            path (Path): path to set
            file_name (str): file name to set
        """
        self.path = str(path)
        self.file_name = file_name

    def set_path(self, path:Path):
        """Set new path to directory for file save.

        Args:
            path (Path): new path to directory
        """
        if isinstance(path, str):
            path = Path(path)
        self.path = str(path)
        path.mkdir(parents=True, exist_ok=True)

    def make_time_dependent_path(self):
        """Set path to new directory with name dependent on current time.
        
        The new directory is created in the current path directory and path
        is set to the newly created directory.
        """
        time = datetime.now()
        time = str(time.date()) + "_" + str(time.hour) + "-" + str(time.minute) + "-" + str(
            time.second)
        path = Path(self.path, "Reports_" + datetime.now().strftime("%yy_%mm_%dd_%HH_%MM"))
        path.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        return path

    def save(self):
        """Save the object as pickle file with name at the path directory using file_name."""
        if self.path is None:
            raise Exception("Set the path to save for", type(self))

        with open(Path(self.path, self.file_name + '.pickle'), "wb+") as file:
            pickle.dump(self, file)
