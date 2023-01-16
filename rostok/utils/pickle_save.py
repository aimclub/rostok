import pickle
from datetime import datetime
from pathlib import Path
from rostok.utils.states import OptimizedGraph


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
        self.path = Path(self.path, "MCTS_report_" + datetime.now().strftime("%yy_%mm_%dd_%HH_%MM"))
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def save(self):
        if self.path is None:
            raise Exception("Set the path to save for", type(self))

        with open(Path(self.path, self.file_name + '.pickle'), "wb+") as file:
            pickle.dump(self, file)

def convert_control_to_list(control):
    if control is None:
        control = []
    elif isinstance(control, (float,int)):
        control = [control]

    return list(control)

class OptimizedGraphReport(Saveable):

    def __init__(self, path = Path("./result")) -> None:
        super().__init__(path, 'optimized_graph_report')
        self.graph_list: list[OptimizedGraph]=[]

    def add_graph(self, graph, reward, control):
        """Add a graph, reward and control to seen_graph

        Args:
            graph (GraphGrammar): the state of the main design
            reward (float): the main reward obtained during MCTS search
            control: parameters of the control for main design"""

        control =  convert_control_to_list(control)
        new_optimized_graph = OptimizedGraph(graph, reward, control)
        self.graph_list.append(new_optimized_graph)
