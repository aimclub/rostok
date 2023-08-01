from pathlib import Path
from tkinter import *
from tkinter import filedialog, ttk
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mcts_run_setup import config_with_standard, config_with_standard_graph

from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.graph_generators.mcts_helper import (MCTSSaveable, OptimizedGraphReport)
from rostok.library.obj_grasp.objects import (get_obj_hard_mesh_piramida,
                                              get_object_parametrized_sphere)
#from rostok.library.rule_sets.ruleset_old_style import create_rules
from rostok.utils.pickle_save import load_saveable


def vis_top_n_mechs(n: int, object: EnvironmentBodyBlueprint):
    root = Tk()
    root.geometry("400x300")
    root.title("Report loader")
    label = ttk.Label(text="Choose path to report!")
    label.pack()
    label_file = ttk.Label(text="Enter path to report:")
    label_file.pack(anchor=NW, padx=8)
    entry_browse = ttk.Entry(width=30, font=12)
    entry_browse.pack(anchor=NW, padx=8)
    entry_browse.place(x=8, y=40)
    report_path = None

    def func_browse():
        path = filedialog.askopenfilename()
        nonlocal entry_browse
        entry_browse.delete(first=0, last=END)
        entry_browse.insert(0, path)

    def func_add():
        nonlocal report_path
        report_path = Path(entry_browse.get())
        nonlocal root
        root.destroy()

    btn_browse = ttk.Button(text="browse", command=func_browse)  # создаем кнопку из пакета ttk
    btn_browse.place(x=300, y=40)
    btn_add = ttk.Button(text="add report", command=func_add)  # создаем кнопку из пакета ttk
    btn_add.place(x=300, y=85)

    root.mainloop()
    report = load_saveable(report_path)
    graph_report = report.seen_graphs
    control_optimizer = config_with_standard(grasp_object_blueprint)
    simulation_rewarder = control_optimizer.rewarder
    simulation_manager = control_optimizer.simulation_scenario
    graph_list = graph_report.graph_list

    sorted_graph_list = sorted(graph_list, key=lambda x: x.reward)
    some_top = sorted_graph_list[-1:-(n + 1):-1]
    for graph in some_top:
        G = graph.graph
        reward = graph.reward
        control = graph.control
        data = {"initial_value": control}
        simulation_output = simulation_manager.run_simulation(G, data, True)
        res = simulation_rewarder.calculate_reward(simulation_output)
        print(reward)
        print(res)
        print()


if __name__ == "__main__":
    grasp_object_blueprint = get_object_parametrized_sphere(0.5)
    vis_top_n_mechs(3, grasp_object_blueprint)
