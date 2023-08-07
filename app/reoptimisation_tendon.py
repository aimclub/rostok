from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tkinter import filedialog, ttk, Tk, NW, END

import matplotlib.pyplot as plt
import numpy as np
from mcts_run_setup import (config_cable_multiobject)

from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.graph_generators.mcts_helper import (MCTSSaveable, OptimizedGraphReport)
from rostok.library.obj_grasp.objects import (get_obj_hard_mesh_piramida,
                                              get_object_parametrized_sphere, get_object_cylinder,
                                              get_object_box, get_object_ellipsoid)
#from rostok.library.rule_sets.ruleset_old_style import create_rules
from rostok.utils.pickle_save import load_saveable


def reoptimize_nth_graph(n: int, objects_and_weights: Tuple[List[EnvironmentBodyBlueprint],List[int]]):
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
    control_optimizer = config_cable_multiobject(*objects_and_weights)
    control_optimizer.limit = 16
    simulation_rewarder = control_optimizer.rewarder
    simulation_managers = control_optimizer.simulation_scenario
    graph_list = graph_report.graph_list
    sorted_graph_list = sorted(graph_list, key=lambda x: -x.reward)
    graph = sorted_graph_list[n]
    G = graph.graph
    reward = graph.reward
    print(graph.control)
    reward, optim_parameters = control_optimizer.calculate_reward(G)
    control = control_optimizer.optim_parameters2data_control(optim_parameters, G)
    print(control)
    simulation_rewarder.verbosity = 1
    i = 0
    for simulation_scenario in simulation_managers:

        simulation_output = simulation_scenario[0].run_simulation(G, control[i], True, False)
        res = simulation_rewarder.calculate_reward(simulation_output)
        print(res)
        print()
        i += 1


if __name__ == "__main__":
    grasp_object_blueprints = [[
        get_object_box(0.8, 1, 0.4, 10),
        get_object_cylinder(0.6, 0.9, 10),
        get_object_parametrized_sphere(0.6)
    ], [1, 1, 1]]
    reoptimize_nth_graph(0, grasp_object_blueprints)