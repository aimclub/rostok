from pathlib import Path
from tkinter import *
from tkinter import filedialog, ttk
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mcts_run_setup import config_with_standard, config_with_standard_graph, config_with_standard_tendon

from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.graph_generators.mcts_helper import (MCTSSaveable, OptimizedGraphReport)
from rostok.library.obj_grasp.objects import (get_obj_hard_mesh_piramida,
                                              get_object_parametrized_sphere, get_object_parametrized_tilt_ellipsoid)
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
    control_optimizer = config_with_standard_tendon(grasp_object_blueprint)
    simulation_rewarder = control_optimizer.rewarder
    simulation_manager = control_optimizer.simulation_scenario
    graph_list = graph_report.graph_list
    simulation_rewarder.verbosity = 1
    sorted_graph_list = sorted(graph_list, key=lambda x: x.reward)
    some_top = sorted_graph_list[-1:-(n + 1):-1]
    for graph in some_top:
        G = graph.graph
        reward = graph.reward
        print(graph.control)
        control = control_optimizer.optim_parameters2data_control(graph.control, G)
        print(control)
        #data = {"initial_value": control}
        simulation_output = simulation_manager.run_simulation(G, control, True, True)
        res = simulation_rewarder.calculate_reward(simulation_output)
        print(reward)
        print(res)
        print()


def save_svg_mean_reward(name: str,
                         objecy_name: str,
                         step_limit=27,
                         path=None,
                         filter: bool = False):
    report_paths = []
    labels = []
    root = Tk()
    root.geometry("400x300")
    root.title("Report loader")
    label = ttk.Label(text="Choose path to report and plot label!")
    label.pack()
    label_file = ttk.Label(text="Enter path to report:")
    label_file.pack(anchor=NW, padx=8)
    entry_browse = ttk.Entry(width=30, font=12)
    entry_browse.pack(anchor=NW, padx=8)
    entry_browse.place(x=8, y=40)
    label_label = ttk.Label(text="Enter plot label:")
    label_label.place(x=8, y=65)

    entry_label = ttk.Entry(font=12)
    entry_label.place(x=8, y=85)

    def func_browse():
        path = filedialog.askopenfilename()
        nonlocal entry_browse
        entry_browse.delete(first=0, last=END)
        entry_browse.insert(0, path)

    def func_add():
        labels.append(entry_label.get())
        report_paths.append(Path(entry_browse.get()))
        entry_browse.delete(first=0, last=END)
        entry_label.delete(first=0, last=END)
        # saved_label = ttk.Label(text=entry_browse.get()+"  "+ entry_label.get())
        # saved_label.pack()

    def func_finish():
        nonlocal root
        root.destroy()

    btn_browse = ttk.Button(text="browse", command=func_browse)  # создаем кнопку из пакета ttk
    btn_browse.place(x=300, y=40)
    btn_add = ttk.Button(text="add report", command=func_add)  # создаем кнопку из пакета ttk
    btn_add.place(x=300, y=85)
    btn_finish = ttk.Button(text="finish", command=func_finish)  # создаем кнопку из пакета ttk

    btn_finish.place(x=150, y=250)
    root.mainloop()
    reporter = []

    for report_path in report_paths:
        report = load_saveable(Path(report_path))
        reporter.append(report)
    legend = []
    for entry in labels:
        legend.append(entry)
    if path is None:
        path = "./results/figures/" + name + ".svg"
    arr_mean_rewards = []

    for report in reporter:
        rewards = []
        for state in report.seen_states.state_list:
            i = state.step
            if len(rewards) == i:
                rewards.append([state.reward])
            else:
                rewards[i].append(state.reward)
        if filter:
            for step, value_step in enumerate(rewards):
                rewards[step] = list(filter(lambda x: x != 0, value_step))
        mean_rewards = [np.mean(on_step_rewards) for on_step_rewards in rewards]
        arr_mean_rewards.append(mean_rewards[0:step_limit])

    if len(legend) < len(reporter):
        for _ in range(len(reporter) - len(legend)):
            legend.append('unknown')
    plt.figure()
    plt.xlabel("Steps")
    plt.ylabel("Rewards")
    plt.title(f"Non-terminal rules: {reporter[0].non_terminal_rules_limit}. Object: {objecy_name}")
    for m_reward in arr_mean_rewards:
        plt.plot(m_reward)
    plt.legend(legend, loc="upper left")
    plt.savefig(path, format="svg")


if __name__ == "__main__":
    #grasp_object_blueprint = get_object_parametrized_sphere(0.4, 1)
    grasp_object_blueprint = get_object_parametrized_tilt_ellipsoid(1, 0.8, 1.4, 10)
    vis_top_n_mechs(5, grasp_object_blueprint)
    # save_svg_mean_reward( name = 'kek', objecy_name='sphere')