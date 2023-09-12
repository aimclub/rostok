from pathlib import Path

from tkinter import filedialog, ttk, Tk, NW, END
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mcts_run_setup import config_combination_force_tendon_multiobject, config_combination_force_tendon_multiobject_parallel

from rostok.graph_generators.environments.design_environment import DesignEnvironment, SubDesignEnvironment, SubStringDesignEnvironment, prepare_state_for_optimal_simulation


from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint
from rostok.library.obj_grasp.objects import (get_object_cylinder,get_object_box, get_object_ellipsoid)
from rostok.library.rule_sets.ruleset_simple_fingers import create_rules

from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.mcts_manager import MCTSManager

RULE_VOCABULARY = create_rules()

import hyperparameters as hp

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
        path = filedialog.askdirectory()
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
    control_optimizer = config_combination_force_tendon_multiobject_parallel(*objects_and_weights)
    env = SubStringDesignEnvironment(RULE_VOCABULARY, control_optimizer, hp.MAX_NUMBER_RULES, verbosity=4)
    mcts = MCTS(env)
    # mcts_manager = MCTSManager(mcts, "reoptimization",verbosity=4, use_date=False)
    mcts.load(report_path)
    best_s = env.get_best_states(n)
    simulation_rewarder = control_optimizer.rewarder
    simulation_managers = control_optimizer.simulation_scenario
    for s in best_s:
        full_reward = 0
        G = env.state2graph[s]
        reward, optim_parameters = control_optimizer.calculate_reward(G)
        control = control_optimizer.optim_parameters2data_control(optim_parameters, G)

        simulation_rewarder.verbosity = 1
        for sim_scen, ctrl in zip(simulation_managers, control):
            simulation_output = env.control_optimizer.simulate_with_control_parameters(ctrl, G, sim_scen[0])
            part_reward = simulation_rewarder.calculate_reward(simulation_output, True)
            reward = simulation_rewarder.calculate_reward(simulation_output)
            full_reward += reward*sim_scen[1]
            print("=====================================")
            print(f"Object: {sim_scen[0].grasp_object_callback}")
            print(f"New control: {ctrl.forces}, old control: {env.terminal_states[s][1]}")
            print(f"Partial reward: {part_reward}")
            print(f"Reward: {reward}")
            print("=====================================")
        print(f"Full reward: {full_reward}, old reward: {env.terminal_states[s][0]}")



if __name__ == "__main__":
    grasp_object_blueprints = ([
        get_object_box(0.25, 0.146, 0.147, 0, mass=0.164),
        get_object_ellipsoid(0.14, 0.14, 0.22, 0, mass=0.188),
        get_object_cylinder(0.155/2, 0.155, 0, mass = 0.261),
    ], [1, 1, 1])
    reoptimize_nth_graph(1, grasp_object_blueprints)
