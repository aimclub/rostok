from pathlib import Path
from tkinter import *
from tkinter import filedialog, ttk
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


from rostok.block_builder_api.block_blueprints import EnvironmentBodyBlueprint

from rostok.graph_generators.environments.design_environment import DesignEnvironment, SubDesignEnvironment, SubStringDesignEnvironment, prepare_state_for_optimal_simulation

from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.mcts_manager import MCTSManager

# =============================================================================
from rostok.library.rule_sets.ruleset_old_style_smc import create_rules
from mcts_run_setup import config_with_standard_graph, config_combination_force_tendon_multiobject

from rostok.library.obj_grasp.objects import (get_object_parametrized_sphere,get_object_cylinder,get_object_box)


INIT_GRAPH = GraphGrammar()
RULE_VOCABULARY = create_rules()


def vis_top_n_mechs(n: int, env: DesignEnvironment):
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
    mcts = MCTS(env)
    mcts_manager = MCTSManager(mcts, "last_seen_results",verbosity=4, use_date=False)
    mcts.load(report_path)
    best_s = env.get_best_states(n)

    simulation_rewarder = env.control_optimizer.rewarder
    simulation_manager = env.control_optimizer.simulation_scenario
    for s in best_s:
        data, graph = prepare_state_for_optimal_simulation(s, env)
        for d, sim_scen in zip(data, simulation_manager):
            simulation_output = env.control_optimizer.simulate_with_control_parameters(d, graph, sim_scen[0])
            reward = simulation_rewarder.calculate_reward(simulation_output, True)
            full_reward = simulation_rewarder.calculate_reward(simulation_output)
            print("=====================================")
            print(f"Object: {simulation_manager.grasp_object_callback()}")
            print(f"Reward: {reward}")
            print(f"Full reward: {full_reward}, old reward: {env.terminal_states[s]}")
            print("=====================================")


if __name__ == "__main__":
    top = 3
    
    grasp_object_blueprint = []
    grasp_object_blueprint.append(get_object_parametrized_sphere(0.11))
    grasp_object_blueprint.append(get_object_cylinder(0.07, 0.09, 0))
    grasp_object_blueprint.append(get_object_box(0.12, 0.12, 0.1, 0))
    
    control_optimizer = config_combination_force_tendon_multiobject(grasp_object_blueprint, [1,1,1])

    env = SubStringDesignEnvironment(RULE_VOCABULARY, control_optimizer, 13, INIT_GRAPH, 4)
    
    vis_top_n_mechs(top, env)
