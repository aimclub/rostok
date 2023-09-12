from pathlib import Path
from tkinter import *
from tkinter import filedialog, ttk
from typing import Optional, Tuple



from rostok.graph_generators.environments.design_environment import DesignEnvironment, SubDesignEnvironment, SubStringDesignEnvironment, prepare_state_for_optimal_simulation

from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.mcts_manager import MCTSManager

# =============================================================================
# from rostok.library.rule_sets.ruleset_old_style_smc import create_rules
from rostok.library.rule_sets.ruleset_simple_fingers import create_rules
from mcts_run_setup import config_combination_force_tendon_multiobject, config_combination_force_tendon_multiobject_parallel

from rostok.library.obj_grasp.objects import (get_object_cylinder,get_object_box, get_object_ellipsoid)

import hyperparameters as hp

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
    # mcts_manager = MCTSManager(mcts, "last_seen_results",verbosity=4, use_date=False)
    mcts.load(report_path)
    best_s = env.get_best_states(n)

    simulation_rewarder = env.control_optimizer.rewarder
    simulation_manager = env.control_optimizer.simulation_scenario

    for s in best_s:
        full_reward = 0
        # env.terminal_states[s] = (env.terminal_states[s], [env.terminal_states[s][1] for __ in range(len(simulation_manager))])
        data, graph = prepare_state_for_optimal_simulation(s, env)
        # data = [data for __ in range(len(simulation_manager))]
        for d, sim_scen in zip(data, simulation_manager):
            simulation_output = env.control_optimizer.simulate_with_control_parameters(d, graph, sim_scen[0])
            part_reward = simulation_rewarder.calculate_reward(simulation_output, True)
            reward = simulation_rewarder.calculate_reward(simulation_output)
            full_reward += reward*sim_scen[1]
            print("=====================================")
            print(f"Object: {sim_scen[0].grasp_object_callback}")
            print(f"Force control: {d.forces}, control params: {env.terminal_states[s][1]}")
            print(f"Partial reward: {part_reward}")
            print(f"Reward: {reward}")
            print("=====================================")
        print(f"Full reward: {full_reward}, old reward: {env.terminal_states[s][0]}")


if __name__ == "__main__":
    top = 3
    
    grasp_object_blueprint = []
    grasp_object_blueprint.append(get_object_box(0.25, 0.146, 0.147, 0, mass=0.164))
    grasp_object_blueprint.append(get_object_ellipsoid(0.14, 0.14, 0.22, 0, mass=0.188))
    grasp_object_blueprint.append(get_object_cylinder(0.155/2, 0.155, 0, mass = 0.261))
    
    # control_optimizer = config_combination_force_tendon_multiobject(grasp_object_blueprint, [1,1,1])
    control_optimizer = config_combination_force_tendon_multiobject_parallel(
        grasp_object_blueprint, [1, 1, 1])

    init_graph = GraphGrammar()
    env = SubStringDesignEnvironment(RULE_VOCABULARY, control_optimizer, hp.MAX_NUMBER_RULES, init_graph, 4)
    
    vis_top_n_mechs(top, env)
