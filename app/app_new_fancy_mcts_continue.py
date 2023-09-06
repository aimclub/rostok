import numpy as np
import hyperparameters as hp
from rostok.graph_generators.environments.design_environment import DesignEnvironment, SubDesignEnvironment, SubStringDesignEnvironment
from rostok.graph_generators.search_algorithms.mcts import MCTS
from rostok.graph_generators.mcts_manager import MCTSManager
from rostok.graph_generators.search_algorithms.random_search import RandomSearch

from rostok.library.rule_sets.ruleset_old_style_smc import create_rules
from rostok.graph_grammar.node import GraphGrammar
from rostok.library.obj_grasp.objects import get_object_parametrized_sphere, get_object_cylinder, get_object_box
import sys

from mcts_run_setup import config_combination_force_tendon_multiobject

from pathlib import Path
from tkinter import *
from tkinter import filedialog, ttk
from typing import Any, Dict, List, Optional, Tuple
import os

root = Tk()
root.geometry("400x300")
root.title("Checkpoint loader")
label = ttk.Label(text="Choose path to directory!")
label.pack()
label_file = ttk.Label(text="Enter path to directory:")
label_file.pack(anchor=NW, padx=8)
entry_browse = ttk.Entry(width=30, font=12)
entry_browse.pack(anchor=NW, padx=8)
entry_browse.place(x=8, y=40)
report_path: Path = None

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

name_directory = report_path.name
btn_browse = ttk.Button(text="browse", command=func_browse)  # создаем кнопку из пакета ttk
btn_browse.place(x=300, y=40)
btn_add = ttk.Button(text="add directory", command=func_add)  # создаем кнопку из пакета ttk
btn_add.place(x=300, y=85)

rule_vocabulary = create_rules()
grasp_object_blueprint = []
grasp_object_blueprint.append(get_object_parametrized_sphere(0.11))
grasp_object_blueprint.append(get_object_cylinder(0.07, 0.09, 0))
grasp_object_blueprint.append(get_object_box(0.12, 0.12, 0.1, 0))
# create reward counter using run setup function
control_optimizer = config_combination_force_tendon_multiobject(grasp_object_blueprint, [ 1, 1, 1])

init_graph = GraphGrammar()
env = SubStringDesignEnvironment(rule_vocabulary, control_optimizer, 13, init_graph, 4)

mcts = MCTS(env)
name_directory = input("enter directory name")
mcts_manager = MCTSManager(mcts, name_directory,verbosity=4)
mcts_manager.save_information_about_search(hp)

for i in range(10):
    mcts_manager.run_search(10, 1, iteration_checkpoint=1, num_test=3)
    mcts_manager.save_results()
# state = env.initial_state
# trajectory = [state]
# while not env.is_terminal_state(state)[0]:
#     for __ in range(10):
#         mcts.search(state)
    
#     pi = mcts.get_policy(state)
#     a = max(env.actions, key=lambda x: pi[x])
#     state, reward, is_terminal_state, __ = env.next_state(state, a)
#     print(f"State: {state}, Reward: {reward}, is_terminal_state: {is_terminal_state}")
#     trajectory.append(state)
# env.save_environment("test")
# mcts.save("test")
# print(f"Trajectory: {trajectory}")