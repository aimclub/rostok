import sys
import time
from pathlib import Path

# chrono imports
import pychrono as chrono
import rule_extention
from control_optimisation import (create_grab_criterion_fun, create_traj_fun, get_object_to_grasp)

from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact, FlagSlipout)
from rostok.graph_generators.mcts_helper import OptimizedGraphReport
from rostok.graph_grammar.graphgrammar_explorer import ruleset_explorer
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import (ConfigRewardFunction, ControlOptimizer)

rule_vocabul, node_features = rule_extention.init_extension_rules()

start = time.time()
out = ruleset_explorer(2, rule_vocabul)
ex = time.time() - start

print(f"time :{ex}")
print(f"Non-uniq graphs :{out[1]}")
print(f"Uniq graphs :{len(out[0])}")

# %% Create config for control optimizer
GAIT = 2.5
WEIGHT = [3, 1, 1, 2]
cfg = ConfigRewardFunction()
cfg.bound = (2, 10)
cfg.iters = 5
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.005
cfg.time_sim = 2
cfg.flags = [FlagMaxTime(2), FlagNotContact(1), FlagSlipout(0.5, 0.5)]
criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)
cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun
control_optimizer = ControlOptimizer(cfg)

report = OptimizedGraphReport(Path("./results/MCTS_nonterminal_depth_4_two_finger"))

path = report.make_time_dependent_path()
with open(Path(path, "counter.txt"), 'w') as file:
    original_stdout = sys.stdout
    sys.stdout = file
    print(f"time :{ex}")
    print(f"Non-uniq graphs :{out[1]}")
    print(f"Uniq graphs :{len(out[0])}")
    sys.stdout = original_stdout

reward_list = []
i = 0
for graph in out[0]:
    result_optimizer = control_optimizer.start_optimisation(graph)
    reward = -result_optimizer[0]
    movments_trajectory = result_optimizer[1]
    report.add_graph(graph, reward, movments_trajectory)
    reward_list.append(reward)
    i += 1
    if i % 100 == 0:
        print(i)

print(max(reward_list))
report.save()
