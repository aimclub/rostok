from pathlib import Path
import mcts

# chrono imports
import pychrono as chrono

from app.control_optimisation import create_grab_criterion_fun, create_traj_fun
from rostok.graph_generators.mcts_helper import make_mcts_step, prepare_mcts_state_and_helper
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagSlipout, FlagNotContact,
                                                FlagFlyingApart)

import rule_grasp_pipe
import pickup_pipes_utils as ppu

PATH_TO_PIPE_OBJ = './examples/models/custom/pipe_mul_10.obj'
PATH_TO_PIPE_XML = './examples/models/custom/pipe.xml'

# # %% Create extension rule vocabulary

rule_vocabul, node_features = rule_grasp_pipe.create_rules_to_pickup_pipe(
    PATH_TO_PIPE_OBJ, PATH_TO_PIPE_XML)

# # %% Create condig optimizing control

GAIT = 2.5
WEIGHT = [3, 1, 1, 2]

max_time = 2
cfg = ConfigRewardFunction()
cfg.bound = (300, 700)
cfg.iters = 2
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.001
cfg.time_sim = max_time
cfg.flags = [FlagMaxTime(max_time),
            FlagNotContact(max_time / 4 - 0.2),
            FlagSlipout(max_time / 4 + 0.2, 0.2),
            FlagFlyingApart(20)]

criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = ppu.create_builder_grab_object(PATH_TO_PIPE_OBJ, PATH_TO_PIPE_XML)
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)
control_optimizer.is_visualize = True

# # %% Init mcts parameters
if __name__ == "__main__":
    # Hyperparameters mctss
    iteration_limit = 3

    # Initialize MCTScl
    searcher = mcts.mcts(iterationLimit=iteration_limit)
    finish = False

    G = GraphGrammar()
    max_numbers_rules = 5 * 3 + 1

    # Create graph environments for algorithm (not gym)
    graph_env = prepare_mcts_state_and_helper(G, rule_vocabul, control_optimizer, max_numbers_rules,
                                            Path("./results"))
    mcts_helper = graph_env.helper
    n_steps = 0
    #%% Run first algorithm
    while not finish:
        finish, graph_env = make_mcts_step(searcher, graph_env, n_steps)
        n_steps += 1
        print(f"number iteration: {n_steps}, counter actions: {graph_env.counter_action} " +
            f"reward: {mcts_helper.report.get_best_info()[1]}")
        
    # %% Save results

    report = mcts_helper.report
    best_graph, reward, best_control = mcts_helper.report.get_best_info()
    func_reward = control_optimizer.create_reward_function(best_graph)
    res = -func_reward(best_control)
    report.plot_means()
    report.make_time_dependent_path()
    report.save()
    report.save_visuals()
    report.save_lists()
