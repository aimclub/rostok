import numpy as np
import rule_extention
import mcts
import matplotlib.pyplot as plt

# imports from standard libs
import networkx as nx
from scipy.spatial.transform import Rotation as R
# chrono imports
import pychrono as chrono

from control_optimisation import create_grab_criterion_fun, create_traj_fun
from rostok.graph_grammar.node import GraphGrammar, BlockWrapper
from rostok.block_builder.node_render import ChronoBodyEnv
import rostok.intexp as intexp
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagSlipout, FlagNotContact,
                                                FlagFlyingApart)
from rostok.block_builder.transform_srtucture import FrameTransform
from rostok.utils.result_saver import MCTSReporter, load_reporter
import rostok.graph_generators.graph_environment as env


def get_pipes():

    # Create 3D mesh and setup parameters from files

    obj = BlockWrapper(ChronoBodyEnv)
    obj_db = intexp.chrono_api.ChTesteeObject()
    obj_db.create_chrono_body_from_file('./examples/models/custom/pipe_mul_10.obj',
                                        './examples/models/custom/pipe.xml')
    obj_db.set_chrono_body_ref_frame_in_point(
        chrono.ChFrameD(chrono.ChVectorD(0, 0, -5), chrono.ChQuaternionD(1, 0, 0, 0)))
    center_coord = obj_db.mesh.get_center()
    longest_axis_size = np.argmax(obj_db.bound_box)
    shorthest_axis_size = np.argmin(obj_db.bound_box)
    
    axis = {0:('x',np.array([1,0,0])),1:('y',np.array([0,1,0])),2:('z',np.array([0,0,1]))}
    
    gen_poses = intexp.poses_generator.gen_cylindrical_surface_around_object_axis(obj_db, 1, obj_db.bound_box[shorthest_axis_size]*1.5,
                                                                      obj_db.bound_box[longest_axis_size]/2, axis[longest_axis_size][0])
    
    
    grab_pos_1 = gen_poses[1][0] + obj_db.bound_box[longest_axis_size]*axis[longest_axis_size][1]*0.25
    
    grab_pos_2 = gen_poses[1][0] - obj_db.bound_box[longest_axis_size]*axis[longest_axis_size][1]*0.25
    
    obj_db.clear_grasping_poses_list()
    obj_db.add_grasping_pose(grab_pos_1, [0, 1, 0, 0])
    obj_db.add_grasping_pose(grab_pos_2, [0, 1, 0, 0])
    
    frame_1 = FrameTransform(grab_pos_1, [0, 1, 0, 0])
    frame_2 = FrameTransform(grab_pos_2, [0, 1, 0, 0])
    robot_frames = [frame_1, frame_2]
    return obj, robot_frames


def plot_graph(graph: GraphGrammar):
    plt.figure()
    nx.draw_networkx(graph,
                     pos=nx.kamada_kawai_layout(graph, dim=2),
                     node_size=800,
                     labels={n: graph.nodes[n]["Node"].label for n in graph})
    plt.savefig('grap_pickup_pipe.png')
    plt.show()


#
length_link = [0.4, 0.6, 0.8]
width_flat = [0.25, 0.35, 0.5]

# # %% Create extension rule vocabulary

rule_vocabul, node_features = rule_extention.init_extension_rules(length_link, width_flat)

# # %% Create condig optimizing control

GAIT = 2.5
WEIGHT = [3, 1, 1, 2]

max_time = 10
cfg = ConfigRewardFunction()
cfg.bound = (750, 1000)
cfg.iters = 2
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.0001
cfg.time_sim = max_time
cfg.flags = [
    FlagMaxTime(max_time),
    FlagNotContact(max_time / 4 - 0.2),
    FlagSlipout(max_time / 4 + 0.2, 0.2),
    FlagFlyingApart(4)
]

criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_pipes
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)

# # %% Init mcts parameters

# Hyperparameters mctss
iteration_limit = 3

# Initialize MCTScl
searcher = mcts.mcts(iterationLimit=iteration_limit)
finish = False

G = GraphGrammar()
max_numbers_rules = 5
# Create graph envirenments for algorithm (not gym)
graph_env = env.GraphVocabularyEnvironment(G, rule_vocabul, max_numbers_rules)

graph_env.set_control_optimizer(control_optimizer)

reporter = MCTSReporter.get_instance()
reporter.rule_vocabulary = rule_vocabul
reporter.initialize()

#%% Run first algorithm
iter = 0
while not finish:
    action = searcher.search(initialState=graph_env)
    finish, final_graph, opt_trajectory, path = graph_env.step(action, False)
    iter += 1
    print(
        f"number iteration: {iter}, counter actions: {graph_env.counter_action}"
    )

path = reporter.dump_results()
reporter = load_reporter('results\MCTS_report_22y_12m_30d_03H_30M')
best_graph, reward, best_control = reporter.get_best_info()
# best_control = [float(x) for x in best_control]
func_reward = control_optimizer.create_reward_function_pickup(best_graph)
res = -func_reward(best_control)
plot_graph(best_graph)
print(res)