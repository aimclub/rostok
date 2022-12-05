# chrono imports
import pychrono as chrono

from stable_baselines3 import PPO
import gymnasium as gym
import rostok.gym_rostok

from app.control_optimisation import create_grab_criterion_fun, create_traj_fun, get_object_to_grasp
import app.rule_extention as rule_extention

# imports from standard libs
import rostok.gym_rostok.envs as env
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact

rule_vocabul, node_features = rule_extention.init_extension_rules()

# %% Create condig optimizing control

GAIT = 2.5
WEIGHT = [1, 1, 1, 1]

cfg = ConfigRewardFunction()
cfg.bound = (0, 10)
cfg.iters = 1
cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
cfg.time_step = 0.0015
cfg.time_sim = 3
cfg.flags = [FlagMaxTime(3), FlagNotContact(1), FlagSlipout(1, 0.5)]
"""Wraps function call"""

criterion_callback = create_grab_criterion_fun(node_features, GAIT, WEIGHT)
traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

cfg.criterion_callback = criterion_callback
cfg.get_rgab_object_callback = get_object_to_grasp
cfg.params_to_timesiries_callback = traj_generator_fun

control_optimizer = ControlOptimizer(cfg)

# Create gym for algorithm
# graph_env = env.GGrammarControlOpimizingEnv(rule_vocabul,
#                                             control_optimizer,
#                                             render_mode="grammar")

# # %%

# model = PPO("MlpPolicy", graph_env)
# model.learn(total_timesteps=100)
# vec_env = model.get_env()
# obs = vec_env.reset()

# # %% Run first algorithm

# graph_env.set_max_number_nonterminal_rules(5)
# for id in range(100):
#     print(f"___{id}___")
#     action, _space = model.predict(obs)
#     s, r, done_iteration, w, info = graph_env.step(action)
#     print(f"{action=:2}: {r=:0.2f}, {done_iteration=}, {info=}")
#     if done_iteration:
#         graph_env.reset()

# %%  Testing module gym

kwargs = {"rule_vocabulary":rule_vocabul, "controller":control_optimizer, "render_mode":"grammar"}
#FIXME: The not found `gym_rostok` module
env = gym.make("gym_rostok/GGrammarControlOpimizingEnv-v0", **kwargs)
env.reset()
action = [0, 4, 4, 5, 20, 11, 29, 29, 30, 30, 8, 9]
for k, a in enumerate(action):
    s, r, done, w, info = env.step(a)
    print(k, a)