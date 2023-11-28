from copy import deepcopy
from rostok.trajectory_optimizer.control_optimizer import TendonForceOptiVar
import tendon_driven_cfg
import rostok.pipeline.generate_grasper_cfgs as pipeline

tendon_optivar_standart = TendonForceOptiVar(tendon_driven_cfg.get_default_tendon_params(),
                                             params_start_pos=-45)
# Standart config 
evaluator_tendon_standart = pipeline.create_reward_calulator(
    tendon_driven_cfg.default_simulation_config, tendon_driven_cfg.default_grasp_objective,
    tendon_optivar_standart, tendon_driven_cfg.brute_force_opti_default_cfg)

# Standart + parallel config
brute_force_opti_default_cfg_parallel = deepcopy(tendon_driven_cfg.brute_force_opti_default_cfg)
brute_force_opti_default_cfg_parallel.num_cpu_workers = 6

evaluator_tendon_standart_parallel = pipeline.create_reward_calulator(
    tendon_driven_cfg.default_simulation_config, tendon_driven_cfg.default_grasp_objective,
    tendon_optivar_standart, brute_force_opti_default_cfg_parallel)


# More powerfull tendon + parallel config
tendon_optivar_strong = TendonForceOptiVar(tendon_driven_cfg.get_default_tendon_params(),
                                           params_start_pos=-60)
brute_force_opti_strong = deepcopy(brute_force_opti_default_cfg_parallel)
brute_force_opti_strong.variants = [30, 40, 50]

simulation_config_strong = deepcopy(tendon_driven_cfg.default_simulation_config)
simulation_config_strong.obj_disturbance_forces = tendon_driven_cfg.get_random_force_with_null_grav(
    150)

evaluator_tendon_strong = pipeline.create_reward_calulator(
    simulation_config_strong, tendon_driven_cfg.default_grasp_objective, tendon_optivar_strong,
    brute_force_opti_strong)

# Fast single thread config for debug
tendon_optivar_debug = deepcopy(tendon_optivar_standart)
brute_force_debug_cfg = deepcopy(tendon_driven_cfg.brute_force_opti_default_cfg)
brute_force_debug_cfg.variants = [40]
tendon_optivar_debug.is_vis = True
evaluator_tendon_fast_debug = pipeline.create_reward_calulator(
    tendon_driven_cfg.fast_mock_simulation_config, tendon_driven_cfg.default_grasp_objective,
    tendon_optivar_debug, brute_force_debug_cfg)


mcts_hyper_default = pipeline.MCTSCfg()
mcts_hyper_short_fast = pipeline.MCTSCfg(max_number_rules=10, full_loop=8)
mcts_hyper_long = pipeline.MCTSCfg(max_number_rules=34)