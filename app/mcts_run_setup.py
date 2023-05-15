


def config_with_standard():
    WEIGHT = hp.CRITERION_WEIGHTS
    # Init configuration of control optimizing
    cfg = ConfigVectorJoints()
    cfg.bound = (6, 15)
    cfg.iters = hp.CONTROL_OPTIMIZATION_ITERATION
    cfg.time_step = hp.TIME_STEP_SIMULATION
    cfg.time_sim = hp.TIME_SIMULATION
    cfg.flags = [FlagMaxTime(cfg.time_sim), 
                 FlagNotContact(hp.FLAG_TIME_NO_CONTACT), 
                 FlagSlipout(hp.FLAG_TIME_NO_CONTACT, hp.FLAG_TIME_SLIPOUT)]
    """Wraps function call"""
    criterion_callback = partial(criterion_calc, weights=WEIGHT)
    traj_generator_fun = partial(create_torque_traj_from_x,
                                 stop_time=cfg.time_sim,
                                 time_step=cfg.time_step)

    cfg.criterion_callback = criterion_callback
    cfg.params_to_timesiries_callback = traj_generator_fun
    return cfg