import hyperparameters as hp
from rostok.simulation_chrono.simulation_scenario import ConstTorqueGrasp
from rostok.block_builder_chrono.block_builder_chrono_api import \
    ChronoBlockCreatorInterface as creator
from rostok.criterion.simulation_flags import (FlagContactTimeOut,
                                               FlagFlyingApart, FlagSlipout)
from rostok.criterion.criterion_calculation import (ForceCriterion,
                                                    ObjectCOGCriterion,
                                                    SimulationReward,
                                                    TimeCriterion)
from rostok.trajectory_optimizer.control_optimizer import CounterWithOptimization, CounterWithOptimizationDirect


def config_with_standard(grasp_object_blueprint):
    # configurate the simulation manager
    simulation_manager = ConstTorqueGrasp(hp.TIME_STEP_SIMULATION, hp.TIME_SIMULATION)
    simulation_manager.grasp_object_callback = lambda :creator.create_environment_body(grasp_object_blueprint)
    simulation_manager.add_flag(FlagContactTimeOut(hp.FLAG_TIME_NO_CONTACT))
    simulation_manager.add_flag(FlagFlyingApart(hp.FLAG_FLYING_APART))
    simulation_manager.add_flag(FlagSlipout(hp.FLAG_TIME_SLIPOUT))
    #create criterion manager
    simulation_rewarder = SimulationReward()
    #create criterions and add them to manager
    simulation_rewarder.add_criterion(TimeCriterion(hp.TIME_STEP_SIMULATION), hp.TIME_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(ForceCriterion(), hp.FORCE_CRITERION_WEIGHT)
    simulation_rewarder.add_criterion(ObjectCOGCriterion(), hp.OBJECT_COG_CRITERION_WEIGHT)

    control_optimizer = CounterWithOptimizationDirect(simulation_manager, simulation_rewarder)
    # Init configuration of control optimizing
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