import multiprocessing
import streamlit as st
import dill
import sys
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import StarmapParallelization
from pathlib import Path
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.problems import (MultiCriteriaProblem,
                                                     SingleCriterionProblem)
from auto_robot_design.optimization.saver import ProblemSaver

if __name__ == "__main__":
    with open(Path(f"./results/optimization_widget/user_{int(sys.argv[1])}/buffer/data.pkl"), "rb") as f:
        data = dill.load(f)
    N_PROCESS = 10
    pool = multiprocessing.Pool(N_PROCESS)
    runner = StarmapParallelization(pool.starmap)
    population_size = 64
    n_generations = 30
    graph_manager = data[0]
    builder = data[1]
    reward_manager = data[3]
    soft_constraint = data[4]
    actuator = builder.actuator['default']
    num_objs = reward_manager.close_trajectories()
    print(num_objs)
    if num_objs > 1:
        # create the problem for the current optimization
        problem = MultiCriteriaProblem(graph_manager, builder, reward_manager,
                                       soft_constraint, elementwise_runner=runner, Actuator=actuator)

        algorithm = AGEMOEA2(pop_size=population_size, save_history=True)
    else:
        problem = SingleCriterionProblem(graph_manager, builder, reward_manager,
                                         soft_constraint, elementwise_runner=runner, Actuator=actuator)
        algorithm = PSO(pop_size=population_size, save_history=True)
    saver = ProblemSaver(
        problem, Path(f"optimization_widget\\user_{int(sys.argv[1])}\\current_results"), False)
    saver.save_nonmutable()
    optimizer = PymooOptimizer(problem, algorithm, saver)

    res = optimizer.run(
        True, **{
            "seed": 2,
            "termination": ("n_gen", n_generations),
            "verbose": True
        })
