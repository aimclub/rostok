from copy import deepcopy
from auto_robot_design.description.builder import jps_graph2urdf_by_bulder
import numpy as np




from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.problems import CalculateMultiCriteriaProblem
from auto_robot_design.optimization.saver import load_checkpoint

def get_optimizer_and_problem(path) -> tuple[PymooOptimizer, CalculateMultiCriteriaProblem]:
    problem = CalculateMultiCriteriaProblem.load(
        path)
    checklpoint = load_checkpoint(path)

    optimizer = PymooOptimizer(problem, checklpoint)
    optimizer.load_history(path)
    res = optimizer.run()

    return optimizer, problem, res


def get_pareto_sample_linspace(res, sample_len: int):

    sample_indices = np.linspace(0, len(res.F) - 1, sample_len, dtype=int)
    sample_x = res.X[sample_indices]
    sample_F = res.F[sample_indices]

    return sample_x, sample_F


def get_pareto_sample_histogram(res, sample_len: int):
    """Histogram uses 0 from reword vector

    Args:
        res (_type_): _description_
        sample_len (int): _description_

    Returns:
        _type_: _description_
    """
    rewards = res.F
    _, bins_edg = np.histogram(rewards[:,0], sample_len)
    bin_indices = np.digitize(rewards[:,0], bins_edg, right=True)
    bins_set_id = [np.where(bin_indices == i)[0]
                   for i in range(1, len(bins_edg))]
    best_in_bins = [i[0] for i in bins_set_id]
    sample_F = rewards[best_in_bins]
    sample_X = res.X[best_in_bins]
    return sample_X, sample_F

def get_urdf_from_problem(sample_X: np.ndarray, problem: CalculateMultiCriteriaProblem):
    problem.mutate_JP_by_xopt(problem.initial_xopt)
    graphs = []
    urdf_j_des_l_des = []
    for x_i in sample_X:
        problem.mutate_JP_by_xopt(x_i)
        mutated_graph = deepcopy(problem.graph)

        robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
            mutated_graph, problem.builder)
        graphs.append(mutated_graph)
        urdf_j_des_l_des.append(
            (robo_urdf, joint_description, loop_description))
    return graphs, urdf_j_des_l_des