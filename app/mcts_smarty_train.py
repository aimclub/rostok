import logging

import numpy as np

import rule_extention as re
from rostok.graph_generators.graph_game import GraphGrammarGame as Game
from rostok.neural_network.wrappers import AlphaZeroWrapper
from MCTS import MCTS

from control_optimisation import (create_grab_criterion_fun, create_traj_fun, get_object_to_grasp)

from rostok.criterion.flags_simualtions import (FlagMaxTime, FlagNotContact, FlagSlipout)
from rostok.graph_generators.mcts_helper import (make_mcts_step, prepare_mcts_state_and_helper)
from rostok.graph_grammar.node import GraphGrammar
import pychrono as chrono
from rostok.trajectory_optimizer.control_optimizer import (ConfigRewardFunction, ControlOptimizer)
from utils import dotdict

log = logging.getLogger(__name__)

CURRENT_PLAYER = 1

mcts_args = dotdict({
    "numMCTSSims" : 10000,
    "cpuct" : 1/np.sqrt(2)
})

def executeEpisode(game, mcts, temp_threshold):

        trainExamples = []
        graph = game.getInitBoard()
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < temp_threshold)

            pi = mcts.getActionProb(graph, temp=temp)

            action = np.random.choice(len(pi), p=pi)
            graph, __ = game.getNextState(graph, CURRENT_PLAYER, action)

            r = game.getGameEnded(graph, CURRENT_PLAYER)

            trainExamples.append([graph, CURRENT_PLAYER, pi, r])
            if r != 0:
                return tuple(trainExamples)

def main():

    rule_vocab, __ = re.init_extension_rules()

    # List of weights for each criterion (force, time, COG)
    WEIGHT = [5, 5, 2]

    # At least 20 iterations are needed for good results
    cfg = ConfigRewardFunction()
    cfg.bound = (-7, 7)
    cfg.iters = 3
    cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
    cfg.time_step = 0.005
    cfg.time_sim = 2
    cfg.flags = [FlagMaxTime(cfg.time_sim), FlagNotContact(1), FlagSlipout(0.5, 0.5)]
    """Wraps function call"""

    criterion_callback = create_grab_criterion_fun(WEIGHT)
    traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

    cfg.criterion_callback = criterion_callback
    cfg.get_rgab_object_callback = get_object_to_grasp
    cfg.params_to_timesiries_callback = traj_generator_fun

    control_optimizer = ControlOptimizer(cfg)

    log.info('Loading %s ...', Game.__name__)
    graph_game = Game(rule_vocab, control_optimizer, 10)
    
    log.info('Loading %s ...', AlphaZeroWrapper.__name__)
    nnet = AlphaZeroWrapper(graph_game)
    
    mcts_searcher = MCTS(graph_game, nnet, mcts_args)
    examples = executeEpisode(graph_game, mcts_searcher, 15)
    None


if __name__ == "__main__":
    main()