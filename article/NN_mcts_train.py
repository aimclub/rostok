from copy import deepcopy
import logging
import pickle
import time

import numpy as np

from rostok.graph_generators.graph_game import GraphGrammarGame as Game
from rostok.neural_network.wrappers import AlphaZeroWrapper
from MCTS import MCTS


from obj_grasp.objects import get_obj_easy_box
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
from utils import dotdict

import hyperparameters as hp
import optmizers_config
from rule_sets import rule_extention, rule_extention_graph

log = logging.getLogger(__name__)

CURRENT_PLAYER = 1

mcts_args = dotdict({
    "numMCTSSims" : 10,
    "cpuct" : 1/np.sqrt(2),
    "epochs": 10
})

def executeEpisode(game, mcts, temp_threshold):

        trainExamples = []
        graph = game.getInitBoard()
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = 1#int(episodeStep < temp_threshold)

            pi = mcts.getActionProb(graph, temp=temp)

            action = np.random.choice(len(pi), p=pi)
            graph, __ = game.getNextState(graph, CURRENT_PLAYER, action)

            r = game.getGameEnded(graph, CURRENT_PLAYER)

            trainExamples.append([graph, CURRENT_PLAYER, pi, r])
            if r != 0:
                return tuple(trainExamples)

def main():
    # List of weights for each criterion (force, time, COG)
    WEIGHT = hp.CRITERION_WEIGHTS

    # At least 20 iterations are needed for good results
    cfg = optmizers_config.get_cfg_standart()
    rule_vocabul = deepcopy(rule_extention_graph.rule_vocab)

    cfg.get_rgab_object_callback = get_obj_easy_box
    control_optimizer = ControlOptimizer(cfg)
    graph_game = Game(rule_vocabul, control_optimizer, hp.MAX_NUMBER_RULES)

    examples = [[] for __ in range(mcts_args.epochs)]
    for epoch in range(mcts_args.epochs):
        log.info('Loading %s ...', Game.__name__)

        log.info('Loading %s ...', AlphaZeroWrapper.__name__)
        nnet = AlphaZeroWrapper(graph_game)

        mcts_searcher = MCTS(graph_game, nnet, mcts_args)
    
        start = time.time()
        examples[epoch] = executeEpisode(graph_game, mcts_searcher, 15)
        ex = time.time() - start
        print(f"epoch: {epoch:3}, time: {ex}")
        
    with open("train_data.pickle", "wb+") as file:
        pickle.dump(examples, file)


if __name__ == "__main__":
    initial_time = time.time()
    main()
    final_ex = time.time() - initial_time
    print(f"full_time :{final_ex}")