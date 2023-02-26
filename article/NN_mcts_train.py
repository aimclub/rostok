from copy import deepcopy
import logging
import pickle
import time

import numpy as np
import torch


from rostok.graph_generators.graph_game import GraphGrammarGame as Game
from rostok.neural_network.wrappers import AlphaZeroWrapper
from MCTS import MCTS
from Coach import Coach


from obj_grasp.objects import get_obj_easy_box, get_object_to_grasp_sphere
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
from utils import dotdict

import hyperparameters as hp
import optmizers_config
# from rule_sets import rule_extention, rule_extention_graph
# from rule_sets.ruleset_old_style_graph import create_rules
from rule_sets.ruleset_old_style_graph_nonails import create_rules

log = logging.getLogger(__name__)

CURRENT_PLAYER = 1

coach_args = dotdict({
    "numMCTSSims" : 100,
    "cpuct" : 5,
    "train_offline_epochs": 100,
    "train_online_epochs":10,
    "num_learn_epochs": 100,
    "tempThreshold":4,
    "maxlenOfQueue":20000,
    "numItersForTrainExamplesHistory":20,
    "offline_iters":200,
    "online_iters":10,
    "update_weights":5,
    "checkpoint": "./temp/"
})


args_train = dotdict({
    "batch_size": 2,
    "cuda": torch.cuda.is_available(),
    "nhid": 512,
    "pooling_ratio": 0.7,
    "dropout_ratio": 0.3
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
            trainExamples.append((graph, pi, None))

            r = game.getGameEnded(graph, CURRENT_PLAYER)

            if r != 0:
                return [(x[0],x[1], r) for x in trainExamples]

def preconfigure():
    # List of weights for each criterion (force, time, COG)
    WEIGHT = hp.CRITERION_WEIGHTS

    # At least 20 iterations are needed for good results
    rule_vocabul, torque_dict = create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)

    cfg.get_rgab_object_callback = get_object_to_grasp_sphere
    control_optimizer = ControlOptimizer(cfg)
    graph_game = Game(rule_vocabul, control_optimizer, hp.MAX_NUMBER_RULES)
    
    return graph_game

def main():

    graph_game = preconfigure()
    examples = [[] for __ in range(coach_args.epochs)]
    for epoch in range(coach_args.epochs):
        log.info('Loading %s ...', Game.__name__)

        log.info('Loading %s ...', AlphaZeroWrapper.__name__)
        nnet = AlphaZeroWrapper(graph_game)

        mcts_searcher = MCTS(graph_game, nnet, coach_args)
    
        start = time.time()
        examples[epoch] = executeEpisode(graph_game, mcts_searcher, 15)
        ex = time.time() - start
        print(f"epoch: {epoch:3}, time: {ex}")
    struct = time.localtime(time.time())
    str_time = time.strftime('%d%m%Y%H%M', struct)
    name_file = f"train_mcts_data_{coach_args.epochs}e_{coach_args.cpuct:f}c_{coach_args.numMCTSSims}mcts_{hp.MAX_NUMBER_RULES}rule_{str_time}t.pickle" 
    with open(name_file, "wb+") as file:
        pickle.dump(examples, file)

def load_train_data(path_to_data):
    with open(path_to_data, "rb") as input_file:
        train = pickle.load(input_file)
    
    formatting_train_data = [[]+list(example) for episode in train for example in episode]
    formatting_train_data = list(map(lambda x: (x[0], x[2], x[3]), formatting_train_data))
    return train

def pretrain_model(path_to_data, list_name_data):
    
    for name_data in list_name_data:
        train_examples = load_train_data(path_to_data+name_data+".pickle")
        graph_game = preconfigure()
        
        log.info('Loading %s ...', AlphaZeroWrapper.__name__)
        nnet = AlphaZeroWrapper(graph_game, args_train)
        
        nnet.train(train_examples)
    
    nnet.save_checkpoint()
if __name__ == "__main__":
    # for idx in range(10):    
    #     initial_time = time.time()
    #     main()
    #     final_ex = time.time() - initial_time
    #     print(f"train {idx} index, full_time: {final_ex}")

    # load_train("train_data_10e_1000mcts_2302.pickle")
    game_graph = preconfigure()
    graph_nnet = AlphaZeroWrapper(game_graph, args_train)
    graph_nnet.load_checkpoint("temp_GraphControl_sphere", "best.pth.tar")
    nnmcts = MCTS(game_graph, graph_nnet, coach_args)
    nnmcts.search()
    # coacher = Coach(game_graph, graph_nnet, coach_args)
    # coacher.learn()